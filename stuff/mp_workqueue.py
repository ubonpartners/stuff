import sys
import io
import multiprocessing as mp
import time
from multiprocessing import Process, Queue
from tqdm.auto import tqdm
import traceback
import logging

class StdoutProxy(io.TextIOBase):
    """
    Redirects stdout prints from a worker process into a multiprocessing queue,
    so that the main process can display them properly.
    """
    def __init__(self, queue, worker_id):
        self.queue = queue
        self.worker_id = worker_id

    def write(self, msg):
        if msg.strip():  # skip pure newlines
            self.queue.put(("print", self.worker_id, msg))

    def flush(self):
        pass  # Needed to be a full file-like object

def mpwq_progress(mpwq_context, desc=None, total=None, update=None):
    """
    Allows workers to report progress back to the main process, either by updating
    a running total or resetting the progress bar description and maximum.
    """
    if update is not None:
        mpwq_context["accum_update"]+=update
    if desc is None and total is None:
        t=time.time()
        if t-mpwq_context["last_update"]<mpwq_context["min_update_interval"]:
            return
        mpwq_context["last_update"]=t
    mpwq_context["result_queue"].put(("progress", mpwq_context["worker_id"], (desc, total, mpwq_context["accum_update"])))
    mpwq_context["accum_update"]=0

def mpwq_worker_fn(work_queue, result_queue, quit_queue, worker_id, worker_fn,min_update_interval=0.1):
    """
    Core worker function that:
    - Redirects stdout to a queue
    - Fetches tasks from the work queue
    - Executes the given worker function
    - Reports results or exceptions back to the main process
    """
    # redirect stdout to the result queue
    # This is needed to capture print statements from the worker
    # and send them to the main process
    # so they can be printed to the console

    logging.debug(f"Worker {worker_id} started")
    sys.stdout = StdoutProxy(result_queue, worker_id)

    mpwq_context={"result_queue": result_queue,
                  "worker_id": worker_id,
                  "last_update":time.time(),
                  "accum_update":0,
                  "min_update_interval": min_update_interval}
    while True:
        try:
            # Get a job from the work queue
            work_item = work_queue.get(timeout=30)
            if work_item is None:
                # None indicates no more jobs
                break
            # Perform the task
            result = worker_fn(work_item, mpwq_context=mpwq_context, mpwq_progress_fn=mpwq_progress)
            # Put the result in the results queue
            result_queue.put(("result", worker_id, result))
        except Exception as e:
            error_info = traceback.format_exc()
            result_queue.put(("exception", worker_id, error_info))
            # Break the loop if queue is empty and timeout occurs
            break

    result_queue.put(("progress", worker_id, ("Done!", 1, None)))
    _ = quit_queue.get(timeout=600)

def mp_workqueue_run(work_to_run, worker_fn,
                     num_workers=1,
                     desc="mp_work",
                     min_update_interval=0.2,
                     result_callback_context=None,
                     result_callback=None):
    """
    Launches multiple worker processes to execute work items concurrently.
    - Sets up communication queues
    - Collects results
    - Displays progress bars per worker
    - Handles output and exceptions
    """
    log=logging.getLogger('mp_workqueue')

    mp.set_start_method('spawn', force=True)


    log.debug(f"Running {len(work_to_run)} tests... with {num_workers} workers")
    output_results=[]
    with tqdm(total=len(work_to_run),
              desc=desc,
              colour="#0000ff",
              position=0,
              leave=True,
              dynamic_ncols=True) as pbar:

        work_queue = Queue()
        result_queue = Queue()
        quit_queue = Queue()
        for item in work_to_run:
            work_queue.put(item)
        for _ in range(num_workers):
            work_queue.put(None)

        prev_stdout=sys.stdout
        #sys.stdout = StdoutProxy(result_queue, -1)
    # Create a list of worker processes
        workers = []
        pbars=[]
        for i in range(num_workers):
            c=30+(200*i)//num_workers
            pbars.append(tqdm(total=100,
              desc=f"{i:02d}: {'Starting....':31s}",
              colour=f"#e0{c:02x}00",
              position=i+1,
              leave=True, dynamic_ncols=True))

            p = Process(target=mpwq_worker_fn,
                        args=(work_queue, result_queue, quit_queue, i, worker_fn, min_update_interval))
            p.start()
            workers.append(p)

        num_results_got=0
        last_pbar_refresh_time=time.time()
        while num_results_got<len(work_to_run):
            # handle messages from worker processes
            # this is a blocking call, so it will wait for a message
            msg=result_queue.get(timeout=300)

            if msg[0] == "print":
                _, worker_id, log_message = msg
                tqdm.write(f"[Worker {worker_id}] {log_message}\n", end="")
                continue

            if msg[0]=="result":
                output_results.append(msg[2])
                if result_callback is not None:
                    result_callback(result_callback_context, msg[2])
                num_results_got+=1
                pbar.update(1)
                continue

            if msg[0]=="exception":
                print(f"Process exception {msg[2]} from worker {msg[1]}")
                exit()

            if msg[0]=="progress":
                # progress message is used to update TQDM progress bar
                # we need to do this in the main process as its impossible
                # to get it working properly for all the processes to try
                # to write to stdout
                worker_id, (desc, total, update) = msg[1], msg[2]
                if desc is not None:
                    pbars[worker_id].reset(total=total)
                    pbars[worker_id].set_description(desc)
                    pbars[worker_id].refresh()
                if update is not None:
                    pbars[worker_id].update(update)
                t=time.time()
                if t-last_pbar_refresh_time>1:
                    for p in pbars:
                        p.refresh()
                    last_pbar_refresh_time=t
                continue

            assert False, f"Unknown message type {msg[0]} from worker {msg[1]}"

        log.debug("All work done, waiting for workers to finish")
        for _ in range(num_workers):
            quit_queue.put(None)

        for p in pbars:
            p.refresh()

        for p in workers:
            p.join()
        pbar.close()

        sys.stdout = prev_stdout
        log.debug("Done")
    return output_results

def test_work_fn(work_item, mpwq_context, mpwq_progress_fn=None):
    """
    Example worker function that:
    - Pretends to process a work item in 10 steps
    - Occasionally prints debug messages
    - Updates the main process about progress
    """
    mpwq_progress_fn(mpwq_context, desc=work_item, total=10)
    time.sleep(0.1)
    for i in range(10):
        if i==5:
            print(f" test print {work_item}")
        time.sleep(0.05)
        mpwq_progress_fn(mpwq_context, update=1)
    return "Fish"

def test_work():
    """
    Test harness for the multiprocessing workqueue framework.
    Runs 40 fake work items using 8 worker processes.
    """
    work_to_run = []
    for i in range(40):
        work_to_run.append(f"Work item {i}")
    results = mp_workqueue_run(work_to_run, test_work_fn, num_workers=8)
    print("Results:", len(results), results)
