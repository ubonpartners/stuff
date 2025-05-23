import ultralytics
import concurrent.futures
import torch
import copy
import cv2
import stuff.ultralytics as ultralytics_stuff
from ensemble_boxes import weighted_boxes_fusion

def merge_detections_wbf(detectors_outputs, iou_thr=0.55, skip_box_thr=0.001):
    all_boxes = []
    all_scores = []
    all_labels = []

    for detections in detectors_outputs:
        boxes = []
        scores = []
        labels = []
        for det in detections:
            boxes.append(det["box"])
            scores.append(det["confidence"])
            labels.append(det["class"])
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr
    )

    # Build output detections
    merged_detections = []
    for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
        merged_detections.append({
            "box": box,
            "confidence": score,
            "class": int(label)
        })

    return merged_detections

def ensemble_merge_results(results):
    num=len(results[0])
    out=[]
    for i in range(num):
        dets=[]
        for r in results:
            dets.append(r[i])
        out.append(merge_detections_wbf(dets))
    return out

class ultralytics_wrapper:
    class_synonyms={'person':['man','woman','boy','girl'],
                    'vehicle':['car','bicycle','motorcycle','train','truck','bus','airplane','boat'],
                    'animal':['cat','dog','horse','sheep','cow','bird','elephant','bear','zebra','giraffe'],
                    'weapon':['gun', 'dagger', 'pistol', 'handgun', 'rifle', 'revolver'],
                    'face':[]}

    def __init__(self,
                 model_name,
                 class_names=None,
                 thr=0.5,
                 nms_iou=0.45,
                 half=True,
                 rect=True,
                 max_det=400,
                 batch_size=32,
                 imgsz=640,
                 attributes=False,
                 yolo_add_faces=False,
                 face_kp=False,
                 pose_kp=False,
                 facepose_kp=False,
                 fold_attributes=False):

        self.ensemble_models=None

        if model_name=="ensemble2":
            model_name=["/mldata/weights/good/yolo11l.pt:960",
                        "/mldata/weights/good/yolo12l.pt:640",
                        "/mldata/weights/good/yolov9e.pt:864",
                        "/mldata/weights/good/yolov5x6u.pt:1280"]
        if model_name=="ensemble3":
            model_name=["/mldata/weights/good/yolov9e.pt:512",
                        "/mldata/weights/good/yolov9e.pt:640,flip-lr",
                        "/mldata/weights/good/yolov9e.pt:864",
                        "/mldata/weights/good/yolov9e.pt:960,flip-lr"]


        if isinstance(model_name, list):
            self.ensemble_models=[]
            for m in model_name:
                yolo=ultralytics_wrapper(m,
                                         class_names=class_names,
                                         thr=thr,
                                         nms_iou=nms_iou,
                                         max_det=max_det,
                                         rect=rect,
                                         imgsz=imgsz,
                                         batch_size=batch_size//2)
                self.ensemble_models.append(yolo)
            self.num_threads=self.ensemble_models[0].num_threads
            self.yolo_batch_size=self.ensemble_models[0].yolo_batch_size
            self.yolo_num_params=sum([x.yolo_num_params for x in self.ensemble_models])
            self.yolo_num_flops=sum([x.yolo_num_flops for x in self.ensemble_models])
            self.yolo_cache={}
            return

        aug_params={}
        if ":" in model_name:
            x=model_name.split(":")
            model_name=x[0]
            params=[x[1]]
            if "," in x[1]:
                params=x[1].split(",")
            for p in params:
                if p=="flip-lr":
                    aug_params["flip-lr"]=True
                else:
                    imgsz=int(p)

        self.ensemble_models=None
        self.model=self.load_ultralytics_model(model_name)
        self.num_gpus=torch.cuda.device_count()
        self.num_threads=self.num_gpus
        self.yolo=[None]*self.num_threads
        self.model_params=[aug_params]*self.num_threads
        self.yolo_det_conf=thr
        self.yolo_nms_iou=nms_iou
        self.yolo_half=half
        self.yolo_rect=rect
        self.yolo_max_det=max_det
        self.yolo_batch_size=batch_size
        self.attributes=attributes
        self.yolo_add_faces=yolo_add_faces
        self.face_kp=face_kp
        self.pose_kp=pose_kp
        self.facepose_kp=facepose_kp
        self.fold_attributes=fold_attributes

        self.imgsz=imgsz
        self.yolo_cache={}

        for i in range(self.num_threads):
            self.yolo[i]=self.load_ultralytics_model(model_name)
            self.yolo[i]=self.yolo[i].to("cuda:"+str(i))
            assert(self.yolo[i] is not None)

        self.yolo_class_names=[self.yolo[0].names[i] for i in range(len(self.yolo[0].names))]
        self.det_class_remap=[-1]*len(self.yolo_class_names)

        self.class_names=class_names
        if self.class_names is None:
            self.class_names=self.yolo_class_names
        self.can_detect=[False]*len(self.class_names)

        # map the detected classes back to our class set
        # assume if our class set has things like 'vehicle' then we would want any standard
        # coco classes like 'car' to map to that

        for i,x in enumerate(self.yolo_class_names):
            if x in self.class_names:
                self.det_class_remap[i]=self.class_names.index(x)
                self.can_detect[self.class_names.index(x)]=True
            else:
                for y in self.class_synonyms:
                    if y in self.class_names:
                        if x in self.class_synonyms[y]:
                            self.det_class_remap[i]=self.class_names.index(y)
                            self.can_detect[self.class_names.index(y)]=True

    def load_ultralytics_model(self, name):
        task="detect"
        if "world" in name:
            extended_classes=copy.deepcopy(self.class_names)
            for c in self.class_names:
                if c in self.class_synonyms:
                    for s in self.class_synonyms[c]:
                        if not s in extended_classes:
                            extended_classes.append(s)

            model=ultralytics.YOLOWorld(name, verbose=False)
            model.set_classes(extended_classes)
            return model
        if "yoloe" in name:
            model=ultralytics.YOLOE(name, task=task, verbose=False)
            model.set_classes(self.class_names, model.get_text_pe(self.class_names))
        if "nas" in name:
            return ultralytics.NAS(name)
        if "rtdetr" in name:
            return ultralytics.RTDETR(name)
        if "pose" in name or "face" in name or "full" in name or "attributes" in name or "dpa" in name:
            task="pose"
        if "dpar" in name:
            task="posereid"

        model=ultralytics.YOLO(name, task=task, verbose=False)
        info=model.info(verbose=False)

        self.yolo_num_params=0
        self.yolo_num_flops=0
        if info is not None:
            self.yolo_num_params=info[1]
            self.yolo_num_flops=info[3]

        #try:
        #    self.yolo_num_params = sum(p.numel() for p in self.yolo[0].model.parameters())
        #except AttributeError:
        #    self.yolo_num_params = 0
        return model

    def infer(self, input_frames, thr=None):

        if self.ensemble_models is not None:
            results=[]
            for m in self.ensemble_models:
                results.append(m.infer(input_frames, thr=thr))
            return ensemble_merge_results(results)

        def process_batch(yolo_fn, frames, params):
            if params is not None:
                if params.get("flip-lr", False):
                    out_frames=[]
                    for f in frames:
                        if isinstance(f, str):
                            f = cv2.imread(f)
                        f=cv2.flip(f, 1)
                        out_frames.append(f)
                    frames=out_frames
            return yolo_fn(
                frames,
                conf=self.yolo_det_conf,
                iou=self.yolo_nms_iou,
                max_det=self.yolo_max_det,
                agnostic_nms=False,
                half=self.yolo_half,
                imgsz=self.imgsz,
                verbose=False,
                rect=self.yolo_rect
            )

        results = []
        index = 0
        if thr is not None:
            self.yolo_det_conf=thr

        if self.num_threads == 1:
            while index < len(input_frames):
                end = min(len(input_frames), index + self.yolo_batch_size)
                batch = input_frames[index:end]
                results += process_batch(self.yolo[0], batch, self.model_params[0])
                index = end
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                while index < len(input_frames):
                    futures = []
                    for i in range(self.num_threads):
                        end = min(len(input_frames), index + self.yolo_batch_size)
                        if index < end:
                            batch = input_frames[index:end]
                            futures.append(executor.submit(process_batch, self.yolo[i], batch, self.model_params[i]))
                        index = end
                    for future in futures:
                        results += future.result()

        ret=[]
        for r in results:
            out_det=ultralytics_stuff.yolo_results_to_dets(r,
                                             det_thr=self.yolo_det_conf,
                                             det_class_remap=self.det_class_remap,
                                             yolo_class_names=self.yolo_class_names,
                                             class_names=self.class_names,
                                             attributes=self.attributes,
                                             add_faces=False,
                                             face_kp=self.face_kp,
                                             pose_kp=self.pose_kp,
                                             facepose_kp=self.facepose_kp,
                                             fold_attributes=self.fold_attributes)

            if self.model_params[0] is not None:
                if self.model_params[0].get("flip-lr", False):
                    for d in out_det:
                        d["box"]=[1.0-d["box"][0],d["box"][1],1.0-d["box"][2],d["box"][3]]
            ret.append(out_det)
        return ret

    def infer_cached(self, index, num_images, get_image_fn, thr=None):
        if index in self.yolo_cache:
            return self.yolo_cache[index]
        self.yolo_cache={}
        num=self.yolo_batch_size*self.num_threads
        num=min(num, num_images-index)
        image_list=[get_image_fn(i+index) for i in range(num)]
        batch_results=self.infer(image_list, thr=thr)
        self.yolo_cache={(i+index):batch_results[i] for i in range(num)}
        assert index in self.yolo_cache
        return self.yolo_cache[index]