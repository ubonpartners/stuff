"""Local isolated workspaces with asynchronous command execution.

A *workspace* is an isolated working directory + environment + resource/policy limits, in which you
run shell command lines as child processes — `python3 ...`, a custom executable, a pipeline — and
then tear it all down cleanly. It is a small, **local, non-cloud** primitive (no scheduler, no
object store, no credential broker), stdlib-only, running on **x86/arm Linux or macOS**.

Threat model: the scripts run here are FIRST-PARTY (operator/agent-authored under the operator's
control). The goal is to contain *accidents*, bound *blast radius*, guarantee *cleanup*, and offer
optional *egress cut* + an *audit trail* — not to defend against determined hostile code escaping a
sandbox. (See docs/workspace.md.)

Isolation is tiered and degrades honestly; the chosen tier is recorded on the workspace and every
result, so a run that fell back to a weaker tier is visible in the logs rather than masquerading:

  * ``none``     — subprocess in the ws dir, own session/process-group, ``setrlimit`` caps,
                   scrubbed env. Portable (Linux + macOS). Resource caps + cleanup, no spatial
                   isolation.
  * ``bwrap``    — Linux only: wrap argv in **bubblewrap** (unprivileged user namespaces): ws dir
                   read-write, system paths read-only, private /tmp, optional ``--unshare-net``.
  * ``seatbelt`` — macOS only: wrap argv in **sandbox-exec** with a generated profile (deny
                   file-write outside the ws, optionally deny network).

On Linux a per-workspace **cgroup v2** (when a delegated/writable cgroup is available) adds HARD
memory/pids/cpu caps that rlimits can't give (a fork bomb evades ``RLIMIT_AS``); absent that, rlimits
are the best-effort fallback.

Quick start::

    from stuff.workspace import WorkspaceManager, Policy

    mgr = WorkspaceManager(root="/var/lib/ubon/workspaces")
    ws  = mgr.create(policy=Policy(network=False, wall_seconds=60), ttl=600)

    h = ws.exec("python3 -c 'print(2+2)'")   # returns immediately (async)
    res = h.wait()                            # block until done -> ExecResult
    print(res.returncode, res.stdout_text())

    mgr.destroy(ws.id)                         # full, idempotent teardown

Interactive (PTY) execution::

    h = ws.exec(["bash", "-i"], tty=True)     # a real pseudo-terminal
    h.send(b"echo hi\n"); ...; print(h.read())
    # or, from a controlling terminal, bridge stdin/stdout to it:
    from stuff.workspace import attach
    attach(h)                                  # Ctrl-] to detach
"""
from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import platform
import select
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tarfile
import termios
import threading
import time
import tty as _tty
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

try:
    import resource as _resource  # POSIX only (Linux + macOS); absent on Windows
except ImportError:  # pragma: no cover - we don't target Windows
    _resource = None

_SYSTEM = platform.system()  # "Linux" | "Darwin" | ...
IS_LINUX = _SYSTEM == "Linux"
IS_MAC = _SYSTEM == "Darwin"

Command = Union[str, Sequence[str]]
DoneCallback = Callable[["ExecResult"], None]

# Default read-only system paths bound into a bwrap sandbox (only those that exist are bound).
_DEFAULT_RO_BINDS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/lib32", "/etc", "/opt")


# ---------------------------------------------------------------------------
# Policy + result types
# ---------------------------------------------------------------------------
@dataclass
class Policy:
    """How a workspace's commands are isolated and limited. All fields optional; the defaults are a
    sensible *contain-accidents* posture for first-party scripts."""
    sandbox: str = "auto"                 # "auto" | "none" | "bwrap" | "seatbelt"
    network: bool = True                  # False -> cut egress (sandbox tiers only)
    allow_cmds: Optional[set] = None      # None = any; else argv[0] basename must be in the set
    env: Optional[dict] = None            # explicit env; None -> a scrubbed minimum (see _build_env)
    env_passthrough: tuple = ("PATH", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR")
    cpu_seconds: Optional[int] = None     # RLIMIT_CPU (cumulative CPU time)
    mem_bytes: Optional[int] = None       # cgroup memory.max (Linux) else RLIMIT_AS
    wall_seconds: Optional[float] = None  # parent-side timeout -> kill the process tree
    pids: Optional[int] = None            # cgroup pids.max (Linux) else RLIMIT_NPROC
    file_bytes: Optional[int] = None      # RLIMIT_FSIZE (max single file size written)
    nofile: Optional[int] = None          # RLIMIT_NOFILE
    max_output: int = 64 * 1024           # per-stream inline cap; beyond this output SPILLS to a file
    ro_binds: tuple = _DEFAULT_RO_BINDS   # bwrap read-only binds
    rw_binds: tuple = ()                  # extra writable binds beyond the ws dir
    cgroup: bool = True                   # Linux: use a per-ws cgroup v2 for hard caps when available


@dataclass
class Output:
    """A captured stream. ``inline`` holds up to Policy.max_output bytes; if the stream was larger it
    SPILLS to ``ref`` (a file path) — full content is never silently truncated."""
    inline: bytes = b""
    ref: Optional[str] = None
    spilled: bool = False
    total_bytes: int = 0

    def text(self, encoding: str = "utf-8") -> str:
        if self.spilled and self.ref:
            return Path(self.ref).read_text(encoding, errors="replace")
        return self.inline.decode(encoding, errors="replace")

    def bytes(self) -> bytes:
        if self.spilled and self.ref:
            return Path(self.ref).read_bytes()
        return self.inline


@dataclass
class ExecResult:
    returncode: int
    stdout: Output
    stderr: Output
    duration: float
    timed_out: bool = False
    killed: bool = False
    tier: str = "none"
    seq: int = 0

    def stdout_text(self, **kw) -> str:
        return self.stdout.text(**kw)

    def stderr_text(self, **kw) -> str:
        return self.stderr.text(**kw)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.killed


# ---------------------------------------------------------------------------
# Resource limits (rlimit) — applied in the child between fork and exec
# ---------------------------------------------------------------------------
def _make_preexec(policy: "Policy", use_cgroup_for_mem_pids: bool, tty: bool):
    """Build a preexec_fn that runs in the child: sets rlimits and (for a tty) makes the pty the
    controlling terminal. ``start_new_session=True`` already does setsid()."""
    if _resource is None:
        return None

    def _preexec():
        def setl(res, val):
            try:
                soft, hard = _resource.getrlimit(res)
                hard = val if hard == _resource.RLIM_INFINITY else min(hard, val)
                _resource.setrlimit(res, (val, hard))
            except (ValueError, OSError):
                pass
        if policy.cpu_seconds:
            setl(_resource.RLIMIT_CPU, int(policy.cpu_seconds))
        if policy.file_bytes:
            setl(_resource.RLIMIT_FSIZE, int(policy.file_bytes))
        if policy.nofile:
            setl(_resource.RLIMIT_NOFILE, int(policy.nofile))
        # When a cgroup is enforcing mem/pids we DON'T also set the blunter rlimits (RLIMIT_AS
        # caps virtual address space, which trips many runtimes spuriously). Without a cgroup,
        # rlimits are the only lever we have.
        if not use_cgroup_for_mem_pids:
            if policy.mem_bytes:
                setl(_resource.RLIMIT_AS, int(policy.mem_bytes))
            if policy.pids and hasattr(_resource, "RLIMIT_NPROC"):
                # NOTE: RLIMIT_NPROC is per real-uid (not per-session), so it's a coarse, shared
                # limit — fine as a mac fallback, but cgroup pids.max is the correct tool on Linux.
                setl(_resource.RLIMIT_NPROC, int(policy.pids))
        if tty:
            try:  # make the slave pty (now fd 0/1/2) the controlling terminal of the new session
                fcntl.ioctl(0, termios.TIOCSCTTY, 0)
            except OSError:
                pass

    return _preexec


# ---------------------------------------------------------------------------
# cgroup v2 (Linux) — hard memory/pids/cpu caps when a writable cgroup is available
# ---------------------------------------------------------------------------
class _CgroupV2:
    """Best-effort per-workspace cgroup v2. Works when the manager runs with a writable/delegated
    cgroup (e.g. as root, or under ``systemd-run --user --scope -p Delegate=yes``). Otherwise
    ``setup()`` returns None and callers fall back to rlimits — recorded, never silent.

    cgroup v2's no-internal-processes rule means a cgroup can't both hold processes and have
    controllers enabled on its subtree. So we move the manager process into a ``.manager`` leaf,
    enable controllers on our base cgroup, and create one ``ws-<id>`` leaf per workspace.
    """
    ROOT = Path("/sys/fs/cgroup")

    def __init__(self, base: Path):
        self.base = base  # our delegated base cgroup; ws leaves are created under it

    @classmethod
    def setup(cls) -> Optional["_CgroupV2"]:
        if not IS_LINUX:
            return None
        try:
            if not (cls.ROOT / "cgroup.controllers").exists():
                return None  # not a cgroup v2 mount
            # our cgroup, from the unified (0::) line of /proc/self/cgroup
            rel = ""
            for line in Path("/proc/self/cgroup").read_text().splitlines():
                if line.startswith("0::"):
                    rel = line[3:].strip()
                    break
            cur = cls.ROOT / rel.lstrip("/")
            if not cur.is_dir():
                return None
            # Re-nest guard: if we're ALREADY inside our own ".manager" leaf (a prior setup in this
            # process), reuse the original base instead of creating .manager/.manager/...
            base = cur.parent if cur.name == ".manager" else cur
            leaf = base / ".manager"
            leaf.mkdir(exist_ok=True)
            if cur != leaf:  # move the manager process into the leaf so the base can carry controllers
                (leaf / "cgroup.procs").write_text(str(os.getpid()))
            avail = (base / "cgroup.controllers").read_text().split()
            want = [c for c in ("memory", "pids", "cpu") if c in avail]
            if want:
                (base / "cgroup.subtree_control").write_text(" ".join("+" + c for c in want))
            return cls(base)
        except (OSError, PermissionError):
            return None

    def create(self, ws_id: str, policy: "Policy") -> Optional[Path]:
        try:
            cg = self.base / f"ws-{ws_id}"
            cg.mkdir(exist_ok=True)
            if policy.mem_bytes:
                (cg / "memory.max").write_text(str(int(policy.mem_bytes)))
            if policy.pids:
                (cg / "pids.max").write_text(str(int(policy.pids)))
            return cg
        except (OSError, PermissionError):
            return None

    @staticmethod
    def add(cg: Path, pid: int) -> None:
        try:
            (cg / "cgroup.procs").write_text(str(pid))
        except OSError:
            pass

    @staticmethod
    def destroy(cg: Optional[Path]) -> None:
        if not cg:
            return
        try:
            # kill any survivors, then rmdir (cgroup dirs are removed with rmdir once empty)
            if (cg / "cgroup.kill").exists():
                (cg / "cgroup.kill").write_text("1")
                time.sleep(0.05)
            cg.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Sandbox tier resolution + argv wrapping
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _bwrap_usable() -> bool:
    """True only if bwrap is present AND can actually create namespaces here. Probing matters:
    bwrap can be installed but unable to run where unprivileged user namespaces are blocked
    (inside another sandbox/container, or kernel.unprivileged_userns_clone=0)."""
    if not (IS_LINUX and shutil.which("bwrap")):
        return False
    try:
        r = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--unshare-user", "--", "/bin/true"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _resolve_tier(requested: str) -> str:
    """Map a requested sandbox to the tier actually used on this OS. Explicit unavailable tiers
    raise (degrade loudly); ``auto`` degrades to the best available and is stamped on the
    workspace/results so logs never overstate the protection."""
    if requested == "none":
        return "none"
    if requested == "bwrap":
        if not (IS_LINUX and shutil.which("bwrap")):
            raise RuntimeError("sandbox='bwrap' requires Linux with bubblewrap (bwrap) installed")
        if not _bwrap_usable():
            raise RuntimeError(
                "bwrap is installed but cannot create namespaces here — unprivileged user "
                "namespaces are likely blocked (running inside another sandbox/container, or "
                "kernel.unprivileged_userns_clone=0). Use sandbox='none' or enable userns.")
        return "bwrap"
    if requested == "seatbelt":
        if not (IS_MAC and shutil.which("sandbox-exec")):
            raise RuntimeError("sandbox='seatbelt' requires macOS with sandbox-exec")
        return "seatbelt"
    if requested == "auto":
        if _bwrap_usable():
            return "bwrap"
        if IS_MAC and shutil.which("sandbox-exec"):
            return "seatbelt"
        return "none"
    raise ValueError(f"unknown sandbox tier {requested!r}")


def _bwrap_prefix(ws: "Workspace") -> list:
    p = ws.policy
    argv = ["bwrap", "--die-with-parent", "--new-session",
            "--unshare-user", "--unshare-ipc", "--unshare-uts", "--unshare-pid",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    if not p.network:
        argv += ["--unshare-net"]
    for d in p.ro_binds:
        if Path(d).exists():
            argv += ["--ro-bind", d, d]
    for d in p.rw_binds:
        if Path(d).exists():
            argv += ["--bind", d, d]
    argv += ["--bind", str(ws.work_dir), str(ws.work_dir),
             "--chdir", str(ws.work_dir)]
    return argv


def _seatbelt_prefix(ws: "Workspace") -> list:
    """Generate a sandbox-exec profile: allow-by-default (first-party), deny file-write outside the
    ws dir, optionally deny network. Written to the ws so it's inspectable/auditable."""
    work = str(ws.work_dir)
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-write* (with no-report))",
        '(allow file-write*',
        f'    (subpath "{work}")',
        '    (subpath "/private/tmp")',
        '    (subpath "/private/var/folders")',
        '    (subpath "/dev"))',
    ]
    if not ws.policy.network:
        lines.append("(deny network*)")
    prof = ws.io_dir / "sandbox.sb"
    prof.write_text("\n".join(lines) + "\n")
    return ["sandbox-exec", "-f", str(prof)]


def _wrap_argv(ws: "Workspace", inner: list) -> list:
    if ws.tier == "bwrap":
        return _bwrap_prefix(ws) + ["--"] + inner
    if ws.tier == "seatbelt":
        return _seatbelt_prefix(ws) + inner
    return inner


# ---------------------------------------------------------------------------
# Stream capture (bounded, spill-to-file)
# ---------------------------------------------------------------------------
class _Capture:
    def __init__(self, path: Path, cap: int):
        self.path = path
        self.cap = cap
        self.buf = bytearray()
        self.total = 0
        self.spilled = False
        self.fh = None

    def feed(self, data: bytes) -> None:
        self.total += len(data)
        if not self.spilled:
            if len(self.buf) + len(data) <= self.cap:
                self.buf += data
                return
            self.spilled = True
            self.fh = open(self.path, "wb")
            self.fh.write(self.buf)
        self.fh.write(data)

    def close(self) -> None:
        if self.fh:
            self.fh.close()
            self.fh = None

    def output(self) -> Output:
        return Output(inline=bytes(self.buf), ref=str(self.path) if self.spilled else None,
                      spilled=self.spilled, total_bytes=self.total)


# ---------------------------------------------------------------------------
# Exec — an asynchronous command handle
# ---------------------------------------------------------------------------
class Exec:
    """A running (or finished) command in a workspace. Returned immediately by ``Workspace.exec``;
    the process runs regardless of whether the caller polls or waits.

    Inspect/await with ``poll()`` (non-blocking), ``wait(timeout)`` (blocking), ``when_done(cb)``
    (callback on completion), ``running``, ``returncode``. Stop with ``kill()``. For ``tty=True``,
    drive the terminal with ``send()`` / ``read()`` / ``resize()``.
    """

    def __init__(self, ws: "Workspace", cmd: Command, *, tty: bool = False,
                 env: Optional[dict] = None, timeout: Optional[float] = None,
                 stdin: Optional[bytes] = None, seq: int = 0,
                 on_output: Optional[Callable[[bytes], None]] = None):
        self.ws = ws
        self.tier = ws.tier
        self.seq = seq
        self.tty = tty
        # live output hook (called with each chunk as it's read) — e.g. to stream a PTY to a
        # websocket terminal. Output is still buffered/spilled for the ExecResult as well.
        self._on_output = on_output
        self.timed_out = False
        self.killed = False
        self._done = threading.Event()
        self._result: Optional[ExecResult] = None
        self._callbacks: list = []
        self._lock = threading.Lock()
        self._master_fd: Optional[int] = None
        self._t0 = time.monotonic()

        argv, shell = self._normalize(cmd)
        if ws.policy.allow_cmds is not None:
            if shell:
                raise ValueError("allow_cmds is set: pass an argv list, not a shell string "
                                 "(a shell line can't be vetted against the allowlist)")
            base = os.path.basename(argv[0])
            if base not in ws.policy.allow_cmds:
                raise PermissionError(f"command {base!r} not in workspace allowlist")
        inner = ["/bin/sh", "-c", cmd] if shell else list(argv)
        full = _wrap_argv(ws, inner)
        run_env = ws._build_env(env)
        use_cg = bool(ws.cgroup_path)
        preexec = _make_preexec(ws.policy, use_cg, tty)

        out_path = ws.io_dir / f"{seq:04d}.out"
        err_path = ws.io_dir / f"{seq:04d}.err"
        self._cap_out = _Capture(out_path, ws.policy.max_output)
        self._cap_err = _Capture(err_path, ws.policy.max_output)
        self._readers: list = []

        if tty:
            self._launch_tty(full, run_env, preexec)
        else:
            self._launch_pipes(full, run_env, preexec, stdin)

        if use_cg:
            _CgroupV2.add(ws.cgroup_path, self.proc.pid)

        wall = timeout if timeout is not None else ws.policy.wall_seconds
        self._timer = threading.Timer(wall, self._on_timeout) if wall else None
        if self._timer:
            self._timer.daemon = True
            self._timer.start()

        self._waiter = threading.Thread(target=self._reap, daemon=True)
        self._waiter.start()

    # -- command normalization ------------------------------------------------
    @staticmethod
    def _normalize(cmd: Command):
        if isinstance(cmd, str):
            return shlex.split(cmd), True   # run via /bin/sh -c (pipes/redirects work)
        return list(cmd), False

    # -- launch ---------------------------------------------------------------
    def _launch_pipes(self, full, env, preexec, stdin):
        self.proc = subprocess.Popen(
            full, cwd=str(self.ws.work_dir), env=env,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, preexec_fn=preexec, close_fds=True,
        )
        if stdin is not None:
            threading.Thread(target=self._feed_stdin, args=(stdin,), daemon=True).start()
        for fh, cap in ((self.proc.stdout, self._cap_out), (self.proc.stderr, self._cap_err)):
            t = threading.Thread(target=self._drain, args=(fh.fileno(), cap), daemon=True)
            t.start()
            self._readers.append(t)

    def _launch_tty(self, full, env, preexec):
        master, slave = os.openpty()
        self._master_fd = master
        try:
            self.proc = subprocess.Popen(
                full, cwd=str(self.ws.work_dir), env=env,
                stdin=slave, stdout=slave, stderr=slave,
                start_new_session=True, preexec_fn=preexec, close_fds=True,
            )
        finally:
            os.close(slave)
        t = threading.Thread(target=self._drain, args=(master, self._cap_out), daemon=True)
        t.start()
        self._readers.append(t)

    def _feed_stdin(self, data: bytes):
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.close()
        except OSError:
            pass

    def _drain(self, fd: int, cap: "_Capture"):
        while True:
            try:
                data = os.read(fd, 65536)
            except OSError:  # EIO when a pty's child exits
                break
            if not data:
                break
            cap.feed(data)
            if self._on_output is not None:
                try:
                    self._on_output(data)
                except Exception:
                    pass

    # -- completion -----------------------------------------------------------
    def _reap(self):
        self.proc.wait()
        for t in self._readers:
            t.join()
        self._cap_out.close()
        self._cap_err.close()
        if self._timer:
            self._timer.cancel()
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None
        result = ExecResult(
            returncode=self.proc.returncode,
            stdout=self._cap_out.output(), stderr=self._cap_err.output(),
            duration=time.monotonic() - self._t0,
            timed_out=self.timed_out, killed=self.killed, tier=self.tier, seq=self.seq,
        )
        with self._lock:
            self._result = result
            cbs = list(self._callbacks)
        self.ws._record(self, result)
        self._done.set()
        for cb in cbs:
            try:
                cb(result)
            except Exception:
                pass

    def _on_timeout(self):
        self.timed_out = True
        self.kill()

    # -- public API -----------------------------------------------------------
    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def running(self) -> bool:
        return not self._done.is_set()

    def poll(self) -> Optional[int]:
        """Non-blocking: returncode if finished, else None."""
        return self._result.returncode if self._done.is_set() else None

    def wait(self, timeout: Optional[float] = None) -> Optional[ExecResult]:
        """Block until done (returns ExecResult) or until ``timeout`` elapses (returns None)."""
        if self._done.wait(timeout):
            return self._result
        return None

    @property
    def returncode(self) -> Optional[int]:
        return self._result.returncode if self._done.is_set() else None

    def result(self) -> Optional[ExecResult]:
        return self._result

    def when_done(self, cb: DoneCallback) -> "Exec":
        """Register a callback fired (once) with the ExecResult when the command completes. If it's
        already done, the callback fires immediately on the calling thread."""
        with self._lock:
            if self._result is None:
                self._callbacks.append(cb)
                return self
            res = self._result
        cb(res)
        return self

    def kill(self, grace: float = 3.0) -> None:
        """Terminate the whole process tree: SIGTERM to the group, then SIGKILL after ``grace``."""
        self.killed = True
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return

        def _hard():
            if self.running:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
        t = threading.Timer(grace, _hard)
        t.daemon = True
        t.start()

    # -- interactive (tty) ----------------------------------------------------
    def send(self, data: bytes) -> None:
        """Write to the terminal's stdin (tty execs only)."""
        if self._master_fd is None:
            raise RuntimeError("send() requires tty=True")
        os.write(self._master_fd, data)

    def read(self, encoding: Optional[str] = None):
        """Return the terminal output captured SO FAR (tty execs). bytes, or str if encoding given."""
        data = self._cap_out.output().bytes()
        return data.decode(encoding, errors="replace") if encoding else data

    def resize(self, rows: int, cols: int) -> None:
        """Set the terminal window size (tty execs)."""
        if self._master_fd is None:
            raise RuntimeError("resize() requires tty=True")
        fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    @property
    def master_fd(self) -> Optional[int]:
        """The pty master fd (tty execs) — bridge this to a websocket for a web terminal."""
        return self._master_fd


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
class Workspace:
    def __init__(self, manager: "WorkspaceManager", ws_id: str, dir: Path, policy: Policy,
                 tier: str, expires: Optional[float], cgroup_path: Optional[Path]):
        self.manager = manager
        self.id = ws_id
        self.dir = dir
        self.work_dir = dir / "work"
        self.io_dir = dir / ".io"
        self.policy = policy
        self.tier = tier
        self.expires = expires
        self.cgroup_path = cgroup_path
        self._seq = 0
        self._execs: list = []
        self._lock = threading.Lock()
        self._transcript = (dir / "transcript.jsonl").open("a")
        self._destroyed = False

    # -- paths / io -----------------------------------------------------------
    def path(self, *parts) -> Path:
        return self.work_dir.joinpath(*parts)

    def write(self, name: str, data: Union[bytes, str]) -> Path:
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            p.write_text(data)
        else:
            p.write_bytes(data)
        return p

    def read(self, name: str) -> bytes:
        return self.path(name).read_bytes()

    def _build_env(self, extra: Optional[dict]) -> dict:
        if self.policy.env is not None:
            env = dict(self.policy.env)
        else:  # scrubbed minimum
            env = {k: os.environ[k] for k in self.policy.env_passthrough if k in os.environ}
            env.setdefault("PATH", "/usr/bin:/bin")
        env["HOME"] = str(self.work_dir)
        env["PWD"] = str(self.work_dir)
        env["TMPDIR"] = str(self.work_dir)
        env.setdefault("TERM", "xterm-256color")
        if extra:
            env.update(extra)
        return env

    # -- exec -----------------------------------------------------------------
    def exec(self, cmd: Command, *, tty: bool = False, env: Optional[dict] = None,
             timeout: Optional[float] = None, stdin: Optional[bytes] = None,
             on_output: Optional[Callable[[bytes], None]] = None) -> Exec:
        """Run a command asynchronously in this workspace. Returns an :class:`Exec` handle at once.

        ``cmd`` may be a string (run via ``/bin/sh -c`` so pipes/redirects work) or an argv list
        (run directly; required when an allowlist is set). ``tty=True`` allocates a pseudo-terminal
        for interactive programs. ``on_output`` (if given) is called with each output chunk as it's
        read — e.g. to stream a PTY to a websocket terminal.
        """
        if self._destroyed:
            raise RuntimeError("workspace destroyed")
        with self._lock:
            seq = self._seq
            self._seq += 1
        h = Exec(self, cmd, tty=tty, env=env, timeout=timeout, stdin=stdin, seq=seq, on_output=on_output)
        with self._lock:
            self._execs.append(h)
        self._transcript.write(json.dumps({
            "t": time.time(), "seq": seq, "event": "exec", "tier": self.tier,
            "cmd": cmd if isinstance(cmd, str) else list(cmd), "tty": tty,
        }) + "\n")
        self._transcript.flush()
        return h

    def shell(self, *, tty: bool = True) -> Exec:
        """Convenience: launch an interactive shell in the workspace (PTY-backed)."""
        return self.exec(["/bin/bash", "-i"] if Path("/bin/bash").exists() else ["/bin/sh", "-i"],
                         tty=tty)

    def execs(self) -> list:
        with self._lock:
            return list(self._execs)

    def _record(self, h: Exec, res: ExecResult) -> None:
        # written by the executor (not the command) so it can't be falsified from inside
        try:
            digest = hashlib.sha256(res.stdout.bytes() + res.stderr.bytes()).hexdigest()[:16]
            self._transcript.write(json.dumps({
                "t": time.time(), "seq": res.seq, "event": "done", "rc": res.returncode,
                "dur": round(res.duration, 3), "timed_out": res.timed_out, "killed": res.killed,
                "out_bytes": res.stdout.total_bytes, "err_bytes": res.stderr.total_bytes,
                "io_hash": digest,
            }) + "\n")
            self._transcript.flush()
        except (ValueError, OSError):
            pass

    # -- context manager ------------------------------------------------------
    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> None:
        self.manager.destroy(self.id)


# ---------------------------------------------------------------------------
# WorkspaceManager
# ---------------------------------------------------------------------------
class WorkspaceManager:
    """Owns a root directory of workspaces: create / get / destroy, a TTL sweeper, and startup
    orphan-GC (dirs whose owning manager process is gone). One per process is typical."""

    def __init__(self, root: Union[str, Path], sweep_interval: float = 15.0,
                 gc_orphans: bool = True, cgroups: bool = False):
        """``cgroups=True`` (Linux) enables per-workspace cgroup v2 hard caps. It is OFF by default
        because enabling it RELOCATES this process into a child cgroup (cgroup v2's
        no-internal-processes rule) and enables controllers on the parent — a process-global side
        effect you should opt into. It needs a writable/delegated cgroup: run under
        ``systemd-run --user --scope -p Delegate=yes`` or as root. Without it, rlimits are used."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ws: dict = {}
        self._lock = threading.Lock()
        self._cgroup = _CgroupV2.setup() if (cgroups and IS_LINUX) else None
        if gc_orphans:
            self._gc_orphans()
        self._stop = threading.Event()
        self._sweeper = threading.Thread(target=self._sweep, args=(sweep_interval,), daemon=True)
        self._sweeper.start()

    @property
    def cgroup_available(self) -> bool:
        return self._cgroup is not None

    # -- lifecycle ------------------------------------------------------------
    def create(self, label: Optional[str] = None, policy: Optional[Policy] = None,
               ttl: Optional[float] = None) -> Workspace:
        policy = policy or Policy()
        tier = _resolve_tier(policy.sandbox)
        ws_id = uuid.uuid4().hex[:12]
        d = self.root / ws_id
        (d / "work").mkdir(parents=True)
        (d / ".io").mkdir()
        expires = (time.time() + ttl) if ttl else None
        cg = self._cgroup.create(ws_id, policy) if (policy.cgroup and self._cgroup) else None
        state = {
            "id": ws_id, "label": label, "tier": tier, "manager_pid": os.getpid(),
            "created": time.time(), "expires": expires, "cgroup": str(cg) if cg else None,
            "policy": _policy_dict(policy),
        }
        (d / "state.json").write_text(json.dumps(state, indent=2))
        ws = Workspace(self, ws_id, d, policy, tier, expires, cg)
        with self._lock:
            self._ws[ws_id] = ws
        return ws

    def get(self, ws_id: str) -> Optional[Workspace]:
        with self._lock:
            return self._ws.get(ws_id)

    def list(self) -> list:
        with self._lock:
            return list(self._ws.values())

    def destroy(self, ws_id: str, reason: str = "") -> bool:
        """Idempotent, total teardown: kill all live execs (whole trees), remove the cgroup, rmtree
        the dir, deregister. Returns False if the id was unknown (already gone)."""
        with self._lock:
            ws = self._ws.pop(ws_id, None)
        if ws is None:
            # may still be an on-disk orphan from a previous run
            d = self.root / ws_id
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                return True
            return False
        ws._destroyed = True
        for h in ws.execs():
            if h.running:
                h.kill(grace=1.0)
        # give trees a moment to exit, then hard-kill via the cgroup if present
        deadline = time.time() + 2.0
        for h in ws.execs():
            h.wait(timeout=max(0.0, deadline - time.time()))
        _CgroupV2.destroy(ws.cgroup_path)
        try:
            ws._transcript.close()
        except Exception:
            pass
        shutil.rmtree(ws.dir, ignore_errors=True)
        return True

    def renew(self, ws_id: str, extend: float) -> bool:
        ws = self.get(ws_id)
        if not ws:
            return False
        ws.expires = time.time() + extend
        return True

    def close(self) -> None:
        """Destroy all workspaces and stop the sweeper."""
        self._stop.set()
        for ws_id in [w.id for w in self.list()]:
            self.destroy(ws_id, reason="manager-close")

    # -- snapshot (lightweight) ----------------------------------------------
    def snapshot(self, ws_id: str, dest: Union[str, Path]) -> Optional[str]:
        """Save the workspace's working dir + a manifest (file hashes, policy, reason) to a
        ``.tar.gz``. Returns the path, or None if the workspace is unknown."""
        ws = self.get(ws_id)
        if not ws:
            return None
        dest = Path(dest)
        manifest = {"id": ws.id, "tier": ws.tier, "policy": _policy_dict(ws.policy),
                    "files": {}, "created": time.time()}
        for p in sorted(ws.work_dir.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(ws.work_dir))
                manifest["files"][rel] = {"size": p.stat().st_size,
                                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        (ws.dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        with tarfile.open(dest, "w:gz") as tf:
            tf.add(ws.work_dir, arcname="work")
            tf.add(ws.dir / "manifest.json", arcname="manifest.json")
        return str(dest)

    def restore(self, snapshot: Union[str, Path], policy: Optional[Policy] = None,
                ttl: Optional[float] = None) -> Workspace:
        """Materialize a snapshot into a NEW workspace (the resume/escalation mechanic)."""
        ws = self.create(policy=policy, ttl=ttl)
        with tarfile.open(snapshot, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.name.startswith("work/")]
            for m in members:
                m.name = m.name[len("work/"):] or "."
                if m.name != ".":
                    tf.extract(m, ws.work_dir, filter="data")
        return ws

    # -- background sweeper + orphan GC --------------------------------------
    def _sweep(self, interval: float):
        while not self._stop.wait(interval):
            now = time.time()
            for ws in self.list():
                if ws.expires and now >= ws.expires:
                    self.destroy(ws.id, reason="lease-expired")

    def _gc_orphans(self):
        for d in self.root.iterdir():
            if not d.is_dir():
                continue
            sf = d / "state.json"
            try:
                pid = json.loads(sf.read_text()).get("manager_pid") if sf.exists() else None
            except (OSError, ValueError):
                pid = None
            if pid is None or not _pid_alive(pid):
                shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _policy_dict(p: Policy) -> dict:
    d = asdict(p)
    d["allow_cmds"] = sorted(p.allow_cmds) if p.allow_cmds else None
    return d


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def attach(handle: Exec, escape: bytes = b"\x1d") -> None:
    """Bridge the current controlling terminal to a tty-backed Exec — a real terminal window onto
    the workspace command. Blocks until the command exits or the ``escape`` byte (default Ctrl-]).

    Requires ``handle.tty=True`` and a real stdin tty (not usable from a non-interactive process).
    """
    if handle.master_fd is None:
        raise RuntimeError("attach() requires an exec started with tty=True")
    if not sys.stdin.isatty():
        raise RuntimeError("attach() requires a controlling terminal")
    master = handle.master_fd
    stdin_fd = sys.stdin.fileno()
    old = termios.tcgetattr(stdin_fd)
    # propagate our window size to the pty
    try:
        sz = fcntl.ioctl(stdin_fd, termios.TIOCGWINSZ, b"\0" * 8)
        rows, cols, _, _ = struct.unpack("HHHH", sz)
        handle.resize(rows, cols)
    except OSError:
        pass
    try:
        _tty.setraw(stdin_fd)
        while handle.running:
            r, _, _ = select.select([stdin_fd, master], [], [], 0.2)
            if stdin_fd in r:
                data = os.read(stdin_fd, 4096)
                if escape and escape in data:
                    break
                os.write(master, data)
            if master in r:
                try:
                    out = os.read(master, 65536)
                except OSError:
                    break
                if not out:
                    break
                os.write(sys.stdout.fileno(), out)
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old)
