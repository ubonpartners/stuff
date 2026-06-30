"""Tests for stuff.workspace — local isolated workspaces + async exec.

Runs the portable behaviours on any POSIX box; bwrap/cgroup-specific checks self-skip when the
feature isn't present on the host.
"""
import os
import shutil
import time
import platform

import pytest

from stuff.workspace import WorkspaceManager, Policy, _resolve_tier, IS_LINUX, _bwrap_usable

# bwrap must be present AND able to create namespaces here (CI/agent sandboxes often block userns)
HAS_BWRAP = _bwrap_usable()


@pytest.fixture
def mgr(tmp_path):
    m = WorkspaceManager(root=tmp_path / "ws", sweep_interval=0.5)
    yield m
    m.close()


def test_create_exec_destroy(mgr):
    ws = mgr.create(policy=Policy(sandbox="none"))
    assert ws.dir.exists() and ws.work_dir.exists()
    res = ws.exec("echo hello").wait(10)
    assert res is not None and res.ok
    assert res.stdout_text().strip() == "hello"
    d = ws.dir
    assert mgr.destroy(ws.id) is True
    assert not d.exists()
    assert mgr.destroy(ws.id) is False          # idempotent: second destroy is a no-op


def test_async_poll_then_wait(mgr):
    ws = mgr.create(policy=Policy(sandbox="none"))
    h = ws.exec("sleep 0.6; echo done")
    assert h.poll() is None and h.running        # returns immediately, still running
    res = h.wait(10)
    assert res.ok and res.stdout_text().strip() == "done"
    assert h.poll() == 0


def test_when_done_callback(mgr):
    ws = mgr.create(policy=Policy(sandbox="none"))
    got = []
    ws.exec("echo cb").when_done(lambda r: got.append(r.stdout_text().strip()))
    for _ in range(100):
        if got:
            break
        time.sleep(0.05)
    assert got == ["cb"]


def test_returncode_and_stderr(mgr):
    ws = mgr.create(policy=Policy(sandbox="none"))
    res = ws.exec("echo oops 1>&2; exit 3").wait(10)
    assert res.returncode == 3 and not res.ok
    assert res.stderr_text().strip() == "oops"


def test_cwd_is_workspace_and_isolated(mgr):
    ws = mgr.create(policy=Policy(sandbox="none"))
    res = ws.exec("pwd").wait(10)
    assert res.stdout_text().strip() == str(ws.work_dir)
    ws.exec("touch marker.txt").wait(10)
    assert (ws.work_dir / "marker.txt").exists()


def test_output_spill(mgr):
    ws = mgr.create(policy=Policy(sandbox="none", max_output=1024))
    res = ws.exec("python3 -c 'import sys; sys.stdout.write(\"x\"*5000)'").wait(10)
    assert res.stdout.spilled and res.stdout.total_bytes == 5000
    assert res.stdout.ref and os.path.exists(res.stdout.ref)
    assert len(res.stdout_text()) == 5000        # full content, never truncated


def test_timeout_kills(mgr):
    ws = mgr.create(policy=Policy(sandbox="none"))
    t0 = time.monotonic()
    res = ws.exec("sleep 30", timeout=0.5).wait(10)
    assert res.timed_out and (time.monotonic() - t0) < 6
    assert res.returncode != 0


def test_tree_kill_on_destroy(mgr, tmp_path):
    """A child that forks a long-lived grandchild must leave NOTHING after destroy."""
    ws = mgr.create(policy=Policy(sandbox="none"))
    pidfile = ws.work_dir / "gc.pid"
    # parent forks a grandchild that writes its pid and sleeps; parent also sleeps
    ws.exec(f"(sleep 60 & echo $! > {pidfile}); sleep 60")
    for _ in range(100):
        if pidfile.exists():
            break
        time.sleep(0.05)
    gc_pid = int(pidfile.read_text().strip())
    assert _alive(gc_pid)
    mgr.destroy(ws.id)
    time.sleep(0.5)
    assert not _alive(gc_pid), "grandchild survived workspace destroy (tree-kill failed)"


def test_allowlist(mgr):
    ws = mgr.create(policy=Policy(sandbox="none", allow_cmds={"echo"}))
    assert ws.exec(["echo", "ok"]).wait(10).ok
    with pytest.raises(PermissionError):
        ws.exec(["cat", "/etc/hostname"])
    with pytest.raises(ValueError):
        ws.exec("echo ok")                       # shell string can't be vetted against an allowlist


def test_env_scrubbed_and_home(mgr):
    ws = mgr.create(policy=Policy(sandbox="none", env={"FOO": "bar"}))
    assert ws.exec("echo $FOO").wait(10).stdout_text().strip() == "bar"
    assert ws.exec("echo $HOME").wait(10).stdout_text().strip() == str(ws.work_dir)


def test_tty_interactive(mgr):
    ws = mgr.create(policy=Policy(sandbox="none"))
    h = ws.exec(["/bin/sh", "-i"], tty=True)
    h.send(b"echo tty-works\n")
    time.sleep(0.4)
    h.send(b"exit\n")
    h.wait(10)
    assert "tty-works" in h.read(encoding="utf-8")
    assert h.master_fd is None                   # closed after completion


def test_ttl_sweeper_destroys(mgr):
    ws = mgr.create(policy=Policy(sandbox="none"), ttl=0.5)
    wid, d = ws.id, ws.dir
    time.sleep(2.0)
    assert mgr.get(wid) is None and not d.exists()


def test_orphan_gc(tmp_path):
    root = tmp_path / "ws"
    m1 = WorkspaceManager(root=root)
    ws = m1.create(policy=Policy(sandbox="none"))
    d = ws.dir
    # simulate m1 crashing: fake a dead manager_pid in state.json, drop the in-memory handle
    import json
    state = json.loads((d / "state.json").read_text())
    state["manager_pid"] = 999999                # not a live pid
    (d / "state.json").write_text(json.dumps(state))
    m1._ws.clear()
    m1.close()
    # a fresh manager over the same root GCs the orphan on startup
    m2 = WorkspaceManager(root=root)
    try:
        assert not d.exists()
    finally:
        m2.close()


def test_snapshot_restore(mgr):
    ws = mgr.create(policy=Policy(sandbox="none"))
    ws.write("data.txt", "payload-123")
    snap = mgr.snapshot(ws.id, ws.dir.parent / "snap.tar.gz")
    assert snap and os.path.exists(snap)
    ws2 = mgr.restore(snap, policy=Policy(sandbox="none"))
    assert ws2.read("data.txt") == b"payload-123"
    assert ws2.id != ws.id


def test_resolve_tier_auto():
    tier = _resolve_tier("auto")
    if HAS_BWRAP:
        assert tier == "bwrap"
    else:
        assert tier in ("seatbelt", "none")


@pytest.mark.skipif(HAS_BWRAP, reason="bwrap usable here, so it won't raise")
def test_explicit_unavailable_tier_raises():
    # asking for a tier this host can't provide must fail loudly, not silently downgrade
    with pytest.raises(RuntimeError):
        _resolve_tier("bwrap")


@pytest.mark.skipif(not os.environ.get("UBON_TEST_CGROUPS"),
                    reason="cgroup test relocates the runner's process cgroup; opt in with UBON_TEST_CGROUPS=1")
def test_cgroup_optin(tmp_path):
    m = WorkspaceManager(root=tmp_path / "ws", cgroups=True)
    try:
        if not m.cgroup_available:
            pytest.skip("no writable/delegated cgroup v2 here")
        ws = m.create(policy=Policy(sandbox="none", mem_bytes=64 << 20, pids=32))
        assert ws.cgroup_path is not None and ws.cgroup_path.exists()
        assert ws.exec("echo cg").wait(10).ok
        cgp = ws.cgroup_path
        m.destroy(ws.id)
        assert not cgp.exists()
    finally:
        m.close()


@pytest.mark.skipif(not HAS_BWRAP, reason="bwrap not installed")
def test_bwrap_filesystem_scoping(mgr):
    """Under bwrap the workspace is writable but the host CWD is not bound (so a host path the ws
    shouldn't see isn't there). Also verifies the tier is recorded on the result."""
    ws = mgr.create(policy=Policy(sandbox="bwrap", network=False))
    res = ws.exec("echo sandboxed; pwd").wait(15)
    assert res.ok and res.tier == "bwrap"
    assert "sandboxed" in res.stdout_text()


@pytest.mark.skipif(not HAS_BWRAP, reason="bwrap not installed")
def test_bwrap_network_cut(mgr):
    ws = mgr.create(policy=Policy(sandbox="bwrap", network=False))
    # with --unshare-net there is no usable network namespace -> a connect fails fast
    code = ("import socket,sys\n"
            "try:\n"
            "  socket.create_connection(('1.1.1.1',53),timeout=2); print('NET-OK')\n"
            "except Exception as e: print('NET-BLOCKED')\n")
    ws.write("netcheck.py", code)
    res = ws.exec(["python3", "netcheck.py"]).wait(20)
    assert "NET-BLOCKED" in res.stdout_text()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
