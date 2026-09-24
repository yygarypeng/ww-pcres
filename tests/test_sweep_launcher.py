import os
import stat
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "sweep" / "run_sweep.sh"


def _write_stub(path, body):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _stub_bin(tmp_path):
    stub = tmp_path / "stubbin"
    stub.mkdir()
    logdir = tmp_path / "calls"
    logdir.mkdir()
    _write_stub(
        stub / "taskset",
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{logdir}/taskset.log"\n'
        'if [ "$1" = "-c" ]; then shift 2; fi\n'
        'exec "$@"\n',
    )
    _write_stub(
        stub / "python",
        f'#!/usr/bin/env bash\necho "$@" >> "{logdir}/python.log"\nexit 0\n',
    )
    return stub, logdir


def _run_script(args, stub, output_dir, extra_env=None):
    env = dict(os.environ)
    env["PATH"] = f"{stub}{os.pathsep}{env['PATH']}"
    env["SWEEP_OUTPUT_DIR"] = str(output_dir)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_no_args_validates_then_backgrounds_and_writes_pid(tmp_path):
    stub, logdir = _stub_bin(tmp_path)
    outdir = tmp_path / "outputs"
    result = _run_script([], stub, outdir)
    assert result.returncode == 0, result.stderr
    deadline = time.time() + 10.0
    python_calls = []
    while time.time() < deadline:
        if (logdir / "python.log").exists():
            python_calls = (logdir / "python.log").read_text().splitlines()
            if len(python_calls) >= 2:
                break
        time.sleep(0.05)
    assert len(python_calls) == 2
    assert "--validate" in python_calls[0]
    assert "--validate" not in python_calls[1]
    assert "sweep.optimize" in python_calls[0]
    taskset_calls = (logdir / "taskset.log").read_text().splitlines()
    assert taskset_calls, "taskset stub was never called"
    assert any("0-9,12-31" in line for line in taskset_calls)
    assert all("0-9,12-15" not in line for line in taskset_calls)
    pid_file = outdir / "sweep.pid"
    assert pid_file.exists()
    assert pid_file.read_text().strip().isdigit()


def test_live_pid_blocks_second_launch(tmp_path):
    stub, logdir = _stub_bin(tmp_path)
    outdir = tmp_path / "outputs"
    outdir.mkdir()
    (outdir / "sweep.pid").write_text(str(os.getpid()))
    result = _run_script([], stub, outdir)
    assert result.returncode != 0
    assert "already running" in (result.stdout + result.stderr).lower()


def test_stale_pid_is_replaced_and_workflow_starts(tmp_path):
    stub, _logdir = _stub_bin(tmp_path)
    outdir = tmp_path / "outputs"
    outdir.mkdir()
    (outdir / "sweep.pid").write_text("99999999")
    result = _run_script([], stub, outdir)
    assert result.returncode == 0, result.stderr
    assert (outdir / "sweep.pid").read_text().strip() != "99999999"


def test_foreground_validate_does_not_write_pid_file(tmp_path):
    stub, logdir = _stub_bin(tmp_path)
    outdir = tmp_path / "outputs"
    result = _run_script(["--validate"], stub, outdir)
    assert result.returncode == 0, result.stderr
    assert not (outdir / "sweep.pid").exists()
    python_calls = (logdir / "python.log").read_text().splitlines()
    assert len(python_calls) == 1
    assert "--validate" in python_calls[0]


DETACH_STUB = """#!/usr/bin/env bash
echo "$@" >> "{logdir}/python.log"
awk '{{print "sid=" $6}}' /proc/$$/stat >> "{logdir}/detach.log"
grep SigIgn /proc/$$/status >> "{logdir}/detach.log"
exit 0
"""


def _detach_records(logdir):
    """One (session id, ignored-signal mask) pair per call the launcher made."""
    sessions, masks = [], []
    for line in (logdir / "detach.log").read_text().splitlines():
        if line.startswith("sid="):
            sessions.append(int(line.split("=", 1)[1]))
        elif line.startswith("SigIgn"):
            masks.append(int(line.split(":", 1)[1].strip(), 16))
    return list(zip(sessions, masks))


def test_background_driver_is_detached_from_the_launching_session(tmp_path):
    """The driver must survive a Ctrl-C or a hangup aimed at the shell that started it."""
    stub, logdir = _stub_bin(tmp_path)
    _write_stub(stub / "python", DETACH_STUB.format(logdir=logdir))
    outdir = tmp_path / "outputs"

    result = _run_script([], stub, outdir)
    assert result.returncode == 0, result.stderr

    deadline = time.time() + 10.0
    records = []
    while time.time() < deadline:
        if (logdir / "detach.log").exists():
            records = _detach_records(logdir)
            if len(records) >= 2:
                break
        time.sleep(0.05)

    assert len(records) == 2, "expected the foreground validation and the background driver"
    (validation_session, _), (driver_session, driver_ignored) = records
    assert validation_session == os.getsid(0), "validation is meant to run in the foreground"
    assert driver_session != os.getsid(0), "the driver still shares the launcher's session"
    assert driver_ignored & 0x3 == 0x3, "the driver does not ignore SIGHUP and SIGINT"
