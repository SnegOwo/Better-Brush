"""``adapter.alive`` - the liveness handshake behind "start it with EffectCraft".

Nothing is registered at logon any more: when EffectCraft opens, the panel
reads this stamp and starts an adapter only if one is not already watching the
bridge. So the stamp has to mean *alive*, not *idle* - a render blocks the
watch loop for the whole job, and a heartbeat that stopped with it would make
the panel start a second adapter mid-job.
"""

from __future__ import annotations

import time

from effectcraft_bridge import pipeline, protocol
from effectcraft_bridge.cli import main
from effectcraft_bridge.pipeline import execute

from test_pipeline import make_job, make_source, write_job

ALIVE = protocol.ALIVE_NAME


def test_touch_alive_writes_an_epoch_the_panel_can_compare(tmp_path):
    """A plain number: the panel compares it to its own clock, no parsing."""
    root = tmp_path / "bridge"
    root.mkdir()

    before = int(time.time() * 1000)
    protocol.touch_alive(root)
    after = int(time.time() * 1000)

    stamp = int((root / ALIVE).read_text(encoding="ascii"))
    assert before <= stamp <= after


def test_touch_alive_survives_a_bridge_it_has_to_create(tmp_path):
    root = tmp_path / "nested" / "bridge"

    protocol.touch_alive(root)

    assert (root / ALIVE).is_file()


def test_once_stamps_the_heartbeat(tmp_path):
    root = tmp_path / "bridge"
    root.mkdir()

    assert main(["--bridge-dir", str(root), "once"]) == 0

    assert (root / ALIVE).is_file()
    assert int((root / ALIVE).read_text(encoding="ascii")) <= int(time.time() * 1000) + 5000


def test_a_running_job_stamps_the_heartbeat(tmp_path):
    """The render blocks the watch loop - the heartbeat must not go stale."""
    root = tmp_path / "bridge"
    root.mkdir()
    job = make_job(root, make_source(tmp_path))
    write_job(root, job)

    stamps: list[int] = []

    def record(_root):
        stamps.append(int(time.time() * 1000))

    original = pipeline.touch_alive
    pipeline.touch_alive = record
    try:
        status = execute(job, root, mode="simulate")
    finally:
        pipeline.touch_alive = original

    assert status.state == "done"
    # Once at the start of the job, and again on every progress report.
    assert len(stamps) >= 2
    # The last stamp is newer than the first, not a single kick at the door.
    assert stamps[-1] >= stamps[0]


# --------------------------------------------------------------------- #
# A start with no console (what start-adapter.vbs does)
# --------------------------------------------------------------------- #
def test_streams_are_replaced_when_the_start_had_no_console(monkeypatch):
    """wscript leaves sys.stdout/sys.stderr as None.

    ``print()`` quietly does nothing then - which is why adapter.log looked
    healthy - while tqdm, in the engine's propagate stage, raised
    ``'NoneType' object has no attribute 'write'`` and failed the job.
    """
    import run_adapter

    monkeypatch.setattr(run_adapter.sys, "stdout", None)
    monkeypatch.setattr(run_adapter.sys, "stderr", None)

    run_adapter.give_the_process_streams()

    assert run_adapter.sys.stdout is not None
    assert run_adapter.sys.stderr is not None
    # The exact operation that used to blow up, mid-render.
    run_adapter.sys.stderr.write("tqdm writes here\n")
    run_adapter.sys.stderr.flush()


def test_tqdm_runs_after_the_streams_are_repaired(monkeypatch):
    import run_adapter
    from tqdm import tqdm

    monkeypatch.setattr(run_adapter.sys, "stderr", None)
    run_adapter.give_the_process_streams()

    assert list(tqdm(range(3))) == [0, 1, 2]


def test_streams_are_left_alone_when_there_is_a_console():
    """A terminal start must keep its own streams: the log goes to both."""
    import run_adapter

    marker = run_adapter.sys.stdout
    if marker is None:  # this suite itself has no console (e.g. under pythonw)
        return
    run_adapter.give_the_process_streams()

    assert run_adapter.sys.stdout is marker
