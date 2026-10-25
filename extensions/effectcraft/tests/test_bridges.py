"""Which bridge folders the adapter serves.

The field failure this guards: a machine had a portable 0.11.1 *and* a 0.17.0,
each with its own panel and its own bridge; the adapter watched only the one
it was recorded against, so the other panel's job sat at "queued — waiting for
the adapter…" forever, with nothing anywhere saying why.
"""

from __future__ import annotations

import argparse

from effectcraft_bridge import cli, protocol


def bridge(root):
    path = root / "EffectCraftData" / protocol.BRIDGE_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------- #
def test_a_portable_install_next_to_the_downloads_is_found(tmp_path):
    install = tmp_path / "effectcraft-0.11.1-windows-x64-portable"
    found = bridge(install)

    assert found in cli.portable_bridges(roots=[tmp_path])


def test_the_nested_unzipped_layout_is_found(tmp_path):
    """Downloads\\effectcraft-x\\effectcraft-x\\EffectCraftData — the real one."""
    inner = tmp_path / "effectcraft-0.17.0-windows-x64-portable" / "effectcraft-0.17.0-windows-x64-portable"
    found = bridge(inner)

    assert found in cli.portable_bridges(roots=[tmp_path])


def test_an_install_buried_too_deep_is_not_searched(tmp_path):
    """Bounded on purpose: this runs at startup, not on every tick."""
    deep = tmp_path / "a" / "b" / "c" / "d" / "e"
    bridge(deep)

    assert cli.portable_bridges(roots=[tmp_path], depth=4) == []


# --------------------------------------------------------------------- #
# What the watcher decides to serve
# --------------------------------------------------------------------- #
def test_an_explicit_bridge_means_only_that_one(tmp_path, monkeypatch):
    """--bridge-dir (and $BETTERBRUSH_BRIDGE_DIR) stay literal: CI and the
    tests must never start touching a real install's bridge."""
    only = tmp_path / "only"
    only.mkdir()
    monkeypatch.setattr(cli, "bridge_dir", lambda explicit=None: only)
    monkeypatch.setattr(cli, "candidate_bridge_dirs", lambda explicit=None: [])
    monkeypatch.setattr(cli, "portable_bridges", lambda **kw: [])

    assert cli.bridge_dir_or_dirs(str(only)) == [only]


def test_every_existing_bridge_is_served(tmp_path, monkeypatch):
    recorded = tmp_path / "recorded"
    portable = tmp_path / "portable"
    ghost = tmp_path / "ghost"
    recorded.mkdir()
    portable.mkdir()

    monkeypatch.setattr(cli, "bridge_dir", lambda explicit=None: recorded)
    monkeypatch.setattr(cli, "candidate_bridge_dirs", lambda explicit=None: [recorded, ghost])
    monkeypatch.setattr(cli, "portable_bridges", lambda **kw: [portable])

    assert cli.bridge_dir_or_dirs(None) == [recorded, portable]


def test_a_bridge_that_does_not_exist_is_not_served(tmp_path, monkeypatch):
    recorded = tmp_path / "recorded"
    recorded.mkdir()
    monkeypatch.setattr(cli, "bridge_dir", lambda explicit=None: recorded)
    monkeypatch.setattr(cli, "candidate_bridge_dirs", lambda explicit=None: [tmp_path / "gone"])
    monkeypatch.setattr(cli, "portable_bridges", lambda **kw: [])

    assert cli.bridge_dir_or_dirs(None) == [recorded]


# --------------------------------------------------------------------- #
# Draining
# --------------------------------------------------------------------- #
def test_once_drains_a_job_from_the_second_bridge(tmp_path, monkeypatch):
    """The exact field symptom: the job lived in the *other* install's bridge."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (second / "job-bb-1.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli, "bridge_dir_or_dirs", lambda explicit: [first, second])
    monkeypatch.setattr(cli, "pending_jobs",
                        lambda root: sorted(root.glob("job-*.json")))
    ran: list[Path] = []
    monkeypatch.setattr(cli, "run_one",
                        lambda root, path, mode: ran.append((root, path)))
    monkeypatch.setattr(cli.appbridge, "handle_requests",
                        lambda root, log=None: 0)

    code = cli.cmd_once(argparse.Namespace(bridge_dir=None, mode=None))

    assert code == 0
    assert ran == [(second, second / "job-bb-1.json")]


def test_a_heartbeat_is_left_in_every_bridge(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    monkeypatch.setattr(cli, "bridge_dir_or_dirs", lambda explicit: [first, second])
    monkeypatch.setattr(cli, "pending_jobs", lambda root: [])
    monkeypatch.setattr(cli.appbridge, "handle_requests",
                        lambda root, log=None: 0)

    cli.cmd_once(argparse.Namespace(bridge_dir=None, mode=None))

    assert (first / protocol.ALIVE_NAME).is_file()
    assert (second / protocol.ALIVE_NAME).is_file()
