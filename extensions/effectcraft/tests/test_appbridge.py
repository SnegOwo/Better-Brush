"""Open in Better Brush: request -> launch -> answer, and where the exe comes from.

The panel cannot start a process, so this is the half of the feature that
can: it finds the desktop app, launches it detached on the clip, and answers
the panel in the bridge. See ``effectcraft_bridge/appbridge.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from effectcraft_bridge import appbridge
from effectcraft_bridge.cli import main


def make_exe(folder: Path, name: str = "BetterBrush.exe") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / name
    exe.write_bytes(b"MZ")
    return exe


def write_request(root: Path, clip: Path, request_id: str = "open1") -> Path:
    path = root / f"app-open-{request_id}.json"
    path.write_text(json.dumps({"id": request_id, "source": str(clip)}),
                    encoding="utf-8")
    return path


def read_answer(root: Path, request_id: str = "open1") -> dict:
    return json.loads((root / f"app-open-status-{request_id}.json")
                      .read_text(encoding="utf-8"))


# --------------------------------------------------------------------- #
# Where the exe comes from
# --------------------------------------------------------------------- #
def test_recorded_path_wins_and_is_not_searched(tmp_path, monkeypatch):
    root = tmp_path / "bridge"
    root.mkdir()
    exe = make_exe(tmp_path / "apps")
    appbridge.remember_exe(root, exe)

    monkeypatch.setattr(appbridge, "candidate_exes", lambda: [tmp_path / "nope.exe"])
    found, message = appbridge.resolve_exe(root)

    assert found == exe
    assert message == ""


def test_recorded_path_that_no_longer_exists_falls_back(tmp_path, monkeypatch):
    root = tmp_path / "bridge"
    root.mkdir()
    (root / appbridge.APP_PATH_FILE).write_text(
        json.dumps({"exe": str(tmp_path / "gone" / "BetterBrush.exe")}),
        encoding="utf-8")
    real = make_exe(tmp_path / "apps")
    monkeypatch.setattr(appbridge, "candidate_exes", lambda: [real])

    found, message = appbridge.resolve_exe(root)

    assert found == real
    assert message == ""
    # ...and the new answer is remembered, so the next request skips the search.
    assert appbridge.read_recorded_exe(root) == real


def test_search_remembering_makes_the_second_call_free(tmp_path, monkeypatch):
    root = tmp_path / "bridge"
    root.mkdir()
    exe = make_exe(tmp_path / "apps")
    monkeypatch.setattr(appbridge, "candidate_exes", lambda: [exe])

    first, _ = appbridge.resolve_exe(root)
    calls = []
    monkeypatch.setattr(appbridge, "candidate_exes",
                        lambda: calls.append(1) or [])
    second, _ = appbridge.resolve_exe(root)

    assert first == exe == second
    assert calls == []  # answered from the bridge, not from a search


def test_no_exe_anywhere_says_what_to_do(tmp_path, monkeypatch):
    root = tmp_path / "bridge"
    root.mkdir()
    monkeypatch.setattr(appbridge, "candidate_exes", lambda: [])

    found, message = appbridge.resolve_exe(root)

    assert found is None
    assert "Paste the path" in message
    assert "BetterBrush.exe" in message


def test_candidates_are_most_specific_first():
    names = [str(p).lower() for p in appbridge.candidate_exes()]
    # An installed copy outranks whatever happens to be lying in the repo.
    installed = [i for i, n in enumerate(names) if "programs" in n or "program files" in n]
    repo = [i for i, n in enumerate(names) if "dist-final" in n or "dist" in n]
    if installed and repo:
        assert min(installed) < max(repo)


# --------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------- #
def test_status_files_are_answers_never_requests(tmp_path):
    root = tmp_path
    (root / "app-open-status-open1.json").write_text("{}", encoding="utf-8")
    assert appbridge.pending_requests(root) == []


def test_a_request_is_served_and_answered(tmp_path, monkeypatch):
    root = tmp_path / "bridge"
    root.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    exe = make_exe(tmp_path / "apps")
    monkeypatch.setattr(appbridge, "candidate_exes", lambda: [exe])
    launched = []
    monkeypatch.setattr(appbridge, "launch", lambda e, c: launched.append((e, c)))
    write_request(root, clip)

    handled = appbridge.handle_requests(root)

    assert handled == 1
    assert launched == [(exe, str(clip))]
    assert read_answer(root) == {"state": "ok", "exe": str(exe), "source": str(clip)}
    # The request is consumed: the watcher must never launch the same clip twice.
    assert appbridge.pending_requests(root) == []


def test_a_missing_clip_is_answered_not_retried(tmp_path):
    root = tmp_path / "bridge"
    root.mkdir()
    write_request(root, tmp_path / "gone.mp4")

    assert appbridge.handle_requests(root) == 0
    assert "no longer on disk" in read_answer(root)["message"]
    assert appbridge.pending_requests(root) == []


def test_a_broken_request_still_gets_an_answer(tmp_path):
    root = tmp_path / "bridge"
    root.mkdir()
    (root / "app-open-open1.json").write_text("{not json", encoding="utf-8")

    assert appbridge.handle_requests(root) == 0
    assert read_answer(root)["state"] == "error"
    assert "Unreadable" in read_answer(root)["message"]


def test_a_launch_failure_is_reported(tmp_path, monkeypatch):
    root = tmp_path / "bridge"
    root.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    exe = make_exe(tmp_path / "apps")
    monkeypatch.setattr(appbridge, "candidate_exes", lambda: [exe])

    def boom(_exe, _clip):
        raise OSError("Access is denied")

    monkeypatch.setattr(appbridge, "launch", boom)
    write_request(root, clip)

    assert appbridge.handle_requests(root) == 0
    assert "Could not start" in read_answer(root)["message"]


def test_requests_are_served_in_order(tmp_path, monkeypatch):
    root = tmp_path / "bridge"
    root.mkdir()
    exe = make_exe(tmp_path / "apps")
    monkeypatch.setattr(appbridge, "candidate_exes", lambda: [exe])
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        clip = tmp_path / name
        clip.write_bytes(b"\x00")
        write_request(root, clip, request_id=name)
    order = []
    monkeypatch.setattr(appbridge, "launch", lambda e, c: order.append(Path(c).name))

    appbridge.handle_requests(root)

    assert order == ["a.mp4", "b.mp4", "c.mp4"]


# --------------------------------------------------------------------- #
# The panel's half of the contract
# --------------------------------------------------------------------- #
def test_the_panel_only_remembers_a_real_exe_name(tmp_path):
    assert appbridge.looks_like_exe(r"C:\apps\BetterBrush.exe")
    assert appbridge.looks_like_exe("C:/apps/Better Brush 2.exe")
    assert not appbridge.looks_like_exe(r"C:\apps\python.exe")
    assert not appbridge.looks_like_exe("")


def test_the_adapter_serves_open_requests_from_watch(tmp_path, monkeypatch):
    """``once`` is what a background adapter run uses; it must serve both."""
    root = tmp_path / "bridge"
    root.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    exe = make_exe(tmp_path / "apps")
    monkeypatch.setattr(appbridge, "candidate_exes", lambda: [exe])
    monkeypatch.setattr(appbridge, "launch", lambda e, c: None)
    write_request(root, clip)

    code = main(["--bridge-dir", str(root), "once"])

    assert code == 0
    assert read_answer(root)["state"] == "ok"


def test_a_frozen_adapter_can_be_told_where_the_repo_is(tmp_path, monkeypatch):
    """A packaged adapter has no ``__file__`` worth walking: env names it."""
    repo = tmp_path / "checkout"
    (repo / "src").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.setenv("BETTERBRUSH_ROOT", str(repo))

    assert appbridge._repo_root() == repo


def test_an_env_root_that_does_not_exist_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("BETTERBRUSH_ROOT", str(tmp_path / "nope"))

    assert appbridge._repo_root() is None


def test_running_from_a_checkout_the_repo_root_is_that_checkout(monkeypatch):
    monkeypatch.delenv("BETTERBRUSH_ROOT", raising=False)
    root = appbridge._repo_root()

    assert root is None or (root / "src").is_dir()
