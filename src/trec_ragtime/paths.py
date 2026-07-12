"""Shared project storage paths that remain stable across Git worktrees."""

from __future__ import annotations

import os
import stat
import tomllib
from pathlib import Path

PROJECT_CACHE_DIR_ENV = "RAGTIME_PROJECT_CACHE_DIR"
PROJECT_CACHE_SUBDIRECTORY = "ragtime-cache"
PROJECT_NAME = "trec-ragtime-2026"


def _absolute_path(value: str | os.PathLike[str], setting: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{setting} must be an absolute path")
    return path


def ensure_private_cache_dir(path: Path) -> Path:
    """Create a cache directory and require owner-only permissions."""

    if not path.is_absolute():
        raise ValueError("cache directory must be absolute")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise PermissionError(f"cache directory is not a directory: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(f"cache directory must be user-only (chmod 700): {path}")
    return path


def _linked_worktree_common_directory(marker: Path) -> Path | None:
    try:
        marker_text = marker.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    prefix = "gitdir: "
    if not marker_text.startswith(prefix):
        return None
    git_directory = Path(marker_text[len(prefix) :].strip())
    if not git_directory.is_absolute():
        git_directory = marker.parent / git_directory
    git_directory = git_directory.resolve()

    common_marker = git_directory / "commondir"
    if not common_marker.is_file():
        return git_directory
    try:
        common_value = common_marker.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not common_value:
        return None
    common_directory = Path(common_value)
    if not common_directory.is_absolute():
        common_directory = git_directory / common_directory
    return common_directory.resolve()


def _is_ragtime_project(directory: Path) -> bool:
    try:
        with (directory / "pyproject.toml").open("rb") as handle:
            metadata = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    project = metadata.get("project")
    return isinstance(project, dict) and project.get("name") == PROJECT_NAME


def _is_git_common_directory(directory: Path) -> bool:
    return (
        directory.is_dir()
        and (directory / "HEAD").is_file()
        and (directory / "config").is_file()
    )


def shared_git_common_dir(start: Path) -> Path | None:
    """Find this project's Git common directory from a source-tree anchor."""

    try:
        candidate = start.expanduser().resolve()
    except OSError:
        return None
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if not _is_ragtime_project(directory):
            continue
        marker = directory / ".git"
        if _is_git_common_directory(marker):
            return marker.resolve()
        if not marker.is_file():
            continue
        common_directory = _linked_worktree_common_directory(marker)
        if common_directory is None or not _is_git_common_directory(common_directory):
            return None
        return common_directory
    return None


def default_project_cache_dir() -> Path:
    """Return one project-level cache directory shared by every worktree.

    An explicit project override takes precedence. Inside this project's Git
    checkout, the default is ``<git-common-dir>/ragtime-cache`` even when XDG
    is configured. Installed packages outside the checkout use the XDG/user
    cache fallback.
    """

    explicit = os.getenv(PROJECT_CACHE_DIR_ENV, "").strip()
    if explicit:
        return _absolute_path(explicit, PROJECT_CACHE_DIR_ENV)

    common_directory = shared_git_common_dir(Path(__file__))
    if common_directory is not None:
        return common_directory / PROJECT_CACHE_SUBDIRECTORY

    xdg_cache_home = os.getenv("XDG_CACHE_HOME", "").strip()
    if xdg_cache_home:
        return _absolute_path(xdg_cache_home, "XDG_CACHE_HOME") / "trec-ragtime-2026"

    return Path.home() / ".cache" / "trec-ragtime-2026"


__all__ = [
    "PROJECT_CACHE_DIR_ENV",
    "default_project_cache_dir",
    "ensure_private_cache_dir",
    "shared_git_common_dir",
]
