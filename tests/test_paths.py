from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trec_ragtime.paths import (
    PROJECT_CACHE_DIR_ENV,
    default_project_cache_dir,
    ensure_private_cache_dir,
    shared_git_common_dir,
)


def write_project(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text(
        '[project]\nname = "trec-ragtime-2026"\n',
        encoding="utf-8",
    )


class ProjectPathTests(unittest.TestCase):
    def test_primary_and_linked_worktrees_resolve_one_git_common_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            linked = root / "linked"
            write_project(primary)
            write_project(linked)
            common = primary / ".git"
            common.mkdir()
            (common / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            (common / "config").write_text("[core]\n", encoding="utf-8")
            linked_git_directory = common / "worktrees" / "linked"
            linked_git_directory.mkdir(parents=True)
            (linked / ".git").write_text(
                f"gitdir: {linked_git_directory}\n",
                encoding="utf-8",
            )
            (linked_git_directory / "commondir").write_text(
                "../..\n",
                encoding="utf-8",
            )

            self.assertEqual(shared_git_common_dir(primary), common.resolve())
            self.assertEqual(shared_git_common_dir(linked), common.resolve())

    def test_project_cache_override_and_xdg_fallback_must_be_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "project-cache"
            with patch.dict(
                os.environ,
                {
                    PROJECT_CACHE_DIR_ENV: str(explicit),
                    "XDG_CACHE_HOME": str(root / "xdg"),
                },
                clear=False,
            ):
                self.assertEqual(default_project_cache_dir(), explicit)

            with patch.dict(
                os.environ,
                {PROJECT_CACHE_DIR_ENV: "relative", "XDG_CACHE_HOME": ""},
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "absolute path"):
                    default_project_cache_dir()

            with (
                patch.dict(
                    os.environ,
                    {PROJECT_CACHE_DIR_ENV: "", "XDG_CACHE_HOME": "relative"},
                    clear=False,
                ),
                patch(
                    "trec_ragtime.paths.shared_git_common_dir",
                    return_value=None,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "absolute path"):
                    default_project_cache_dir()

            with (
                patch.dict(
                    os.environ,
                    {
                        PROJECT_CACHE_DIR_ENV: "",
                        "XDG_CACHE_HOME": str(root / "xdg"),
                    },
                    clear=False,
                ),
                patch(
                    "trec_ragtime.paths.shared_git_common_dir",
                    return_value=None,
                ),
            ):
                self.assertEqual(
                    default_project_cache_dir(),
                    root / "xdg" / "trec-ragtime-2026",
                )

            common = root / ".git"
            with (
                patch.dict(
                    os.environ,
                    {
                        PROJECT_CACHE_DIR_ENV: "",
                        "XDG_CACHE_HOME": str(root / "ignored-xdg"),
                    },
                    clear=False,
                ),
                patch(
                    "trec_ragtime.paths.shared_git_common_dir",
                    return_value=common,
                ),
            ):
                self.assertEqual(
                    default_project_cache_dir(),
                    common / "ragtime-cache",
                )

    def test_private_cache_directory_permissions_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = ensure_private_cache_dir(root / "new-cache")
            self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o700)

            permissive = root / "permissive"
            permissive.mkdir(mode=0o755)
            permissive.chmod(0o755)
            with self.assertRaisesRegex(PermissionError, "user-only"):
                ensure_private_cache_dir(permissive)

    def test_non_project_git_checkout_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            (root / "pyproject.toml").write_text(
                '[project]\nname = "different-project"\n',
                encoding="utf-8",
            )

            self.assertIsNone(shared_git_common_dir(root))

            write_project(root)
            self.assertIsNone(shared_git_common_dir(root))


if __name__ == "__main__":
    unittest.main()
