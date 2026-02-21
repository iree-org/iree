# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pre-commit validations for tools/utils/ (cross-platform, no bash).

Checks:
  1) Permissions: only bin/* is executable (POSIX); bin/* must have python3 shebang
  2) Forbidden imports: no "from lit ..." in utils (use lit_tools);
     allow in lit_tools/core/lit_wrapper.py and integration tests
  3) JSON purity: no print() immediately after an "if args.json" branch in lit_tools

Outputs short diagnostics and exits non-zero on failure.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def repo_root() -> Path:
    """Find repository root by searching for .git or WORKSPACE file.

    Returns:
        Path to repository root
    """
    p = Path(__file__).resolve()
    for q in [p.parents[3], p.parents[2], p.parents[1], p.parents[0]]:
        if (q / ".git").exists() or (q / "WORKSPACE").exists():
            return q
    return p.parents[3]


# Ensure utils modules are importable at module import time (no inline imports).
ROOT = repo_root()
UTILS_DIR = ROOT / "tools" / "utils"
sys.path.insert(0, str(UTILS_DIR.resolve()))
from common import console  # type: ignore  # noqa: E402


def is_executable(path: Path) -> bool:
    """Check if file has executable permission (POSIX only).

    Args:
        path: Path to file to check

    Returns:
        True if file is executable, False otherwise (including on Windows)
    """
    if os.name == "nt":
        return False
    try:
        return bool(path.stat().st_mode & 0o111)
    except OSError:
        return False


def check_permissions(root: Path) -> list[str]:
    """Check that only bin/ files are executable and have python3 shebang.

    Args:
        root: Repository root path

    Returns:
        List of error messages
    """
    errors: list[str] = []
    utils = root / "tools" / "utils"
    # 1) Non-bin Python files must not be executable (POSIX only)
    for py in utils.rglob("*.py"):
        rel = py.relative_to(root)
        if "tools/utils/bin/" in str(rel).replace("\\", "/"):
            continue
        if is_executable(py):
            errors.append(f"Non-bin Python file is executable: {rel}")

    # 2) Bin wrappers must be executable (POSIX) and have python3 shebang
    bin_dir = utils / "bin"
    if bin_dir.exists():
        for f in bin_dir.iterdir():
            if f.is_file():
                if os.name != "nt" and not is_executable(f):
                    errors.append(f"Bin wrapper not executable: {f.relative_to(root)}")
                try:
                    first = f.read_text(encoding="utf-8", errors="ignore").splitlines()[
                        :1
                    ]
                except OSError:
                    first = []
                if not (first and first[0].strip() == "#!/usr/bin/env python3"):
                    errors.append(
                        f"Bin wrapper missing python3 shebang: {f.relative_to(root)}"
                    )
    return errors


def check_forbidden_imports(root: Path) -> list[str]:
    """Check that code uses lit_tools instead of direct lit imports.

    Args:
        root: Repository root path

    Returns:
        List of error messages
    """
    errors: list[str] = []
    utils = root / "tools" / "utils"
    allowlist = {
        str((utils / "lit_tools" / "core" / "lit_wrapper.py").resolve()),
    }

    # Integration tests may import lit directly; allow those
    def allowed(path: Path) -> bool:
        p = str(path.resolve())
        if p in allowlist:
            return True
        norm = p.replace("\\", "/")
        # Allow integration tests (test_*_integration.py pattern)
        return norm.endswith("_integration.py")

    pattern = re.compile(r"^\s*from\s+lit(\.|\s)", re.MULTILINE)
    for py in utils.rglob("*.py"):
        if allowed(py):
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if pattern.search(text):
            errors.append(
                f"Forbidden import 'from lit ...' in {py.relative_to(root)}; use 'from lit_tools'"
            )
    return errors


def check_json_purity(root: Path) -> list[str]:
    """Check that print() is never called inside JSON output branches.

    Args:
        root: Repository root path

    Returns:
        List of error messages
    """
    errors: list[str] = []
    lt = root / "tools" / "utils" / "lit_tools"
    # Heuristic: look for 'if args.json' followed by a 'print(' within next 1 line
    json_guard = re.compile(r"if\s+args\.json[^\n]*\n\s*print\(", re.MULTILINE)
    for py in lt.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if json_guard.search(text):
            errors.append(
                f"print() inside JSON branch in {py.relative_to(root)}; never print to stdout when --json is set"
            )
    return errors


def check_forbidden_sys_stream_writes(root: Path) -> list[str]:
    """Disallow direct sys.stdout/sys.stderr writes in implementation.

    Primary output must go through lit_tools.core.console helpers to ensure
    consistent behavior and easy policy changes. Allow writes only in the
    console module itself.

    Args:
        root: Repository root path

    Returns:
        List of error messages
    """
    errors: list[str] = []
    lt = root / "tools" / "utils" / "lit_tools"
    for py in lt.rglob("*.py"):
        # Allow in core/console.py (the only place where touching sys.* is OK).
        norm = str(py.resolve()).replace("\\", "/")
        if norm.endswith("/lit_tools/core/console.py"):
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(r"\bsys\.(stdout|stderr)\.", text):
            rel = py.relative_to(root)
            errors.append(
                f"Forbidden sys.(stdout|stderr) usage in {rel}; use console.out/console.write or console.error/warn/note/success"
            )
        if re.search(
            r"^\s*from\s+sys\s+import\s+(stdout|stderr)\b", text, re.MULTILINE
        ):
            rel = py.relative_to(root)
            errors.append(
                f"Forbidden 'from sys import stdout/stderr' in {rel}; use console helpers"
            )
    return errors


def main() -> int:
    """Main entry point for pre-commit validation checks.

    Returns:
        0 if all checks pass, 1 if any violations found
    """
    root = ROOT

    problems: list[str] = []
    problems += check_permissions(root)
    problems += check_forbidden_imports(root)
    problems += check_json_purity(root)
    problems += check_forbidden_sys_stream_writes(root)
    if problems:
        for p in problems:
            console.error(str(p))
        return 1
    # Be silent on success to avoid noisy pre-commit output.
    return 0


if __name__ == "__main__":
    sys.exit(main())
