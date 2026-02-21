# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Custom lint hooks for pre-commit (scoped to tools/utils/*).

Usage (pre-commit local hook): pass staged file paths as arguments. Exits non-zero
if any violations are found and prints a short diagnostic per violation.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

LICENSE_LINE = "Licensed under the Apache License v2.0 with LLVM Exceptions."
SEE_LINE = "See https://llvm.org/LICENSE.txt for license information."
SPDX_LINE = "SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception"


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(errors="ignore")


def check_license_header(p: Path, text: str, problems: list[str]) -> None:
    """Check that file has standard IREE license header.

    Args:
        p: Path to file being checked
        text: File contents
        problems: List to append error messages to
    """
    if p.name == "__init__.py":
        return
    # Shebang is optional; required only for top-level entrypoints. We don't enforce it here.
    head = text.splitlines()[:8]
    if not (
        any(LICENSE_LINE in line for line in head)
        and any(SEE_LINE in line for line in head)
        and any(SPDX_LINE in line for line in head)
    ):
        problems.append(f"{p}: missing standard IREE license header")


def check_shebang_policy(p: Path, text: str, problems: list[str]) -> None:
    """Check that shebangs only appear in bin/ entrypoints.

    Args:
        p: Path to file being checked
        text: File contents
        problems: List to append error messages to
    """
    # Only allow shebangs in true CLI entry points under tools/utils/bin/
    if text.startswith("#!/") and not p.as_posix().startswith("tools/utils/bin/"):
        problems.append(f"{p}: shebang found in non-entrypoint module; remove it")


def check_first_line_is_copyright(p: Path, text: str, problems: list[str]) -> None:
    """Check that copyright header is on first line (or immediately after shebang).

    Args:
        p: Path to file being checked
        text: File contents
        problems: List to append error messages to
    """
    # Ensure there is no leading blank line before the copyright header
    if p.name == "__init__.py":
        return
    if text.startswith("#!/"):
        # Shebang allowed only in bin/, but even there we want copyright next
        # line without extra blank lines.
        lines = text.splitlines()
        if len(lines) >= 2 and not lines[1].startswith("# Copyright "):
            problems.append(
                f"{p}: copyright header must immediately follow shebang or be first line"
            )
        return
    if not text.startswith("# Copyright "):
        problems.append(f"{p}: file must start with the standard IREE copyright header")


def check_no_stderr_print(p: Path, tree: ast.AST, problems: list[str]) -> None:
    """Check that code uses console helpers instead of print(..., file=sys.stderr).

    Args:
        p: Path to file being checked
        tree: AST of file
        problems: List to append error messages to
    """
    if p.as_posix().endswith("tools/utils/lit_tools/core/console.py"):
        return

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                for kw in node.keywords:
                    if (
                        kw.arg == "file"
                        and isinstance(kw.value, ast.Attribute)
                        and isinstance(kw.value.value, ast.Name)
                        and kw.value.value.id == "sys"
                        and kw.value.attr == "stderr"
                    ):
                        problems.append(
                            f"{p}:{node.lineno}:{node.col_offset}: use console.error/warn/note instead of print(..., file=sys.stderr)"
                        )
            self.generic_visit(node)

    V().visit(tree)


def check_no_numeric_sys_exit(p: Path, tree: ast.AST, problems: list[str]) -> None:
    """Check that code uses exit_codes constants instead of numeric sys.exit.

    Args:
        p: Path to file being checked
        tree: AST of file
        problems: List to append error messages to
    """

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            is_sys_exit = False
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                is_sys_exit = node.func.value.id == "sys" and node.func.attr == "exit"
            if (
                is_sys_exit
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, int)
            ):
                problems.append(
                    f"{p}:{node.lineno}:{node.col_offset}: use exit_codes.SUCCESS/ERROR/NOT_FOUND instead of numeric sys.exit"
                )
            self.generic_visit(node)

    V().visit(tree)


def check_open_without_encoding(p: Path, tree: ast.AST, problems: list[str]) -> None:
    """Check that open() calls for writing include encoding parameter.

    Args:
        p: Path to file being checked
        tree: AST of file
        problems: List to append error messages to
    """

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                mode = None
                if (
                    len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    mode = node.args[1].value
                for kw in node.keywords:
                    if (
                        kw.arg == "mode"
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)
                    ):
                        mode = kw.value.value
                has_encoding = any(kw.arg == "encoding" for kw in node.keywords)
                if mode and any(ch in mode for ch in ("w", "a")) and not has_encoding:
                    problems.append(
                        f"{p}:{node.lineno}:{node.col_offset}: open(..., '{mode}') without encoding= — use fs.safe_write_text or add encoding='utf-8'"
                    )
            self.generic_visit(node)

    V().visit(tree)


def check_cli_common_flags(p: Path, text: str, problems: list[str]) -> None:
    """Check that CLI tools use cli.add_common_output_flags.

    Args:
        p: Path to file being checked
        text: File contents
        problems: List to append error messages to
    """
    # Heuristic: if ArgumentParser appears and cli.add_common_output_flags does not, warn.
    if "# lint: disable=cli-flags" in text:
        return
    if "ArgumentParser(" in text and "cli.add_common_output_flags(" not in text:
        problems.append(
            f"{p}: add common flags with cli.add_common_output_flags(parser)"
        )


def main(argv: list[str]) -> int:
    """Main entry point for lint hooks.

    Args:
        argv: Command line arguments (file paths to check)

    Returns:
        0 if no problems found, 1 otherwise
    """
    problems: list[str] = []
    for arg in argv[1:]:
        p = Path(arg)
        if p.suffix != ".py" or not p.exists():
            continue
        text = _read_text(p)
        try:
            tree = ast.parse(text, filename=str(p))
        except SyntaxError as e:
            problems.append(f"{p}:{e.lineno}:{e.offset}: syntax error: {e.msg}")
            continue
        # Checks
        check_license_header(p, text, problems)
        check_no_stderr_print(p, tree, problems)
        check_no_numeric_sys_exit(p, tree, problems)
        check_open_without_encoding(p, tree, problems)
        check_cli_common_flags(p, text, problems)
        check_shebang_policy(p, text, problems)
        check_first_line_is_copyright(p, text, problems)

    if problems:
        print("\n".join(problems))  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
