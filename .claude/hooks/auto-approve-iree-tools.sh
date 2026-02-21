#!/usr/bin/env bash
# =============================================================================
# auto-approve-iree-tools.sh - Auto-approve IREE tool invocations
# =============================================================================
#
# PURPOSE:
#   Automatically approve permission dialogs for IREE tool invocations in
#   Claude Code, eliminating repeated permission prompts during development.
#
# WHEN IT ACTIVATES:
#   When Claude Code attempts to run a `Bash` command where:
#   1. The command basename starts with `iree-` (e.g., iree-compile, iree-opt)
#   2. The current working directory is within the IREE project
#      (checked via $CLAUDE_PROJECT_DIR)
#
# WHAT IT ENABLES:
#   - Seamless use of IREE tools without repeated permission prompts
#   - Heredoc syntax: `iree-lit-test << 'EOF' ... EOF`
#   - Environment prefixes: `PYTHONPATH=foo iree-opt ...`
#   - Flexible build directories: `../any-build-dir/tools/iree-compile`
#
# SCOPE AND SECURITY:
#   - Execution context: Must be working in IREE project directory
#   - Tool location: Binary can exist anywhere (build dirs, $PATH, submodules)
#   - Risk: If a malicious binary named `iree-*` exists, it will be
#     auto-approved when working in IREE
#
# =============================================================================
set -euo pipefail

# If python3 is not available in this shell environment, gracefully no-op.
if ! command -v python3 >/dev/null 2>&1; then
  # Exit 0 with no stdout => hook is treated as "no output, no decision".
  exit 0
fi

# Capture stdin from Claude Code.
input=$(cat)

# Debug: Uncomment to log stdin to file for debugging.
# echo "$input" > /tmp/iree-hook-debug-stdin.json

# Run the JSON logic in an inline python3 block, passing JSON as argument.
python3 -c '
import json
import os
import re
import shlex
import sys
from typing import Optional

ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")

def parse_command_word(command: str) -> Optional[str]:
    """Return first non-env-assignment token from a shell command string."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None

    if not tokens:
        return None

    idx = 0
    n = len(tokens)

    # Skip leading env assignments like FOO=bar BAR=baz.
    while idx < n and ASSIGNMENT_RE.fullmatch(tokens[idx]):
        idx += 1

    if idx >= n:
        return None

    return tokens[idx]

def is_within_project() -> bool:
    """Return True if current working directory is within IREE project."""
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if not project_dir:
        return False
    cwd = os.path.realpath(os.getcwd())
    project_root = os.path.realpath(project_dir)
    return cwd.startswith(project_root)

def is_iree_tool_command(command: str) -> bool:
    cmd_word = parse_command_word(command)
    if not cmd_word:
        return False
    basename = os.path.basename(cmd_word)
    return basename.startswith("iree-")

try:
    # Read JSON from command-line argument.
    data = json.loads(sys.argv[1])
except (json.JSONDecodeError, IndexError):
    # Invalid input => no-op.
    sys.exit(0)

tool_name = data.get("tool_name", "")
tool_input = data.get("tool_input", {}) or {}
command = tool_input.get("command", "") or ""

# Only care about Bash tool permission dialogs with a non-empty command.
if tool_name != "Bash" or not command.strip():
    sys.exit(0)

# Only match IREE tools (basename starts with "iree-").
if not is_iree_tool_command(command):
    sys.exit(0)

# Only auto-approve when executing from within the IREE project.
if not is_within_project():
    sys.exit(0)

# Emit PreToolUse decision JSON to auto-approve this call.
output = {
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow"
    }
}
json.dump(output, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()
' "$input"
