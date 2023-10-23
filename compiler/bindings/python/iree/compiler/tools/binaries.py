# Lint-as: python3
"""Utilities for locating and invoking compiler tool binaries."""

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import io
import logging
import os
import platform
import shlex
import subprocess
import sys
import textwrap
import threading

from typing import List, Optional, Union

__all__ = [
    "find_tool",
    "invoke_immediate",
    "invoke_pipeline",
    "get_tool_path",
    "CompilerToolError",
]

_BUILTIN_TOOLS = [
    "iree-compile",
    "iree-lld",
]

# In normal distribution circumstances, each named tool is associated with
# a python module that provides a `get_tool` function for getting its absolute
# path. This dictionary maps the tool name to the module.
_TOOL_MODULE_MAP = {
    # Note that iree-compile is builtin, but if not found, it can be resolved
    # in the external 'core' module. This is used for some outside packaging
    # options.
    "iree-compile": "iree.tools.core",
    "iree-lld": "iree.tools.core",
    "iree-import-tflite": "iree.tools.tflite",
    "iree-import-tf": "iree.tools.tf",
}

# Map of tool module to package name as distributed to archives (used for
# error messages).
_TOOL_MODULE_PACKAGES = {
    "iree.tools.core": "<none>",
    "iree.tools.tf": "iree-tools-tf",
    "iree.tools.tflite": "iree-tools-tflite",
}

# Environment variable holding directories to be searched for named tools.
# Delimitted by os.pathsep.
_TOOL_PATH_ENVVAR = "IREE_TOOL_PATH"

# We do complicated logging so retain our own Logger instance.
logger = logging.getLogger(__name__)


class CompilerToolError(Exception):
    """Compiler exception that preserves the command line and error output."""

    def __init__(self, process: subprocess.CompletedProcess):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)  # Decode error or other: best we can do.

        tool_name = os.path.basename(process.args[0])
        super().__init__(
            f"Error invoking IREE compiler tool {tool_name}\n"
            f"Error code: {process.returncode}\n"
            f"Diagnostics:\n{errs}\n\n"
            f"Invoked with:\n {tool_name} {' '.join(process.args)}\n\n"
            f"Need more information? Set IREE_SAVE_TEMPS=/some/dir "
            f"in your environment to save all artifacts and "
            f"reproducers.\n"
        )


def get_tool_path() -> List[str]:
    """Returns list of paths to search for tools."""
    list_str = os.environ.get(_TOOL_PATH_ENVVAR)
    if not list_str:
        return []
    return list_str.split(os.pathsep)


def find_tool(exe_name: str) -> str:
    """Finds a tool by its (extension-less) executable name.

    Args:
      exe_name: The name of the executable (extension-less).
    Returns:
      An absolute path to the tool.
    Raises:
      ValueError: If the tool is not known or not found.
    """
    is_builtin = exe_name in _BUILTIN_TOOLS
    if not is_builtin and exe_name not in _TOOL_MODULE_MAP:
        raise ValueError(f"IREE compiler tool '{exe_name}' is not a known tool")

    # First search an explicit tool path (from environment).
    tool_path = get_tool_path()
    for path_entry in tool_path:
        if not path_entry:
            continue
        candidate_exe = os.path.join(path_entry, exe_name)
        if _is_executable(candidate_exe):
            return candidate_exe

    if is_builtin:
        # Get builtin tool.
        candidate_exe = _get_builtin_tool(exe_name)
        if _is_executable(candidate_exe):
            return candidate_exe

    # Fall-through and attempt to find it via a tools module.
    # Attempt to load the tool module.
    tool_module_name = _TOOL_MODULE_MAP[exe_name]
    tool_module_package = _TOOL_MODULE_PACKAGES[tool_module_name]
    try:
        tool_module = importlib.import_module(tool_module_name)
    except ModuleNotFoundError:
        raise ValueError(
            f"IREE compiler tool '{exe_name}' is not installed (it should have been "
            f"found in the python module '{tool_module_name}', typically installed "
            f"via the package {tool_module_package}).\n\n"
            f"Either install the package or set the {_TOOL_PATH_ENVVAR} environment "
            f"variable to contain the path of the tool executable "
            f"(current {_TOOL_PATH_ENVVAR} = {repr(tool_path)})."
        ) from None

    # Ask the module for its tool.
    candidate_exe = tool_module.get_tool(exe_name)

    if not _is_executable(candidate_exe):
        raise ValueError(
            f"IREE compiler tool '{exe_name}' was located in module "
            f"'{tool_module_name}' but the file was not found or not executable: "
            f"{candidate_exe}"
        )
    return candidate_exe


def _get_builtin_tool(exe_name: str) -> Optional[str]:
    if platform.system() == "Windows":
        exe_name = exe_name + ".exe"
    this_path = os.path.dirname(__file__)
    tool_path = os.path.join(this_path, "..", "_mlir_libs", exe_name)
    return tool_path


def _is_executable(candidate_exe: str) -> bool:
    if not candidate_exe:
        return False
    if not os.path.isfile(candidate_exe):
        return False
    if not os.access(candidate_exe, os.X_OK):
        return False
    return True


def invoke_immediate(
    command_line: List[str], *, input_file: Optional[bytes] = None, immediate_input=None
):
    """Invokes an immediate command.

    This is separate from invoke_pipeline as it is simpler and supports more
    complex input redirection, using recommended facilities for sub-processes
    (less magic).

    Note that this differs from the usual way of using subprocess.run or
    subprocess.Popen().communicate() because we need to pump all of the error
    streams individually and only pump pipes not connected to a different stage.
    Uses threads to pump everything that is required.
    """
    if logger.isEnabledFor(logging.INFO):
        logging.info("Invoke IREE Tool: %s", _quote_command_line(command_line))
    run_args = {}
    input_file_handle = None
    stderr_handle = sys.stderr
    try:
        # Redirect input.
        if input_file is not None:
            input_file_handle = open(input_file, "rb")
            run_args["stdin"] = input_file_handle
        elif immediate_input is not None:
            run_args["input"] = immediate_input

        process = subprocess.run(command_line, capture_output=True, **run_args)
        if process.returncode != 0:
            raise CompilerToolError(process)
        # Emit stderr contents.
        _write_binary_stderr(stderr_handle, process.stderr)
        return process.stdout
    finally:
        if input_file_handle:
            input_file_handle.close()


def invoke_pipeline(command_lines: List[List[str]], immediate_input=None):
    """Invoke a pipeline of commands.

    The first stage of the pipeline will have its stdin set to DEVNULL and each
    subsequent stdin will derive from the prior stdout. The final stdout will
    be accumulated and returned. All stderr contents are accumulated and printed
    to stderr on completion or the first failing stage of the pipeline will have
    an exception raised with its stderr output.
    """
    logging.info(
        "Invoke IREE Pipeline:\n  %s",
        "\n  ".join([_quote_command_line(line) for line in command_lines]),
    )

    stages = []
    pipeline_input = subprocess.DEVNULL if immediate_input is None else subprocess.PIPE
    prev_out = pipeline_input
    stderr_handle = sys.stderr

    # Create all stages.
    for i in range(len(command_lines)):
        command_line = command_lines[i]
        popen_args = {
            "stdin": prev_out,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        process = subprocess.Popen(command_line, **popen_args)
        prev_out = process.stdout
        capture_output = i == (len(command_lines) - 1)
        stages.append(_PipelineStage(process, capture_output))

    # Start stages.
    for stage in stages:
        stage.start()

    # Pump input.
    pipe_success = True
    if immediate_input is not None:
        try:
            pipe_success = False
            stages[0].process.stdin.write(immediate_input)
            pipe_success = True
        finally:
            stages[0].process.stdin.close()

    # Join.
    for stage in stages:
        stage.join()

    # Check for errors.
    for stage in stages:
        assert stage.completed
        if stage.completed.returncode != 0:
            raise CompilerToolError(stage.completed)

    # Broken pipe.
    if not pipe_success:
        raise CompilerToolError(stages[0].completed)

    # Print any stderr output.
    for stage in stages:
        _write_binary_stderr(stderr_handle, stage.errs)
    return stages[-1].outs


class _PipelineStage(threading.Thread):
    """Wraps a process and pumps its handles, waiting for completion."""

    def __init__(self, process, capture_output):
        super().__init__()
        self.process = process
        self.capture_output = capture_output
        self.completed: Optional[subprocess.CompletedProcess] = None
        self.outs = None
        self.errs = None

    def pump_stderr(self):
        self.errs = self.process.stderr.read()

    def pump_stdout(self):
        self.outs = self.process.stdout.read()

    def run(self):
        stderr_thread = threading.Thread(target=self.pump_stderr)
        stderr_thread.start()
        if self.capture_output:
            stdout_thread = threading.Thread(target=self.pump_stdout)
            stdout_thread.start()
        self.process.wait()
        stderr_thread.join()
        if self.capture_output:
            stdout_thread.join()
        self.completed = subprocess.CompletedProcess(
            self.process.args, self.process.returncode, self.outs, self.errs
        )
        self.process.stderr.close()
        self.process.stdout.close()


def _write_binary_stderr(out_handle, contents):
    # Fast-paths buffered text-io (which stderr is by default) while allowing
    # full decode for non buffered and binary io.
    if hasattr(out_handle, "buffer"):
        out_handle.buffer.write(contents)
    elif isinstance(out_handle, io.TextIOBase):
        out_handle.write(contents.decode("utf-8"))
    else:
        out_handle.write(contents)


def _quote_command_line(command_line: List[str]) -> str:
    return " ".join([shlex.quote(token) for token in command_line])
