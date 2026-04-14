#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Captures one process invocation with Tracy."""

from __future__ import annotations

import argparse
import datetime
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def eprint(message: str) -> None:
    print(f"[iree-tracy-capture] {message}", file=sys.stderr)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Starts tracy-capture, runs one command, and writes a .tracy file.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(".tracy"))
    parser.add_argument("--name")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--capture-tool", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("expected -- <command> [args...]")
    return args


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sanitize_name(value: str) -> str:
    value = value.strip().replace("//", "").replace(":", "-").replace("/", "-")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return value or "trace"


def choose_port(requested_port: int) -> int:
    if requested_port:
        return requested_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listen_socket:
        listen_socket.bind(("127.0.0.1", 0))
        return int(listen_socket.getsockname()[1])


def executable(path: Path) -> bool:
    return path.exists() and os.access(path, os.X_OK)


def find_capture_tool(explicit_tool: Path | None) -> Path:
    root = repository_root()
    candidates: list[Path] = []
    if explicit_tool is not None:
        candidates.append(explicit_tool)
    if os.environ.get("IREE_TRACY_CAPTURE"):
        candidates.append(Path(os.environ["IREE_TRACY_CAPTURE"]))
    candidates.extend(
        [
            root / "third_party/tracy/capture/build/tracy-capture",
            root / "build_tools/third_party/tracy/build/iree-tracy-capture",
        ]
    )
    path_from_path = shutil.which("tracy-capture")
    if path_from_path:
        candidates.append(Path(path_from_path))
    for candidate in candidates:
        if executable(candidate):
            return candidate
    raise RuntimeError(
        "Could not find tracy-capture; set IREE_TRACY_CAPTURE or build the "
        "checkout's third_party/tracy/capture tool"
    )


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def run_workload(
    command: list[str],
    *,
    env: dict[str, str],
    capture_process: subprocess.Popen,
) -> int:
    workload_process = subprocess.Popen(command, env=env)
    try:
        while True:
            workload_return_code = workload_process.poll()
            if workload_return_code is not None:
                return int(workload_return_code)
            if capture_process.poll() is not None:
                terminate(workload_process)
                raise RuntimeError("tracy-capture exited before the workload")
            time.sleep(0.1)
    except BaseException:
        terminate(workload_process)
        raise


def read_capture_output(capture_output) -> str:
    capture_output.flush()
    capture_output.seek(0)
    return capture_output.read().strip()


def main() -> int:
    args = parse_arguments()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    trace_name = sanitize_name(args.name or Path(args.command[0]).name)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{timestamp}-{trace_name}.tracy"
    trace_port = choose_port(args.port)

    run_env = os.environ.copy()
    run_env["TRACY_NO_EXIT"] = "1"
    run_env["TRACY_ONLY_IPV4"] = "1"
    run_env["TRACY_ONLY_LOCALHOST"] = "1"
    run_env["TRACY_PORT"] = str(trace_port)

    capture_output = None
    capture_process: subprocess.Popen | None = None
    workload_return_code = 1
    try:
        capture_tool = find_capture_tool(args.capture_tool)
        capture_command = [
            str(capture_tool),
            "-o",
            str(trace_path),
            "-f",
            "-a",
            args.address,
            "-p",
            str(trace_port),
        ]
        capture_output = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        eprint(shlex.join(capture_command))
        capture_process = subprocess.Popen(
            capture_command,
            stdout=capture_output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(0.2)
        if capture_process.poll() is not None:
            raise RuntimeError("tracy-capture exited before the workload")

        eprint(shlex.join(args.command))
        workload_return_code = run_workload(
            args.command,
            env=run_env,
            capture_process=capture_process,
        )

        eprint("waiting for tracy-capture to finish")
        capture_return_code = int(capture_process.wait())
        if capture_return_code != 0:
            raise RuntimeError(
                f"tracy-capture failed with exit code {capture_return_code}"
            )
        if not trace_path.exists():
            raise RuntimeError(f"tracy-capture did not produce {trace_path}")

        eprint(f"trace: {trace_path}")
        return workload_return_code
    except KeyboardInterrupt:
        if capture_process is not None:
            terminate(capture_process)
        raise
    except Exception as exc:
        eprint(str(exc))
        if capture_output is not None:
            output = read_capture_output(capture_output)
            if output:
                eprint("tracy-capture output:")
                sys.stderr.write(output + "\n")
        return workload_return_code if workload_return_code != 0 else 1
    finally:
        if capture_process is not None and capture_process.poll() is None:
            terminate(capture_process)
        if capture_output is not None:
            capture_output.close()


if __name__ == "__main__":
    sys.exit(main())
