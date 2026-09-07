#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compare effective ABI settings in host and installed-plugin compilations.

This checks selected compiler macros, not full binary compatibility.
"""

import json
import pathlib
import shlex
import subprocess
import sys

_MACROS = (
    "NDEBUG",
    "__GXX_RTTI",
    "__EXCEPTIONS",
    "__cplusplus",
    "__SIZEOF_POINTER__",
    "__BYTE_ORDER__",
    "_GLIBCXX_USE_CXX11_ABI",
    "_GLIBCXX_DEBUG",
    "_LIBCPP_VERSION",
    "_LIBCPP_ABI_VERSION",
    "LLVM_ENABLE_ABI_BREAKING_CHECKS",
    "LLVM_ENABLE_REVERSE_ITERATION",
)
_SOURCE = """
#include <string>
#define LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING 1
#include "llvm/Config/abi-breaking.h"
"""


def settings(entry):
    args = entry.get("arguments") or shlex.split(entry["command"])
    command = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg in ("-o", "-MF", "-MT", "-MQ"):
            skip = True
        elif arg not in ("-c", "-MD", "-MMD", entry["file"]):
            command.append(arg)
    result = subprocess.run(
        command + ["-dM", "-E", "-x", "c++", "-"],
        input=_SOURCE,
        text=True,
        capture_output=True,
        cwd=entry["directory"],
        check=True,
    )
    macros = {}
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) >= 2 and fields[0] == "#define":
            macros[fields[1]] = fields[2] if len(fields) == 3 else ""
    return {name: macros.get(name) for name in _MACROS}


def main():
    host = json.loads(pathlib.Path(sys.argv[1]).read_text())
    plugin = json.loads(pathlib.Path(sys.argv[2]).read_text())
    host_entry = next(
        entry
        for entry in host
        if entry["file"].endswith("/compiler/PluginAPI/Client.cpp")
    )
    expected = settings(host_entry)
    for filename in ("plugin.cpp", "helper.cpp"):
        entry = next(e for e in plugin if pathlib.Path(e["file"]).name == filename)
        actual = settings(entry)
        differences = [
            f"{key}: host={expected[key]!r}, plugin={actual[key]!r}"
            for key in _MACROS
            if expected[key] != actual[key]
        ]
        if differences:
            raise SystemExit(
                f"{filename}: incompatible build settings:\n" + "\n".join(differences)
            )
    print("PASS: plugin and helper match the host's checked ABI settings")


if __name__ == "__main__":
    main()
