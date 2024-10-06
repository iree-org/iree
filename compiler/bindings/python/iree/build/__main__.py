# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence

import argparse
import importlib
import importlib.util
from pathlib import Path
import sys

from iree.build.executor import Entrypoint, BuildContext, Executor


class ModuleWrapper:
    """Wraps a raw, loaded module with access to discovered details."""

    def __init__(self, mod):
        self.mod = mod
        self.entrypoints = self._collect_entrypoints()

    @staticmethod
    def load_module(module_name: str) -> "ModuleWrapper":
        return ModuleWrapper(importlib.import_module(module_name))

    def load_py_file(module_path: str) -> "ModuleWrapper":
        spec = importlib.util.spec_from_file_location("__sfbuild__", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return ModuleWrapper(mod)

    def _collect_entrypoints(self) -> dict[str, Entrypoint]:
        results: dict[str, Entrypoint] = {}
        for attr_name, attr_value in self.mod.__dict__.items():
            if isinstance(attr_value, Entrypoint):
                results[attr_name] = attr_value
        return results


def command_list(mod: ModuleWrapper, args, rest_argv):
    for target_name in mod.targets.keys():
        print(target_name)


def command_build(mod: ModuleWrapper, args, rest_argv):
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path.cwd()
    executor = Executor(output_dir)

    # Analyze.
    executor.analyze(*mod.entrypoints.values())

    # Build.
    # TODO: Should resolve them from arguments vs just building all.
    executor.build(*executor.entrypoints)


def parse_top_level_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description="Shortfin program build driver")
    p.add_argument(
        "-m",
        action="store_true",
        help="Interpret the build definitions argument as a module (vs a file)",
    )
    p.add_argument(
        "module_or_file",
        help="The Python file or module from which to load build definitions",
    )

    subp = p.add_subparsers(required=True)
    list_p = subp.add_parser("list", help="List targets")
    list_p.set_defaults(func=command_list)

    build_p = subp.add_parser("build", help="Build targets")
    build_p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the build tree",
    )
    build_p.add_argument("target_names", nargs="+", help="Target names to build")
    build_p.set_defaults(func=command_build, needs_rest_argv=True)

    args, rest = p.parse_known_args(args=argv)
    return args, rest


def main(argv: Sequence[str] | None = None):
    args, rest_argv = parse_top_level_args(argv)

    is_module = args.m or _is_module_like(args.module_or_file)
    if is_module:
        mod = ModuleWrapper.load_module(args.module_or_file)
    else:
        mod = ModuleWrapper.load_py_file(args.module_or_file)

    if not hasattr(args, "needs_rest_argv") and rest_argv:
        print(f"ERROR: Unexpected command line arguments: {rest_argv}", file=sys.stderr)
        sys.exit(1)
    args.func(mod, args, rest_argv)


def _is_module_like(s: str) -> bool:
    return "/" not in s and "\\" not in s and not s.endswith(".py")


if __name__ == "__main__":
    main()
