# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, IO

import argparse
import importlib
import importlib.util
from pathlib import Path
import sys

from iree.build.args import (
    argument_namespace_context,
    configure_arg_parser,
    run_global_arg_handlers,
)
from iree.build.executor import BuildEntrypoint, BuildFile, Entrypoint, Executor

__all__ = [
    "iree_build_main",
    "load_build_module",
]


def iree_build_main(
    module="__main__",
    args: list[str] | None = None,
    stdout: IO | None = None,
    stderr: IO | None = None,
):
    """Make a build module invoke iree.build on itself when run.

    Typically, if you have a module that declares build entrypoints, you will
    add a stanza at the end:

    .. code-block:: python
        from iree.build import *

        if __name__ == "__main__":
            iree_build_main()
    """
    main = CliMain(module=module, args=args, stdout=stdout, stderr=stderr)
    main.run()


def load_build_module(module_path: Path | str):
    """Loads a build module by path, evaling and returning it."""
    spec = importlib.util.spec_from_file_location("__iree_build__", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CliMain:
    """Composes command line programs."""

    def __init__(
        self,
        *,
        args: list[str] | None = None,
        module=None,
        stdout: IO | None = None,
        stderr: IO | None = None,
    ):
        self.stdout = stdout if stdout is not None else sys.stdout
        self.stderr = stderr if stderr is not None else sys.stderr
        if args is None:
            args = sys.argv[1:]
        if module is not None and isinstance(module, str):
            module = __import__(module)
        module = module

        p = self.arg_parser = argparse.ArgumentParser(
            description="IREE program build driver"
        )
        if module is None:
            args, self.top_module = self._resolve_module_arguments(args)
        else:
            self.top_module = ModuleWrapper(module)

        p.add_argument(
            "--output-dir",
            type=Path,
            default=Path.cwd(),
            help="Output directory for the build tree (defaults to current directory)",
        )

        cmd_group_desc = p.add_argument_group(
            title="Build command",
            description="Selects a build sub-command to invoke (default '--build')",
        )
        cmd_group = cmd_group_desc.add_mutually_exclusive_group()
        cmd_group.add_argument(
            "--build",
            dest="command",
            action="store_const",
            const=self.build_command,
            help="Executes build actions",
        )
        cmd_group.add_argument(
            "--list",
            dest="command",
            action="store_const",
            const=self.list_command,
            help="Lists top level build actions",
        )

        cmd_group.add_argument(
            "--list-all",
            dest="command",
            action="store_const",
            const=self.list_all_command,
            help="Lists all build actions",
        )

        p.add_argument(
            "action_path",
            nargs="*",
            help="Paths of actions to build (default to top-level actions)",
        )

        configure_arg_parser(p)
        self._define_action_arguments(p)
        self.args = self.arg_parser.parse_args(args)

    def abort(self):
        sys.exit(1)

    def _define_action_arguments(self, p: argparse.ArgumentParser):
        user_group = p.add_argument_group("Action defined options")
        for ep in self.top_module.entrypoints.values():
            for cl_arg in ep.cl_arg_defs:
                cl_arg.define_arg(user_group)

    def _resolve_module_arguments(
        self, args: list[str]
    ) -> tuple[list[str], "ModuleWrapper"]:
        p = argparse.ArgumentParser(
            add_help=False,
            usage="python -m iree.build [-m] build_module [... additional module specific options ...]",
            prog="python -m iree.build",
        )
        # Invoked as a standalone tool: need the user to specify the
        # module.
        p.add_argument(
            "-m",
            dest="parse_as_module",
            action="store_true",
            help="Interpret the build definitions argument as a module (vs a file)",
        )
        p.add_argument(
            "build_module",
            help="The Python file or module from which to load build definitions",
        )

        bootstrap_args, rem_args = p.parse_known_args(args)
        # Resolve from arguments.
        is_module = bootstrap_args.parse_as_module or _is_module_like_str(
            bootstrap_args.build_module
        )
        if is_module:
            try:
                top_module = ModuleWrapper.load_module(bootstrap_args.build_module)
            except ModuleNotFoundError as e:
                print(
                    f"ERROR: Module '{bootstrap_args.build_module}' not found: {e}",
                    file=self.stderr,
                )
                self.abort()
        else:
            top_module = ModuleWrapper.load_py_file(bootstrap_args.build_module)
        return rem_args, top_module

    def _create_executor(self) -> Executor:
        executor = Executor(self.args.output_dir, stderr=self.stderr)
        executor.analyze(*self.top_module.entrypoints.values())
        return executor

    def run(self):
        with argument_namespace_context(self.args):
            run_global_arg_handlers(self.args)
            command = self.args.command
            if command is None:
                command = self.build_command
            command()

    def build_command(self):
        executor = self._create_executor()

        if not self.args.action_path:
            # Default to all.
            build_actions = list(executor.entrypoints)
        else:
            # Look up each requested and add it.
            build_actions = []
            for action_path in self.args.action_path:
                try:
                    build_actions.append(executor.all[action_path])
                except KeyError:
                    all_paths = "\n".join(executor.all.keys())
                    print(
                        f"ERROR: Action '{action_path}' not found. Available: \n{all_paths}",
                        file=self.stderr,
                    )
                    self.abort()
        executor.build(*build_actions)

        for build_action in build_actions:
            if isinstance(build_action, BuildEntrypoint):
                for output in build_action.outputs:
                    print(output.get_fs_path(), file=self.stdout)
            elif isinstance(build_action, BuildFile):
                print(build_action.get_fs_path(), file=self.stdout)

    def list_command(self):
        executor = self._create_executor()
        for ep in executor.entrypoints:
            print(ep.path, file=self.stdout)

    def list_all_command(self):
        executor = self._create_executor()
        for name in executor.all.keys():
            if name:
                print(name, file=self.stdout)


class ModuleWrapper:
    """Wraps a raw, loaded module with access to discovered details."""

    def __init__(self, mod):
        self.mod = mod
        self.entrypoints = self._collect_entrypoints()

    @staticmethod
    def load_module(module_name: str) -> "ModuleWrapper":
        return ModuleWrapper(importlib.import_module(module_name))

    @staticmethod
    def load_py_file(module_path: Path | str) -> "ModuleWrapper":
        return ModuleWrapper(load_build_module(str(module_path)))

    def _collect_entrypoints(self) -> dict[str, Entrypoint]:
        results: dict[str, Entrypoint] = {}
        for attr_name, attr_value in self.mod.__dict__.items():
            if isinstance(attr_value, Entrypoint):
                results[attr_name] = attr_value
        return results


def _is_module_like_str(s: str) -> bool:
    return "/" not in s and "\\" not in s and not s.endswith(".py")
