# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import shlex

import iree.compiler.api as compiler_api
import iree.compiler.tools as compiler_tools

from iree.build.args import (
    expand_cl_arg_defaults,
    register_arg_handler_callback,
    register_arg_parser_callback,
    cl_arg_ref,
)

from iree.build.executor import (
    BuildAction,
    BuildContext,
    BuildFile,
    BuildFileLike,
    FileNamespace,
)

from iree.build.metadata import CompileSourceMeta
from iree.build.target_machine import compute_target_machines_from_flags

__all__ = [
    "compile",
]


@register_arg_parser_callback
def _(p: argparse.ArgumentParser):
    g = p.add_argument_group(
        title="IREE Compiler Options",
        description="Global options controlling invocation of iree-compile",
    )
    g.add_argument(
        "--iree-compile-out-of-process",
        action=argparse.BooleanOptionalAction,
        help="Invokes iree-compiler as an out of process executable (the default is to "
        "invoke it in-process via API bindings). This can make debugging somewhat "
        "easier and also grants access to global command line options that may not "
        "otherwise be available.",
    )
    g.add_argument(
        "--iree-compile-extra-args",
        help="Extra arguments to pass to iree-compile. When running in-process, these "
        "will be passed as globals to the library and effect all compilation in the "
        "process. These are split with shlex rules.",
    )


@register_arg_handler_callback
def _(ns: argparse.Namespace):
    in_process = not ns.iree_compile_out_of_process
    extra_args_str = ns.iree_compile_extra_args
    if in_process and extra_args_str:
        # TODO: This is very unsafe. If called multiple times (i.e. in a library)
        # or with illegal arguments, the program will abort. The safe way to do
        # this is to spawn one child process and route all "in process" compilation
        # there. This would allow explicit control of startup/shutdown and would
        # provide isolation in the event of a compiler crash. It is still important
        # for a single process to handle all compilation activities since this
        # allows global compiler resources (like threads) to be pooled and not
        # saturate the machine resources.
        extra_args_list = shlex.split(extra_args_str)
        compiler_api._initializeGlobalCL("unused_prog_name", *extra_args_list)


class CompilerInvocation:
    @expand_cl_arg_defaults
    def __init__(
        self,
        *,
        input_file: BuildFile,
        output_file: BuildFile,
        out_of_process: bool = cl_arg_ref("iree_compile_out_of_process"),
        extra_args_str=cl_arg_ref("iree_compile_extra_args"),
    ):
        self.input_file = input_file
        self.output_file = output_file
        # We manage most flags as keyword values that can have at most one
        # setting.
        self.kw_flags: dict[str, str | None] = {}
        # Flags can also be set free-form. These are always added to the command
        # line after the kw_flags.
        self.extra_flags: list[str] = []
        self.out_of_process = out_of_process

        if extra_args_str:
            self.extra_args = shlex.split(extra_args_str)
        else:
            self.extra_args = []

    def run(self):
        raw_flags: list[str] = []

        # Set any defaults derived from the input_file metadata. These are set
        # first because they can be overriden by explicit flag settings.
        meta = CompileSourceMeta.get(self.input_file)
        raw_flags.append(f"--iree-input-type={meta.input_type}")

        # Process kw_flags.
        for key, value in self.kw_flags.items():
            if value is None:
                raw_flags.append(f"--{key}")
            else:
                raw_flags.append(f"--{key}={value}")

        # Process extra_flags.
        for raw in self.extra_flags:
            raw_flags.append(raw)

        if self.out_of_process:
            self.run_out_of_process(raw_flags)
        else:
            self.run_inprocess(raw_flags)

    def run_inprocess(self, flags: list[str]):
        with compiler_api.Session() as session:
            session.set_flags(*flags)
            with compiler_api.Invocation(session) as inv, compiler_api.Source.open_file(
                session, str(self.input_file.get_fs_path())
            ) as source, compiler_api.Output.open_file(
                str(self.output_file.get_fs_path())
            ) as output:
                inv.enable_console_diagnostics()
                inv.parse_source(source)
                if not inv.execute():
                    raise RuntimeError("COMPILE FAILED (TODO)")
                inv.output_vm_bytecode(output)
                output.keep()

    def run_out_of_process(self, flags: list[str]):
        # TODO: This Python executable wrapper is really long in the tooth. We should
        # just invoke iree-compile directly (which would also let us have a flag for
        # the path to it).
        all_extra_args = self.extra_args + flags
        compiler_tools.compile_file(
            str(self.input_file.get_fs_path()),
            output_file=str(self.output_file.get_fs_path()),
            extra_args=self.extra_args + flags,
        )


def compile(
    *,
    name: str,
    source: BuildFileLike,
    target_default: bool = True,
) -> tuple[BuildFile]:
    """Invokes iree-compile on a source file, producing binaries for one or more target
    machines.

    Args:
      name: The logical name of the compilation command. This is used as the stem
        for multiple kinds of output files.
      source: Input source file.
      target_default: Whether to use command line arguments to compute a target
        machine configuration (default True). This would be set to False to explicitly
        depend on target information contained in the source file and not require
        any target flags passed to the build tool.
    """
    context = BuildContext.current()
    input_file = context.file(source)
    if target_default:
        # Compute the target machines from flags and create one compilation for each.
        tms = compute_target_machines_from_flags()
        output_files: list[BuildFile] = []
        for tm in tms:
            output_file = context.allocate_file(
                f"{name}_{tm.target_spec}.vmfb", namespace=FileNamespace.BIN
            )
            inv = CompilerInvocation(input_file=input_file, output_file=output_file)
            inv.extra_flags.extend(tm.flag_list)
            CompileAction(
                inv,
                desc=f"Compiling {input_file} (for {tm.target_spec})",
                executor=context.executor,
            )
            output_files.append(output_file)
        return output_files
    else:
        # The compilation is self contained, so just directly compile it.
        output_file = context.allocate_file(f"{name}.vmfb", namespace=FileNamespace.BIN)
        inv = CompilerInvocation(input_file=input_file, output_file=output_file)
        CompileAction(
            inv,
            desc=f"Compiling {name}",
            executor=context.executor,
        )
        return output_file


class CompileAction(BuildAction):
    def __init__(self, inv: CompilerInvocation, **kwargs):
        super().__init__(**kwargs)
        self.inv = inv
        self.inv.output_file.deps.add(self)
        self.deps.add(self.inv.input_file)

    def _invoke(self):
        self.inv.run()
