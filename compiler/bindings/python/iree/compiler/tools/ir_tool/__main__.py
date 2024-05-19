# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import logging
import sys

from ...api import Invocation, Session, Source, Output


def load_source(inv: Invocation, input_file: str) -> Source:
    source = Source.open_file(inv.session, input_file)
    if not inv.parse_source(source):
        raise RuntimeError(f"Error parsing source file {input_file}")
    return source


def write_output(inv: Invocation, output: Output, args, keep: bool = True):
    if args.emit_bytecode:
        inv.output_ir_bytecode(output, args.bytecode_version)
    else:
        inv.output_ir(output)
    if keep:
        output.keep()


###############################################################################
# CLI handling
###############################################################################


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="IREE IR Tool")
    subparsers = parser.add_subparsers(
        help="sub-command help", required=True, dest="sub_command"
    )

    def add_output_options(subparser):
        subparser.add_argument(
            "--emit-bytecode", action="store_true", help="Emit bytecode"
        )
        subparser.add_argument(
            "--bytecode-version",
            default=-1,
            type=int,
            help="Bytecode version to emit or -1 for latest",
        )

    # copy (cp) command.
    copy_parser = subparsers.add_parser(
        "copy",
        aliases=["cp"],
        help="Read a file and then output it using the given options, without "
        "modification",
    )
    add_output_options(copy_parser)
    copy_parser.add_argument("input_file", help="File to process")
    copy_parser.add_argument(
        "-o", required=True, dest="output_file", help="Output file"
    )
    copy_parser.set_defaults(func=do_copy)

    # strip-data command.
    strip_data_parser = subparsers.add_parser(
        "strip-data",
        help="Strip large constants and values, "
        "replacing them with pseudo data suitable for interactive "
        "debugging of IR",
    )
    add_output_options(strip_data_parser)
    strip_data_parser.add_argument(
        "--no-import",
        action="store_true",
        help="Disable import of public dialects to internal",
    )
    strip_data_parser.add_argument("input_file", help="File to process")
    strip_data_parser.add_argument(
        "-o", required=True, dest="output_file", help="Output file"
    )
    strip_data_parser.set_defaults(func=do_strip_data)

    args = parser.parse_args(argv)
    return args


def main(args) -> int:
    args.func(args)
    return 0


def do_copy(args) -> int:
    session = Session()
    output = Output.open_file(args.output_file)
    inv = session.invocation()
    inv.enable_console_diagnostics()
    load_source(inv, args.input_file)
    write_output(inv, output, args)
    return 0


def do_strip_data(args) -> int:
    session = Session()
    output = Output.open_file(args.output_file)
    inv = session.invocation()
    inv.enable_console_diagnostics()
    load_source(inv, args.input_file)
    if not args.no_import:
        if not inv.execute_text_pass_pipeline(
            "iree-import-public, iree-import-ml-program"
        ):
            return 1
    if not inv.execute_text_pass_pipeline(
        "iree-flow-outline-constants, iree-util-strip-and-splat-constants"
    ):
        return 2
    write_output(inv, output, args)
    return 0


def _cli_main():
    sys.exit(main(parse_arguments()))


if __name__ == "__main__":
    _cli_main()
