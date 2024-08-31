# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Console tool for converting an ONNX proto to torch IR.

Typically, when installed from a wheel, this can be invoked as:

  iree-import-onnx some.pb

Or from Python:

  python -m iree.compiler.tools.import_onnx ...
"""
import argparse
from pathlib import Path
import sys
from ..onnx import compile_saved_model


def main(args: argparse.Namespace):
    output_file = None if args.output_file == "-" else args.output_file

    compile_saved_model(args.input_file,
                        output_file=output_file,
                        import_only=True,
                        use_bytecode=False,
                        verify_module=not args.no_verify,
                        data_prop=args.data_prop,
                        data_dir=args.data_dir)


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IREE ONNX import tool")
    parser.add_argument("input_file", help="ONNX protobuf input", type=Path)
    parser.add_argument(
        "-o", dest="output_file", help="Output path (or '-' for stdout)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable verification prior to printing",
    )
    parser.add_argument(
        "--data-prop",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Toggle data propogation for onnx shape inference",
    )
    parser.add_argument(
        "--data-dir",
        help="Path to the base directory of the data."
        " Defaults to the directory of the input file.",
        type=Path,
    )
    args = parser.parse_args(argv)
    return args


def _cli_main():
    sys.exit(main(parse_arguments()))


if __name__ == "__main__":
    _cli_main()
