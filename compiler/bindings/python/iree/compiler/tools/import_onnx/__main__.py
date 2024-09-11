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
                        min_opset_version=args.min_opset_version,
                        preprocess_model=args.preprocess_model,
                        entry_point_name=args.entry_point_name,
                        module_name=args.module_name,
                        import_only=True,
                        save_temp_iree_input=args.save_temp_iree_input,
                        verify_module=not args.no_verify,
                        use_bytecode=args.use_bytecode,
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
    parser.add_argument(
        "--min-opset-version",
        help="Minimum ONNX opset version. Model with lower opset version will"
        " be converted to this version",
        type=int,
        default=17,
        required=False,
    )
    parser.add_argument(
        "--preprocess-model",
        help="Perform shape inference when importing the model.",
        type=bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--entry-point-name",
        help="Name of the entry point for the exported graph",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--module-name",
        help="Name for the exported MLIR module",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--import-only",
        help="Only import the ONNX graph do not run the iree compiler",
        type=bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--save-temp-iree-input",
        help="Save intermediate files to the given directory",
        type=Path,
        required=False,
    )
    parser.add_argument(
        "--use-bytecode",
        help="Use MLIR bytecode as the output format",
        type=bool,
        default=False,
        required=False,
    )
    args = parser.parse_args(argv)
    return args


def _cli_main():
    sys.exit(main(parse_arguments()))


if __name__ == "__main__":
    _cli_main()
