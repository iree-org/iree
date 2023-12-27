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

try:
    import onnx
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"iree-import-onnx requires that the `onnx` Python package is installed "
        f"(typically `{sys.executable} -m pip install onnx`)"
    ) from e

try:
    from ...extras import onnx_importer
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "iree-import-onnx is only available if IREE was built with Torch support"
    ) from e

from ...ir import (
    Context,
)


def main(args):
    model_proto = load_onnx_model(args.input_file)
    context = Context()
    model_info = onnx_importer.ModelInfo(model_proto)
    m = model_info.create_module(context=context)
    imp = onnx_importer.NodeImporter.define_function(model_info.main_graph, m)
    imp.import_all()
    if not args.no_verify:
        m.verify()

    # TODO: This isn't very efficient output. If these files ever
    # get large, enable bytecode and direct binary emission to save
    # some copies.
    if args.output_file and args.output_file != "-":
        with open(args.output_file, "wt") as f:
            print(m.get_asm(assume_verified=not args.no_verify), file=f)
    else:
        print(m.get_asm(assume_verified=not args.no_verify))


def load_onnx_model(file_path: Path) -> onnx.ModelProto:
    raw_model = onnx.load(file_path)
    inferred_model = onnx.shape_inference.infer_shapes(raw_model)
    return inferred_model


def parse_arguments(argv=None):
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
    args = parser.parse_args(argv)
    return args


def _cli_main():
    sys.exit(main(parse_arguments()))


if __name__ == "__main__":
    _cli_main()
