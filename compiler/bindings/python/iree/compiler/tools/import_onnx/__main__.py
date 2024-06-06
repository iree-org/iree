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
import os
from pathlib import Path
import sys
import tempfile

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


def main(args: argparse.Namespace):
    model_proto = load_onnx_model(args)
    context = Context()
    model_info = onnx_importer.ModelInfo(model_proto)
    m = model_info.create_module(context=context).operation
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


def load_onnx_model(args: argparse.Namespace) -> onnx.ModelProto:
    input_dir = os.path.dirname(os.path.abspath(args.input_file))

    # Load the model, with possible external data coming from the default
    # location, or the location specified on the command line.
    if args.data_dir is None:
        raw_model = onnx.load(args.input_file)
    else:
        raw_model = onnx.load(args.input_file, load_external_data=False)
        onnx.load_external_data_for_model(raw_model, str(args.data_dir))

    # Do shape inference two ways.  First, attempt in-memory to avoid redundant
    # loading and the need for writing a temporary file somewhere.  If that
    # fails, typically because of the 2 GB protobuf size limit, try again via
    # files.  See
    # https://onnx.ai/onnx/repo-docs/PythonAPIOverview.html#shape-inference-a-large-onnx-model-2gb
    # for details about the file-based technique.

    # Run the checker to test whether the file is above the threshold for
    # in-memory shape inference.  If not, go ahead and do the shape inference.
    try:
        onnx.checker.check_model(raw_model)
        inferred_model = onnx.shape_inference.infer_shapes(
            raw_model, data_prop=args.data_prop
        )
        return inferred_model
    except ValueError:
        pass

    # Model is too big for in-memory inference: do file-based shape inference
    # to a temp file.
    # Make a temp dir for all the temp files we'll be generating as a side
    # effect of infering shapes. For now, the only file is a new .onnx holding
    # the revised model with shapes.
    with tempfile.TemporaryDirectory(dir=input_dir) as temp_dir_name:
        temp_dir_path = Path(temp_dir_name)
        temp_inferred_file = temp_dir_path / "temp-inferred.onnx"
        onnx.shape_inference.infer_shapes_path(
            args.input_file, temp_inferred_file, data_prop=args.data_prop
        )

        # Load the temp file and the external data.
        inferred_model = onnx.load(temp_inferred_file, load_external_data=False)
        data_dir = Path(input_dir if args.data_dir is None else args.data_dir)
        onnx.load_external_data_for_model(inferred_model, str(data_dir))

        return inferred_model


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
