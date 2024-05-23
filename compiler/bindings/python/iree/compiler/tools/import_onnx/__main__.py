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
import shutil
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
    # Do shape inference two ways.  First, attempt in-memory to avoid redundant
    # loading and the need for writing a temporary file somewhere.  If that
    # fails, typically because of the 2 GB protobuf size limit, try again via
    # files.  See
    # https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#shape-inference-a-large-onnx-model-2gb
    # for details about the file-based technique.

    # Make a temp dir for all the temp files we'll be generating as a side
    # effect of infering shapes.  For now, the only file is a new .onnx holding
    # the revised model with shapes.
    #
    # TODO: If the program temp_dir is None, we should be using an ephemeral
    # temp directory instead of a hard-coded path in order to avoid data races
    # by default.
    input_dir = os.path.dirname(os.path.abspath(args.input_file))
    temp_dir = (
        Path(input_dir if args.temp_dir is None else args.temp_dir)
        / "onnx-importer-temp"
    )
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(exist_ok=True)

    # Load the model, with possible external data coming from the default
    # location, or the location specified on the conmand line.
    if args.data_dir is None:
        raw_model = onnx.load(args.input_file)
    else:
        raw_model = onnx.load(args.input_file, load_external_data=False)
        onnx.load_external_data_for_model(raw_model, args.data_dir)

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

    # The following code was an attempt to work around the bug where models
    # with external data produce invalid output shapes after infer_shapes_path.
    # It works with small models but threw an error for llama seeming to
    # indicate that the protobuf is corrupt.
    #
    # temp_raw_file = temp_dir / "raw.onnx"
    # onnx.save(raw_model, temp_raw_file, save_as_external_data=False)
    # onnx.shape_inference.infer_shapes_path(temp_raw_file, temp_inferred_file)
    # inferred_model = onnx.load(temp_inferred_file)

    # Model is too big for in-memory inference: do file-based shape inference
    # to a temp file.
    temp_inferred_file = temp_dir / "inferred.onnx"
    onnx.shape_inference.infer_shapes_path(
        args.input_file, temp_inferred_file, data_prop=args.data_prop
    )

    # Sanity check the shape-inferred model to be sure we have a good model
    # for the importer.  This call uses the file-based method, as the
    # in-memory method (passing the loaded model) fails due to the 2 GB limit.
    #
    # TODO: this call throws an exception because it can't find the external
    # data files, and there doesn't appear to be a way to let the checker know
    # where to find them.
    #
    # onnx.checker.check_model(temp_inferred_file)

    # Load the temp file and the external data.
    inferred_model = onnx.load(temp_inferred_file, load_external_data=False)
    data_dir = Path(input_dir if args.temp_dir is None else args.data_dir)
    onnx.load_external_data_for_model(inferred_model, data_dir)

    # Remove the inferred shape file unless asked to keep it
    if not args.keep_temps:
        shutil.rmtree(temp_dir)

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
        dest="data_prop",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Toggle data propogation for onnx shape inference",
    )
    parser.add_argument(
        "--keep-temps", action="store_true", help="Keep intermediate files"
    )
    parser.add_argument(
        "--temp-dir",
        help="Pre-existing directory in which to create temporary files."
        ' For example, to place temporaries under the directory "foo/bar",'
        ' specify --temp-dir=foo/bar.  "foo/bar" must already exist.'
        " Defaults to the directory of the input file.",
        type=Path,
    )
    parser.add_argument(
        "--data-dir",
        help="Path between CWD and the base directory of the data,"
        " excluding the directories given in the 'location' argument of "
        " convert_model_to_external_data.  For example, if 'location' was"
        ' "data/data.bin" and the relative path from CWD to that .bin file is'
        ' a/b/data/data.bin, then set data-dir to "a/b".'
        " Defaults to the directory of the input file.",
        type=Path,
    )
    args = parser.parse_args(argv)
    return args


def _cli_main():
    sys.exit(main(parse_arguments()))


if __name__ == "__main__":
    _cli_main()
