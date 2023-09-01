# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import inspect
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "saved_model_path", help="Path to the saved model directory to import."
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        required=True,
        help="Path to the mlir file name to output.",
    )
    parser.add_argument(
        "--tf-savedmodel-exported-names",
        required=False,
        help="List of exported names for cases that the model has ambiguous exports",
    )
    parser.add_argument(
        "--tf-import-type",
        default="savedmodel_v2",
        help="Import type for legacy saved models ('savedmodel_v2' or 'savedmodel_v1')",
    )
    parser.add_argument(
        "--tf-savedmodel-tags",
        default="serve",
        help="Tags used to indicate which MetaGraphDef to import, separated by " "','",
    )

    # Deprecated and unused.  Kept in place so callers of the old tool don't break
    # when using the new tool.
    parser.add_argument(
        "--output-format", dest="_", required=False, help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    saved_model_dir = args.saved_model_path
    exported_names = args.tf_savedmodel_exported_names
    output_path = args.output_path
    import_type = args.tf_import_type
    tags = args.tf_savedmodel_tags
    import_saved_model(
        output_path=output_path,
        saved_model_dir=saved_model_dir,
        exported_names=exported_names,
        import_type=import_type,
        tags=tags,
    )


def import_saved_model(
    *, output_path, saved_model_dir, exported_names, import_type, tags
):
    # From here there be dragons.
    try:
        # Available from TF 2.14.
        from tensorflow.mlir.experimental import (
            convert_saved_model,
            convert_saved_model_v1,
            run_pass_pipeline,
            write_bytecode,
        )
    except ImportError as e:
        # Try the old names for the same API instead, e.g. with TF 2.12.
        # Yes, this is brittle. Yes, we may need to completely change this
        # if/when these APIs change again. Such is working with TensorFlow.
        from tensorflow.python import pywrap_mlir

        convert_saved_model = pywrap_mlir.experimental_convert_saved_model_to_mlir
        convert_saved_model_v1 = pywrap_mlir.experimental_convert_saved_model_v1_to_mlir
        run_pass_pipeline = pywrap_mlir.experimental_run_pass_pipeline
        write_bytecode = pywrap_mlir.experimental_write_bytecode

    if import_type == "savedmodel_v2":
        result = convert_saved_model(
            saved_model_dir, exported_names=exported_names, show_debug_info=False
        )
    elif import_type == "savedmodel_v1":
        # You saw it here, folks: The TF team just adds random positional params
        # without explanation or default. So we detect and default them on our
        # own. Because this is normal and fine.
        sig = inspect.signature(convert_saved_model_v1)
        dumb_extra_kwargs = {}
        if "include_variables_in_initializers" in sig.parameters:
            dumb_extra_kwargs["include_variables_in_initializers"] = False
        if "upgrade_legacy" in sig.parameters:
            dumb_extra_kwargs["upgrade_legacy"] = False
        if "lift_variables" in sig.parameters:
            dumb_extra_kwargs["lift_variables"] = True
        result = convert_saved_model_v1(
            saved_model_dir,
            exported_names=exported_names,
            tags=tags,
            show_debug_info=False,
            **dumb_extra_kwargs,
        )
    else:
        raise ValueError(f"Unsupported import type: '{import_type}'")
    # The import to MLIR produces public functions like __inference__{name}_2222
    # but the conversion pipeline requires a single public @main function.
    # Not sure how this was allowed to happen, but regex to the rescue.
    # This is fine and normal, and totally to be expected. :(
    result = re.sub(r"func @__inference_(.+)_[0-9]+\(", r"func @\1(", result)
    pipeline = ["tf-lower-to-mlprogram-and-hlo"]
    result = run_pass_pipeline(result, ",".join(pipeline), show_debug_info=False)
    write_bytecode(output_path, result)


if __name__ == "__main__":
    main()
