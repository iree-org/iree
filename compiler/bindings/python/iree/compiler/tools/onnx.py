# Lint-as: python3
# Copyright 20204 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Imports and compiles onnx models."""

import os
import sys
import tempfile
import logging

from dataclasses import dataclass
from typing import IO, Any, Optional, Union
from pathlib import Path

from .. import CompilerOptions, InputType, TempFileSaver
from ..ir import Context, StringAttr
from .binaries import invoke_pipeline
from .core import build_compile_command_line

try:
    import onnx
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"iree-import-onnx requires that the `onnx` Python package is installed "
        f"(typically `{sys.executable} -m pip install onnx`)"
    ) from e

try:
    from ..extras import onnx_importer
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "iree-import-onnx is only available if IREE was built with Torch support"
    ) from e


__all__ = [
    "compile_saved_model",
    "compile_model"
]


logger = logging.getLogger(__name__)


def load_onnx_model(model_path: str | os.PathLike,
                    data_prop: bool = True,
                    data_dir: str | os.PathLike | None = None,
                    preprocess_model: bool = True) -> onnx.ModelProto:
    input_dir = os.path.dirname(os.path.abspath(model_path))

    # Load the model, with possible external data coming from the default
    # location, or the location specified on the command line.
    if data_dir is None:
        model = onnx.load(model_path)
    else:
        model = onnx.load(model_path, load_external_data=False)
        onnx.load_external_data_for_model(model, str(data_dir))

    if not preprocess_model:
        return model

    # Do shape inference two ways.  First, attempt in-memory to avoid redundant
    # loading and the need for writing a temporary file somewhere.  If that
    # fails, typically because of the 2 GB protobuf size limit, try again via
    # files.  See
    # https://onnx.ai/onnx/repo-docs/PythonAPIOverview.html#shape-inference-a-large-onnx-model-2gb
    # for details about the file-based technique.

    # Run the checker to test whether the file is above the threshold for
    # in-memory shape inference.  If not, go ahead and do the shape inference.
    try:
        onnx.checker.check_model(model)
        inferred_model = onnx.shape_inference.infer_shapes(
            model, data_prop=data_prop
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
            model_path, temp_inferred_file, data_prop=data_prop
        )

        # Load the temp file and the external data.
        inferred_model = onnx.load(
            temp_inferred_file, load_external_data=False)
        data_dir = Path(input_dir if data_dir is None else data_dir)
        onnx.load_external_data_for_model(inferred_model, str(data_dir))

        return inferred_model


@dataclass
class ImportOptions(CompilerOptions):
    """Options for Onnx imports

    Args:
        input_type: The input dialect used to import the model. 
            Default is InputType.ONNX.
        min_opset_version: The minimum opset version to convert the model to.
            Default is 17.
        entry_point_name: The name of the entry point for the exported graph.
        model_name: The symbol name for the exported MLIR module.
        import_only: Only import the ONNX graph do not run the IREE compiler.
            Default is False.
        save_temp_iree_input: Save the IR resulting from the import before
            compilation. Default is None.
        verify_module: Verify the module after importing. Default is False.
        use_bytecode: Use MLIR bytecode instead of textual IR. 
            Default is False.
        data_prop: Toggle data propagation for ONNX shape inference.
            Default is True.
        data_dir: Path to the base directory of the model data.
            Default is None.
    """
    input_type: Union[InputType, str] = InputType.ONNX
    preprocess_model: bool = True
    min_opset_version: int = 17
    entry_point_name: Optional[str] = None
    module_name: Optional[str] = None
    import_only: bool = False
    save_temp_iree_input: Optional[str] = None
    verify_module: bool = False
    use_bytecode: bool = False
    data_prop: bool = True
    data_dir: Optional[Path] = None


def compile_saved_model(model_path: str | os.PathLike, **kwargs) -> None | str | IO[bytes]:
    """Import and compile an ONNX model.

    Args:
        model_path onnx.ModelProto: The path for the ONNX model to
            import and compile.

    Returns:
        None | str | IO[bytes]: If no output file is specified, the compiled
            model as a string or bytes depending on `output_format` and `use_bytecode`.
    """
    options = ImportOptions(**kwargs)

    with TempFileSaver.implicit() as tfs, tempfile.TemporaryDirectory() as tmpdir:
        if options.import_only and options.output_file:
            # Importing to a file and stopping, write to that file directly.
            onnx_iree_input = options.output_file
        elif options.save_temp_iree_input:
            # Saving the file, use tfs.
            extension = ".mlirbc" if options.use_bytecode else ".mlir"
            onnx_iree_input = tfs.alloc_optional(
                "onnx-iree-input" + extension, export_as=options.save_temp_iree_input
            )
        else:
            # Not saving the file, so generate a loose temp file without tfs.
            onnx_iree_input = os.path.join(tmpdir, "onnx-iree-input.mlirbc")
            # onnx_iree_input = io.BytesIO()

        model = load_onnx_model(
            model_path, options.data_prop, options.data_dir, options.preprocess_model)

        # convert onnx model to version if needed 17
        opset_version = model.opset_import[0].version
        if opset_version < options.min_opset_version:
            logger.info("Converting onnx model opset version from %s to %s",
                        opset_version, options.min_opset_version)
            try:
                converted_model = onnx.version_converter.convert_version(
                    model, options.min_opset_version)
            except:
                # Conversion failed. Do our best with the original file.
                logger.warning("Converting onnx model opset version from %s "
                               "to %s failed. Continuning without conversion.",
                               opset_version, options.min_opset_version)
                converted_model = model
        else:
            # No conversion needed.
            converted_model = model

        if options.entry_point_name:
            converted_model.graph.name = options.entry_point_name

        logger.info("Importing graph: '%s' from onnx model",
                    model_proto.graph.name)
        context = Context()
        model_info = onnx_importer.ModelInfo(model_proto)
        module = model_info.create_module(context=context).operation

        if options.module_name:
            module.attributes["sym_name"] = StringAttr.get(
                options.module_name, context)

        importer = onnx_importer.NodeImporter.define_function(
            model_info.main_graph, module)
        importer.import_all()

        if options.verify_module:
            module.verify()

        if options.use_bytecode:
            with open(onnx_iree_input, "wb") as f:
                module.write_bytecode(f)
        else:
            with open(onnx_iree_input, "wt", encoding="utf-8") as f:
                print(module.get_asm(assume_verified=options.verify_module), file=f)

        if options.import_only:
            if options.output_file:
                return None
            if options.use_bytecode:
                with open(onnx_iree_input, "rb") as f:
                    return f.read()
            else:
                with open(onnx_iree_input, "rt", encoding="utf-8") as f:
                    return f.read()

        # compile onnx model
        compile_cl = build_compile_command_line(onnx_iree_input, tfs, options)
        result = invoke_pipeline([compile_cl])
        if options.output_file:
            return None
        return result


def compile_model(model: onnx.ModelProto, **kwargs) -> None | str | IO[bytes]:
    """Import and compile an ONNX model.

    Args:
        model str | os.PathLike: The ONNX model to import and compile.

    Returns:
        None | str | IO[bytes]: If no output file is specified, the compiled
            model as a string or bytes depending on `output_format` and `use_bytecode`.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir_path = Path(tmpdir)
        temp_model_file = temp_dir_path / "temp.onnx"
        onnx.save(model, temp_model_file)
        return compile_saved_model(temp_model_file, **kwargs)
