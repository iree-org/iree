import os
import tempfile
import logging
import onnx

from dataclasses import dataclass
from typing import IO, Any, Optional, Union
from pathlib import Path

from .. import CompilerOptions, InputType, TempFileSaver
from ..extras import onnx_importer
from ..ir import Context, StringAttr
from .binaries import invoke_pipeline
from .core import build_compile_command_line
from ...runtime import Config, VmInstance, VmModule, load_vm_module

__all__ = [
    "compile_saved_model",
    "compile_model"
]


logger = logging.getLogger(__name__)


def load_onnx_model(raw_model: onnx.ModelProto,
                    data_prop: bool = True,
                    data_dir: IO[bytes] | str | os.PathLike | None = None):
    # Load the model, with possible external data coming from the default
    # location, or the location specified on the command line.
    if data_dir:
        onnx.load_external_data_for_model(raw_model, str(data_dir))

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
            raw_model, data_prop=data_prop
        )
        return inferred_model
    except ValueError:
        pass

    # Model is too big for in-memory inference: do file-based shape inference
    # to a temp file.
    # Make a temp dir for all the temp files we'll be generating as a side
    # effect of infering shapes. For now, the only file is a new .onnx holding
    # the revised model with shapes.
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir_path = Path(temp_dir_name)
        temp_input_file = temp_dir_path / "temp.onnx"
        temp_inferred_file = temp_dir_path / "temp-inferred.onnx"
        onnx.save(raw_model, temp_input_file)
        onnx.shape_inference.infer_shapes_path(
            temp_input_file, temp_inferred_file, data_prop=data_prop
        )

        # Load the temp file and the external data.
        inferred_model = onnx.load(
            temp_inferred_file, load_external_data=False)
        data_dir = Path(temp_input_file if data_dir is None else data_dir)
        onnx.load_external_data_for_model(inferred_model, str(data_dir))

        return inferred_model


@dataclass
class ImportOptions(CompilerOptions):
    input_type: Union[InputType, str] = InputType.ONNX
    min_opset_version: int = 17
    entry_point_name: Optional[str] = None
    module_name: Optional[str] = None
    import_only: bool = False
    save_temp_iree_input: Optional[str] = None
    verify_module: bool = False
    use_bytecode: bool = False


def compile_saved_model(model_path: IO[bytes] | str | os.PathLike, **kwargs) -> None | str | IO[bytes]:
    options = ImportOptions(**kwargs)

    with TempFileSaver.implicit() as tfs, tempfile.TemporaryDirectory() as tmpdir:
        if options.import_only and options.output_file:
            # Importing to a file and stopping, write to that file directly.
            onnx_iree_input = options.output_file
        elif options.save_temp_iree_input:
            # Saving the file, use tfs.
            onnx_iree_input = tfs.alloc_optional(
                "onnx-iree-input.mlirbc", export_as=options.save_temp_iree_input
            )
        else:
            # Not saving the file, so generate a loose temp file without tfs.
            onnx_iree_input = os.path.join(tmpdir, "onnx-iree-input.mlirbc")
            # onnx_iree_input = io.BytesIO()

        # convert onnx model to version if needed 17
        original_model = onnx.load_model(model_path)
        opset_version = original_model.opset_import[0].version
        if opset_version < options.min_opset_version:
            logger.info("Converting onnx model opset version from %s to %s",
                        opset_version, options.min_opset_version)
            try:
                converted_model = onnx.version_converter.convert_version(
                    original_model, options.min_opset_version)
            except:
                # Conversion failed. Do our best with the original file.
                logger.warning("Converting onnx model opset version from %s to %s failed. Continuning without conversion.",
                               opset_version, options.min_opset_version)
                converted_model = original_model
        else:
            # No conversion needed.
            converted_model = original_model

        if options.entry_point_name:
            converted_model.graph.name = options.entry_point_name

        # import onnx model
        model_proto = load_onnx_model(converted_model)

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
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir_path = Path(tmpdir)
        temp_model_file = temp_dir_path / "temp.onnx"
        onnx.save(model, temp_model_file)
        return compile_saved_model(temp_model_file, **kwargs)
