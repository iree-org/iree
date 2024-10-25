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
import copy
import random
import iree.runtime as rt
import string

from ...dialects import util
from typing import Optional, Tuple, Any

try:
    import onnx
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"iree-import-onnx requires that the `onnx` Python package is installed "
        f"(typically `{sys.executable} -m pip install onnx`)"
    ) from e

from onnx import numpy_helper

try:
    from ...extras import onnx_importer
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "iree-import-onnx is only available if IREE was built with Torch support"
    ) from e

from ...ir import (
    Context,
    Type as IrType,
    TypeAttr,
    RankedTensorType,
    StringAttr,
    Attribute,
    Operation,
    Location,
    InsertionPoint,
    Value,
    SymbolTable,
    IntegerType,
)


class IREENodeImporter(onnx_importer.NodeImporter):
    def __init__(
        self,
        graph_info: onnx_importer.GraphInfo,
        *,
        parent_op: Operation,
        block: onnx_importer.Block,
        context_cache: "onnx_importer.ContextCache",
        module_op: Operation,
        module_cache: "onnx_importer.ModuleCache",
        numel_threshold: int,
    ):
        super().__init__(
            graph_info,
            parent_op=parent_op,
            block=block,
            context_cache=context_cache,
            module_op=module_op,
            module_cache=module_cache,
        )
        self.last_global_op = None
        self.symbol_table = SymbolTable(module_op)
        self.symbol_table.insert(parent_op)
        self.numel_threshold = numel_threshold
        self.param_archive = rt.ParameterIndex()

    def sanitize_name(self, name: str) -> str:
        # There are often some initializers in the models that have no name
        # labels, or contain substrings like '::', which can cause conflicts,
        # and invalid symbol names for symbolic references. This function will
        # remove substrings like '::' when the name is not empty, and generate
        # a random string when it is, as a placeholder.
        new_name: str = ""
        for c in range(len(name)):
            if name[c] == ":":
                new_name += "_"
            else:
                new_name += name[c]

        if len(new_name) == 0:
            alpha = string.ascii_lowercase
            ch = random.choice(alpha)
            new_name = str(random.randrange(1, 1000)) + "__" + ch
        return new_name

    def create_tensor_global(
        self,
        t: onnx.TensorProto,
    ) -> Tuple[str, IrType]:
        # Always create globals at the top. Then after created, if there was
        # a prior one, move the new one to after it to maintain declaration
        # order.
        name = self.sanitize_name(t.name)
        with InsertionPoint.at_block_begin(
            self._m.regions[0].blocks[0]
        ), Location.unknown():
            # After lowering to linalg-on-tensors, the data type needs to be signless.
            # So, we construct the globals to have signless types, and use
            # torch_c.from_builtin_tensor to convert to the correct frontend type.
            vtensor_type = RankedTensorType.get(
                tuple(t.dims), ELEM_TYPE_TO_SIGNLESS_IR_TYPE[t.data_type]()
            )
            ir_attrs = {
                "sym_name": StringAttr.get(name),
                "sym_visibility": StringAttr.get("private"),
                "type": TypeAttr.get(vtensor_type),
            }

            external_scope_attr = StringAttr.get("model")
            external_name_attr = StringAttr.get(name)
            ir_attrs["initial_value"] = Attribute.parse(
                f"#stream.parameter.named<{external_scope_attr}::{external_name_attr}> : {vtensor_type}"
            )
            global_op = util.GlobalOp(
                ir_attrs["sym_name"],
                ir_attrs["type"],
                sym_visibility=ir_attrs["sym_visibility"],
                initial_value=ir_attrs["initial_value"],
            )
            self.symbol_table.insert(global_op)
            if self.last_global_op is not None:
                global_op.move_after(self.last_global_op)
            self.last_global_op = global_op
            actual_symbol_name = StringAttr(global_op.attributes["sym_name"]).value
        return actual_symbol_name, vtensor_type

    @classmethod
    def define_function(
        cls,
        graph_info: onnx_importer.GraphInfo,
        module_op: Operation,
        numel_threshold: int,
        context_cache: Optional["onnx_importer.ContextCache"] = None,
        module_cache: Optional["onnx_importer.ModuleCache"] = None,
        private: bool = False,
    ) -> "IREENodeImporter":
        cc = (
            context_cache
            if context_cache is not None
            else onnx_importer.ContextCache(module_op.context)
        )
        mc = (
            module_cache
            if module_cache is not None
            else onnx_importer.ModuleCache(module_op, cc)
        )
        with module_op.context, Location.name(f"graph:{graph_info.graph_proto.name}"):
            body = module_op.regions[0].blocks[0]
            func_name = graph_info.graph_proto.name
            input_types = [
                cc.type_proto_to_type(inp.type) for inp in graph_info.input_map.values()
            ]
            output_types = [
                cc.type_proto_to_type(out.type)
                for out in graph_info.output_map.values()
            ]
            ftype = onnx_importer.FunctionType.get(input_types, output_types)
            func_op = onnx_importer.func_dialect.FuncOp(
                func_name,
                ftype,
                ip=InsertionPoint(body),
                visibility="private" if private else None,
            )
            block = func_op.add_entry_block(
                [Location.name(k) for k in graph_info.input_map.keys()]
            )
        imp = IREENodeImporter(
            graph_info,
            parent_op=func_op,
            block=block,
            context_cache=cc,
            module_op=module_op,
            module_cache=mc,
            numel_threshold=numel_threshold,
        )
        for node_name, input_value in zip(graph_info.input_map.keys(), block.arguments):
            imp._nv_map[node_name] = input_value
        imp._populate_graph_attrs(func_op)
        return imp

    def import_initializer(
        self, initializer: onnx.TensorProto, extern_name: Optional[str] = None
    ) -> Value:
        # If an explicitly specified name is given, use that; otherwise, pick
        # up the name from the tensor proto itself
        iname = extern_name if extern_name else initializer.name
        dims = list(initializer.dims)
        numel = 1
        for d in dims:
            numel = numel * d
        if numel < self.numel_threshold:
            imported_tensor = super().import_initializer(initializer)
            self._nv_map[iname] = imported_tensor
            return imported_tensor

        x, t = self.create_tensor_global(initializer)
        vtensor_type = self._cc.get_vtensor_type(
            tuple(initializer.dims), self._cc.tensor_element_type(initializer.data_type)
        )

        with InsertionPoint(self._b), Location.name(iname):
            old_op = util.GlobalLoadOp(t, x)
            converted_value = Operation.create(
                "torch_c.from_builtin_tensor",
                results=[vtensor_type],
                operands=[old_op.result],
            ).result

        self._nv_map[iname] = converted_value
        tensor_as_array = numpy_helper.to_array(initializer)
        self.param_archive.add_buffer(x, tensor_as_array)
        return converted_value


def main(args: argparse.Namespace):
    model_proto = load_onnx_model(args)
    context = Context()
    model_info = onnx_importer.ModelInfo(model_proto)
    m = model_info.create_module(context=context).operation

    imp: Any = None
    if args.externalize_params:
        imp = IREENodeImporter.define_function(
            model_info.main_graph, m, args.numel_threshold
        )
    else:
        imp = onnx_importer.NodeImporter.define_function(model_info.main_graph, m)
    imp.import_all()

    if not args.no_verify:
        m.verify()

    if args.externalize_params:
        default_param_path = Path(args.output_file).parent / Path(args.output_file).stem
        param_path = (
            (str(default_param_path) + "_params.irpa")
            if args.save_params_to is None
            else args.save_params_to
        )
        imp.param_archive.create_archive_file(param_path)

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

    # Only change the opset version if it is greater than the current one.
    if args.opset_version and args.opset_version > raw_model.opset_import[0].version:
        raw_model = onnx.version_converter.convert_version(
            raw_model, args.opset_version
        )

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


ELEM_TYPE_TO_SIGNLESS_IR_TYPE = copy.deepcopy(onnx_importer.ELEM_TYPE_TO_IR_TYPE_CB)

ELEM_TYPE_TO_SIGNLESS_IR_TYPE[
    onnx.TensorProto.DataType.INT64
] = lambda: IntegerType.get_signless(64)
ELEM_TYPE_TO_SIGNLESS_IR_TYPE[
    onnx.TensorProto.DataType.INT32
] = lambda: IntegerType.get_signless(32)
ELEM_TYPE_TO_SIGNLESS_IR_TYPE[
    onnx.TensorProto.DataType.INT16
] = lambda: IntegerType.get_signless(16)
ELEM_TYPE_TO_SIGNLESS_IR_TYPE[
    onnx.TensorProto.DataType.INT8
] = lambda: IntegerType.get_signless(8)
ELEM_TYPE_TO_SIGNLESS_IR_TYPE[
    onnx.TensorProto.DataType.INT4
] = lambda: IntegerType.get_signless(4)
ELEM_TYPE_TO_SIGNLESS_IR_TYPE[
    onnx.TensorProto.DataType.UINT8
] = lambda: IntegerType.get_signless(8)
ELEM_TYPE_TO_SIGNLESS_IR_TYPE[
    onnx.TensorProto.DataType.UINT4
] = lambda: IntegerType.get_signless(4)
ELEM_TYPE_TO_SIGNLESS_IR_TYPE[
    onnx.TensorProto.DataType.UINT16
] = lambda: IntegerType.get_signless(16)
ELEM_TYPE_TO_SIGNLESS_IR_TYPE[
    onnx.TensorProto.DataType.UINT64
] = lambda: IntegerType.get_signless(64)
ELEM_TYPE_TO_SIGNLESS_IR_TYPE[
    onnx.TensorProto.DataType.UINT32
] = lambda: IntegerType.get_signless(32)


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
        "--opset-version",
        help="Allows specification of a newer opset_version to update the model"
        " to before importing to MLIR. This can sometime assist with shape inference.",
        type=int,
    )
    parser.add_argument(
        "--numel-threshold",
        help="Minimum number of elements for an initializer to be externalized. Only has an effect if 'externalize-params' is true.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--externalize-params",
        help="Externalize large parameters and store them on the disk, to load at runtime.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-params-to",
        help="Location to save the externalized parameters. When not set, the parameters will be written to '<output_file_name>_params.irpa'.",
        default=None,
    )
    args = parser.parse_args(argv)
    return args


def _cli_main():
    sys.exit(main(parse_arguments()))


if __name__ == "__main__":
    _cli_main()
