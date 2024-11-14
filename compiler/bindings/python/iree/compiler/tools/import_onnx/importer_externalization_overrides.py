# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import random
import string
import iree.runtime as rt

from ...dialects import util
from typing import Optional, Tuple, Any

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

from onnx import numpy_helper

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
        num_elements_threshold: int,
        params_scope: str,
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
        self.num_elements_threshold = num_elements_threshold
        self.param_archive = rt.ParameterIndex()
        self.params_scope = params_scope

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

            external_scope_attr = StringAttr.get(self.params_scope)
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
        num_elements_threshold: int,
        params_scope: str,
        context_cache: Optional["onnx_importer.ContextCache"] = None,
        module_cache: Optional["onnx_importer.ModuleCache"] = None,
        private: bool = False,
    ) -> "IREENodeImporter":
        # Recover per-context caches of various attributes.
        # Allows modifications in the same context without
        # loss of current state.
        cc = (
            context_cache
            if context_cache is not None
            else onnx_importer.ContextCache(module_op.context)
        )
        # Recover per-module caches of various attributes.
        # Allows modification in the same module_op without
        # loss of current state.
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
            num_elements_threshold=num_elements_threshold,
            params_scope=params_scope,
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
        initializer_name = extern_name if extern_name else initializer.name
        dims = list(initializer.dims)
        num_elements = 1
        for d in dims:
            num_elements = num_elements * d
        if num_elements < self.num_elements_threshold:
            imported_tensor = super().import_initializer(initializer)
            self._nv_map[initializer_name] = imported_tensor
            return imported_tensor

        actual_symbol_name, tensor_type = self.create_tensor_global(initializer)
        vtensor_type = self._cc.get_vtensor_type(
            tuple(initializer.dims), self._cc.tensor_element_type(initializer.data_type)
        )

        with InsertionPoint(self._b), Location.name(initializer_name):
            old_op = util.GlobalLoadOp(tensor_type, actual_symbol_name)
            converted_value = Operation.create(
                "torch_c.from_builtin_tensor",
                results=[vtensor_type],
                operands=[old_op.result],
            ).result

        self._nv_map[initializer_name] = converted_value
        tensor_as_array = numpy_helper.to_array(initializer)
        self.param_archive.add_buffer(actual_symbol_name, tensor_as_array)
        return converted_value


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
