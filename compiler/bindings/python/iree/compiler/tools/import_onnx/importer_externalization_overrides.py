# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import logging
import random
import string
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Any, NamedTuple, Union

import numpy

try:
    import onnx
    from onnx import numpy_helper
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

from ...dialects import util

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

logger = logging.getLogger(__name__)


class ParamData(NamedTuple):
    param_bit_threshold: Optional[int]
    num_elements_threshold: int
    params_scope: str
    data_dir: str
    param_path: str
    input_index_threshold: Optional[int]


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
        param_data: ParamData,
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
        self.param_data = param_data
        self.globals = []

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

    def get_type_info_from_type(self, tp: onnx.TypeProto):
        tt = tp.tensor_type
        if tt.elem_type:
            dims = tuple(
                (d.dim_value if not d.dim_param else None) for d in tt.shape.dim
            )
        return dims, tt.elem_type

    def create_tensor_global(
        self,
        t: Union[onnx.TensorProto, onnx.ValueInfoProto],
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
            if isinstance(t, onnx.TensorProto):
                dims = tuple(t.dims)
                data_type = t.data_type
            elif isinstance(t, onnx.ValueInfoProto):
                dims, data_type = self.get_type_info_from_type(t.type)
            else:
                raise TypeError(
                    f"Expected an onnx.TensorProto or an onnx.ValueInfoProto, recieved {type(t)} from {name}"
                )

            vtensor_type = RankedTensorType.get(
                dims, ELEM_TYPE_TO_SIGNLESS_IR_TYPE[data_type]()
            )
            ir_attrs = {
                "sym_name": StringAttr.get(name),
                "sym_visibility": StringAttr.get("private"),
                "type": TypeAttr.get(vtensor_type),
            }

            external_scope_attr = StringAttr.get(self.param_data.params_scope)
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
        param_data: ParamData,
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
        all_input_map = copy.deepcopy(graph_info.input_map)
        num_inputs = len(all_input_map.items())
        if param_data.input_index_threshold is not None:
            if param_data.input_index_threshold not in range(0, num_inputs):
                raise ValueError(
                    f"input_index_threshold must be in the range [0,num_inputs={num_inputs})"
                )
            for _index in range(param_data.input_index_threshold, num_inputs):
                _discarded = graph_info.input_map.popitem()

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
            param_data=param_data,
        )
        for node_name, input_value in zip(graph_info.input_map.keys(), block.arguments):
            imp._nv_map[node_name] = input_value
        imp._populate_graph_attrs(func_op)
        imp._gi.input_map = all_input_map
        return imp

    def import_all(self, func=True):
        num_inputs = len(self._gi.input_map.items())
        if self.param_data.input_index_threshold is not None:
            for i in range(self.param_data.input_index_threshold, num_inputs):
                self.import_initializer(list(self._gi.input_map.values())[i])
        super().import_all(func)

    def import_initializer(
        self,
        initializer: Union[onnx.TensorProto, onnx.ValueInfoProto],
        extern_name: Optional[str] = None,
    ) -> Value:
        # If an explicitly specified name is given, use that; otherwise, pick
        # up the name from the tensor proto itself
        initializer_name = extern_name if extern_name else initializer.name
        if isinstance(initializer, onnx.TensorProto):
            dims = tuple(initializer.dims)
            num_elements = 1
            for d in dims:
                num_elements = num_elements * d
            if num_elements < self.param_data.num_elements_threshold:
                imported_tensor = super().import_initializer(initializer)
                self._nv_map[initializer_name] = imported_tensor
                return imported_tensor
            data_type = initializer.data_type
        elif isinstance(initializer, onnx.ValueInfoProto):
            dims, data_type = self.get_type_info_from_type(initializer.type)
        else:
            raise TypeError(
                f"Expected an onnx.TensorProto or an onnx.ValueInfoProto, recieved {type(initializer)} from {initializer_name}"
            )

        actual_symbol_name, tensor_type = self.create_tensor_global(initializer)
        vtensor_type = self._cc.get_vtensor_type(
            dims, self._cc.tensor_element_type(data_type)
        )

        with InsertionPoint(self._b), Location.name(initializer_name):
            old_op = util.GlobalLoadOp(tensor_type, actual_symbol_name)
            converted_value = Operation.create(
                "torch_c.from_builtin_tensor",
                results=[vtensor_type],
                operands=[old_op.result],
            ).result

        self._nv_map[initializer_name] = converted_value
        if isinstance(initializer, onnx.TensorProto):
            self.globals.append((initializer_name, actual_symbol_name))
        return converted_value

    def save_params(self):
        """
        Only gets called if the arg `--save-params` is set to `True`.
        Saving params requires iree-runtime, so putting the import here will avoid requiring it uniformly for the importer.
        """
        try:
            import iree.runtime as rt
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "iree-import-onnx requires iree runtime api for externalizing parameters. "
                "For example: `pip install iree-base-runtime`"
            ) from e

        param_archive = rt.ParameterIndex()
        # if we don't need to save params in batches, gather all tensors in param_archive
        if self.param_data.param_bit_threshold is None:
            for name, actual_symbol_name in self.globals:
                initializer = self._gi.initializer_map[name]
                tensor_as_array = numpy_helper.to_array(
                    initializer, base_dir=self.param_data.data_dir
                )
                param_archive.add_buffer(actual_symbol_name, tensor_as_array)
            param_archive.create_archive_file(self.param_data.param_path)
            return

        # else we need to save in batches:
        #    1. setup a temporary directory to save smaller param files
        #    2. keep a target_index for storing references to saved tensors
        #    3. create an archive file for target_index at param_path to gather all temp data
        target_index = rt.ParameterIndex()
        in_memory_param_bits = 0
        iter = 0
        t00 = time.time()
        with tempfile.TemporaryDirectory(
            dir=Path(self.param_data.param_path).parent
        ) as temp_dir_name:
            get_curr_path = lambda: str(Path(temp_dir_name) / f"params_{iter}.irpa")
            for name, actual_symbol_name in self.globals:
                initializer = self._gi.initializer_map[name]
                tensor_as_array = numpy_helper.to_array(
                    initializer, base_dir=self.param_data.data_dir
                )
                param_archive.add_buffer(actual_symbol_name, tensor_as_array)
                # get the new param size
                elem_dtype = tensor_as_array.dtype
                elem_kind = elem_dtype.kind
                if elem_kind not in ["i", "f"]:
                    raise TypeError(f"Unhandled numpy dtype: {elem_dtype}")
                elem_info = (
                    numpy.iinfo(elem_dtype)
                    if elem_kind == "i"
                    else numpy.finfo(elem_dtype)
                )
                elem_bits = elem_info.bits
                param_bits = tensor_as_array.size * elem_bits
                # update the running total memory use
                in_memory_param_bits += param_bits
                if param_bits >= self.param_data.param_bit_threshold:
                    logger.warning(
                        f"Single parameter {name} is {param_bits} bits, "
                        + f"which exceeds threshold of {self.param_data.param_bit_threshold} bits."
                    )
                # flush the param archive to a temp file if the threshold is exceeded
                if in_memory_param_bits >= self.param_data.param_bit_threshold:
                    t0 = time.time()
                    param_archive.create_archive_file(
                        get_curr_path(),
                        target_index=target_index,
                    )
                    logger.info(
                        f"iter {iter} with {in_memory_param_bits} bits took {time.time() - t0}s to flush"
                    )
                    iter += 1
                    del param_archive
                    param_archive = rt.ParameterIndex()
                    in_memory_param_bits = 0

            # write any remaining params to a temp file
            t0 = time.time()
            param_archive.create_archive_file(
                get_curr_path(), target_index=target_index
            )
            logger.info(
                f"iter {iter} with {in_memory_param_bits} bits took {time.time() - t0}s to flush"
            )
            # combine all temporary param files into the final result
            t0 = time.time()
            target_index.create_archive_file(self.param_data.param_path)
            logger.info(f"combining {iter + 1} irpa files took {time.time() - t0}s")
            logger.info(f"total time to save params: {time.time()-t00}")


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
