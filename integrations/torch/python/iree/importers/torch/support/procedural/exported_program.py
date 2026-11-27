# Copyright 2024 Advanced Micro Devices, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
import math
from typing import Any

import torch
from iree.compiler.extras.fx_importer import (
    FxImporter,
    FxImporterHooks,
    GraphNodeImporter,
    InputInfo,
)
from iree.importers.torch.support._logging import logger
from torch.utils._pytree import (
    tree_flatten,
)

from iree.importers.torch.support.ir_utils import (
    GlobalAttributes,
    ModuleBuilder,
)
from iree.importers.torch.support.procedural.base import (
    CallableIntrinsic,
)
from iree.importers.torch.support.procedural.globals import (
    GlobalsDef,
    MaterializedGlobal,
)
from iree.importers.torch.support.tensor_traits import (
    ExternalTensorTrait,
)
from iree.importers.torch.support.ir_imports import (
    Operation,
    Value,
    util_d,
)

# Limit of tensor volumes. Over this limit, otherwise uncategorized tensor
# constants will be emitted out-of-line. Under the limit, inline.
INLINE_TENSOR_VOLUME_LIMIT = 1024


class ExportedProgramIntrinsic(CallableIntrinsic):
    def __init__(
        self,
        entry_func_op: Operation,
        entry_sig: torch.export.ModuleCallSignature,
        user_output_dtypes: list[torch.dtype | None],
    ):
        self.entry_func_op = entry_func_op
        self.entry_sig = entry_sig
        self.user_output_dtypes = user_output_dtypes


def import_exported_program(
    module_builder: ModuleBuilder,
    exported_program: torch.export.ExportedProgram,
    symbol_name: str,
    symbol_visibility: str | None,
) -> ExportedProgramIntrinsic:
    fx_importer = _create_fx_importer(module_builder)
    entry_func_op = fx_importer.import_program(
        exported_program,
        func_name=symbol_name,
        func_visibility=symbol_visibility,
        import_symbolic_shape_expressions=module_builder.options.import_symbolic_shape_expressions,
    )

    module_call_graph = exported_program.module_call_graph
    assert len(module_call_graph) >= 1, "Expected at least one module call signature"
    entry_module_call_entry = module_call_graph[0]
    assert (
        entry_module_call_entry.fqn == ""
    ), "Expected first module call entry to be unnamed"

    # We want additional torch-level metadata about any user outputs.
    # This will help us create a true python fake without loss of information.
    # TODO: It is unclear how much switchiness is actually needed here as
    # modern use is pretty constrained. Potentially streamline the body of
    # the for loop once done with full test cases available.
    user_output_dtypes: list[torch.dtype | None] = []
    node_map: dict[str, torch.fx.Node] = {
        n.name: n for n in exported_program.graph.nodes
    }
    for user_output in exported_program.graph_signature.user_outputs:
        if user_output is None:
            raise NotImplementedError(
                "Importing a program with an output of `None` is not yet implemented."
            )
        output_node = node_map[user_output]
        tensor_meta = output_node.meta.get("tensor_meta")
        fake_val = output_node.meta.get("val")
        dtype = None
        if tensor_meta is not None:
            dtype = tensor_meta.dtype
        elif fake_val is not None:
            dtype = fake_val.dtype
        user_output_dtypes.append(dtype)

    return ExportedProgramIntrinsic(
        entry_func_op, entry_module_call_entry.signature, user_output_dtypes
    )


class _Hooks(FxImporterHooks):
    def __init__(self, module_builder: ModuleBuilder):
        self.module_builder = module_builder

    def store_produced_value(
        self,
        gni: GraphNodeImporter,
        py_value: Any,
        produced_ir_value: Any,
        info: InputInfo,
    ):
        # Write-only buffers may not have been materialized by a prior load.
        materialized_global = self._lift_tensor_to_global(py_value, info)
        if materialized_global is None:
            raise ValueError(f"Cannot store value to unmapped global for: {info}")
        materialized_global_op_opview = materialized_global.global_op.opview
        assert isinstance(materialized_global_op_opview, util_d.GlobalOp)
        materialized_global_op_opview.is_mutable = True
        converted_value = Operation.create(
            "torch_c.to_builtin_tensor",
            results=[materialized_global.ir_type],
            operands=[produced_ir_value],
        ).result
        util_d.GlobalStoreOp(converted_value, materialized_global.symbol_name)

    def resolve_literal(
        self, gni: GraphNodeImporter, literal: Any, info: InputInfo | None = None
    ) -> Value | None:
        # We support resolution of tracked reference types. Currently this
        # only includes Tensors. All others we let the importer do what it
        # is going to do.
        if not isinstance(literal, torch.Tensor):
            return None

        # See if we know about it.
        materialized_global = self._lift_tensor_to_global(literal, info)
        if not materialized_global:
            # If it is unknown, just let the default importer take it on.
            return None

        # Emit a global load and conversion.
        vtensor_type = gni._cc.tensor_to_vtensor_type(literal)
        loaded_value = util_d.GlobalLoadOp(
            materialized_global.ir_type, materialized_global.symbol_name
        ).result
        converted_value = Operation.create(
            "torch_c.from_builtin_tensor",
            results=[vtensor_type],
            operands=[loaded_value],
        ).result
        return converted_value

    def _lift_tensor_to_global(
        self, literal: torch.Tensor, info: InputInfo | None
    ) -> MaterializedGlobal | None:
        module_builder = self.module_builder
        mapping = module_builder.global_ref_tracker.track(literal)
        if not mapping.is_empty:
            # Already materialized.
            logger.debug(f"Resolved defined global for literal {mapping}")
            materialized_global: MaterializedGlobal = mapping.value  # type: ignore
            return materialized_global

        # Policy check: Should we auto-import? Generally, we keep "small"
        # tensors as inline as they can be optimized.
        external_trait = ExternalTensorTrait.get(literal)
        if not self._should_lift_tensor_to_global(literal, external_trait, info):
            return None

        # If it is a tensor we haven't seen yet, materialize it
        # as a global and return.
        if external_trait is not None:
            # If it is an external tensor, we can generate a nicer
            # symbol name.
            name = external_trait.external_name
        else:
            # Otherwise, generate a name based on what we have.
            shape_desc = "_".join([str(d) for d in literal.shape])
            name = f"constant_{shape_desc}_{str(literal.dtype)}"

        name = module_builder.unique_auto_symbol(name)
        # TODO: We may want to unique this somehow in the module builder.
        auto_def = AutoGlobalTensorDef(name, literal, GlobalAttributes())
        materialized_global = auto_def.track(module_builder, "_auto")
        assert isinstance(materialized_global, MaterializedGlobal)
        return materialized_global

    def _should_lift_tensor_to_global(
        self,
        literal: torch.Tensor,
        external_trait: ExternalTensorTrait | None,
        info: InputInfo | None,
    ) -> bool:
        if info is not None and info.store_producer_node:
            return True
        if external_trait is not None:
            return True
        volume = math.prod(literal.shape)
        return volume > INLINE_TENSOR_VOLUME_LIMIT


class AutoGlobalTensorDef(GlobalsDef):
    """Global definition that is used for arbitrary tensor literals encountered
    during processing.
    """

    __slots__ = [
        "_name",
        "_value",
        "_schema",
    ]

    def __init__(self, name: str, value: torch.Tensor, attrs: GlobalAttributes):
        super().__init__(attrs)
        self._name = name
        self._value = value
        _, self._schema = tree_flatten(self._value)

    def items(self):
        yield (self._name, self._value)

    def schema(self):
        return self._schema


# In https://github.com/llvm/torch-mlir/pull/3046, the FxImporter was
# extended to accept a "module_op" as an Operation (vs a Module). Switch for
# compatibility.
_fx_importer_accepts_module_op = (
    "module_op" in inspect.getfullargspec(FxImporter).kwonlyargs
)


def _create_fx_importer(module_builder: ModuleBuilder) -> FxImporter:
    hooks = _Hooks(module_builder)
    if _fx_importer_accepts_module_op:
        # New path.
        return FxImporter(
            module_op=module_builder.module_op,
            config_check=False,
            py_attr_tracker=module_builder.fx_py_attr_tracker,
            hooks=hooks,
        )
    else:
        # Legacy path.
        class FakeModule:
            def __init__(self, op):
                self._op = module_builder.module_op

            @property
            def context(self):
                return self._op.context

            @property
            def operation(self):
                return self._op

            @property
            def body(self):
                return self._op.regions[0].blocks[0]

        return FxImporter(
            module=FakeModule(module_builder.module_op),
            config_check=False,
            py_attr_tracker=module_builder.fx_py_attr_tracker,
            hooks=hooks,
        )
