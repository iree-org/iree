# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from iree.compiler.extras.fx_importer import (
    RefTracker,
)
from iree.importers.torch.support._logging import logger

from iree.importers.torch.support.tensor_traits import (
    ExternalTensorTrait,
)
from iree.importers.torch.support.conversions import (
    TORCH_DTYPE_TO_IREE_TYPE,
)
from iree.importers.torch.support.ir_imports import (
    Attribute,
    DenseElementsAttr,
    DenseResourceElementsAttr,
    F16Type,
    F32Type,
    F64Type,
    FloatAttr,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    IrType,
    Location,
    MLIRError,
    Operation,
    RankedTensorType,
    StringAttr,
    SymbolTable,
    TypeAttr,
    UnitAttr,
)

###############################################################################
# Configuration
###############################################################################

# Maps a name to an altered name. If returns None, then the original
# name is used (this lets dict.get serve as a NameMapCallback).
NameMapCallback = Callable[[str], Optional[str]]


class GlobalAttributes:
    """Settings for how to initialize the global."""

    __slots__ = [
        "mutable",
        "external",
        "external_scope",
        "name_mapper",
        "noinline",
        "uninitialized",
    ]

    def __init__(
        self,
        mutable: bool = False,
        external: bool | None = None,
        external_scope: str | None = None,
        name_mapper: NameMapCallback | None = None,
        noinline: bool = False,
        uninitialized: bool | None = None,
    ):
        if external and uninitialized:
            raise ValueError(
                "Globals with external=True cannot also have uninitialized=True"
            )
        if uninitialized and not mutable:
            raise ValueError(
                "Globals with uninitialized=True must also be mutable=True"
            )
        self.mutable = mutable
        self.external = external
        self.external_scope = external_scope
        self.name_mapper = name_mapper
        self.noinline = noinline
        self.uninitialized = uninitialized

    def map_name(self, name: str) -> str:
        if self.name_mapper:
            new_name = self.name_mapper(name)
            if new_name is not None:
                return new_name
        return name

    def infer_external_from_tensor(
        self, t: torch.Tensor
    ) -> tuple[bool, str | None, str | None]:
        """If externality is not specified, infers it from the tensor."""
        # We check for the first item in a list because this lets us in the
        # future extend the list by unwrapping.
        check_tensors = [t]
        for check_t in check_tensors:
            trait = ExternalTensorTrait.get(check_t)
            if trait is None:
                continue
            try:
                external_scope = trait.external_scope
                external_name = trait.external_name
            except AttributeError:
                raise AttributeError(
                    f"Tensor defines an external tensor trait but not other fields: {type(t)} = {t}"
                )
            return (
                True,
                external_scope if self.external_scope is None else self.external_scope,
                external_name,
            )
        return bool(self.external), self.external_scope, None


###############################################################################
# Builders
###############################################################################


@dataclass
class ModuleBuilderOptions:
    # Whether to import torch symbolic shape expressions for ExportedPrograms.
    import_symbolic_shape_expressions: bool = False


class ModuleBuilder:
    """Wrapper around module and IR accounting for a module being built."""

    __slots__ = [
        "body",
        "cache",
        "context",
        "fx_py_attr_tracker",
        "last_global_op",
        "ip",
        "module_op",
        "options",
        "symbol_table",
        "global_ref_tracker",
        "native_type_converter",
        "_auto_symbol_counts",
    ]

    def __init__(
        self, module_op: Operation, *, options: ModuleBuilderOptions | None = None
    ):
        self.module_op = module_op
        self.options = options or ModuleBuilderOptions()
        self.context = module_op.context
        self.body = module_op.regions[0].blocks[0]
        self.symbol_table = SymbolTable(module_op)
        # We organize globals in order of declaration at the top of the module.
        # To do so, record the last one emitted so that newly created ones
        # can be ordered properly.
        self.last_global_op: Operation | None = None
        self.ip = InsertionPoint(self.body)
        # Tracks global references to a MaterializedGlobal.
        self.global_ref_tracker = RefTracker()
        # Usually the FxImporter makes a new ref tracker for each invocation,
        # but we want to preserve it across individual JIT evaluations so
        # as to better intern tensors to attributes.
        self.fx_py_attr_tracker = RefTracker()
        self._auto_symbol_counts: dict[str, int] = {}

    def unique_auto_symbol(self, requested_name: str) -> str:
        if requested_name not in self._auto_symbol_counts:
            self._auto_symbol_counts[requested_name] = 0
            return requested_name
        count = self._auto_symbol_counts[requested_name] + 1
        self._auto_symbol_counts[requested_name] = count
        return f"{requested_name}${count}"

    def handle_mlir_error(self, op: Operation, e: MLIRError, message: str):
        # TODO: Replace with a real dumping facility.
        # See: https://github.com/nod-ai/SHARK-ModelDev/issues/136
        dump_path = Path(tempfile.gettempdir()) / "iree_torch_importer_error.mlir"
        logger.exception(f"{message} (dumping to {dump_path})")
        try:
            with open(dump_path, "wb") as f:
                op.print(
                    file=f,
                    binary=True,
                    print_generic_op_form=True,
                    large_elements_limit=100,
                )
            logger.debug(f"Dump complete to {dump_path}")
        except Exception:
            logger.exception("Error generating dump file")

    def finalize_construct(self):
        try:
            self.module_op.verify()
        except MLIRError as e:
            self.handle_mlir_error(self.module_op, e, "module failed to verify")
            raise

    def torch_dtype_to_iree_type(self, dtype: torch.dtype) -> IrType:
        try:
            with self.context:
                return TORCH_DTYPE_TO_IREE_TYPE[dtype]()
        except KeyError:
            raise TypeError(f"Could not map Torch dtype {dtype} to an IREE type")

    def create_tensor_global(
        self,
        symbol_name: str,
        t: torch.Tensor,
        *,
        attrs: GlobalAttributes,
        logical_name: str | None = None,
    ) -> tuple[str, Operation, IrType]:
        element_type = self.torch_dtype_to_iree_type(t.dtype)
        external, external_scope, external_name = attrs.infer_external_from_tensor(t)

        # Always create globals at the top. Then after created, if there was
        # a prior one, move the new one to after it to maintain declaration
        # order.
        with InsertionPoint.at_block_begin(self.body), Location.unknown():
            tensor_type = RankedTensorType.get(list(t.shape), element_type)
            ir_attrs = {
                "sym_name": StringAttr.get(symbol_name),
                "sym_visibility": StringAttr.get("private"),
                "type": TypeAttr.get(tensor_type),
            }
            if attrs.noinline:
                ir_attrs["noinline"] = UnitAttr.get()
            if attrs.mutable:
                ir_attrs["is_mutable"] = UnitAttr.get()

            if external:
                # Emit named external reference.
                external_scope_attr = StringAttr.get(external_scope or "model")
                external_name = (
                    external_name
                    if external_name is not None
                    else attrs.map_name(
                        logical_name if logical_name is not None else symbol_name
                    )
                )
                external_name_attr = StringAttr.get(external_name)
                # TODO: Have real Python builders for this.
                ir_attrs["initial_value"] = Attribute.parse(
                    f"#flow.parameter.named<{external_scope_attr}::{external_name_attr}> : {tensor_type}"
                )
            elif attrs.uninitialized:
                # Emit uninitialized initial_value to signal that the memory
                # is valid but has undefined contents.
                # TODO: Have real Python builders for this.
                ir_attrs["initial_value"] = Attribute.parse(
                    f"#util.uninitialized : {tensor_type}"
                )
            else:
                # Emit inline initialized.
                # This is the line originally used here before copying
                # the tensor to a memoryview. We also need to make the tensor
                # contiguous here to get the correct strides. We further detach
                # the tensor from the current graph. Returned Tensor shares the
                # same storage with the original one. We further use .cpu()
                # to copy the tensor into CPU memory. If this object is already
                # in CPU memory and on the correct device, then no copy
                # is performed and the original object is returned.
                detached_tensor = t.detach().contiguous()
                blob_name = symbol_name
                if hasattr(DenseResourceElementsAttr, "get_from_ndarray"):
                    contents = torch.utils.dlpack.to_dlpack(detached_tensor)
                    # Pass the alignment explicitly: the binding otherwise
                    # infers it from the last stride, which a rank-zero tensor
                    # does not have.
                    elements_attr = DenseResourceElementsAttr.get_from_ndarray(
                        contents,
                        blob_name,
                        tensor_type,
                        alignment=detached_tensor.element_size(),
                    )
                else:
                    # Bindings without the DLPack entry point take a Python
                    # buffer, which cannot describe every dtype (bfloat16).
                    try:
                        buffer = memoryview(detached_tensor.cpu().numpy())
                    except TypeError as e:
                        raise NotImplementedError(
                            f"cannot embed a tensor of type {detached_tensor.dtype} "
                            "with these MLIR Python bindings; externalize the "
                            "parameters instead"
                        ) from e
                    elements_attr = DenseResourceElementsAttr.get_from_buffer(
                        buffer, blob_name, tensor_type
                    )
                ir_attrs["initial_value"] = elements_attr

            global_op = Operation.create("util.global", attributes=ir_attrs)
            self.symbol_table.insert(global_op)
            if self.last_global_op is not None:
                global_op.move_after(self.last_global_op)
            self.last_global_op = global_op
            actual_symbol_name = StringAttr(global_op.attributes["sym_name"]).value
            return actual_symbol_name, global_op, tensor_type

    def _create_initial_value_for_type(self, t: IrType) -> Attribute:
        # TODO: Implement something upstream for this (it exists in the C++ API).
        # See: https://github.com/nod-ai/AMD-SHARK-ModelDev/issues/169
        # and use it.
        if isinstance(t, RankedTensorType):
            rtt = RankedTensorType(t)
            if not rtt.has_static_shape:
                raise ValueError(
                    "Cannot create initialization value for dynamic shaped tensor"
                )
            element_attr = self._create_initial_value_for_type(rtt.element_type)
            return DenseElementsAttr.get_splat(t, element_attr)
        elif isinstance(t, IntegerType):
            return IntegerAttr.get(t, 0)
        elif isinstance(t, (F32Type, F64Type, F16Type)):
            # TODO: There should be a common way to check if a FloatType.
            # See: https://github.com/nod-ai/AMD-SHARK-ModelDev/issues/170
            return FloatAttr.get(t, 0.0)
        elif isinstance(t, IndexType):
            return IntegerAttr.get(IndexType.get(), 0)
        else:
            raise ValueError(
                f"Cannot create a default initialization value for type {t}"
            )
