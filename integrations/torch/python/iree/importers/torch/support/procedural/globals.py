# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Global references in a module.

from collections.abc import Generator, Sequence
from typing import (
    Any,
)

import torch
from iree.importers.torch.support._logging import logger
from torch.utils._pytree import (
    TreeSpec,
    tree_unflatten,
)

from iree.importers.torch.support.ir_utils import (
    GlobalAttributes,
    ModuleBuilder,
)
from iree.importers.torch.support.procedural.base import IrTrace
from iree.importers.torch.support.procedural.primitives import (
    IrTensor,
)
from iree.importers.torch.support.ir_imports import (
    IrType,
    Operation,
    Value,
    util_d,
)

###############################################################################
# Globals
###############################################################################


class GlobalsDef:
    """Base class for all exporting descriptors."""

    __slots__ = [
        "_attrs",
    ]

    def __init__(self, attrs: GlobalAttributes):
        self._attrs = attrs

    def items(self) -> Generator[tuple[str, Any], None, None]:
        """Yields tuples of name/value exports."""
        raise NotImplementedError

    def schema(self) -> TreeSpec:
        """A schema used to unflatten for access from Python."""
        raise NotImplementedError

    def track(self, module_builder: ModuleBuilder, export_namespace: str) -> Any:
        """Track the given pack of globals, returning a struct that can be used to access them."""
        flat_globals = []
        for name, value in self.items():
            # Switch on types we support.
            fq_name = f"{export_namespace}.{name}"
            if isinstance(value, torch.Tensor):
                mapping = module_builder.global_ref_tracker.track(value)
                if not mapping.is_empty:
                    logger.debug(
                        f"IGNORE EXISTING TRACKED TENSOR({fq_name}): {mapping}"
                    )
                    flat_globals.append(mapping.value)
                    continue
                (
                    actual_symbol_name,
                    global_op,
                    global_type,
                ) = module_builder.create_tensor_global(
                    f"_{fq_name}",
                    value,
                    attrs=self._attrs,
                    logical_name=fq_name,
                )
                mapping.value = IrGlobalTensor(
                    fq_name,
                    self,
                    symbol_name=actual_symbol_name,
                    global_op=global_op,
                    global_type=global_type,
                    dtype=value.dtype,
                )
                logger.debug(f"TRACK NEW TENSOR({fq_name}): {mapping}")
                flat_globals.append(mapping.value)
                continue

            raise TypeError(f"Unsupported global type: {value.__class__}")
        tree_globals = tree_unflatten(flat_globals, self.schema())
        assert isinstance(tree_globals, MaterializedGlobal)
        return tree_globals


class MaterializedGlobal:
    """Tags an Ir* that is duck-typed as a global."""

    ir_type: IrType
    symbol_name: str
    global_op: Operation
    global_type: IrType


class IrGlobalTensor(IrTensor, MaterializedGlobal):
    """An IrScalar that is loaded from a global and associated with its aggregate."""

    __slots__ = [
        "global_op",
        "info",
        "export_name",
        "symbol_name",
    ]

    def __init__(
        self,
        export_name: str,
        info: GlobalsDef,
        *,
        symbol_name: str,
        global_op: Operation,
        global_type: IrType,
        dtype: torch.dtype,
    ):
        super().__init__(global_type, dtype)
        self.info = info
        self.export_name = export_name
        self.symbol_name = symbol_name
        self.global_op = global_op

    def resolve_ir_values(self, trace: "IrTrace") -> Sequence[Value]:
        with trace.loc, trace.ip:
            value = util_d.GlobalLoadOp(self.ir_type, self.symbol_name).result
        return [value]

    def __repr__(self):
        return f"<MaterializedGlobal {self.export_name} = {self.symbol_name}:{self.ir_type}>"
