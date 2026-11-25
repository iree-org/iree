# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from typing import Any

from iree.importers.torch.support._logging import logger
from torch.export import ExportedProgram

from iree.importers.torch.support.ir_utils import (
    ModuleBuilder,
    ModuleBuilderOptions,
)
from iree.importers.torch.support.procedural.exported_program import (
    import_exported_program,
)
from iree.importers.torch.support.ir_imports import (
    Context,
    Location,
    Module,
    Operation,
    StringAttr,
)

__all__ = ["ExportContext"]


class ExportContext:
    """
    Manages the process of exporting PyTorch ExportedPrograms into an MLIR module.

    This class simplifies the original CompiledModule structure by removing
    metaclasses and class-level definitions. Exports are added explicitly
    to an instance of this context.
    """

    def __init__(
        self,
        *,
        module_name: str = "torch_module",
        options: ModuleBuilderOptions | None = None,
    ):
        """
        Initializes the ExportContext.

        Args:
            module_name: The desired 'sym_name' for the top-level MLIR module.
            options: Configuration options for the ModuleBuilder.
        """
        self.options = options or ModuleBuilderOptions()
        self._imported_items: dict[str, Any] = {}  # Stores results of imports

        self.context = Context()
        self.module_op = self._create_module(module_name)

        self.module_builder = ModuleBuilder(self.module_op, options=self.options)

    def _create_module(self, module_name: str) -> Operation:
        """Helper to create a new MLIR module operation."""
        with self.context:
            loc = Location.unknown(context=self.context)
            module = Module.create(loc)
            module_op = module.operation
            module_op.attributes["sym_name"] = StringAttr.get(
                module_name, context=self.context
            )
        return module_op

    def add_exported_program(
        self,
        export_name: str,
        exported_program: ExportedProgram,
        public: bool = True,
    ):
        """
        Imports a PyTorch ExportedProgram into the MLIR module.

        Args:
            export_name: The name to give the exported function symbol in MLIR.
            exported_program: The PyTorch ExportedProgram object.
            public: If True, the function visibility is public, otherwise private.
        """
        if export_name in self._imported_items:
            logger.warning(f"Export name '{export_name}' already exists. Overwriting.")

        logger.info(f"Adding ExportedProgram '{export_name}' (public={public})")
        visibility = (
            None if public else "private"
        )  # import_exported_program uses None for public

        # The actual import happens here using the helper function
        imported_repr = import_exported_program(
            self.module_builder,
            exported_program,
            symbol_name=export_name,
            symbol_visibility=visibility,
        )

        self._imported_items[export_name] = imported_repr  # Store reference/result
        # Note: finalize_construct might be needed after all adds, depending on ModuleBuilder impl.
        # Calling it here assumes adds can be interleaved with finalization steps,
        # or that it's idempotent/safe to call multiple times.
        # If not, move self.module_builder.finalize_construct() to *after* the loop
        # where users add all their programs, perhaps in run_import or a dedicated finalize method.
        self.module_builder.finalize_construct()

    def get_mlir_module(self) -> Operation | None:
        """Returns the current MLIR Module Operation."""
        return self.module_op

    def save_mlir(self, path: str | Path):
        """
        Saves the current MLIR module to a file.

        Args:
            path: The file path. If the extension is '.mlirbc', saves as bytecode,
                  otherwise saves as textual MLIR.
        """
        if not self.module_op:
            logger.error("Cannot save MLIR module, it is not available.")
            return

        path = Path(path)
        bytecode = path.suffix == ".mlirbc"
        mode = "wb"  # Always binary mode for open()

        logger.info(
            f"Saving MLIR module to {path} {'(bytecode)' if bytecode else '(textual)'}"
        )
        try:
            with open(path, mode) as f:
                if bytecode:
                    self.module_op.write_bytecode(f)
                else:
                    # Print requires a file opened in binary mode for its internal encoding
                    self.module_op.print(f, binary=True)
            logger.info(f"Successfully saved MLIR to {path}")
        except Exception as e:
            logger.error(f"Failed to save MLIR module to {path}: {e}")
            raise
