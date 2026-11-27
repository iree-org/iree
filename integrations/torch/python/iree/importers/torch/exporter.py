# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import platform
import warnings
from pathlib import Path
from typing import Any, Union, overload

import torch
from iree.compiler.api import (
    Output,
)

from iree.importers.torch import decompositions
from iree.importers.torch.export_context import (
    ExportContext,
    ModuleBuilderOptions,
)
from iree.importers.torch.fx_programs import FxPrograms
from iree.importers.torch.support.ir_imports import (
    Context,
    Operation,
    PassManager,
)

_is_windows = platform.system() == "Windows"

__all__ = ["ExportOutput", "export"]

ModuleLike = Union[
    torch.nn.Module,
    torch.export.ExportedProgram,
    FxPrograms,
]
SaveableTarget = Union[str, Path, None, Output]


class ExportOutput:
    """Wrapper around a CompiledModule produced by `export`."""

    def __init__(
        self,
        module: Operation,
    ):
        self.module = module

    @property
    def mlir_module(self) -> Operation:
        """Gets the MLIR module resulting from the last compilation phase."""
        return self.module

    def verify(self):
        """Runs the verifier on the module, raising an exception on failure."""
        self.mlir_module.verify()

    def print_readable(self, large_elements_limit: int = 50):
        """Prints a human readable version of the current compilation IR."""
        self.mlir_module.print(large_elements_limit=large_elements_limit)

    def save_mlir(self, file_path: str | Path):
        """Saves the current compilation IR to a path on disk.

        Args:
            file_path: Path to save the file. If it has a ".mlirbc"
              extension, it will be saved as bytecode. Otherwise as
              text.
        """
        file_path = Path(file_path)
        with open(file_path, "wb") as f:
            if file_path.suffix == ".mlirbc":
                self.mlir_module.write_bytecode(f)
            else:
                self.mlir_module.print(file=f, binary=True)

    def lower_to_iree(self, *, emit_async_entry_points: bool = True):
        """Converts the module from the torch dialect to IREE input IR.

        Args:
            emit_async_entry_points: Whether each function also gets an
              asynchronous entry point.
        """
        async_option = "true" if emit_async_entry_points else "false"
        pipeline = (
            "builtin.module("
            "func.func(torch-match-quantized-custom-ops),"
            f"torch-to-iree{{emit-async-entry-points={async_option}}})"
        )
        with self.mlir_module.context:
            PassManager.parse(pipeline).run(self.mlir_module)

    @staticmethod
    def load(file_path: str | Path) -> "ExportOutput":
        """Loads the ExportOutput from disc"""
        with Context() as ctx:
            return ExportOutput(
                Operation.parse(Path(file_path).read_text(), context=ctx)
            )


@overload
def export(
    module: torch.nn.Module,
    /,
    *,
    args: tuple | None = None,
    kwargs: dict[str, Any] | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None,
    module_name: str | None = None,
    function_name: str | None = None,
    strict_export: bool = True,
    import_symbolic_shape_expressions: bool = False,
) -> ExportOutput:
    """Exports a torch.nn.Module.

    This is done by first invoking torch.export.export with args, kwargs,
    and dynamic_shapes.
    """
    ...


@overload
def export(
    exported_program: torch.export.ExportedProgram,
    /,
    *,
    module_name: str | None = None,
    function_name: str | None = None,
) -> ExportOutput:
    """Exports a single entry-point module consisting of an ExportedProgram."""
    ...


@overload
def export(
    exported_programs: FxPrograms,
    /,
    *,
    module_name: str | None = None,
) -> ExportOutput:
    """Exports a multi entry-point ExportedProgram."""
    ...


def export(
    mdl: ModuleLike,
    /,
    args: tuple | None = None,
    kwargs: dict[str, Any] | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None,
    module_name: str | None = None,
    function_name: str | None = None,
    strict_export: bool = True,
    import_symbolic_shape_expressions: bool = False,
) -> ExportOutput:
    """Generic export of supported entities.

    See a more specific overload for accepted forms.

    This function behaves differently based on the type of the `mdl` argument:

    * nn.Module: The module is traced with torch.export.export passing it
      `args`, `kwargs`, and `dynamic_shapes`.
    * CompiledModule: The module is imported to IR. Additional arguments are
      illegal in this case.
    * torch.export.ExportedProgram: A pre-exported program can be passed and
      it will be used to construct a single-entrypoint module.

    Args:
      mdl: The nn.Module to export.
      *example_args: Example tensors.
      args: Example arguments to torch.export (if present, then *example_args
        must be empty.
      kwargs: Example keyword arguments.
      dynamic_shapes: Dynamic shape specs to pass to torch.export.
      arg_device: device affinities for the exported function
        arguments. On what devices should the program expect its arguments.
        It is a mapping of argument index to device affinity of the flattened
        arguments.

    Returns:
      An ExportOutput object that wraps the compilation and provides
      easy access.
    """
    if not strict_export:
        warnings.warn(
            "When strict_export == False, Python interpreter is used and code will execute exactly as it would in eager mode. "
            "Therefore, graph optimizations and safety are not garaunteed to be at the same level as TorchDynamo. "
            "This is an experimental feature in PyTorch that the IREE torch importer is still evaluating. Please report issues or experiences."
        )

    ctx = ExportContext(
        module_name=module_name or "module",
        options=ModuleBuilderOptions(
            import_symbolic_shape_expressions=import_symbolic_shape_expressions
        ),
    )
    current_decomps = decompositions.current_aot_decompositions()
    if isinstance(mdl, torch.export.ExportedProgram):
        if current_decomps:
            mdl = mdl.run_decompositions(current_decomps)
        ctx.add_exported_program(function_name or "main", mdl)
    elif isinstance(mdl, FxPrograms):
        for name, prog in mdl.programs.items():
            ctx.add_exported_program(name, prog)
    elif isinstance(mdl, torch.nn.Module):
        assert args
        nn_module = mdl
        exported_program = torch.export.export(
            nn_module,
            args=args,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict_export,
        )
        if current_decomps:
            exported_program = exported_program.run_decompositions(current_decomps)
        ctx.add_exported_program(function_name or "main", exported_program)
    else:
        raise TypeError(f"mdl argument (type: {type(mdl)}) is not a supported type")

    module = ctx.get_mlir_module()
    assert module
    return ExportOutput(module)
