# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import of programs that were saved with ``torch.export.save``."""

from collections.abc import Iterator, Mapping
from pathlib import Path

import torch

from iree.importers.torch.decompositions import current_aot_decompositions
from iree.importers.torch.exporter import ExportOutput, export
from iree.importers.torch.fx_programs import FxPrograms
from iree.importers.torch.import_decompositions import import_decompositions
from iree.importers.torch.params import ParameterArchiveBuilder
from iree.importers.torch.support.tensor_traits import ExternalTensorTrait

__all__ = [
    "externalize_program_parameters",
    "import_programs",
    "load_pt2",
    "save_program_parameters",
    "share_program_tensors",
]


def load_pt2(path: str | Path) -> torch.export.ExportedProgram:
    """Load a program saved with ``torch.export.save``.

    Custom operators and pytree types that the program uses must be registered
    before it is loaded, and the device it was exported on must be available.
    """
    return torch.export.load(str(path))


def _program_tensors(
    program: torch.export.ExportedProgram,
) -> Iterator[tuple[Mapping, str, torch.Tensor]]:
    for holder in (program.state_dict, program.constants):
        for name, value in holder.items():
            if isinstance(value, torch.Tensor):
                yield holder, name, value


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def share_program_tensors(
    programs: Mapping[str, torch.export.ExportedProgram],
) -> int:
    """Make programs share the tensors that they have in common.

    Programs that are loaded from separate files each hold a copy of their
    parameters and buffers. A tensor is shared when programs hold it under the
    same name with the same type, shape and contents, so that it is imported
    once and state is common to the programs. Tensors within one program are
    never merged.

    Returns:
        The number of tensors that were replaced by a shared one.
    """
    shared: dict[str, torch.Tensor] = {}
    replaced = 0
    for program in programs.values():
        added: dict[str, torch.Tensor] = {}
        for holder, name, tensor in list(_program_tensors(program)):
            existing = shared.get(name)
            if existing is None:
                added[name] = tensor
            elif existing is not tensor and _same_tensor(existing, tensor):
                holder[name] = existing
                replaced += 1
        shared.update(added)
    return replaced


def _named_program_tensors(
    programs: Mapping[str, torch.export.ExportedProgram],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield each distinct tensor of the programs with an unambiguous name."""
    names: dict[str, torch.Tensor] = {}
    visited: set[int] = set()
    for program_name, program in programs.items():
        for _, name, tensor in _program_tensors(program):
            if id(tensor) in visited:
                continue
            visited.add(id(tensor))
            if name in names:
                name = f"{program_name}.{name}"
            if name in names:
                raise ValueError(f"cannot name tensor {name!r} unambiguously")
            names[name] = tensor
            yield name, tensor


def externalize_program_parameters(
    programs: Mapping[str, torch.export.ExportedProgram],
    *,
    external_scope: str = "",
) -> dict[str, torch.Tensor]:
    """Mark the parameters and buffers of programs as external.

    Marked tensors are imported as named references into a parameter archive
    instead of having their contents embedded in the module.

    Returns:
        The marked tensors by the name they are referenced under.
    """
    marked = dict(_named_program_tensors(programs))
    for name, tensor in marked.items():
        ExternalTensorTrait(external_scope=external_scope, external_name=name).set(
            tensor
        )
    return marked


def save_program_parameters(
    file_path: str | Path, tensors: Mapping[str, torch.Tensor]
) -> None:
    """Write tensors, as ``externalize_program_parameters`` returns them, to a
    parameter archive."""
    builder = ParameterArchiveBuilder()
    for name, tensor in tensors.items():
        builder.add_tensor(name, tensor)
    builder.save(file_path)


def import_programs(
    programs: Mapping[str, torch.export.ExportedProgram],
    *,
    module_name: str | None = None,
    decompose: bool = True,
    share_tensors: bool = True,
    externalize_parameters: bool = False,
    save_parameters_to: str | Path | None = None,
    import_symbolic_shape_expressions: bool = True,
) -> ExportOutput:
    """Import exported programs as the functions of one module.

    Args:
        programs: The programs by the name of the function each becomes.
        module_name: Name of the module.
        decompose: Whether to run the import decompositions on the programs.
        share_tensors: Whether to share the tensors that programs have in
            common, see ``share_program_tensors``.
        externalize_parameters: Whether to keep the parameters and buffers out
            of the module, see ``externalize_program_parameters``.
        save_parameters_to: Parameter archive to write the external parameters
            to. Requires ``externalize_parameters`` and ``iree.runtime``.
        import_symbolic_shape_expressions: Whether to import the relations
            between dynamic dimensions.

    Returns:
        The module in the torch dialect.
    """
    if not programs:
        raise ValueError("no programs to import")
    if save_parameters_to is not None and not externalize_parameters:
        raise ValueError("save_parameters_to requires externalize_parameters")

    programs = dict(programs)
    if decompose:
        with import_decompositions():
            table = current_aot_decompositions()
            programs = {
                name: program.run_decompositions(table)
                for name, program in programs.items()
            }
    if share_tensors and len(programs) > 1:
        share_program_tensors(programs)
    if externalize_parameters:
        external = externalize_program_parameters(programs)
        if save_parameters_to is not None:
            save_program_parameters(save_parameters_to, external)

    fx_programs = FxPrograms()
    fx_programs.programs.update(programs)
    return export(
        fx_programs,
        module_name=module_name,
        import_symbolic_shape_expressions=import_symbolic_shape_expressions,
    )
