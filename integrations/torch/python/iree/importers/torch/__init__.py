# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import of PyTorch programs into IREE."""

from iree.importers.torch.decompositions import (
    current_aot_decompositions,
    extend_aot_decompositions,
)
from iree.importers.torch.export_context import ExportContext
from iree.importers.torch.exporter import ExportOutput, export
from iree.importers.torch.fx_programs import (
    FxPrograms,
    FxProgramsBuilder,
    SharedStateTensor,
)
from iree.importers.torch.import_decompositions import (
    import_decomposition_table,
    import_decompositions,
    register_import_decomposition,
)
from iree.importers.torch.params import (
    ParameterArchive,
    ParameterArchiveBuilder,
    ParameterArchiveEntry,
    externalize_module_parameters,
    save_module_parameters,
)
from iree.importers.torch.pt2 import (
    externalize_program_parameters,
    import_programs,
    load_pt2,
    save_program_parameters,
    share_program_tensors,
)

__all__ = [
    "ExportContext",
    "ExportOutput",
    "FxPrograms",
    "FxProgramsBuilder",
    "ParameterArchive",
    "ParameterArchiveBuilder",
    "ParameterArchiveEntry",
    "SharedStateTensor",
    "current_aot_decompositions",
    "export",
    "extend_aot_decompositions",
    "externalize_module_parameters",
    "externalize_program_parameters",
    "import_decomposition_table",
    "import_decompositions",
    "import_programs",
    "load_pt2",
    "register_import_decomposition",
    "save_module_parameters",
    "save_program_parameters",
    "share_program_tensors",
]
