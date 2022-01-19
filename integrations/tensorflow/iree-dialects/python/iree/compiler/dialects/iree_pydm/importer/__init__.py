# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The importer is capable of introspecting running Python programs and converting
parts of them to the `iree_pydm` dialect for compilation. It effectively
defines a toolkit for constructing Python program extraction DSLs that can
be compiled into independent entities.
"""
from .util import (create_context, DefaultImportHooks, FuncProvidingIntrinsic,
                   ImportContext, ImportHooks, ImportStage)
from .importer import Importer
from .intrinsic_def import (def_ir_macro_intrinsic, def_pattern_call_intrinsic,
                            def_pyfunc_intrinsic)
