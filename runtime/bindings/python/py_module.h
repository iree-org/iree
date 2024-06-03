// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_IREE_RT_PY_MODULE_H_
#define IREE_BINDINGS_PYTHON_IREE_RT_PY_MODULE_H_

#include <vector>

#include "./binding.h"

namespace iree::python {

void SetupPyModuleBindings(py::module_ &m);

}  // namespace iree::python

#endif  // IREE_BINDINGS_PYTHON_IREE_RT_PY_MODULE_H_
