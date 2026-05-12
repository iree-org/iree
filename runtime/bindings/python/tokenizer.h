// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_IREE_RT_TOKENIZER_H_
#define IREE_BINDINGS_PYTHON_IREE_RT_TOKENIZER_H_

#include "./binding.h"

namespace iree::python {

void SetupTokenizerBindings(py::module_& m);

}  // namespace iree::python

#endif  // IREE_BINDINGS_PYTHON_IREE_RT_TOKENIZER_H_
