// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.h"

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.cpp.inc"
#include "compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_specs_amdgpu.h"

namespace mlir::iree_compiler::IREE::ROCM {

void ROCMDialect::initialize() {
  registerAttributes();

  // Initialize the mapping from builtin filenames to data.
  const iree_file_toc_t *toc = iree_default_tuning_specs_amdgpu_create();
  for (size_t i = 0, e = iree_default_tuning_specs_amdgpu_size(); i != e; ++i) {
    builtins.addFile(toc[i].name, llvm::StringRef{toc[i].data, toc[i].size});
  }
}

bool ROCMDialect::hasBuiltin(llvm::StringRef name) {
  return builtins.getFile(name).has_value();
}

} // namespace mlir::iree_compiler::IREE::ROCM
