// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.h"
#include "compiler/plugins/target/ROCM/builtins/mlir_ukernel/iree_mlir_ukernel_patterns_amdgpu.h"
#include "compiler/plugins/target/ROCM/builtins/mlir_ukernel/iree_mlir_ukernels_amdgpu.h"
#include "compiler/plugins/target/ROCM/builtins/specialization/iree_specialization_patterns_amdgpu.h"
#include "compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_specs_amdgpu.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.cpp.inc"

namespace mlir::iree_compiler::IREE::ROCM {

void ROCMDialect::initialize() {
  registerAttributes();

  // Initialize the mapping from builtin filenames to data.
  {
    const iree_file_toc_t *toc = iree_default_tuning_specs_amdgpu_create();
    for (size_t i = 0, e = iree_default_tuning_specs_amdgpu_size(); i != e;
         ++i) {
      builtins.addFile(toc[i].name, StringRef{toc[i].data, toc[i].size});
    }
  }

  {
    const iree_file_toc_t *toc = iree_specialization_patterns_amdgpu_create();
    for (size_t i = 0, e = iree_specialization_patterns_amdgpu_size(); i != e;
         ++i) {
      builtins.addFile(toc[i].name, StringRef{toc[i].data, toc[i].size});
    }
  }

  {
    const iree_file_toc_t *toc = iree_mlir_ukernel_patterns_amdgpu_create();
    for (size_t i = 0, e = iree_mlir_ukernel_patterns_amdgpu_size(); i != e;
         ++i) {
      builtins.addFile(toc[i].name, StringRef{toc[i].data, toc[i].size});
    }
  }

  {
    const iree_file_toc_t *toc = iree_mlir_ukernels_amdgpu_create();
    for (size_t i = 0, e = iree_mlir_ukernels_amdgpu_size(); i != e; ++i) {
      builtins.addFile(toc[i].name, StringRef{toc[i].data, toc[i].size});
    }
  }
}

bool ROCMDialect::hasBuiltin(StringRef name) {
  return builtins.getFile(name).has_value();
}

std::optional<StringRef> ROCMDialect::getBuiltin(StringRef name) {
  return builtins.getFile(name);
}

SmallVector<StringRef> ROCMDialect::getBuiltinNames() {
  return to_vector(builtins.getMap().keys());
}

const ArrayRef<Util::FuncOp> ROCMDialect::getMlirUKernels() {
  // Issue #22842: Avoid doing nontrivial MLIR work (such as parsing) in a
  // critical section. Due to how MLIR threading works, any threaded workload
  // may result in yielding and scheduling another task on the same thread,
  // potentially reentering this code on the same thread, resulting in
  // deadlocks. That is why the code below is structured with two separate
  // critical sections leaving the MLIR parsing itself outside. It was
  // specifically the verifier that was being threaded here, and we could have
  // set verifyAfterParse=false, but we actually care about the verifier running
  // here, and it is unsafe to assume that it will always be the only threaded
  // thing here.

  {
    // Critical section: check if already have mlirUkernels.
    std::lock_guard<std::mutex> guard(mlirUkernelsMutex);
    if (!mlirUkernels.empty()) {
      return mlirUkernels;
    }
  }

  // Do the parsing outside of critical sections, so that reentry will not
  // deadlock.
  SmallVector<Util::FuncOp> localMlirUkernels;
  const iree_file_toc_t *toc = iree_mlir_ukernels_amdgpu_create();
  SmallVector<ModuleOp> result;
  for (size_t i = 0, e = iree_mlir_ukernels_amdgpu_size(); i != e; ++i) {
    FailureOr<ModuleOp> moduleOp = this->getOrLoadBuiltinModule(toc[i].name);
    if (failed(moduleOp)) {
      // parseSourceString should have reported an error already.
      continue;
    }
    moduleOp->walk(
        [&](Util::FuncOp funcOp) { localMlirUkernels.push_back(funcOp); });
  }

  // Critical section: set mlirUkernels.
  std::lock_guard<std::mutex> guard(mlirUkernelsMutex);
  if (mlirUkernels.empty()) {
    mlirUkernels = localMlirUkernels;
  }
  return mlirUkernels;
}

} // namespace mlir::iree_compiler::IREE::ROCM
