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
      builtins.addFile(toc[i].name, llvm::StringRef{toc[i].data, toc[i].size});
    }
  }

  {
    const iree_file_toc_t *toc = iree_specialization_patterns_amdgpu_create();
    for (size_t i = 0, e = iree_specialization_patterns_amdgpu_size(); i != e;
         ++i) {
      builtins.addFile(toc[i].name, llvm::StringRef{toc[i].data, toc[i].size});
    }
  }

  {
    const iree_file_toc_t *toc = iree_mlir_ukernel_patterns_amdgpu_create();
    for (size_t i = 0, e = iree_mlir_ukernel_patterns_amdgpu_size(); i != e;
         ++i) {
      builtins.addFile(toc[i].name, llvm::StringRef{toc[i].data, toc[i].size});
    }
  }

  {
    const iree_file_toc_t *toc = iree_mlir_ukernels_amdgpu_create();
    for (size_t i = 0, e = iree_mlir_ukernels_amdgpu_size(); i != e; ++i) {
      builtins.addFile(toc[i].name, llvm::StringRef{toc[i].data, toc[i].size});
    }
  }
}

bool ROCMDialect::hasBuiltin(llvm::StringRef name) {
  return builtins.getFile(name).has_value();
}

std::optional<StringRef> ROCMDialect::getBuiltin(llvm::StringRef name) {
  return builtins.getFile(name);
}

SmallVector<StringRef> ROCMDialect::getBuiltinNames() {
  return llvm::to_vector(builtins.getMap().keys());
}

const ArrayRef<Util::FuncOp> ROCMDialect::getMlirUKernels() {
  // Issue #22842: This needs to be a recursive mutex because parsing MLIR
  // ukernels triggers the verifier, which schedules work as futures and then
  // yields to the thread pool, allowing other tasks to be scheduled on the
  // current thread, potentially recursing back here on the same thread, so this
  // used to deadlock with std::mutex.
  //
  // A downside of using std::recursive_mutex to allow such recursion, though
  // is that we are now potentially entering this critical section multiple
  // times (albeit on the same thread). We do not want to be parsing MLIR
  // ukernels twice and inserting them into `mlirUkernels` twice. To prevent
  // that, we condition this on a boolean `mlirUkernelsParsingStarted` that is
  // set *before* the call to getOrLoadBuiltinModule potentially recursing.
  std::lock_guard<std::recursive_mutex> guard(mlirUkernelsMutex);
  if (!mlirUkernelsParsingStarted) {
    mlirUkernelsParsingStarted = true;
    const iree_file_toc_t *toc = iree_mlir_ukernels_amdgpu_create();
    llvm::SmallVector<ModuleOp> result;
    for (size_t i = 0, e = iree_mlir_ukernels_amdgpu_size(); i != e; ++i) {
      auto moduleOp = this->getOrLoadBuiltinModule(toc[i].name);
      if (failed(moduleOp)) {
        // parseSourceString should have reported an error already.
        continue;
      }
      moduleOp->walk(
          [&](Util::FuncOp funcOp) { mlirUkernels.push_back(funcOp); });
    }
  }
  assert(!mlirUkernels.empty() &&
         "Zero MLIR ukernels found after parsing builtins?!");
  return mlirUkernels;
}

} // namespace mlir::iree_compiler::IREE::ROCM
