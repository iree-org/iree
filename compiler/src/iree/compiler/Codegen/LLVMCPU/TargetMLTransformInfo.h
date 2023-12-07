// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_TARGETMLTRANSFORMINFO_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_TARGETMLTRANSFORMINFO_H_

#include <limits>

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

namespace mlir::iree_compiler {

/// Holds target specific information to specialize ML transformations.
// TODO(dcaballe): Move to a Concept-Model implementation when it's worth it.
struct TargetMLTransformInfo {
  unsigned defaultMaxUnrollFactor = 8;
  unsigned defaultMaxTransposeUnrollFactor =
      std::numeric_limits<unsigned>::max();

  static const TargetMLTransformInfo
  getTargetMLTransformInfo(IREE::HAL::ExecutableTargetAttr targetAttr);
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_TARGETMLTRANSFORMINFO_H_
