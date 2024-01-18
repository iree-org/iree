// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/TargetMLTransformInfo.h"

#include "iree/compiler/Codegen/LLVMCPU/Utils.h"

namespace mlir::iree_compiler {

namespace {

struct RISCVTargetMLTransformInfo : TargetMLTransformInfo {
  RISCVTargetMLTransformInfo() {
    defaultMaxUnrollFactor = 8;
    defaultMaxTransposeUnrollFactor = 1;
  }
};

} // namespace

const TargetMLTransformInfo TargetMLTransformInfo::getTargetMLTransformInfo(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (isRISCV(targetAttr)) {
    return RISCVTargetMLTransformInfo();
  }

  return TargetMLTransformInfo();
};

} // namespace mlir::iree_compiler
