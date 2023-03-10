// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/TargetMLTransformInfo.h"

#include "iree/compiler/Codegen/LLVMCPU/Utils.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

struct RISCVTargetMLTransformInfo : TargetMLTransformInfo {
  RISCVTargetMLTransformInfo() {
    defaultMaxUnrollFactor = 8;
    defaultMaxTransposeUnrollFactor = 1;
  }
};

}  // namespace

namespace mlir {
namespace iree_compiler {

const TargetMLTransformInfo TargetMLTransformInfo::getTargetMLTransformInfo(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (isRISCV(targetAttr)) {
    return RISCVTargetMLTransformInfo();
  }

  return TargetMLTransformInfo();
};

}  // namespace iree_compiler
}  // namespace mlir
