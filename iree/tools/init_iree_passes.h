// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file defines a helper to trigger the registration of passes to
// the system.
//
// Based on MLIR's InitAllPasses but for IREE passes.

#ifndef IREE_TOOLS_INIT_IREE_PASSES_H_
#define IREE_TOOLS_INIT_IREE_PASSES_H_

#include <cstdlib>

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Bindings/Native/Transforms/Passes.h"
#include "iree/compiler/Bindings/TFLite/Transforms/Passes.h"
#include "iree/compiler/ConstEval/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Modules/VMVX/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Analysis/TestPasses.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "iree/compiler/InputConversion/TOSA/Passes.h"
#include "iree/compiler/Translation/IREEVM.h"

namespace mlir {
namespace iree_compiler {

// Registers IREE passes with the global registry.
inline void registerAllIreePasses() {
  IREE::ABI::registerPasses();
  IREE::ABI::registerTransformPassPipeline();

  IREE::TFLite::registerPasses();
  IREE::TFLite::registerTransformPassPipeline();

  registerCommonInputConversionPasses();
  MHLO::registerMHLOConversionPasses();
  registerTOSAConversionPasses();
  ConstEval::registerConstEvalPasses();

  IREE::Flow::registerFlowPasses();
  IREE::HAL::registerHALPasses();
  IREE::LinalgExt::registerPasses();
  IREE::Stream::registerStreamPasses();
  IREE::Util::registerTransformPasses();
  IREE::VM::registerVMPasses();
  IREE::VM::registerVMAnalysisTestPasses();
  IREE::VM::registerVMTestPasses();
  IREE::VMVX::registerVMVXPasses();
  registerIREEVMTransformPassPipeline();
}
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_TOOLS_INIT_IREE_PASSES_H_
