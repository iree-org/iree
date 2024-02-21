// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file defines a helper to trigger the registration of passes to
// the system.
//
// Based on MLIR's InitAllPasses but for IREE passes.

#ifndef IREE_COMPILER_TOOLS_INIT_IREE_PASSES_H_
#define IREE_COMPILER_TOOLS_INIT_IREE_PASSES_H_

#include <cstdlib>

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Bindings/Native/Transforms/Passes.h"
#include "iree/compiler/Bindings/TFLite/Transforms/Passes.h"
#include "iree/compiler/ConstEval/Passes.h"
#include "iree/compiler/Dialect/Flow/Conversion/MeshToFlow/MeshToFlow.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Analysis/TestPasses.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/Modules/HAL/Inline/Transforms/Passes.h"
#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h"
#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/Preprocessing/Passes.h"

#ifdef IREE_HAVE_C_OUTPUT_FORMAT
// TODO: Remove these once rolled up into explicit registration.
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"
#endif // IREE_HAVE_C_OUTPUT_FORMAT

namespace mlir::iree_compiler {

// Registers IREE passes with the global registry.
inline void registerAllIreePasses() {
  IREE::ABI::registerPasses();
  IREE::ABI::registerTransformPassPipeline();

  IREE::TFLite::registerPasses();
  IREE::TFLite::registerTransformPassPipeline();

  registerCommonInputConversionPasses();
  ConstEval::registerConstEvalPasses();
  GlobalOptimization::registerGlobalOptimizationPipeline();
  Preprocessing::registerPreprocessingPasses();
  IREE::Flow::registerFlowPasses();
  IREE::Flow::registerMeshToFlowPasses();
  IREE::HAL::registerHALPasses();
  IREE::HAL::Inline::registerHALInlinePasses();
  IREE::HAL::Loader::registerHALLoaderPasses();
  IREE::IO::Parameters::registerParametersPasses();
  IREE::LinalgExt::registerPasses();
  IREE::Stream::registerStreamPasses();
  IREE::Util::registerTransformPasses();
  IREE::VM::registerVMPasses();
  IREE::VM::registerVMAnalysisTestPasses();
  IREE::VM::registerVMTestPasses();
  IREE::VMVX::registerVMVXPasses();
  registerIREEVMTransformPassPipeline();

  // We have some dangling passes that don't use explicit
  // registration and that we need to force instantiation
  // of in order to register.
  // TODO: Eliminate these.
#ifdef IREE_HAVE_C_OUTPUT_FORMAT
  IREE::VM::createConvertVMToEmitCPass();
#endif // IREE_HAVE_C_OUTPUT_FORMAT
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_TOOLS_INIT_IREE_PASSES_H_
