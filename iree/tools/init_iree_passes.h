// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file defines a helper to trigger the registration of passes to
// the system.
//
// Based on MLIR's InitAllPasses but for IREE passes.

#ifndef IREE_TOOLS_INIT_IREE_PASSES_H_
#define IREE_TOOLS_INIT_IREE_PASSES_H_

#include <cstdlib>

#include "iree/compiler/Bindings/TFLite/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Analysis/TestPasses.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Conversion/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Analysis/TestPasses.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Dialect/VMLA/Transforms/Passes.h"
#include "iree/compiler/Translation/IREEVM.h"

namespace mlir {
namespace iree_compiler {

// Registers IREE passes with the global registry.
inline void registerAllIreePasses() {
  IREE::TFLite::registerPasses();
  IREE::TFLite::registerTransformPassPipeline();

  IREE::Flow::registerFlowPasses();
  IREE::Flow::registerFlowAnalysisTestPasses();
  IREE::HAL::registerHALPasses();
  IREE::registerTransformPasses();
  Shape::registerShapeConversionPasses();
  Shape::registerShapePasses();
  IREE::VM::registerVMPasses();
  IREE::VM::registerVMAnalysisTestPasses();
  IREE::VM::registerVMTestPasses();
  IREE::VMLA::registerVMLAPasses();
  registerIREEVMTransformPassPipeline();
}
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_TOOLS_INIT_IREE_PASSES_H_
