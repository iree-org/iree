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
// Based on MLIR's InitAllPasses but without passes we don't care about.

#ifndef IREE_TOOLS_INIT_PASSES_H_
#define IREE_TOOLS_INIT_PASSES_H_

#include <cstdlib>

#include "iree/compiler/Dialect/Flow/Analysis/TestPasses.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Conversion/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Analysis/TestPasses.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Dialect/VMLA/Transforms/Passes.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRVPass.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LoopOps/Passes.h"
#include "mlir/Dialect/Quant/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

// This function may be called to register the MLIR passes with the
// global registry.
// If you're building a compiler, you likely don't need this: you would build a
// pipeline programmatically without the need to register with the global
// registry, since it would already be calling the creation routine of the
// individual passes.
// The global registry is interesting to interact with the command-line tools.
inline void registerMlirPasses() {
  // Init general passes
#define GEN_PASS_REGISTRATION_AffineLoopFusion
#define GEN_PASS_REGISTRATION_AffinePipelineDataTransfer
#define GEN_PASS_REGISTRATION_CSE
#define GEN_PASS_REGISTRATION_Canonicalizer
#define GEN_PASS_REGISTRATION_Inliner
#define GEN_PASS_REGISTRATION_LocationSnapshot
#define GEN_PASS_REGISTRATION_LoopCoalescing
#define GEN_PASS_REGISTRATION_LoopInvariantCodeMotion
#define GEN_PASS_REGISTRATION_MemRefDataFlowOpt
#define GEN_PASS_REGISTRATION_ParallelLoopCollapsing
#define GEN_PASS_REGISTRATION_PrintOpStats
#define GEN_PASS_REGISTRATION_StripDebugInfo
#define GEN_PASS_REGISTRATION_SymbolDCE
#include "mlir/Transforms/Passes.h.inc"

  // Conversion passes
#define GEN_PASS_REGISTRATION_ConvertAffineToStandard
#define GEN_PASS_REGISTRATION_ConvertSimpleLoopsToGPU
#define GEN_PASS_REGISTRATION_ConvertLoopsToGPU
#define GEN_PASS_REGISTRATION_ConvertLinalgToLLVM
#include "mlir/Conversion/Passes.h.inc"

  // Affine
#define GEN_PASS_REGISTRATION_AffineVectorize
#define GEN_PASS_REGISTRATION_AffineLoopUnroll
#define GEN_PASS_REGISTRATION_AffineLoopUnrollAndJam
#define GEN_PASS_REGISTRATION_SimplifyAffineStructures
#define GEN_PASS_REGISTRATION_AffineLoopInvariantCodeMotion
#define GEN_PASS_REGISTRATION_AffineLoopTiling
#define GEN_PASS_REGISTRATION_AffineDataCopyGeneration
#include "mlir/Dialect/Affine/Passes.h.inc"

  // GPU
#define GEN_PASS_REGISTRATION_GpuKernelOutlining
#include "mlir/Dialect/GPU/Passes.h.inc"

  // Linalg
#define GEN_PASS_REGISTRATION_LinalgFusion
#define GEN_PASS_REGISTRATION_LinalgTiling
#define GEN_PASS_REGISTRATION_LinalgTilingToParallelLoops
#define GEN_PASS_REGISTRATION_LinalgPromotion
#define GEN_PASS_REGISTRATION_LinalgLowerToLoops
#define GEN_PASS_REGISTRATION_LinalgLowerToParallelLoops
#define GEN_PASS_REGISTRATION_LinalgLowerToAffineLoops
#include "mlir/Dialect/Linalg/Passes.h.inc"

  // Loop
#define GEN_PASS_REGISTRATION_LoopParallelLoopFusion
#define GEN_PASS_REGISTRATION_LoopParallelLoopTiling
#include "mlir/Dialect/LoopOps/Passes.h.inc"

  // Quant
  quant::createConvertSimulatedQuantPass();
  quant::createConvertConstPass();

  // SPIR-V
  spirv::createLowerABIAttributesPass();
  createConvertGPUToSPIRVPass();
  createConvertStandardToSPIRVPass();
  createLegalizeStdOpsForSPIRVLoweringPass();
  createLinalgToSPIRVPass();
}

}  // namespace mlir

namespace mlir {
namespace iree_compiler {

// This function may be called to register the IREE passes with the
// global registry.
inline void registerAllIreePasses() {
  IREE::Flow::registerFlowPasses();
  IREE::Flow::registerFlowAnalysisTestPasses();
  IREE::HAL::registerHALPasses();
  IREE::registerIreePasses();
  Shape::registerShapeConversionPasses();
  Shape::registerShapePasses();
  IREE::VM::registerVMPasses();
  IREE::VM::registerVMAnalysisTestPasses();
  IREE::VM::registerVMTestPasses();
  IREE::VMLA::registerVMLAPasses();
}
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_TOOLS_INIT_PASSES_H_
