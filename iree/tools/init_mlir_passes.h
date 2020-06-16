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

#ifndef IREE_TOOLS_INIT_MLIR_PASSES_H_
#define IREE_TOOLS_INIT_MLIR_PASSES_H_

#include <cstdlib>

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRVPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Quant/Passes.h"
#include "mlir/Dialect/SCF/Passes.h"
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
  // A local workaround until we can use individual pass registration from
  // https://reviews.llvm.org/D77322
  ::mlir::registerPass(
      "affine-loop-fusion", "Fuse affine loop nests",
      []() -> std::unique_ptr<Pass> { return mlir::createLoopFusionPass(); });
  ::mlir::registerPass("affine-pipeline-data-transfer",
                       "Pipeline non-blocking data transfers between "
                       "explicitly managed levels of the memory hierarchy",
                       []() -> std::unique_ptr<Pass> {
                         return mlir::createPipelineDataTransferPass();
                       });
  ::mlir::registerPass(
      "cse", "Eliminate common sub-expressions",
      []() -> std::unique_ptr<Pass> { return mlir::createCSEPass(); });
  ::mlir::registerPass("canonicalize", "Canonicalize operations",
                       []() -> std::unique_ptr<Pass> {
                         return mlir::createCanonicalizerPass();
                       });
  ::mlir::registerPass(
      "inline", "Inline function calls",
      []() -> std::unique_ptr<Pass> { return mlir::createInlinerPass(); });
  ::mlir::registerPass("snapshot-op-locations",
                       "Generate new locations from the current IR",
                       []() -> std::unique_ptr<Pass> {
                         return mlir::createLocationSnapshotPass();
                       });
  ::mlir::registerPass(
      "loop-coalescing",
      "Coalesce nested loops with independent bounds into a single loop",
      []() -> std::unique_ptr<Pass> {
        return mlir::createLoopCoalescingPass();
      });
  ::mlir::registerPass("loop-invariant-code-motion",
                       "Hoist loop invariant instructions outside of the loop",
                       []() -> std::unique_ptr<Pass> {
                         return mlir::createLoopInvariantCodeMotionPass();
                       });
  ::mlir::registerPass("memref-dataflow-opt",
                       "Perform store/load forwarding for memrefs",
                       []() -> std::unique_ptr<Pass> {
                         return mlir::createMemRefDataFlowOptPass();
                       });
  ::mlir::registerPass(
      "parallel-loop-collapsing",
      "Collapse parallel loops to use less induction variables",
      []() -> std::unique_ptr<Pass> {
        return mlir::createParallelLoopCollapsingPass();
      });
  ::mlir::registerPass(
      "print-op-stats", "Print statistics of operations",
      []() -> std::unique_ptr<Pass> { return mlir::createPrintOpStatsPass(); });
  ::mlir::registerPass("strip-debuginfo",
                       "Strip debug info from all operations",
                       []() -> std::unique_ptr<Pass> {
                         return mlir::createStripDebugInfoPass();
                       });
  ::mlir::registerPass(
      "symbol-dce", "Eliminate dead symbols",
      []() -> std::unique_ptr<Pass> { return mlir::createSymbolDCEPass(); });
  createCanonicalizerPass();
  createCSEPass();
  createSuperVectorizePass({});
  createLoopUnrollPass();
  createLoopUnrollAndJamPass();
  createSimplifyAffineStructuresPass();
  createLoopFusionPass();
  createLoopInvariantCodeMotionPass();
  createAffineLoopInvariantCodeMotionPass();
  createPipelineDataTransferPass();
  createLowerAffinePass();
  createLoopTilingPass(0);
  createLoopCoalescingPass();
  createAffineDataCopyGenerationPass(0, 0);
  createMemRefDataFlowOptPass();
  createInlinerPass();
  createSymbolDCEPass();
  createLocationSnapshotPass({});

  // Linalg
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Linalg/Passes.h.inc"

  // Loop
  createParallelLoopFusionPass();
  createParallelLoopTilingPass();

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

#endif  // IREE_TOOLS_INIT_MLIR_PASSES_H_
