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

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Quant/Passes.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

// This function may be called to register the MLIR passes with the global
// registry.
// If you're building a compiler, you likely don't need this: you would build a
// pipeline programmatically without the need to register with the global
// registry, since it would already be calling the creation routine of the
// individual passes.
// The global registry is interesting to interact with the command-line tools.
inline void registerMlirPasses() {
  // Core Transforms
  registerCanonicalizerPass();
  registerCSEPass();
  registerInlinerPass();
  registerLocationSnapshotPass();
  registerLoopCoalescingPass();
  registerLoopInvariantCodeMotionPass();
  registerMemRefDataFlowOptPass();
  registerParallelLoopCollapsingPass();
  registerPrintOpStatsPass();
  registerStripDebugInfoPass();
  registerSymbolDCEPass();

  // Affine
  registerAffinePasses();
  registerAffineLoopFusionPass();
  registerAffinePipelineDataTransferPass();
  registerConvertAffineToStandardPass();

  // Linalg
  registerLinalgPasses();

  // MemRef
  memref::registerMemRefPasses();

  // SCF
  registerSCFParallelLoopFusionPass();
  registerSCFParallelLoopTilingPass();

  // Quant
  quant::registerQuantPasses();

  // Shape
  registerShapePasses();

  // SPIR-V
  spirv::registerSPIRVLowerABIAttributesPass();
  registerConvertGPUToSPIRVPass();
  registerConvertStandardToSPIRVPass();
  registerConvertLinalgToSPIRVPass();

  // TOSA.
  registerTosaToLinalgOnTensorsPass();
}

}  // namespace mlir

#endif  // IREE_TOOLS_INIT_MLIR_PASSES_H_
