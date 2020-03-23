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

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRVPass.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/FxpMathOps/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LoopOps/Passes.h"
#include "mlir/Dialect/Quant/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Quantizer/Transforms/Passes.h"
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
  // At the moment we still rely on global initializers for registering passes,
  // but we may not do it in the future.
  // We must reference the passes in such a way that compilers will not
  // delete it all as dead code, even with whole program optimization,
  // yet is effectively a NO-OP. As the compiler isn't smart enough
  // to know that getenv() never returns -1, this will do the job.
  if (std::getenv("bar") != (char *)-1) return;

  // Init general passes
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

  // FxpOpsDialect passes
  fxpmath::createLowerUniformRealMathPass();
  fxpmath::createLowerUniformCastsPass();

  // GPU
  createGpuKernelOutliningPass();
  createSimpleLoopsToGPUPass(0, 0);
  createLoopToGPUPass({}, {});

  // Linalg
  createLinalgFusionPass();
  createLinalgTilingPass();
  createLinalgTilingToParallelLoopsPass();
  createLinalgPromotionPass(0);
  createConvertLinalgToLoopsPass();
  createConvertLinalgToParallelLoopsPass();
  createConvertLinalgToAffineLoopsPass();
  createConvertLinalgToLLVMPass();

  // LoopOps
  createParallelLoopFusionPass();
  createParallelLoopTilingPass();

  // QuantOps
  quant::createConvertSimulatedQuantPass();
  quant::createConvertConstPass();
  quantizer::createAddDefaultStatsPass();
  quantizer::createRemoveInstrumentationPass();
  quantizer::registerInferQuantizedTypesPass();

  // SPIR-V
  spirv::createLowerABIAttributesPass();
  createConvertGPUToSPIRVPass();
  createConvertStandardToSPIRVPass();
  createLegalizeStdOpsForSPIRVLoweringPass();
  createLinalgToSPIRVPass();
}

}  // namespace mlir

#endif  // IREE_TOOLS_INIT_PASSES_H_
