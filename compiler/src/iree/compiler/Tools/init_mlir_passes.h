// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file defines a helper to trigger the registration of passes to
// the system.
//
// Based on MLIR's InitAllPasses but without passes we don't care about.

#ifndef IREE_COMPILER_TOOLS_INIT_MLIR_PASSES_H_
#define IREE_COMPILER_TOOLS_INIT_MLIR_PASSES_H_

#include <cstdlib>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
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
  affine::registerLoopCoalescingPass();
  registerLoopInvariantCodeMotionPass();
  affine::registerAffineScalarReplacementPass();
  registerPrintOpStatsPass();
  registerViewOpGraphPass();
  registerStripDebugInfoPass();
  registerSymbolDCEPass();

  // Generic conversions
  registerReconcileUnrealizedCastsPass();

  // Affine
  affine::registerAffinePasses();
  affine::registerAffineLoopFusionPass();
  affine::registerAffinePipelineDataTransferPass();
  registerConvertAffineToStandardPass();

  // Linalg
  registerLinalgPasses();

  // LLVM
  registerConvertArmNeon2dToIntrPass();

  // MemRef
  memref::registerMemRefPasses();

  // SCF
  registerSCFParallelLoopFusionPass();
  registerSCFParallelLoopTilingPass();
  registerSCFToControlFlowPass();

  // Shape
  registerShapePasses();

  // SPIR-V
  spirv::registerSPIRVLowerABIAttributesPass();
  registerConvertGPUToSPIRVPass();
  registerConvertControlFlowToSPIRVPass();
  registerConvertFuncToSPIRVPass();
}

}  // namespace mlir

#endif  // IREE_COMPILER_TOOLS_INIT_MLIR_PASSES_H_
