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

#include "mlir/Conversion/AVX512ToLLVM/ConvertAVX512ToLLVM.h"
#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRVPass.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/FxpMathOps/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LoopOps/Passes.h"
#include "mlir/Dialect/Quant/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Quantizer/Transforms/Passes.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "mlir/Transforms/ViewRegionGraph.h"

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
#define GEN_PASS_REGISTRATION
#include "mlir/Transforms/Passes.h.inc"

  // Conversion passes
#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/Passes.h.inc"

  // FxpMath
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/FxpMathOps/Passes.h.inc"

  // GPU
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/GPU/Passes.h.inc"

  // Linalg
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Linalg/Passes.h.inc"

  // Loop
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/LoopOps/Passes.h.inc"

  // Quant
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Quant/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "mlir/Quantizer/Transforms/Passes.h.inc"

  // SPIR-V
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SPIRV/Passes.h.inc"
}

}  // namespace mlir

#endif  // IREE_TOOLS_INIT_PASSES_H_
