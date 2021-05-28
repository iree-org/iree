// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONVERSION_INIT_CONVERSIONS_H_
#define IREE_COMPILER_CONVERSION_INIT_CONVERSIONS_H_

#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Conversion/LinalgToLinalg/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Conversion/LinalgToVector/Passes.h"
#include "iree/compiler/Conversion/Passes.h"
#include "iree/compiler/Conversion/VectorToLLVM/Passes.h"

namespace mlir {
namespace iree_compiler {

// These functions should be called before creating any MLIRContext if one
// expects all the possible conversions to be made available to the context
// automatically.

inline void registerCommonConversionPasses() {
  static bool init_once = []() {
    // Common
    createFlattenMemRefSubspanPass();
    createForOpCanonicalizationPass();
    createLinalgBufferizePass();
    createSetNumWorkgroupsPass();
    return true;
  }();
  (void)init_once;
}

inline void registerLinalgToVectorPasses() {
  static bool init_once = []() {
    createVectorizeLinalgConvPass();
    return true;
  }();
  (void)init_once;
}

inline void registerLinalgToSPIRVPasses() {
  static bool init_once = []() {
    // LinalgToSPIRV
    createConvertToGPUPass();
    createFoldProcessorIDUsesPass();
    createTileAndVectorizeInOneWorkgroupPass(SPIRVCodegenOptions());
    createVectorToGPUPass();
    createVectorizeMemrefLoadStorePass();
    return true;
  }();
  (void)init_once;
}

inline void registerLinalgToLLVMPasses() {
  static bool init_once = []() {
    createLowerExecutableTargetPass(LLVMCodegenOptions());
    // LinalgToLLVM
    createLinalgTileAndVectorizeWorkgroupsPass();
    createUnfusedFMAOpsPass();
    createPadLinalgWorkgroupTilesPass();
    return true;
  }();
  (void)init_once;
}

inline void registerLinalgToLinalgPasses() {
  static bool init_once = []() {
    // LinalgToLinalg
    createConvert1x1ConvToMatmulPass();
    createConvertConv2DToImg2ColPass();
    return true;
  }();
  (void)init_once;
}

inline void registerVectorToLLVMPasses() {
  // VectorToLLVM
  static bool init_once = []() {
    createVectorToAArch64InlineAssemblyPass();
    return true;
  }();
  (void)init_once;
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_INIT_CONVERSIONS_H_
