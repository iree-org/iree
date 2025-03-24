// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This files defines a helper to trigger the registration of dialects to
// the system.
//
// Based on MLIR's InitAllDialects but for IREE dialects.

#ifndef IREE_COMPILER_TOOLS_INIT_IREE_DIALECTS_H_
#define IREE_COMPILER_TOOLS_INIT_IREE_DIALECTS_H_

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Interfaces/Interfaces.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/TransformOps/UtilTransformOps.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/ExternalInterfaces/Interfaces.h"
#include "iree/compiler/GlobalOptimization/Interfaces/Interfaces.h"
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineDialect.h"
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderDialect.h"
#include "iree/compiler/Modules/IO/Parameters/IR/IOParametersDialect.h"
#include "iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.h"

namespace mlir::iree_compiler {

// Add all the IREE dialects to the provided registry.
inline void registerIreeDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<IREE::CPU::IREECPUDialect,
                  IREE::Codegen::IREECodegenDialect,
                  IREE::Flow::FlowDialect,
                  IREE::GPU::IREEGPUDialect,
                  IREE::HAL::HALDialect,
                  IREE::HAL::Inline::HALInlineDialect,
                  IREE::HAL::Loader::HALLoaderDialect,
                  IREE::IO::Parameters::IOParametersDialect,
                  IREE::LinalgExt::IREELinalgExtDialect,
                  IREE::Encoding::IREEEncodingDialect,
                  IREE::Stream::StreamDialect,
                  IREE::Util::UtilDialect,
                  IREE::VM::VMDialect,
                  IREE::VMVX::VMVXDialect,
                  IREE::VectorExt::IREEVectorExtDialect>();
  // clang-format on

  // External models.
  registerExternalInterfaces(registry);
  registerCodegenInterfaces(registry);
  registerGlobalOptimizationInterfaces(registry);
  registerUKernelBufferizationInterface(registry);

  // Register transform dialect extensions.
  registerTransformDialectPreprocessingExtension(registry);
  IREE::Util::registerTransformDialectExtension(registry);
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_TOOLS_INIT_IREE_DIALECTS_H_
