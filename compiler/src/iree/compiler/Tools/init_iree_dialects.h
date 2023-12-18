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

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Interfaces/Interfaces.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilExternalModels.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"
#include "iree/compiler/GlobalOptimization/Interfaces/Interfaces.h"
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineDialect.h"
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderDialect.h"
#include "iree/compiler/Modules/IO/Parameters/IR/IOParametersDialect.h"
#include "iree/compiler/Modules/IO/Stream/IR/IOStreamDialect.h"
#include "mlir/IR/Dialect.h"

namespace mlir::iree_compiler {

// Add all the IREE dialects to the provided registry.
inline void registerIreeDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<IREE::Codegen::IREECodegenDialect,
                  IREE::Flow::FlowDialect,
                  IREE::HAL::HALDialect,
                  IREE::HAL::Inline::HALInlineDialect,
                  IREE::HAL::Loader::HALLoaderDialect,
                  IREE::IO::Parameters::IOParametersDialect,
                  IREE::IO::Stream::IOStreamDialect,
                  IREE::Input::IREEInputDialect,
                  IREE::LinalgExt::IREELinalgExtDialect,
                  IREE::Stream::StreamDialect,
                  IREE::Util::UtilDialect,
                  IREE::VM::VMDialect,
                  IREE::VMVX::VMVXDialect,
                  IREE::VectorExt::IREEVectorExtDialect,
                  IREE::Vulkan::VulkanDialect>();
  // clang-format on

  // External models.
  IREE::Util::registerUtilExternalModels(registry);
  registerCodegenInterfaces(registry);
  registerGlobalOptimizationInterfaces(registry);
  registerUKernelBufferizationInterface(registry);
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_TOOLS_INIT_IREE_DIALECTS_H_
