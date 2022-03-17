// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TRANSLATION_IREEVM_H_
#define IREE_COMPILER_TRANSLATION_IREEVM_H_

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Pipelines/Options.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

// Builds the translation pipeline with defaults.
void buildDefaultIREEVMTransformPassPipeline(OpPassManager &passManager);

// Registration hooks.
void registerIREEVMTransformPassPipeline();
void registerIREEVMTranslation();
void registerIREEVMTranslationFlags();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_IREEVM_H_
