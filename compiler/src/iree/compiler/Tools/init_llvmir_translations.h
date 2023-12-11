// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Configures a context with hooks to translate custom extensions to LLVMIR.
// Note that this has nothing to do with the named translations that are
// globally registered as part of init_translations.h for the purpose of
// driving iree-compile. This is maintained separately to other dialect
// initializations because it causes a transitive dependency on LLVMIR.

#ifndef IREE_COMPILER_TOOLS_INIT_LLVMIR_TRANSLATIONS_H_
#define IREE_COMPILER_TOOLS_INIT_LLVMIR_TRANSLATIONS_H_

#include "mlir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

namespace mlir::iree_compiler {

inline void registerLLVMIRTranslations(DialectRegistry &registry) {
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerArmNeonDialectTranslation(registry);
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_TOOLS_INIT_LLVMIR_TRANSLATIONS_H_
