// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file defines a helper to trigger the registration of all translations
// in and out of MLIR to the system.
//
// Based on MLIR's InitAllTranslations but without translations we don't care
// about.
//
// Note that this performs registration of named translations for the use of
// iree-translate. This is different from "LLVM IR Translations", which are
// registered on a context and provide hooks for populating LLVM IR for
// certain dialects. See init_llvmir_translations.h.

#ifndef IREE_TOOLS_INIT_TRANSLATIONS_H_
#define IREE_TOOLS_INIT_TRANSLATIONS_H_

#include "iree/compiler/Translation/HALExecutable.h"
#include "iree/compiler/Translation/IREEVM.h"

namespace mlir {

void registerToSPIRVTranslation();

// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerMlirTranslations() {
  static bool init_once = []() {
    registerToSPIRVTranslation();
    return true;
  }();
  (void)init_once;
}

namespace iree_compiler {

// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerIreeTranslations() {
  static bool init_once = []() {
    registerHALExecutableTranslation();
    registerIREEVMTranslation();
    return true;
  }();
  (void)init_once;
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_TOOLS_INIT_TRANSLATIONS_H_
