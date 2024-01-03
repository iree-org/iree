// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"
#include "iree/compiler/Dialect/VM/Target/C/TranslationFlags.h"
#include "mlir/Tools/mlir-translate/Translation.h"

namespace mlir::iree_compiler::IREE::VM {

void registerToCTranslation() {
  TranslateFromMLIRRegistration toCModule(
      "iree-vm-ir-to-c-module", "Translates a vm.module to a c module",
      [](mlir::ModuleOp moduleOp, llvm::raw_ostream &output) {
        return translateModuleToC(moduleOp, getCTargetOptionsFromFlags(),
                                  output);
      });
}

} // namespace mlir::iree_compiler::IREE::VM
