// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"

#include <utility>

#include "llvm/Support/raw_ostream.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::EmptyPluginOptions);

namespace mlir::iree_compiler {

PipelineExtensions::~PipelineExtensions() = default;

AbstractPluginRegistration::~AbstractPluginRegistration() = default;
AbstractPluginSession::~AbstractPluginSession() = default;

LogicalResult AbstractPluginSession::activate(MLIRContext *context) {
  if (this->context) {
    // Already activated - ignore. But verify in debug mode that activated
    // with the same context (which is a non-user triggerable error).
    assert(context == this->context &&
           "duplicate plugin activation with different context");
    return success();
  }
  this->context = context;
  return onActivate();
}

void PluginRegistrar::registerPlugin(
    std::unique_ptr<AbstractPluginRegistration> registration) {
  // Need to copy the id since in the error case, the registration will be
  // deleted before reporting the error message.
  std::string id = std::string(registration->getPluginId());
  auto foundIt = registrations.insert(
      std::make_pair(llvm::StringRef(id), std::move(registration)));
  if (!foundIt.second) {
    llvm::errs() << "ERROR: Duplicate plugin registration for '" << id << "'\n";
    abort();
  }
}

}  // namespace mlir::iree_compiler
