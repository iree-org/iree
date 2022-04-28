// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/Interfaces.h"

#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Interfaces/ProcessorOpInterfaces.h"
// TODO: Remove this dependency once the transform dialect extensions
// have a better registration mechanism.
#include "iree/compiler/Codegen/TransformDialectExtensions/TransformDialectExtensions.h"

namespace mlir {
namespace iree_compiler {

void registerCodegenInterfaces(DialectRegistry &registry) {
  registerProcessorOpInterfaceExternalModels(registry);
  registerBufferizationInterfaces(registry);
  // TODO: Remove this dependency once the transform dialect extensions
  // have a better registration mechanism.
  // TODO: when warranted, move to its own file.
  registerLinalgTransformDialectExtension(registry);
}

}  // namespace iree_compiler
}  // namespace mlir
