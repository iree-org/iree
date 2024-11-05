// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/Interfaces.h"

#include "iree/compiler/ExternalInterfaces/FlowExternalModels.h"
#include "iree/compiler/ExternalInterfaces/LinalgExtExternalModels.h"
#include "iree/compiler/ExternalInterfaces/StreamExternalModels.h"
#include "iree/compiler/ExternalInterfaces/UtilExternalModels.h"

namespace mlir::iree_compiler {

void registerExternalInterfaces(DialectRegistry &registry) {
  registerFlowExternalModels(registry);
  registerLinalgExtExternalModels(registry);
  registerStreamExternalModels(registry);
  registerUtilExternalModels(registry);
}

} // namespace mlir::iree_compiler
