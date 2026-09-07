// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_PLUGIN_FROM_INSTALL_HELPER_H_
#define IREE_TESTING_PLUGIN_FROM_INSTALL_HELPER_H_

namespace mlir {
class MLIRContext;
}  // namespace mlir

// In a second library, so the plugin loads only if that was renamed too.
bool helperTouchesContext(mlir::MLIRContext* context);

#endif  // IREE_TESTING_PLUGIN_FROM_INSTALL_HELPER_H_
