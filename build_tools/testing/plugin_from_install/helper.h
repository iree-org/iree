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

// Defined in a second library, so the plugin only works if that library was
// renamed alongside the first.
bool helperTouchesContext(mlir::MLIRContext* context);

#endif  // IREE_TESTING_PLUGIN_FROM_INSTALL_HELPER_H_
