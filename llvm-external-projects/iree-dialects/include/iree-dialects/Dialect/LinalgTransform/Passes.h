// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Support/LLVM.h"
#include <memory>

namespace mlir {
namespace linalg {
namespace transform {

void registerTransformDialectInterpreterPass();
void registerLinalgTransformExpertExpansionPass();
void registerDropSchedulePass();

} // namespace transform
} // namespace linalg
} // namespace mlir

namespace mlir {
class Pass;

// Pass to schedule a dispatch region by using the transform dialect.
// The schedule is specified by the transform module that is parsed from
// `transformFileName`.
std::unique_ptr<Pass> createTransformDialectInterpreterPass(
    llvm::StringRef transformFileName = llvm::StringRef());
std::unique_ptr<Pass> createDropSchedulePass();
} // namespace mlir
