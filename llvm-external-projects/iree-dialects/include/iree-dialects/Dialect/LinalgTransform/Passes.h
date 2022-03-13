//===-- Passes.h - LinalgTransform passes -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory>

namespace mlir {
namespace linalg {
namespace transform {

void registerLinalgTransformInterpreterPass();
void registerLinalgTransformExpertExpansionPass();
void registerDropScheduleFromModulePass();

}  // namespace transform
}  // namespace linalg
}  // namespace mlir

namespace mlir {
class Pass;
std::unique_ptr<Pass> createLinalgTransformInterpreterPass();
std::unique_ptr<Pass> createDropScheduleFromModulePass();
}  // namespace mlir
