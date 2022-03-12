//===-- LinalgExtBufferization.h - Linalg Extension bufferization ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_BUFFERIZATION_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_BUFFERIZATION_H_

namespace mlir {

class DialectRegistry;

namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

}  // namespace LinalgExt
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_DIALECTS_DIALECT_LINALGEXT_BUFFERIZATION_H_
