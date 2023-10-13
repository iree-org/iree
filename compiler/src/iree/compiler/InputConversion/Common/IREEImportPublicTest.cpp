// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/testing/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::iree_compiler {

TEST(IREEImportPublicTest, CheckElementTypeValue) {
  MLIRContext ctx;
  OpBuilder b(&ctx);

  auto assert_eq = [&](Type type) {
    ASSERT_EQ(IREE::HAL::getElementTypeValue(type),
              IREE::Input::getElementTypeValue(type));
  };

  assert_eq(b.getIntegerType(1));
  assert_eq(b.getIntegerType(8));
  assert_eq(b.getIntegerType(32));
  assert_eq(b.getIntegerType(64));
  assert_eq(b.getBF16Type());
  assert_eq(b.getF16Type());
  assert_eq(b.getF32Type());
  assert_eq(b.getF64Type());
  assert_eq(ComplexType::get(b.getF32Type()));
  assert_eq(ComplexType::get(b.getF64Type()));
}

} // namespace mlir::iree_compiler
