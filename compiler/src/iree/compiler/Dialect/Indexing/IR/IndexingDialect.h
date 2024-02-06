// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_INDEXING_IR_INDEXINGDIALECT_H_
#define IREE_COMPILER_DIALECT_INDEXING_IR_INDEXINGDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::Indexing {

class IndexingDialect : public Dialect {
public:
  explicit IndexingDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "indexing"; }

  void getCanonicalizationPatterns(RewritePatternSet &results) const override;

private:
  void registerAttributes();
  void registerTypes();
};

} // namespace mlir::iree_compiler::IREE::Indexing

#endif // IREE_COMPILER_DIALECT_INDEXING_IR_INDEXINGDIALECT_H_
