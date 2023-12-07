// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VULKAN_IR_VULKANDIALECT_H_
#define IREE_COMPILER_DIALECT_VULKAN_IR_VULKANDIALECT_H_

#include "mlir/IR/Dialect.h"

namespace mlir::iree_compiler::IREE::Vulkan {

class VulkanDialect : public Dialect {
public:
  explicit VulkanDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "vk"; }

  //===--------------------------------------------------------------------===//
  // Attribute
  //===--------------------------------------------------------------------===//

  /// Parses an attribute registered to this dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  /// Prints an attribute registered to this dialect.
  void printAttribute(Attribute, DialectAsmPrinter &printer) const override;

private:
  /// Register the attributes of this dialect.
  void registerAttributes();
};

} // namespace mlir::iree_compiler::IREE::Vulkan

#endif // IREE_COMPILER_DIALECT_VULKAN_IR_VULKANDIALECT_H_
