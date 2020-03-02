// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_COMPILER_DIALECT_VULKAN_IR_VULKANDIALECT_H_
#define IREE_COMPILER_DIALECT_VULKAN_IR_VULKANDIALECT_H_

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Vulkan {

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
};

}  // namespace Vulkan
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VULKAN_IR_VULKANDIALECT_H_
