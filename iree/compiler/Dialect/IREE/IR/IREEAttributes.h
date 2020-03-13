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

#ifndef IREE_COMPILER_DIALECT_IREE_IR_IREEATTRIBUTES_H_
#define IREE_COMPILER_DIALECT_IREE_IR_IREEATTRIBUTES_H_

#include "mlir/IR/Attributes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

namespace AttrKind {
enum Kind {
  FIRST_HAL_ATTR = Attribute::FIRST_IREE_ATTR + 10,
  FIRST_VULKAN_ATTR = Attribute::FIRST_IREE_ATTR + 20,
};
}  // namespace AttrKind

namespace HAL {
namespace AttrKind {
enum Kind {
  DescriptorSetLayoutBindingAttr = IREE::AttrKind::FIRST_HAL_ATTR,
};
}  // namespace AttrKind
}  // namespace HAL

namespace Vulkan {
namespace AttrKind {
enum Kind {
  TargetEnv = IREE::AttrKind::FIRST_VULKAN_ATTR,
};
}  // namespace AttrKind
}  // namespace Vulkan

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_IR_IREEATTRIBUTES_H_
