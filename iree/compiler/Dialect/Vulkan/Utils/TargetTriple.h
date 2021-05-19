// Copyright 2021 Google LLC
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

#ifndef IREE_COMPILER_DIALECT_VULKAN_UTILS_TARGETTRIPLE_H_
#define IREE_COMPILER_DIALECT_VULKAN_UTILS_TARGETTRIPLE_H_

#include <string>

#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Vulkan {

/// GPU triple definitions to describe GPU targets for compilers.
///
/// We use "triple" here to match common compiler language: historically one
/// would describe a CPU compiler target as a string containing exactly three
/// fields. But here the configuration is for GPU and there can exist a lot of
/// architectures/vendors/products/systems. What matters differ from CPU
/// triples. We define it in the form of:
///
///   <vendor/arch>-<product>-<os>
///
/// For example:
///   ampere-rtx3080-windows
///   rdna1-5700xt-linux
///   adreno-a650-android11
///   valhall-unknown-android11
///   cpu-swiftshader-unknown
///
/// Vendor and architecture are combined together because:
/// * Typically each GPU vendor has its own set of architectures. So given the
///   architecture we know which vendor it is from. This is different from CPU
///   land where the the same architecture can be implemented by mulitple
///   vendors.
/// * There are vendors that we don't have public information regarding its
///   architectures.
/// We need a field for product to differentiate the cases where the
/// architecture is unknown or ambiguous.
class TargetTriple {
 public:
  static TargetTriple get(const char *triple);

  TargetTriple(TargetTripleArch, TargetTripleProduct, TargetTripleOS);

  TargetTripleArch getArch() const { return arch; }
  TargetTripleProduct getProduct() const { return product; }
  TargetTripleOS getOS() const { return os; }

  /// Returns the triple string.
  std::string getTriple() const;

  TargetEnvAttr getTargetEnv(MLIRContext *context) const;

 private:
  TargetTripleArch arch;
  TargetTripleProduct product;
  TargetTripleOS os;
};

}  // namespace Vulkan
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VULKAN_UTILS_TARGETTRIPLE_H_
