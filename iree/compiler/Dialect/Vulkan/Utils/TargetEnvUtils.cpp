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

#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvUtils.h"

#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Vulkan {

namespace {
/// Gets the corresponding SPIR-V version for the ggiven Vulkan target
/// environment.
spirv::Version convertVersion(Vulkan::TargetEnvAttr vkTargetEnv) {
  // Vulkan 1.2 supports up to SPIR-V 1.5 by default.
  if (vkTargetEnv.getVersion() == Version::V_1_2) return spirv::Version::V_1_5;

  // Special extension to enable SPIR-V 1.4.
  if (llvm::is_contained(vkTargetEnv.getExtensions(),
                         Extension::VK_KHR_spirv_1_4))
    return spirv::Version::V_1_4;

  switch (vkTargetEnv.getVersion()) {
    case Version::V_1_0:
      // Vulkan 1.0 only supports SPIR-V 1.0 by default.
      return spirv::Version::V_1_0;
    case Version::V_1_1:
      // Vulkan 1.1 supports up to SPIR-V 1.3 by default.
      return spirv::Version::V_1_3;
    default:
      break;
  }
  llvm_unreachable("unhandled Vulkan version!");
}

/// Gets the corresponding SPIR-V extensions for the given Vulkan target
/// environment.
void convertExtensions(Vulkan::TargetEnvAttr vkTargetEnv,
                       SmallVectorImpl<spirv::Extension> &extensions) {
  extensions.clear();

  for (Extension ext : vkTargetEnv.getExtensions()) {
    switch (ext) {
      case Extension::VK_KHR_spirv_1_4:
        // This extension only affects SPIR-V version.
        break;
      case Extension::VK_KHR_storage_buffer_storage_class:
        extensions.push_back(
            spirv::Extension::SPV_KHR_storage_buffer_storage_class);
        break;
    }
  }
}

/// Gets the corresponding SPIR-V resource limits for the given Vulkan target
/// environment.
spirv::ResourceLimitsAttr convertResourceLimits(
    Vulkan::TargetEnvAttr vkTargetEnv) {
  MLIRContext *context = vkTargetEnv.getContext();
  auto core10 = vkTargetEnv.getCore10Properties();
  return spirv::ResourceLimitsAttr::get(core10.maxComputeWorkGroupInvocations(),
                                        core10.maxComputeWorkGroupSize(),
                                        context);
}

// TODO(antiagainst): move this into MLIR core as spirv::TargetEnvAttr::get().
spirv::TargetEnvAttr getSpirvTargetEnv(spirv::Version version,
                                       ArrayRef<spirv::Extension> extensions,
                                       ArrayRef<spirv::Capability> capabilities,
                                       DictionaryAttr limits) {
  Builder b(limits.getContext());

  auto versionAttr = b.getI32IntegerAttr(static_cast<uint32_t>(version));

  SmallVector<Attribute, 4> extAttrs;
  extAttrs.reserve(extensions.size());
  for (spirv::Extension ext : extensions)
    extAttrs.push_back(b.getStringAttr(spirv::stringifyExtension(ext)));

  SmallVector<Attribute, 4> capAttrs;
  capAttrs.reserve(capabilities.size());
  for (spirv::Capability cap : capabilities)
    capAttrs.push_back(b.getI32IntegerAttr(static_cast<uint32_t>(cap)));

  return spirv::TargetEnvAttr::get(versionAttr, b.getArrayAttr(extAttrs),
                                   b.getArrayAttr(capAttrs), limits);
}
}  // anonymous namespace

// TODO(antiagainst): change this to the real SwiftShader target environment.
const char *swiftShaderTargetEnvAssembly =
    "#vk.target_env<v1.1, r(130), [], {"
    "maxComputeWorkGroupInvocations = 128: i32, "
    "maxComputeWorkGroupSize = dense<[64, 4, 4]>: vector<3xi32>"
    "}>";

spirv::TargetEnvAttr convertTargetEnv(Vulkan::TargetEnvAttr vkTargetEnv) {
  auto spvVersion = convertVersion(vkTargetEnv);

  SmallVector<spirv::Extension, 4> spvExtensions;
  convertExtensions(vkTargetEnv, spvExtensions);

  SmallVector<spirv::Capability, 8> spvCapabilities;

  auto spvLimits = convertResourceLimits(vkTargetEnv);

  return getSpirvTargetEnv(spvVersion, spvExtensions, spvCapabilities,
                           spvLimits);
}

}  // namespace Vulkan
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
