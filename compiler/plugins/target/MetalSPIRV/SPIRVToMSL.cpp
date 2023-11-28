// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./SPIRVToMSL.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

// Disable exception handling in favor of assertions.
#define SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
#include "third_party/spirv_cross/spirv_msl.hpp"

#define DEBUG_TYPE "spirv-to-msl"

/// The [[buffer(N)]] index for push constants.
/// Note that this MUST be kept consistent with the Metal HAL driver.
#define IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX 3

namespace mlir {
namespace iree_compiler {

namespace {
class SPIRVToMSLCompiler : public SPIRV_CROSS_NAMESPACE::CompilerMSL {
public:
  using CompilerMSL::CompilerMSL;

  MetalShader::ThreadGroupSize
  getWorkgroupSizeForEntryPoint(StringRef entryName) {
    const auto &entryPoint = get_entry_point(
        entryName.str(), spv::ExecutionModel::ExecutionModelGLCompute);
    const auto &workgroupSize = entryPoint.workgroup_size;
    // TODO(antiagainst): support specialization constant.
    if (workgroupSize.constant != 0)
      return {0, 0, 0};
    return {workgroupSize.x, workgroupSize.y, workgroupSize.z};
  }

  // A struct containing a resource descriptor's information.
  struct Descriptor {
    uint32_t set;
    uint32_t binding;

    Descriptor(uint32_t s, uint32_t b) : set(s), binding(b) {}

    friend bool operator<(const Descriptor &l, const Descriptor &r) {
      return std::tie(l.set, l.binding) < std::tie(r.set, r.binding);
    }
  };

  // Updates `descriptors` with resource set and binding number pairs in
  // increasing order, and `hasPushConstant` if with push constants.
  // Returns true if no unsupported cases are encountered.
  bool getResources(SmallVectorImpl<Descriptor> *descriptors,
                    bool *hasPushConstant) {
    descriptors->clear();
    *hasPushConstant = false;

    // Iterate over all variables in the SPIR-V blob.
    bool hasUnknownCase = false;
    ir.for_each_typed_id<SPIRV_CROSS_NAMESPACE::SPIRVariable>(
        [&](uint32_t id, SPIRV_CROSS_NAMESPACE::SPIRVariable &var) {
          auto storage = var.storage;
          switch (storage) {
            // Non-interface variables. We don't care.
          case spv::StorageClassFunction:
          case spv::StorageClassPrivate:
          case spv::StorageClassWorkgroup:
            // Builtin variables. We don't care either.
          case spv::StorageClassInput:
            return;
          case spv::StorageClassPushConstant:
            *hasPushConstant = true;
            return;
          case spv::StorageClassUniform:
          case spv::StorageClassStorageBuffer: {
            uint32_t setNo = get_decoration(id, spv::DecorationDescriptorSet);
            uint32_t bindingNo = get_decoration(id, spv::DecorationBinding);
            descriptors->emplace_back(setNo, bindingNo);
            return;
          }
          default:
            break;
          }
          hasUnknownCase = true;
        });

    llvm::sort(*descriptors);
    return !hasUnknownCase;
  }

  Options getCompilationOptions(IREE::HAL::MetalTargetPlatform platform) {
    // TODO(antiagainst): fill out the following according to the Metal GPU
    // family.
    SPIRVToMSLCompiler::Options spvCrossOptions;
    switch (platform) {
    case IREE::HAL::MetalTargetPlatform::macOS:
      spvCrossOptions.platform = SPIRVToMSLCompiler::Options::Platform::macOS;
      break;
    case IREE::HAL::MetalTargetPlatform::iOS:
    case IREE::HAL::MetalTargetPlatform::iOSSimulator:
      spvCrossOptions.platform = SPIRVToMSLCompiler::Options::Platform::iOS;
      break;
    }
    spvCrossOptions.msl_version =
        SPIRVToMSLCompiler::Options::make_msl_version(3, 0);
    // Eanble using Metal argument buffers. It is more akin to Vulkan descriptor
    // sets, which is how IREE HAL models resource bindings and mappings.
    spvCrossOptions.argument_buffers = true;
    return spvCrossOptions;
  }
};
} // namespace

std::optional<std::pair<MetalShader, std::string>>
crossCompileSPIRVToMSL(IREE::HAL::MetalTargetPlatform targetPlatform,
                       llvm::ArrayRef<uint32_t> spvBinary,
                       StringRef entryPoint) {
  SPIRVToMSLCompiler spvCrossCompiler(spvBinary.data(), spvBinary.size());

  // All spirv-cross operations work on the current entry point. It should be
  // set right after the cross compiler construction.
  spvCrossCompiler.set_entry_point(
      entryPoint.str(), spv::ExecutionModel::ExecutionModelGLCompute);

  SmallVector<SPIRVToMSLCompiler::Descriptor> descriptors;
  bool hasPushConstant = false;
  if (!spvCrossCompiler.getResources(&descriptors, &hasPushConstant))
    return std::nullopt;

  // Explicitly set the argument buffer [[id(N)]] location for each SPIR-V
  // resource variable.
  for (const auto &descriptor : descriptors) {
    SPIRV_CROSS_NAMESPACE::MSLResourceBinding binding = {};
    binding.stage = spv::ExecutionModelGLCompute;
    binding.desc_set = descriptor.set;
    binding.binding = descriptor.binding;
    // We only interact with buffers in IREE.
    binding.msl_buffer = descriptor.binding;

    spvCrossCompiler.add_msl_resource_binding(binding);
  }
  // If push constants are used, explicitly set its [[buffer(N)]] location too.
  if (hasPushConstant) {
    SPIRV_CROSS_NAMESPACE::MSLResourceBinding binding = {};
    binding.stage = spv::ExecutionModelGLCompute;
    binding.desc_set =
        SPIRV_CROSS_NAMESPACE::ResourceBindingPushConstantDescriptorSet;
    binding.binding = SPIRV_CROSS_NAMESPACE::ResourceBindingPushConstantBinding;
    binding.msl_buffer = IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX;

    spvCrossCompiler.add_msl_resource_binding(binding);
  }

  auto spvCrossOptions = spvCrossCompiler.getCompilationOptions(targetPlatform);
  spvCrossCompiler.set_msl_options(spvCrossOptions);

  std::string mslSource = spvCrossCompiler.compile();
  // Get the revised entry point name. Cross compiling to MSL generates source
  // code, where we may run into the case that we are using reserved keyword for
  // the entry point name, e.g., `abs`. Under such circumstances, it will be
  // revised to avoid collision.
  const auto &spirvEntryPoint = spvCrossCompiler.get_entry_point(
      entryPoint.str(), spv::ExecutionModel::ExecutionModelGLCompute);
  LLVM_DEBUG({
    llvm::dbgs() << "Original entry point name: '" << spirvEntryPoint.orig_name
                 << "'\n";
    llvm::dbgs() << "Revised entry point name: '" << spirvEntryPoint.name
                 << "'\n";
    llvm::dbgs() << "Generated MSL:\n-----\n" << mslSource << "\n-----\n";
  });

  auto workgroupSize =
      spvCrossCompiler.getWorkgroupSizeForEntryPoint(entryPoint);
  if (!workgroupSize.x || !workgroupSize.y || !workgroupSize.z) {
    return std::nullopt;
  }
  return std::make_pair(MetalShader{std::move(mslSource), workgroupSize},
                        spirvEntryPoint.name);
}

} // namespace iree_compiler
} // namespace mlir
