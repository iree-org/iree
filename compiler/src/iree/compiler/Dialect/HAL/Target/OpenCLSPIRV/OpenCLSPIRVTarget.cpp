// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/OpenCLSPIRV/OpenCLSPIRVTarget.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/level_zero_executable_def_builder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Linking/ModuleCombiner.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/SPIRV/Serialization.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

OpenCLSPIRVTargetOptions getOpenCLSPIRVTargetOptionsFromFlags() {
  static llvm::cl::opt<std::string> clOpenCLTargetTriple(
      "iree-opencl-target-triple", llvm::cl::desc("OpenCL target triple"),
      llvm::cl::init("spir-unknown-unknown"));

  static llvm::cl::opt<bool> clOpenCLUsePhysical32(
      "iree-opencl-physical32-addressing",
      llvm::cl::desc("Use Physical32 addressing with OpenCL"),
      llvm::cl::init(false));

  OpenCLSPIRVTargetOptions targetOptions;
  targetOptions.openCLTargetTriple = clOpenCLTargetTriple;
  targetOptions.openCLUsePhysical32 = clOpenCLUsePhysical32;

  return targetOptions;
}

// Returns the Vulkan target environment for conversion.
static spirv::TargetEnvAttr getSPIRVTargetEnv(
    const std::string &openCLTargetTriple, MLIRContext *context) {
  // if (!openCLTargetTriple.empty()) {
  //   return convertTargetEnv(
  //       Vulkan::getTargetEnvForTriple(context, openCLTargetTriple));
  // }
  auto triple = spirv::VerCapExtAttr::get(
      spirv::Version::V_1_4,
      {spirv::Capability::Kernel,
       spirv::Capability::Addresses,
       spirv::Capability::SubgroupDispatch,
       spirv::Capability::Float16Buffer,
       spirv::Capability::Int16,
       spirv::Capability::Int8,
       spirv::Capability::Vector16,
       spirv::Capability::GenericPointer,
       spirv::Capability::Groups,
       spirv::Capability::ImageBasic,
       spirv::Capability::Float16,
       spirv::Capability::Linkage,
       spirv::Capability::Int64Atomics,
       spirv::Capability::Int64,
       spirv::Capability::Float64,
       spirv::Capability::GroupNonUniform,
       spirv::Capability::GroupNonUniformVote,
       spirv::Capability::GroupNonUniformBallot,
       spirv::Capability::GroupNonUniformArithmetic,
       spirv::Capability::GroupNonUniformShuffle,
       spirv::Capability::GroupNonUniformShuffleRelative,
       spirv::Capability::GroupNonUniformClustered,
       spirv::Capability::AtomicFloat16AddEXT,
       spirv::Capability::AtomicFloat32AddEXT,
       spirv::Capability::AtomicFloat64AddEXT,
       spirv::Capability::LiteralSampler,
       spirv::Capability::Sampled1D,
       spirv::Capability::Image1D,
       spirv::Capability::SampledBuffer,
       spirv::Capability::ImageBuffer,
       spirv::Capability::ImageReadWrite},
      {spirv::Extension::SPV_INTEL_subgroups,
       spirv::Extension::SPV_EXT_shader_atomic_float_add,
       spirv::Extension::SPV_EXT_shader_atomic_float16_add,
       spirv::Extension::SPV_EXT_shader_atomic_float_min_max,
       spirv::Extension::SPV_KHR_linkonce_odr},
      context);
  return spirv::TargetEnvAttr::get(triple, spirv::Vendor::Unknown,
                                   spirv::DeviceType::Unknown,
                                   spirv::TargetEnvAttr::kUnknownDeviceID,
                                   spirv::getDefaultResourceLimits(context));
  return {};
}

class OpenCLSPIRVTargetBackend : public TargetBackend {
 public:
  OpenCLSPIRVTargetBackend(OpenCLSPIRVTargetOptions options)
      : options_(std::move(options)) {}

  std::string name() const override { return "opencl"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect, spirv::SPIRVDialect,
                    gpu::GPUDialect>();
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    // Indicates that the runtime HAL driver operates only in the legacy
    // synchronous mode.
    configItems.emplace_back(b.getStringAttr("legacy_sync"), b.getUnitAttr());

    configItems.emplace_back(b.getStringAttr("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    bool use64bitIndex = true;
    if (options_.openCLUsePhysical32) use64bitIndex = false;

    buildSPIRVCodegenPassPipeline(passManager, /*enableFastMath=*/false,
                                  use64bitIndex);
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp innerModuleOp = variantOp.getInnerModule();
    auto spirvModuleOps = innerModuleOp.getOps<spirv::ModuleOp>();
    if (!llvm::hasSingleElement(spirvModuleOps)) {
      return variantOp.emitError()
             << "should only contain exactly one spv.module op";
    }
    auto spvModuleOp = *spirvModuleOps.begin();

    FlatbufferBuilder builder;
    iree_LEVEL_ZEROExecutableDef_start_as_root(builder);

    // Serialize the spirv::ModuleOp into the binary that we will embed in the
    // final FlatBuffer.
    SmallVector<uint32_t, 256> spvBinary;
    if (failed(spirv::serialize(spvModuleOp, spvBinary)) || spvBinary.empty()) {
      return variantOp.emitError() << "failed to serialize spv.module";
    }

    // if (!options.dumpBinariesPath.empty()) {
    //   dumpDataToPath<uint32_t>(options.dumpBinariesPath,
    //   options.dumpBaseName,
    //                            variantOp.getName(), ".spv", spvBinary);
    // }

    auto spvCodeRef = flatbuffers_uint32_vec_create(builder, spvBinary.data(),
                                                    spvBinary.size());

    // The sequencer and runtime use ordinals instead of names. We provide the
    // list of entry point names here that are then passed in
    // zeModuleCreate.
    SmallVector<StringRef, 8> entryPointNames;
    std::vector<SmallVector<int32_t, 3>> workgroupSizes;
    spvModuleOp.walk([&](spirv::ExecutionModeOp executionModelOp) {
      entryPointNames.push_back(executionModelOp.getFn());
      ArrayAttr workGroupSizeAttr = executionModelOp.getValues();
      assert(workGroupSizeAttr.size() == 3 &&
             "workgroup size is expected to be 3");
      workgroupSizes.push_back(
          {int(workGroupSizeAttr[0].dyn_cast<IntegerAttr>().getInt()),
           int(workGroupSizeAttr[1].dyn_cast<IntegerAttr>().getInt()),
           int(workGroupSizeAttr[2].dyn_cast<IntegerAttr>().getInt())});
    });
    // if (!options.dumpBinariesPath.empty()) {
    // dumpDataToPath<uint32_t>("/tmp", entryPointNames[0],
    //                          variantOp.getName(), ".spv", spvBinary);
    // }

    auto entryPointsRef = builder.createStringVec(entryPointNames);
    iree_LEVEL_ZEROBlockSizeDef_vec_start(builder);
    auto blockSizes = workgroupSizes.begin();
    for (int i = 0, e = entryPointNames.size(); i < e; ++i) {
      iree_LEVEL_ZEROBlockSizeDef_vec_push_create(
          builder, (*blockSizes)[0], (*blockSizes)[1], (*blockSizes)[2]);
      ++blockSizes;
    }
    auto blockSizesRef = iree_LEVEL_ZEROBlockSizeDef_vec_end(builder);

    iree_LEVEL_ZEROExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_LEVEL_ZEROExecutableDef_block_sizes_add(builder, blockSizesRef);
    iree_LEVEL_ZEROExecutableDef_level_zero_image_add(builder, spvCodeRef);
    iree_LEVEL_ZEROExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

 private:
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    targetAttrs.push_back(getExecutableTarget(
        context, getSPIRVTargetEnv(options_.openCLTargetTriple, context)));
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr getExecutableTarget(
      MLIRContext *context, spirv::TargetEnvAttr targetEnv) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getStringAttr(spirv::getTargetEnvAttrName()),
                             targetEnv);

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("opencl"), b.getStringAttr("opencl-spirv-fb"),
        configAttr);
  }

  OpenCLSPIRVTargetOptions options_;
};

void registerOpenCLSPIRVTargetBackends(
    std::function<OpenCLSPIRVTargetOptions()> queryOptions) {
  getOpenCLSPIRVTargetOptionsFromFlags();
  auto backendFactory = [=]() {
    return std::make_shared<OpenCLSPIRVTargetBackend>(queryOptions());
  };
  // #hal.device.target<"opencl", ...
  static TargetBackendRegistration registration0("opencl", backendFactory);
  // #hal.executable.target<"opencl-spirv", ...
  static TargetBackendRegistration registration1("opencl-spirv",
                                                 backendFactory);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
