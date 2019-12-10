// Copyright 2019 Google LLC
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

#include "iree/compiler/Translation/SPIRV/SPIRVExecutableTranslation.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "iree/compiler/Translation/SPIRV/EmbeddedKernels.h"
#include "iree/compiler/Translation/SPIRV/IREEToSPIRVPass.h"
#include "iree/compiler/Utils/OpUtils.h"
#include "iree/compiler/Utils/TranslationUtils.h"
#include "iree/schemas/executable_def_generated.h"
#include "iree/schemas/spirv_executable_def_generated.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Translation.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace iree_compiler {

namespace {

class SPIRVTranslator {
 public:
  explicit SPIRVTranslator(ExecutableTranslationOptions options)
      : options_(options) {}

  const ExecutableTranslationOptions &options() const { return options_; }

  // Returns a populated ExecutableDef or nullptr if translation is
  // unsuccessful.
  std::unique_ptr<iree::ExecutableDefT> translateExecutable(
      IREE::ExecutableOp executableOp);

 private:
  // Returns a list of entry point names matching the expected export ordinals.
  std::vector<std::string> populateEntryPointNames(
      IREE::ExecutableOp executableOp);

  // Translates the input module into the SPIR-V dialect and returns the
  // serialized code words or empty if translation failed.
  std::vector<uint32_t> translateAndSerializeShaderModule(
      IREE::ExecutableOp executableOp);

  // Returns a pipeline layout definition based on the bindings required.
  std::unique_ptr<iree::VkPipelineLayoutDefT> populatePipelineLayout(
      spirv::ModuleOp spirvModuleOp);

  ExecutableTranslationOptions options_;
};

std::unique_ptr<iree::ExecutableDefT> SPIRVTranslator::translateExecutable(
    IREE::ExecutableOp executableOp) {
  // Try first to match against an embedded kernel (such as matmul) and
  // otherwise fall back to generating the kernel.
  iree::SpirVExecutableDefT spirvExecutableDef;
  if (!tryEmbeddedKernelRewrite(executableOp.getInnerModule(),
                                &spirvExecutableDef)) {
    // The sequencer and runtime use ordinals instead of names. We provide the
    // list of entry point names here that are then passed in
    // VkShaderModuleCreateInfo.
    spirvExecutableDef.entry_points = populateEntryPointNames(executableOp);

    // Translate the module and generate the SPIR-V code.
    // The module is expected to be modified and must contain the metadata
    // required to enable the following information needed for the
    // SpirVExecutableDef to be extracted.
    spirvExecutableDef.code = translateAndSerializeShaderModule(executableOp);
    if (spirvExecutableDef.code.empty()) {
      executableOp.emitError()
          << "Failed to translate and serialize SPIR-V executable";
      return {};
    }

    // Reflect against the entry thunk to identify the required pipeline
    // layout based on binding information. This is used by the runtime to
    // create the VkPipelineLayout.
    for (auto spirvModuleOp :
         executableOp.getBlock().getOps<spirv::ModuleOp>()) {
      spirvExecutableDef.pipeline_layout =
          populatePipelineLayout(spirvModuleOp);
      if (!spirvExecutableDef.pipeline_layout) {
        spirvModuleOp.emitError()
            << "Failed to generate pipeline for SPIR-V module";
        return {};
      }
      break;
    }
  }

  // Pack the executable definition and get the bytes with the proper header.
  // The header is used to verify the contents at runtime.
  ::flatbuffers::FlatBufferBuilder fbb;
  auto executableOffset =
      iree::SpirVExecutableDef::Pack(fbb, &spirvExecutableDef);
  iree::FinishSpirVExecutableDefBuffer(fbb, executableOffset);
  std::vector<uint8_t> bytes;
  bytes.resize(fbb.GetSize());
  std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

  OpBuilder builder(executableOp);
  executableOp.setAttr("format", builder.getI32IntegerAttr(static_cast<int32_t>(
                                     IREE::ExecutableFormat::SpirV)));

  auto executableDef = std::make_unique<iree::ExecutableDefT>();
  executableDef->format = static_cast<uint32_t>(IREE::ExecutableFormat::SpirV);
  executableDef->contents = std::move(bytes);
  return executableDef;
}

std::vector<std::string> SPIRVTranslator::populateEntryPointNames(
    IREE::ExecutableOp executableOp) {
  auto module = executableOp.getInnerModule();
  DenseMap<unsigned, StringRef> entryPoints;
  for (auto funcOp : module.getOps<FuncOp>()) {
    if (!funcOp.getAttr("iree.executable.export")) continue;
    auto ordinalAttr = funcOp.getAttrOfType<IntegerAttr>("iree.ordinal");
    entryPoints[ordinalAttr.getInt()] = funcOp.getName();
  }
  std::vector<std::string> entryPointNames(entryPoints.size());
  for (auto &entry : entryPoints) {
    entryPointNames[entry.first] = entry.second.str();
  }
  return entryPointNames;
}

std::vector<uint32_t> SPIRVTranslator::translateAndSerializeShaderModule(
    IREE::ExecutableOp executableOp) {
  auto module = executableOp.getInnerModule();

  // We can use the workload hint to know what the expected dispatch workload
  // is. If we want to remap this to make more sense for the operations we are
  // performing we can do that here.
  //
  // Note that workloads are computed per entry point. There may be some
  // dimensions of the workload that are static (in which case workloadAttr will
  // have non-dynamic dims) and others that need to be taken from an argument
  // shape (in which case workloadRef is the argument ordinal to take dynamic
  // dimensions from).
  // TODO(benvanik): make it just an arg instead? iree.workload special op?
  // TODO(benvanik): instead of FuncOp have an iree.entry_point op with these.
  for (auto funcOp : module.getOps<FuncOp>()) {
    // TODO(ravishankarm): FuncOps in executable that are not dispatch functions
    // are not lowered to SPIR-V. Fix this limitation.
    if (!funcOp.getAttr("iree.executable.export")) continue;
    auto workloadAttr =
        funcOp.getAttrOfType<ElementsAttr>("iree.executable.workload");
    auto workloadRefAttr =
        funcOp.getAttrOfType<IntegerAttr>("iree.executable.workload_ref");
    std::array<int32_t, 3> staticWorkloadDims = {-1, -1, -1};
    if (workloadAttr) {
      for (unsigned i = 0; i < 3; ++i) {
        if (auto dimAttr =
                workloadAttr.getValue({i}).dyn_cast_or_null<IntegerAttr>()) {
          staticWorkloadDims[i] = dimAttr.getInt();
        }
      }
    }
    std::array<BlockArgument *, 3> dynamicWorkloadDimRefs;
    if (workloadRefAttr) {
      for (unsigned i = 0; i < 3; ++i) {
        if (staticWorkloadDims[i] == -1) {
          dynamicWorkloadDimRefs[i] =
              funcOp.getArgument(workloadRefAttr.getInt());
        }
      }
    }

    // Now staticWorkloadDims will have non-negative values for known dimensions
    // and any dim with -1 will need to be pulled from the corresponding shape
    // dimension of dynamicWorkloadDimRefs.

    // TODO(b/137868263): use this information to map from workgroup to
    // invocation and perform indexing.
  }

  // Lower module to spirv::ModuleOp.
  auto spirvGenPasses = createPassManager(module.getContext(), options());
  spirvGenPasses->addPass(xla_hlo::createLegalizeToStdPass());
  spirvGenPasses->addPass(createIndexComputationPass());
  spirvGenPasses->addPass(createIREEToSPIRVPass());
  spirvGenPasses->addPass(createAdjustIntegerWidthPass());
  if (failed(runPassPipeline(options(), spirvGenPasses.get(), module))) {
    executableOp.emitError() << "Failed to generate spv.module";
    return {};
  }

  auto spvModules = module.getOps<spirv::ModuleOp>();
  if (std::distance(spvModules.begin(), spvModules.end()) != 1) {
    executableOp.emitError()
        << "Expected a single spv.module for an IREE executable op";
    return {};
  }

  // Serialize the spirv::ModuleOp into the binary that we will embed in the
  // final flatbuffer.
  std::vector<uint32_t> spvBinaries;
  for (auto spvModule : spvModules) {
    SmallVector<uint32_t, 256> spvBinary;
    if (failed(spirv::serialize(spvModule, spvBinary))) {
      executableOp.emitError() << "Failed to serialize spv.module";
      return {};
    }
    spvBinaries.insert(spvBinaries.end(), spvBinary.begin(), spvBinary.end());

    // Clone the module into executableOp directly.
    auto clonedModule = spvModule.clone();
    executableOp.getBlock().getOperations().insert(
        std::prev(executableOp.getBlock().getOperations().end()), clonedModule);
  }
  // Remove the original code.
  module.erase();

  return spvBinaries;
}

std::unique_ptr<iree::VkPipelineLayoutDefT>
SPIRVTranslator::populatePipelineLayout(spirv::ModuleOp spirvModuleOp) {
  // NOTE: we currently make some assumptions about this based on the expected
  // ABI of the runtime. If we wanted to support more general shaders with more
  // complex I/O we'd need to find a better way to communicate this through the
  // VkPipelineLayoutDef.
  auto pipelineLayoutDef = std::make_unique<iree::VkPipelineLayoutDefT>();
  pipelineLayoutDef->buffer_binding_set = 0;

  // Build a set of descriptor_set -> binding -> variable.
  // This makes it easier to write out the descriptor in a logical order, even
  // though this is not strictly required.
  int64_t maxDescriptorSetOrdinal = -1;
  std::map<int32_t, std::map<int32_t, spirv::GlobalVariableOp>> descriptorSets;
  for (auto globalVar :
       spirvModuleOp.getBlock().getOps<spirv::GlobalVariableOp>()) {
    auto descriptorSetAttr =
        globalVar.getAttrOfType<IntegerAttr>("descriptor_set");
    auto bindingAttr = globalVar.getAttrOfType<IntegerAttr>("binding");
    if (!descriptorSetAttr || !bindingAttr) {
      // Not something the runtime cares about.
      continue;
    }
    maxDescriptorSetOrdinal =
        std::max(descriptorSetAttr.getInt(), maxDescriptorSetOrdinal);
    auto &descriptorSet = descriptorSets[descriptorSetAttr.getInt()];
    descriptorSet[bindingAttr.getInt()] = globalVar;
  }

  // Create the individual layout and binding defs.
  pipelineLayoutDef->descriptor_set_layouts.resize(maxDescriptorSetOrdinal + 1);
  for (auto &descriptorSetBindings : descriptorSets) {
    int32_t descriptorSet = descriptorSetBindings.first;
    auto dsl = std::make_unique<iree::VkDescriptorSetLayoutDefT>();

    for (auto &globalVarBinding : descriptorSetBindings.second) {
      auto binding = std::make_unique<iree::VkDescriptorSetLayoutBindingDefT>();
      binding->binding = globalVarBinding.first;
      binding->descriptor_count = 1;
      // TODO(benvanik): pull from type info.
      binding->descriptor_type = 7;       // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
      binding->stage_flags = 0x00000020;  // VK_SHADER_STAGE_COMPUTE_BIT
      dsl->bindings.push_back(std::move(binding));
    }

    pipelineLayoutDef->descriptor_set_layouts[descriptorSet] = std::move(dsl);
  }

  return pipelineLayoutDef;
}

}  // namespace

llvm::Optional<ExecutableTranslationResult>
translateExecutableToSPIRVExecutable(ArrayRef<IREE::ExecutableOp> executableOps,
                                     ExecutableTranslationOptions options) {
  SPIRVTranslator translator(options);
  ExecutableTranslationResult translationResult;
  for (auto executableOp : llvm::make_early_inc_range(executableOps)) {
    auto executableDef = translator.translateExecutable(executableOp);
    if (!executableDef) {
      executableOp.emitError() << "Failed to translate one or more executables";
      return llvm::None;
    }
    translationResult.executable_defs.push_back(std::move(executableDef));
  }
  return translationResult;
}

static ExecutableTranslationRegistration SPIRVExecutableTranslationRegistration(
    "vulkan-spirv", translateExecutableToSPIRVExecutable);

}  // namespace iree_compiler
}  // namespace mlir
