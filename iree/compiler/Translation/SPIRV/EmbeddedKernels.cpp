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

#include "iree/compiler/Translation/SPIRV/EmbeddedKernels.h"

#include "iree/compiler/Translation/SPIRV/Kernels/Kernels.h"
#include "iree/schemas/spirv_executable_def_generated.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Reads the SPIR-V code for the embedded kernel with the given file name.
// If the kernel under Kernels/ is 'matmul.comp' then |kernelName| would be
// 'matmul.spv' (because it's been compiled).
std::vector<uint32_t> readEmbeddedKernelCode(std::string kernelName) {
  auto *fileToc = spirv_kernels::Kernels_create();
  for (int i = 0; i < spirv_kernels::Kernels_size(); ++i) {
    if (std::strcmp(fileToc[i].name, kernelName.c_str()) == 0) {
      std::vector<uint32_t> code;
      code.resize(fileToc[i].size / 4);
      std::memcpy(code.data(), fileToc[i].data, fileToc[i].size);
      return code;
    }
  }
  return {};
}

// Adds a storage buffer binding to the descriptor set layout.
void addDescriptorSetLayoutBinding(uint32_t binding,
                                   iree::VkDescriptorSetLayoutDefT *dsl) {
  auto bindingDef = std::make_unique<iree::VkDescriptorSetLayoutBindingDefT>();
  bindingDef->binding = binding;
  bindingDef->descriptor_count = 1;
  bindingDef->descriptor_type = 7;       // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
  bindingDef->stage_flags = 0x00000020;  // VK_SHADER_STAGE_COMPUTE_BIT
  dsl->bindings.push_back(std::move(bindingDef));
}

// Adds a specialization map entry for |constant_id| set to a 4-byte int value.
void addSpecializationMapEntry(
    uint32_t constant_id, uint32_t value,
    iree::VkSpecializationInfoDefT *specializationInfoDef) {
  auto specValue = std::make_unique<iree::VkSpecializationMapEntryDefT>();
  specValue->constant_id = constant_id;
  specValue->uint32_value = value;
  specializationInfoDef->map_entries.push_back(std::move(specValue));
}

void addSpecializationMapEntryVector(
    uint32_t constant_start, const std::vector<int> &values,
    iree::VkSpecializationInfoDefT *specializationInfoDef) {
  for (int i = 0; i < values.size(); ++i) {
    addSpecializationMapEntry(constant_start + i,
                              *reinterpret_cast<const uint32_t *>(&values[i]),
                              specializationInfoDef);
  }
}

LogicalResult buildReductionExecutable(ModuleOp moduleOp, FuncOp entryFuncOp,
                                       iree::SpirVExecutableDefT *outDef) {
  auto funcType = entryFuncOp.getType();
  auto arg0 = funcType.getInput(0).cast<ShapedType>();
  if (!arg0.getElementType().isF32()) {
    // When we do other types we'll need other shaders.
    return entryFuncOp.emitOpError()
           << "only floating point reduction is implemented";
  }

  auto applyFuncAttr = entryFuncOp.getAttrOfType<FlatSymbolRefAttr>(
      "iree.executable.reduction.apply");
  auto applyFuncOp = moduleOp.lookupSymbol(applyFuncAttr.getValue());

  // TODO(benvanik): specialize (template on shapes/types/etc).
  std::string kernelName = "reduce_untiled.spv";
  llvm::Optional<uint32_t> operationId;
  applyFuncOp->walk([&](Operation *op) {
    if (isa<xla_hlo::AddOp>(op)) {
      operationId = 0;
    } else if (isa<xla_hlo::MaxOp>(op)) {
      operationId = 1;
    } else if (isa<xla_hlo::MinOp>(op)) {
      operationId = 2;
    }
  });
  if (!operationId.hasValue()) {
    applyFuncOp->dump();
    return applyFuncOp->emitOpError() << "unsupported reduction operator";
  }

  outDef->tag = "__reduce__";
  outDef->entry_points = {"main"};

  outDef->code = readEmbeddedKernelCode(kernelName);

  // arg0, arg1, ret0
  auto pipelineLayoutDef = std::make_unique<iree::VkPipelineLayoutDefT>();
  pipelineLayoutDef->buffer_binding_set = 0;
  auto dsl = std::make_unique<iree::VkDescriptorSetLayoutDefT>();
  addDescriptorSetLayoutBinding(0, dsl.get());
  addDescriptorSetLayoutBinding(1, dsl.get());
  addDescriptorSetLayoutBinding(2, dsl.get());
  pipelineLayoutDef->descriptor_set_layouts.push_back(std::move(dsl));
  outDef->pipeline_layout = std::move(pipelineLayoutDef);

  // See the shader source for documentation on the values of A/B/C/R.
  int64_t reductionDimension =
      entryFuncOp
          .getAttrOfType<IntegerAttr>("iree.executable.reduction.dimension")
          .getInt();
  uint32_t r = arg0.getDimSize(reductionDimension);
  uint32_t a = 1;
  for (int i = 0; i < reductionDimension; ++i) {
    a *= arg0.getDimSize(i);
  }
  uint32_t b = 1;
  for (int i = reductionDimension + 1; i < arg0.getRank(); ++i) {
    b *= arg0.getDimSize(i);
  }
  uint32_t c = b;

  auto specializationInfoDef =
      std::make_unique<iree::VkSpecializationInfoDefT>();
  addSpecializationMapEntry(/*kOperationId*/ 100, operationId.getValue(),
                            specializationInfoDef.get());
  addSpecializationMapEntry(/*kA*/ 101, a, specializationInfoDef.get());
  addSpecializationMapEntry(/*kB*/ 102, b, specializationInfoDef.get());
  addSpecializationMapEntry(/*kC*/ 103, c, specializationInfoDef.get());
  addSpecializationMapEntry(/*kR*/ 104, r, specializationInfoDef.get());
  outDef->specialization_info = std::move(specializationInfoDef);

  return success();
}

LogicalResult buildConvExecutable(ModuleOp moduleOp, FuncOp entryFuncOp,
                                  xla_hlo::ConvOp convOp,
                                  iree::SpirVExecutableDefT *outDef) {
  auto lhs = convOp.lhs().getType().cast<ShapedType>();
  auto rhs = convOp.rhs().getType().cast<ShapedType>();
  if (convOp.feature_group_count() != 1) {
    return entryFuncOp.emitOpError()
           << "only feature group counts of 1 supported";
  }
  if (lhs.getRank() != 4 || rhs.getRank() != 4) {
    return entryFuncOp.emitOpError() << "only Conv2d supported";
  }

  auto specializationInfoDef =
      std::make_unique<iree::VkSpecializationInfoDefT>();
  // Get the padding specializations.
  {
    std::vector<int> paddings;
    if (convOp.padding().hasValue()) {
      for (const auto &elm : convOp.padding().getValue().getIntValues()) {
        paddings.push_back(elm.getSExtValue());
      }
    }
    addSpecializationMapEntryVector(100, paddings, specializationInfoDef.get());
  }
  // LHS (image) dimensions in NCHW order - should map to NHWC ie [0,3,1,2].
  {
    std::vector<int> lhsOrdering{
        static_cast<int>(
            convOp.dimension_numbers().input_batch_dimension().getInt()),
        static_cast<int>(
            convOp.dimension_numbers().input_feature_dimension().getInt())};
    for (const auto &dim :
         convOp.dimension_numbers().input_spatial_dimensions()) {
      lhsOrdering.push_back(dim.getSExtValue());
    }
    if (lhsOrdering.size() != 4 || lhsOrdering[0] != 0 || lhsOrdering[1] != 3 ||
        lhsOrdering[2] != 1 || lhsOrdering[3] != 2) {
      return entryFuncOp.emitOpError() << "only NHWC tensor ordering supported";
    }

    // Extents in buffer order.
    std::vector<int> lhsExtents(lhsOrdering.size());
    for (int i = 0; i < lhs.getRank(); ++i) {
      lhsExtents[i] = lhs.getDimSize(i);
    }
    addSpecializationMapEntryVector(110, lhsExtents,
                                    specializationInfoDef.get());
  }
  // RHS (kernel) dimension OIHW - should map to HWIO ie [3,2,0,1].
  {
    std::vector<int> rhsOrdering{
        static_cast<int>(convOp.dimension_numbers()
                             .kernel_output_feature_dimension()
                             .getInt()),
        static_cast<int>(convOp.dimension_numbers()
                             .kernel_input_feature_dimension()
                             .getInt())};
    for (const auto &dim :
         convOp.dimension_numbers().kernel_spatial_dimensions()) {
      rhsOrdering.push_back(dim.getSExtValue());
    }
    if (rhsOrdering.size() != 4 || rhsOrdering[0] != 3 || rhsOrdering[1] != 2 ||
        rhsOrdering[2] != 0 || rhsOrdering[3] != 1) {
      return entryFuncOp.emitOpError() << "only HWIO kernel ordering supported";
    }

    // Extents in buffer order.
    std::vector<int> rhsExtents(rhsOrdering.size());
    for (int i = 0; i < rhs.getRank(); ++i) {
      rhsExtents[i] = rhs.getDimSize(i);
    }
    addSpecializationMapEntryVector(120, rhsExtents,
                                    specializationInfoDef.get());
  }
  // Result dimension order NCHW - should map to NHWC ie [0,3,1,2].
  {
    std::vector<int> retOrdering{
        static_cast<int>(
            convOp.dimension_numbers().output_batch_dimension().getInt()),
        static_cast<int>(
            convOp.dimension_numbers().output_feature_dimension().getInt())};
    for (const auto &dim :
         convOp.dimension_numbers().output_spatial_dimensions()) {
      retOrdering.push_back(dim.getSExtValue());
    }
    if (retOrdering.size() != 4 || retOrdering[0] != 0 || retOrdering[1] != 3 ||
        retOrdering[2] != 1 || retOrdering[3] != 2) {
      return entryFuncOp.emitOpError() << "only HWIO kernel ordering supported";
    }
  }

  outDef->tag = "__conv2d_nhwc__";
  outDef->entry_points = {"main"};
  outDef->code = readEmbeddedKernelCode("conv2d_nhwc.spv");

  auto pipelineLayoutDef = std::make_unique<iree::VkPipelineLayoutDefT>();
  pipelineLayoutDef->buffer_binding_set = 0;
  auto dsl = std::make_unique<iree::VkDescriptorSetLayoutDefT>();
  addDescriptorSetLayoutBinding(0, dsl.get());
  addDescriptorSetLayoutBinding(1, dsl.get());
  addDescriptorSetLayoutBinding(2, dsl.get());
  pipelineLayoutDef->descriptor_set_layouts.push_back(std::move(dsl));
  outDef->pipeline_layout = std::move(pipelineLayoutDef);
  outDef->specialization_info = std::move(specializationInfoDef);

  return success();
}

// Builds a SPIR-V executable from a well-known matmul executable.
// |outDef| will be populated with all required information for serialization.
LogicalResult buildMatMulExecutable(ModuleOp moduleOp, FuncOp entryFuncOp,
                                    xla_hlo::DotOp dotOp,
                                    iree::SpirVExecutableDefT *outDef) {
  auto arg0 = dotOp.getOperand(0).getType().cast<ShapedType>();
  auto arg1 = dotOp.getOperand(1).getType().cast<ShapedType>();

  outDef->tag = "__matmul__";
  outDef->entry_points = {"main"};

  // TODO(benvanik): specialize (template on shapes/types/etc).
  outDef->code = readEmbeddedKernelCode("matmul.spv");

  // arg0, arg1, ret0
  auto pipelineLayoutDef = std::make_unique<iree::VkPipelineLayoutDefT>();
  pipelineLayoutDef->buffer_binding_set = 0;
  auto dsl = std::make_unique<iree::VkDescriptorSetLayoutDefT>();
  addDescriptorSetLayoutBinding(0, dsl.get());
  addDescriptorSetLayoutBinding(1, dsl.get());
  addDescriptorSetLayoutBinding(2, dsl.get());
  pipelineLayoutDef->descriptor_set_layouts.push_back(std::move(dsl));
  outDef->pipeline_layout = std::move(pipelineLayoutDef);

  // Shapes of [arg0, arg1, ret0].
  //   arg0 = [b0, m, k]
  //   arg1 = [b0, k, n]
  //   ret0 = [b0, m, n]
  // Note that we handle both batched (rank 3) and unbatched (rank 2).
  uint32_t m = arg0.getRank() == 3 ? arg0.getDimSize(1) : arg0.getDimSize(0);
  uint32_t k = arg0.getRank() == 3 ? arg0.getDimSize(2) : arg0.getDimSize(1);
  uint32_t n = arg1.getRank() == 3 ? arg1.getDimSize(2) : arg1.getDimSize(1);
  auto specializationInfoDef =
      std::make_unique<iree::VkSpecializationInfoDefT>();
  addSpecializationMapEntry(/*kMatrixM*/ 100, m, specializationInfoDef.get());
  addSpecializationMapEntry(/*kMatrixK*/ 101, k, specializationInfoDef.get());
  addSpecializationMapEntry(/*kMatrixN*/ 102, n, specializationInfoDef.get());
  outDef->specialization_info = std::move(specializationInfoDef);

  return success();
}

}  // namespace

bool tryEmbeddedKernelRewrite(ModuleOp moduleOp,
                              iree::SpirVExecutableDefT *outDef) {
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    if (funcOp.getAttr("iree.executable.reduction")) {
      if (failed(buildReductionExecutable(moduleOp, funcOp, outDef))) {
        moduleOp.emitOpError() << "failed to splat in the reduction kernel";
        return false;
      }
      return true;
    }

    for (auto &block : funcOp) {
      for (auto &op : block) {
        if (auto convOp = dyn_cast_or_null<xla_hlo::ConvOp>(&op)) {
          if (failed(buildConvExecutable(moduleOp, funcOp, convOp, outDef))) {
            moduleOp.emitOpError() << "failed to splat in the conv kernel";
            return false;
          }
          return true;
        } else if (auto dotOp = dyn_cast_or_null<xla_hlo::DotOp>(&op)) {
          if (failed(buildMatMulExecutable(moduleOp, funcOp, dotOp, outDef))) {
            moduleOp.emitOpError() << "failed to splat in the matmul kernel";
            return false;
          }
          return true;
        }
      }
    }
  }
  return false;
}

}  // namespace iree_compiler
}  // namespace mlir
