// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>
#include "iree/compiler/dialects/iree_codegen.h"
#include "iree/compiler/dialects/iree_gpu.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Target/LLVMIR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"

static const char *kCodegenModuleImportPath =
    MAKE_MLIR_PYTHON_QUALNAME("dialects.iree_codegen");
static const char *kGpuModuleImportPath =
    MAKE_MLIR_PYTHON_QUALNAME("dialects.iree_gpu");

static const char *kMMAIntrinsicEnumName = "MMAIntrinsic";
static const char *kVirtualMMAIntrinsicEnumName = "VirtualMMAIntrinsic";

namespace py = nanobind;
using namespace nanobind::literals;
using namespace mlir::python::nanobind_adaptors;

static std::vector<MlirOperation>
ireeCodegenGetExecutableVariantOpsBinding(MlirModule module) {
  size_t numOps = 0;
  ireeCodegenGetExecutableVariantOps(module, &numOps, nullptr);
  std::vector<MlirOperation> ops(numOps);
  ireeCodegenGetExecutableVariantOps(module, &numOps, ops.data());

  return ops;
}

static std::vector<MlirOperation>
ireeCodegenGetTunerRootOpsBinding(MlirModule module) {
  size_t numOps = 0;
  ireeCodegenGetTunerRootOps(module, &numOps, nullptr);
  std::vector<MlirOperation> ops(numOps);
  ireeCodegenGetTunerRootOps(module, &numOps, ops.data());

  return ops;
}

static std::vector<int64_t> getIntArrayAttrValues(MlirAttribute attr) {
  if (mlirAttributeIsNull(attr) || !mlirAttributeIsAArray(attr))
    return {};

  std::vector<int64_t> result;
  size_t n = mlirArrayAttrGetNumElements(attr);
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    MlirAttribute elem = mlirArrayAttrGetElement(attr, i);
    int64_t val = mlirIntegerAttrGetValueInt(elem);
    result.push_back(val);
  }
  return result;
}

NB_MODULE(_ireeCompilerDialects, m) {
  m.doc() = "iree-compiler dialects python extension";

  auto iree_codegen_module =
      m.def_submodule("iree_codegen", "iree_codegen dialect bindings");

  //===-------------------------------------------------------------------===//
  // CodegenDispatchLoweringPassPipelineAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(
      iree_codegen_module, "DispatchLoweringPassPipelineAttr",
      ireeAttributeIsACodegenDispatchLoweringPassPipelineAttr,
      ireeCodegenDispatchLoweringPassPipelineAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, uint32_t value, MlirContext ctx) {
            return ireeCodegenDispatchLoweringPassPipelineAttrGet(ctx, value);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets an #iree_codegen.dispatch_lowering_pass_pipeline from "
          "parameters.")
      .def_property_readonly(
          "raw_value", ireeCodegenDispatchLoweringPassPipelineAttrGetValue)
      .def_property_readonly("value", [](MlirAttribute self) -> py::object {
        uint32_t rawValue =
            ireeCodegenDispatchLoweringPassPipelineAttrGetValue(self);
        return py::module_::import_(kCodegenModuleImportPath)
            .attr("DispatchLoweringPassPipeline")(rawValue);
      });

  //===-------------------------------------------------------------------===//
  // CodegenTranslationInfoAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_codegen_module, "TranslationInfoAttr",
                          ireeAttributeIsACodegenTranslationInfoAttr,
                          ireeCodegenTranslationInfoAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, MlirAttribute passPipeline,
             std::optional<MlirAttribute> codegenSpec,
             std::optional<std::vector<int64_t>> workgroupSize,
             std::optional<int64_t> subgroupSize,
             std::optional<MlirAttribute> configuration, MlirContext ctx) {
            ireeCodegenTranslationInfoParameters parameters = {};
            parameters.passPipeline = passPipeline;
            parameters.codegenSpec =
                codegenSpec.value_or(mlirAttributeGetNull());
            if (workgroupSize.has_value()) {
              parameters.workgroupSize = workgroupSize->data();
              parameters.numWorkgroupSizeElements = workgroupSize->size();
            }
            parameters.subgroupSize = subgroupSize.value_or(0);
            parameters.configuration =
                configuration.value_or(mlirAttributeGetNull());

            return ireeCodegenTranslationInfoAttrGet(ctx, parameters);
          },
          "cls"_a, "pass_pipeline"_a, "codegen_spec"_a = py::none(),
          "workgroup_size"_a = py::none(), "subgroup_size"_a = py::none(),
          "configuration"_a = py::none(), py::kw_only(), "ctx"_a = py::none(),
          "Gets an #iree_codegen.translation_info from parameters.")
      .def_property_readonly(
          "pass_pipeline",
          [](MlirAttribute self) -> MlirAttribute {
            auto parameters = ireeCodegenTranslationInfoAttrGetParameters(self);
            return parameters.passPipeline;
          })
      .def_property_readonly(
          "codegen_spec",
          [](MlirAttribute self) -> std::optional<MlirAttribute> {
            auto parameters = ireeCodegenTranslationInfoAttrGetParameters(self);
            if (mlirAttributeIsNull(parameters.codegenSpec)) {
              return std::nullopt;
            }
            return parameters.codegenSpec;
          })
      .def_property_readonly(
          "workgroup_size",
          [](MlirAttribute self) -> std::vector<int64_t> {
            auto parameters = ireeCodegenTranslationInfoAttrGetParameters(self);
            return {parameters.workgroupSize,
                    parameters.workgroupSize +
                        parameters.numWorkgroupSizeElements};
          })
      .def_property_readonly(
          "subgroup_size",
          [](MlirAttribute self) -> int64_t {
            auto parameters = ireeCodegenTranslationInfoAttrGetParameters(self);
            return parameters.subgroupSize;
          })
      .def_property_readonly(
          "configuration",
          [](MlirAttribute self) -> std::optional<MlirAttribute> {
            auto parameters = ireeCodegenTranslationInfoAttrGetParameters(self);
            if (mlirAttributeIsNull(parameters.configuration)) {
              return std::nullopt;
            }
            return parameters.configuration;
          });

  //===-------------------------------------------------------------------===//
  // CodegenCompilationInfoAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_codegen_module, "CompilationInfoAttr",
                          ireeAttributeIsACodegenCompilationInfoAttr,
                          ireeCodegenCompilationInfoAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, MlirAttribute loweringConfig,
             MlirAttribute translationInfo, MlirContext ctx) {
            ireeCodegenCompilationInfoParameters parameters = {};
            parameters.loweringConfig = loweringConfig;
            parameters.translationInfo = translationInfo;
            return ireeCodegenCompilationInfoAttrGet(ctx, parameters);
          },
          "cls"_a, "lowering_config"_a, "translation_info"_a,
          "ctx"_a = py::none(),
          "Gets an #iree_codegen.compilation_info from parameters.")
      .def_property_readonly(
          "lowering_config",
          [](MlirAttribute self) -> MlirAttribute {
            auto parameters = ireeCodegenCompilationInfoAttrGetParameters(self);
            return parameters.loweringConfig;
          })
      .def_property_readonly(
          "translation_info", [](MlirAttribute self) -> MlirAttribute {
            auto parameters = ireeCodegenCompilationInfoAttrGetParameters(self);
            return parameters.translationInfo;
          });

  //===--------------------------------------------------------------------===//

  auto iree_gpu_module =
      m.def_submodule("iree_gpu", "iree_gpu dialect bindings");

  //===-------------------------------------------------------------------===//
  // GPUReorderWorkgroupsStrategyAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "ReorderWorkgroupsStrategyAttr",
                          ireeAttributeIsAGPUReorderWorkgroupsStrategyAttr,
                          ireeGPUReorderWorkgroupsStrategyAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, uint32_t value, MlirContext ctx) {
            return ireeGPUReorderWorkgroupsStrategyAttrGet(ctx, value);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets an #iree_gpu.reorder_workgroups_strategy from parameters.")
      .def_property_readonly("raw_value",
                             ireeGPUReorderWorkgroupsStrategyAttrGetValue)
      .def_property_readonly("value", [](MlirAttribute self) -> py::object {
        uint32_t rawValue = ireeGPUReorderWorkgroupsStrategyAttrGetValue(self);
        return py::module_::import_(kGpuModuleImportPath)
            .attr("ReorderWorkgroupsStrategy")(rawValue);
      });

  //===-------------------------------------------------------------------===//
  // GPUPipelineOptionsAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "PipelineOptionsAttr",
                          ireeAttributeIsAGPUPipelineOptionsAttr,
                          ireeGPUPipelineOptionsAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, std::optional<int64_t> prefetchNumStages,
             std::optional<bool> noReduceSharedMemoryBankConflicts,
             std::optional<bool> useIgemmConvolution,
             std::optional<MlirAttribute> reorderWorkgroupsStrategy,
             MlirContext ctx) {
            return ireeGPUPipelineOptionsAttrGet(
                ctx,
                prefetchNumStages.has_value() ? &*prefetchNumStages : nullptr,
                noReduceSharedMemoryBankConflicts.has_value()
                    ? &*noReduceSharedMemoryBankConflicts
                    : nullptr,
                useIgemmConvolution.has_value() ? &*useIgemmConvolution
                                                : nullptr,
                reorderWorkgroupsStrategy.has_value()
                    ? &*reorderWorkgroupsStrategy
                    : nullptr);
          },
          "cls"_a, "prefetch_num_stages"_a = py::none(),
          "no_reduce_shared_memory_bank_conflicts"_a = py::none(),
          "use_igemm_convolution"_a = py::none(),
          "reorder_workgroups_strategy"_a = py::none(), py::kw_only(),
          "ctx"_a = py::none(),
          "Gets an #iree_gpu.pipeline_options from parameters.")
      .def_property_readonly(
          "prefetch_num_stages",
          [](MlirAttribute self) -> std::optional<int64_t> {
            auto attr = ireeGPUPipelineOptionsAttrGetPrefetchNumStages(self);
            if (!mlirAttributeIsNull(attr))
              return mlirIntegerAttrGetValueInt(attr);
            return std::nullopt;
          })
      .def_property_readonly(
          "no_reduce_shared_memory_bank_conflicts",
          [](MlirAttribute self) -> std::optional<bool> {
            auto attr =
                ireeGPUPipelineOptionsAttrGetNoReduceSharedMemoryBankConflicts(
                    self);
            if (!mlirAttributeIsNull(attr))
              return mlirBoolAttrGetValue(attr);
            return std::nullopt;
          })
      .def_property_readonly(
          "use_igemm_convolution",
          [](MlirAttribute self) -> std::optional<bool> {
            auto attr = ireeGPUPipelineOptionsAttrGetUseIgemmConvolution(self);
            if (!mlirAttributeIsNull(attr))
              return mlirBoolAttrGetValue(attr);
            return std::nullopt;
          })
      .def_property_readonly(
          "reorder_workgroups_strategy",
          [](MlirAttribute self) -> std::optional<MlirAttribute> {
            auto attr =
                ireeGPUPipelineOptionsAttrGetReorderWorkgroupsStrategy(self);
            if (!mlirAttributeIsNull(attr))
              return attr;
            return std::nullopt;
          });

  //===-------------------------------------------------------------------===//
  // GPUMMAIntrinsicAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "MMAIntrinsicAttr",
                          ireeAttributeIsAGPUMMAIntrinsicAttr,
                          ireeGPUMMAIntrinsicAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, uint32_t value, MlirContext ctx) {
            return ireeGPUMMAIntrinsicAttrGet(ctx, value);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets an #iree_gpu.mma_intrinsic from parameters.")
      .def_property_readonly("raw_value", ireeGPUMMAIntrinsicAttrGetValue)
      .def_property_readonly("value",
                             [](MlirAttribute self) -> py::object {
                               uint32_t rawValue =
                                   ireeGPUMMAIntrinsicAttrGetValue(self);
                               return py::module_::import_(kGpuModuleImportPath)
                                   .attr(kMMAIntrinsicEnumName)(rawValue);
                             })
      .def_property_readonly("mma", [](MlirAttribute self) -> MlirAttribute {
        uint32_t value = ireeGPUMMAIntrinsicAttrGetValue(self);
        return ireeGPUMMAAttrGet(mlirAttributeGetContext(self), value);
      });

  //===-------------------------------------------------------------------===//
  // GPUMMAAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "MMAAttr",
                          ireeAttributeIsAGPUMMAAttr, ireeGPUMMAAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, uint32_t value, MlirContext ctx) {
            return ireeGPUMMAAttrGet(ctx, value);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets an #iree_gpu.mma from parameters.")
      .def_property_readonly(
          "abc_element_types",
          [](MlirAttribute self) -> py::tuple {
            ireeGPUMMAInfo info = ireeGPUMMAAttrGetInfo(self);
            return py::make_tuple(info.aElementType, info.bElementType,
                                  info.cElementType);
          })
      .def_property_readonly(
          "abc_vector_types",
          [](MlirAttribute self) -> py::tuple {
            ireeGPUMMAInfo info = ireeGPUMMAAttrGetInfo(self);
            return py::make_tuple(info.aVectorType, info.bVectorType,
                                  info.cVectorType);
          })
      .def_property_readonly(
          "mnk_shape",
          [](MlirAttribute self) -> py::tuple {
            ireeGPUMMAInfo info = ireeGPUMMAAttrGetInfo(self);
            return py::make_tuple(info.mElements, info.nElements,
                                  info.kElements);
          })
      .def(
          "get_virtual_intrinsics",
          [](MlirAttribute self) {
            MlirAttribute rawArrayAttr =
                ireeGPUMMAAttrGetVirtualMMAIntrinsic(self);
            if (mlirAttributeIsNull(rawArrayAttr)) {
              return std::vector<py::object>{};
            }

            static py::object virtualEnum =
                py::module_::import_(kGpuModuleImportPath)
                    .attr(kVirtualMMAIntrinsicEnumName);

            std::vector<py::object> result;
            for (int64_t val : getIntArrayAttrValues(rawArrayAttr)) {
              result.push_back(virtualEnum(static_cast<uint32_t>(val)));
            }
            return result;
          },
          "Returns a list of virtual intrinsics  associated with this "
          "MMAAttr.");

  //===-------------------------------------------------------------------===//
  // GPUVirtualMMAIntrinsicAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "VirtualMMAIntrinsicAttr",
                          ireeAttributeIsAGPUVirtualMMAIntrinsicAttr,
                          ireeGPUVirtualMMAIntrinsicAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, uint32_t value, MlirContext ctx) {
            return ireeGPUVirtualMMAIntrinsicAttrGet(ctx, value);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets an #iree_gpu.virtual_mma_intrinsic from parameters.")
      .def_property_readonly("raw_value",
                             ireeGPUVirtualMMAIntrinsicAttrGetValue)
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> py::object {
            uint32_t rawValue = ireeGPUVirtualMMAIntrinsicAttrGetValue(self);
            return py::module_::import_(kGpuModuleImportPath)
                .attr(kVirtualMMAIntrinsicEnumName)(rawValue);
          })
      .def_property_readonly("mma", [](MlirAttribute self) -> MlirAttribute {
        uint32_t value = ireeGPUVirtualMMAIntrinsicAttrGetValue(self);
        return ireeGPUVirtualMMAAttrGet(mlirAttributeGetContext(self), value);
      });

  //===-------------------------------------------------------------------===//
  // GPUVirtualMMAAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "VirtualMMAAttr",
                          ireeAttributeIsAGPUVirtualMMAAttr,
                          ireeGPUVirtualMMAAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, uint32_t value, MlirContext ctx) {
            return ireeGPUVirtualMMAAttrGet(ctx, value);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets an #iree_gpu.virtualmma from parameters.")
      .def_property_readonly(
          "abc_element_types",
          [](MlirAttribute self) -> py::tuple {
            ireeGPUMMAInfo info = ireeGPUMMAAttrGetInfo(self);
            return py::make_tuple(info.aElementType, info.bElementType,
                                  info.cElementType);
          })
      .def_property_readonly(
          "abc_vector_types",
          [](MlirAttribute self) -> py::tuple {
            ireeGPUMMAInfo info = ireeGPUMMAAttrGetInfo(self);
            return py::make_tuple(info.aVectorType, info.bVectorType,
                                  info.cVectorType);
          })
      .def_property_readonly("mnk_shape", [](MlirAttribute self) -> py::tuple {
        ireeGPUMMAInfo info = ireeGPUMMAAttrGetInfo(self);
        return py::make_tuple(info.mElements, info.nElements, info.kElements);
      });

  //===-------------------------------------------------------------------===//
  // GPULoweringConfigAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "LoweringConfigAttr",
                          ireeAttributeIsAGPULoweringConfigAttr,
                          ireeGPULoweringConfigAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, MlirAttribute attributeDictionary,
             MlirContext ctx) {
            return ireeGPULoweringConfigAttrGet(ctx, attributeDictionary);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets an #iree_gpu.lowering_config from parameters.")
      .def_property_readonly("attributes",
                             ireeGPULoweringConfigAttrGetAttributes)
      .def_property_readonly(
          "workgroup_tile_sizes",
          [](MlirAttribute self) -> std::vector<int64_t> {
            auto tilesizes = ireeGPULoweringConfigAttrGetTileSizes(self);
            return getIntArrayAttrValues(tilesizes.workgroupAttr);
          })
      .def_property_readonly(
          "reduction_tile_sizes",
          [](MlirAttribute self) -> std::vector<int64_t> {
            auto tilesizes = ireeGPULoweringConfigAttrGetTileSizes(self);
            return getIntArrayAttrValues(tilesizes.reductionAttr);
          })
      .def_property_readonly(
          "subgroup_basis",
          [](MlirAttribute self)
              -> std::tuple<std::vector<int64_t>, std::vector<int64_t>> {
            ireeGPUSubgroupBasisInfo basisInfo =
                ireeGPULoweringConfigAttrGetSubgroupBasis(self);
            std::vector<int64_t> counts;
            std::vector<int64_t> mapping;
            if (!mlirAttributeIsNull(basisInfo.countsAttr)) {
              counts = getIntArrayAttrValues(basisInfo.countsAttr);
            }
            if (!mlirAttributeIsNull(basisInfo.mappingAttr)) {
              mapping = getIntArrayAttrValues(basisInfo.mappingAttr);
            }
            return std::make_tuple(counts, mapping);
          })
      .def_property_readonly(
          "mma_kind", [](MlirAttribute self) -> std::optional<MlirAttribute> {
            auto attr = ireeGPULoweringConfigAttrGetMmaKind(self);
            if (!mlirAttributeIsNull(attr))
              return attr;
            return std::nullopt;
          });

  //===-------------------------------------------------------------------===//
  // Binding to query target info
  //===-------------------------------------------------------------------===//

  py::class_<ireeGPUTargetInfo>(iree_gpu_module, "TargetInfo")
      .def(
          "__init__",
          [](ireeGPUTargetInfo *self, MlirContext context,
             const std::string &arch,
             const std::vector<int32_t> &subgroupChoices,
             const std::vector<int32_t> &workgroupSizes, int32_t threadCount,
             int32_t memoryBytes, int32_t wgpCount, int32_t simdsPerWgp,
             const py::list &mmaIntrinsicObjs) {
            std::vector<mma_intrinsic_enum_t> mmaIntrinsicVals;
            py::module_ gpuModule = py::module_::import_(kGpuModuleImportPath);
            py::object mmaIntrinsicClass =
                gpuModule.attr(kMMAIntrinsicEnumName);
            py::object virtualMmaIntrinsicClass =
                gpuModule.attr(kVirtualMMAIntrinsicEnumName);

            for (py::handle item : mmaIntrinsicObjs) {
              if (!py::isinstance(item, mmaIntrinsicClass) &&
                  !py::isinstance(item, virtualMmaIntrinsicClass)) {
                throw py::type_error("All items must be MMA atributes");
              }
              mmaIntrinsicVals.push_back(
                  py::cast<mma_intrinsic_enum_t>(item.attr("value")));
            }

            if (wgpCount < 0) {
              throw py::value_error("workgroup_count must be non-negative");
            }

            *self = ireeGPUTargetInfoGet(
                context, arch.c_str(), subgroupChoices.data(),
                subgroupChoices.size(), workgroupSizes.data(),
                workgroupSizes.size(), threadCount, memoryBytes, wgpCount,
                simdsPerWgp, mmaIntrinsicVals.data(), mmaIntrinsicVals.size());
          },
          "context"_a, "arch"_a, "subgroup_size_choices"_a,
          "max_workgroup_sizes"_a, "max_thread_count_per_workgroup"_a,
          "max_workgroup_memory_bytes"_a, "workgroup_count"_a,
          "simds_per_workgroup"_a, "mma_intrinsics"_a = py::list{},
          "Create a GPUTargetInfo with the given parameters")
      .def_static(
          "get_gpu_target_info", &ireeHALExecutableTargetAttrGetGPUTargetInfo,
          "executable_target_attr"_a,
          "Get GPU target information from an executable target attribute")
      .def_prop_ro("arch",
                   [](const ireeGPUTargetInfo &self) -> std::string {
                     MlirStringRef strRef = mlirIdentifierStr(self.arch);
                     return std::string(strRef.data, strRef.length);
                   })
      .def_prop_ro("subgroup_size_choices",
                   [](const ireeGPUTargetInfo &self) -> std::vector<int64_t> {
                     return getIntArrayAttrValues(self.subgroupSizeChoices);
                   })
      .def_prop_ro("max_thread_count_per_workgroup",
                   [](const ireeGPUTargetInfo &self) -> int64_t {
                     return self.maxThreadCountPerWorkgroup;
                   })
      .def_prop_ro("max_workgroup_sizes",
                   [](const ireeGPUTargetInfo &self) -> std::vector<int64_t> {
                     return getIntArrayAttrValues(self.maxWorkgroupSizes);
                   })
      .def_prop_ro("max_workgroup_memory_bytes",
                   [](const ireeGPUTargetInfo &self) -> int64_t {
                     return self.maxWorkgroupMemoryBytes;
                   })
      .def_prop_ro("workgroup_count",
                   [](const ireeGPUTargetInfo &self) -> int64_t {
                     return self.wgpCount;
                   })
      .def_prop_ro("simds_per_workgroup",
                   [](const ireeGPUTargetInfo &self) -> int64_t {
                     return self.simdsPerWgp;
                   })
      .def_prop_ro(
          "mma_intrinsics", [](const ireeGPUTargetInfo &self) -> py::list {
            if (mlirAttributeIsNull(self.mmaIntrinsics) ||
                !mlirAttributeIsAArray(self.mmaIntrinsics)) {
              return py::list();
            }

            size_t numElements =
                mlirArrayAttrGetNumElements(self.mmaIntrinsics);

            std::vector<mma_intrinsic_enum_t> mmaIntrinsicVals(numElements);
            // Use uint8_t instead of bool because std::vector<bool> is a
            // specialized template that doesn't provide .data() method.
            std::vector<uint8_t> virtualMmaIntrinsicTags(numElements);
            ireeGPUTargetInfoGetMMAIntrinsics(self.mmaIntrinsics,
                                              mmaIntrinsicVals.data(),
                                              virtualMmaIntrinsicTags.data());

            py::list intrinsics;
            py::module_ gpuModule = py::module_::import_(kGpuModuleImportPath);
            py::object mmaIntrinsicEnum = gpuModule.attr(kMMAIntrinsicEnumName);
            py::object virtualMmaIntrinsicEnum =
                gpuModule.attr(kVirtualMMAIntrinsicEnumName);

            for (size_t i = 0; i < numElements; ++i) {
              if (virtualMmaIntrinsicTags[i]) {
                py::object virtualMmaIntrinsic =
                    virtualMmaIntrinsicEnum(mmaIntrinsicVals[i]);
                intrinsics.append(virtualMmaIntrinsic);
                continue;
              }
              py::object mmaIntrinsic = mmaIntrinsicEnum(mmaIntrinsicVals[i]);
              intrinsics.append(mmaIntrinsic);
            }

            return intrinsics;
          });

  //===-------------------------------------------------------------------===//
  // Binding to utility function getSingleSubgroupLayout
  //===-------------------------------------------------------------------===//
  py::class_<ireeGPUMMASingleSubgroupLayout>(iree_gpu_module,
                                             "GPUMMASingleSubgroupLayout")
      .def_prop_ro("outer",
                   [](const ireeGPUMMASingleSubgroupLayout &self) {
                     return getIntArrayAttrValues(self.outer);
                   })
      .def_prop_ro("thread",
                   [](const ireeGPUMMASingleSubgroupLayout &self) {
                     return getIntArrayAttrValues(self.thread);
                   })
      .def_prop_ro("tstrides",
                   [](const ireeGPUMMASingleSubgroupLayout &self) {
                     return getIntArrayAttrValues(self.tstrides);
                   })
      .def_prop_ro("element", [](const ireeGPUMMASingleSubgroupLayout &self) {
        return getIntArrayAttrValues(self.element);
      });

  iree_gpu_module.def(
      "get_single_subgroup_layout",
      [](MlirAttribute attr, int fragment) {
        return ireeGPUGetSingleSubgroupLayout(attr, fragment);
      },
      "Returns the single subgroup layout (element, thread, outer, "
      "tstrides) for a given MMA or VirtualMMA intrinsic and fragment. ",
      py::arg("attr"), py::arg("fragment"));

  //===-------------------------------------------------------------------===//
  // Binding to utility function getExecutableVariantOps
  //===-------------------------------------------------------------------===//

  iree_codegen_module.def(
      "get_executable_variant_ops", &ireeCodegenGetExecutableVariantOpsBinding,
      "Gets the executable variant operations from a module.",
      py::arg("module"));

  //===-------------------------------------------------------------------===//
  // Binding to utility function ireeCodegenGetTunerRootOps
  //===-------------------------------------------------------------------===//

  iree_codegen_module.def("get_tuner_root_ops",
                          &ireeCodegenGetTunerRootOpsBinding,
                          "Get the operations marked with the tuner root op "
                          "attribute from a module.",
                          py::arg("module"));

  //===-------------------------------------------------------------------===//
  // Binding to utility function ireeCodegenGetAttentionOpDetail
  //===-------------------------------------------------------------------===//
  py::class_<ireeCodegenAttentionOpDetail>(iree_codegen_module,
                                           "AttentionOpDetail")
      .def_prop_ro("batch_dims",
                   [](const ireeCodegenAttentionOpDetail &self) {
                     return getIntArrayAttrValues(self.batch);
                   })
      .def_prop_ro("m_dims",
                   [](const ireeCodegenAttentionOpDetail &self) {
                     return getIntArrayAttrValues(self.m);
                   })
      .def_prop_ro("k1_dims",
                   [](const ireeCodegenAttentionOpDetail &self) {
                     return getIntArrayAttrValues(self.k1);
                   })
      .def_prop_ro("k2_dims",
                   [](const ireeCodegenAttentionOpDetail &self) {
                     return getIntArrayAttrValues(self.k2);
                   })
      .def_prop_ro("n_dims",
                   [](const ireeCodegenAttentionOpDetail &self) {
                     return getIntArrayAttrValues(self.n);
                   })
      .def_prop_ro("domain_rank", [](const ireeCodegenAttentionOpDetail &self) {
        return self.domainRank;
      });

  iree_codegen_module.def(
      "get_attention_op_detail", &ireeCodegenGetAttentionOpDetail,
      "Infers the structure of an attention operation from affine indexing "
      "maps.",
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("o"));

  iree_codegen_module.def(
      "isa_attention_op", &ireeCodegenMlirOperationIsACodegenAttentionOp,
      "Checks if the given operation is an IREE LinalgExt attention op.",
      py::arg("op"));

  //===-------------------------------------------------------------------===//
  // Binding to utility function ireeCodegenGetIGEMMGenericConvDetails
  //===-------------------------------------------------------------------===//
  py::class_<ireeCodegenIGEMMGenericConvDetails>(iree_codegen_module,
                                                 "IGEMMGenericConvDetails")
      .def_prop_ro("igemm_contraction_maps",
                   [](const ireeCodegenIGEMMGenericConvDetails &self) {
                     return self.igemmContractionMaps;
                   })
      .def_prop_ro("igemm_loop_bounds",
                   [](const ireeCodegenIGEMMGenericConvDetails &self) {
                     return getIntArrayAttrValues(self.igemmLoopBounds);
                   })
      .def_prop_ro("igemm_loop_iterators",
                   [](const ireeCodegenIGEMMGenericConvDetails &self) {
                     return self.igemmLoopIterators;
                   })
      .def_prop_ro("im2col_output_perm",
                   [](const ireeCodegenIGEMMGenericConvDetails &self) {
                     return getIntArrayAttrValues(self.im2colOutputPerm);
                   })
      .def_prop_ro(
          "filter_reassoc_indices",
          [](const ireeCodegenIGEMMGenericConvDetails &self)
              -> std::vector<std::vector<int64_t>> {
            MlirAttribute attr = self.filterReassocIndices;
            assert(!mlirAttributeIsNull(attr) && mlirAttributeIsAArray(attr) &&
                   "filterReassocIndices should be a valid ArrayAttr");
            size_t n = mlirArrayAttrGetNumElements(attr);
            std::vector<std::vector<int64_t>> result;
            result.reserve(n);
            for (size_t i = 0; i < n; ++i) {
              MlirAttribute innerArrayAttr = mlirArrayAttrGetElement(attr, i);
              result.push_back(getIntArrayAttrValues(innerArrayAttr));
            }
            return result;
          })
      .def_prop_ro("is_output_channel_first",
                   [](const ireeCodegenIGEMMGenericConvDetails &self) {
                     return self.isOutputChannelFirst;
                   })
      .def_prop_ro(
          "conv_to_igemm_dim_map",
          [](const ireeCodegenIGEMMGenericConvDetails &self) -> py::dict {
            py::dict result;
            MlirAttribute attr = self.convToIgemmDimMap;
            assert(!mlirAttributeIsNull(attr) && mlirAttributeIsAArray(attr) &&
                   "convToIgemmDimMap should be a valid ArrayAttr");
            size_t n = mlirArrayAttrGetNumElements(attr);
            for (size_t i = 0; i < n; ++i) {
              MlirAttribute pairAttr = mlirArrayAttrGetElement(attr, i);
              assert(mlirAttributeIsAArray(pairAttr) &&
                     mlirArrayAttrGetNumElements(pairAttr) == 2 &&
                     "Each pair should be [conv_dim, igemm_dim]");
              MlirAttribute keyAttr = mlirArrayAttrGetElement(pairAttr, 0);
              MlirAttribute valueAttr = mlirArrayAttrGetElement(pairAttr, 1);
              int64_t key = mlirIntegerAttrGetValueInt(keyAttr);
              int64_t value = mlirIntegerAttrGetValueInt(valueAttr);
              result[py::int_(key)] = py::int_(value);
            }
            return result;
          });

  iree_codegen_module.def(
      "get_igemm_generic_conv_details",
      [](MlirOperation op)
          -> std::optional<ireeCodegenIGEMMGenericConvDetails> {
        if (!ireeCodegenHasIGEMMGenericConvDetails(op)) {
          return std::nullopt;
        }
        return ireeCodegenGetIGEMMGenericConvDetails(op);
      },
      "Gets IGEMM details for a linalg operation. "
      "Returns None if failed to infer IGEMM convolution details.",
      py::arg("linalg_op"));

  //===-------------------------------------------------------------------===//
  // Binding to utility function ireeCodegenGetScaledContractionDetails
  //===-------------------------------------------------------------------===//
  iree_codegen_module.def("isa_scaled_contraction_op",
                          &ireeCodegenMlirOperationIsAScaledContractionOp,
                          "Checks if the given operation is an IREE LinalgExt "
                          "scaled contraction op.",
                          py::arg("op"));

  //===-------------------------------------------------------------------===//
  // Binding to struct ireeCodegenScaledContractionDimensions
  //===-------------------------------------------------------------------===//
  py::class_<ireeCodegenScaledContractionDimensions>(
      iree_codegen_module, "ScaledContractionDimensions")
      .def_prop_ro("batch",
                   [](const ireeCodegenScaledContractionDimensions &self) {
                     return getIntArrayAttrValues(self.batch);
                   })
      .def_prop_ro("m",
                   [](const ireeCodegenScaledContractionDimensions &self) {
                     return getIntArrayAttrValues(self.m);
                   })
      .def_prop_ro("n",
                   [](const ireeCodegenScaledContractionDimensions &self) {
                     return getIntArrayAttrValues(self.n);
                   })
      .def_prop_ro("k",
                   [](const ireeCodegenScaledContractionDimensions &self) {
                     return getIntArrayAttrValues(self.k);
                   })
      .def_prop_ro("kB",
                   [](const ireeCodegenScaledContractionDimensions &self) {
                     return getIntArrayAttrValues(self.kB);
                   });

  iree_codegen_module.def(
      "infer_scaled_contraction_dimensions",
      &ireeCodegenInferScaledContractionDimensions,
      "Infers the scaled contraction dimensions for a given operation.",
      py::arg("op"));
}
