// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <optional>
#include <vector>
#include "iree/compiler/dialects/iree_codegen.h"
#include "iree/compiler/dialects/iree_gpu.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"

static const char *kCodegenModuleImportPath =
    MAKE_MLIR_PYTHON_QUALNAME("dialects.iree_codegen");
static const char *kGpuModuleImportPath =
    MAKE_MLIR_PYTHON_QUALNAME("dialects.iree_gpu");

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

static std::vector<py::object>
ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
  size_t numMMAs = 0;
  ireeCodegenQueryMMAIntrinsics(op, &numMMAs, nullptr);
  std::vector<uint32_t> mmaIntrinsics(numMMAs);
  ireeCodegenQueryMMAIntrinsics(op, &numMMAs, mmaIntrinsics.data());

  py::object mmaIntrinsicEnum =
      py::module_::import_(kGpuModuleImportPath).attr("MMAIntrinsic");
  std::vector<py::object> mmaList(numMMAs);
  for (size_t i = 0; i < numMMAs; ++i) {
    mmaList[i] = mmaIntrinsicEnum(mmaIntrinsics[i]);
  }

  return mmaList;
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

  //===-------------------------------------------------------------------===//
  // CodegenExecutableTargetAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_codegen_module, "HAL_ExecutableTargetAttr",
                          ireeAttributeIsACodegenHALExecutableTargetAttr,
                          ireeCodegenHALExecutableTargetAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, MlirAttribute backend, MlirAttribute format,
             std::optional<MlirAttribute> configuration, MlirContext ctx) {
            ireeCodegenHALExecutableTargetInfo info = {};
            info.backend = backend;
            info.format = format;
            info.configuration = configuration.value_or(mlirAttributeGetNull());
            return ireeCodegenHALExecutableTargetAttrGet(ctx, info);
          },
          "cls"_a, "backend"_a, "format"_a, "configuration"_a = py::none(),
          "ctx"_a = py::none(),
          "Gets an #iree_codegen.executable_target from parameters.")
      .def_property_readonly(
          "backend",
          [](MlirAttribute self) -> MlirAttribute {
            auto info = ireeCodegenHALExecutableTargetAttrGetInfo(self);
            return info.backend;
          })
      .def_property_readonly(
          "format",
          [](MlirAttribute self) -> MlirAttribute {
            auto info = ireeCodegenHALExecutableTargetAttrGetInfo(self);
            return info.format;
          })
      .def_property_readonly(
          "configuration", [](MlirAttribute self) -> MlirAttribute {
            auto info = ireeCodegenHALExecutableTargetAttrGetInfo(self);
            return info.configuration;
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
          [](const py::object &, std::optional<bool> prefetchSharedMemory,
             std::optional<bool> noReduceSharedMemoryBankConflicts,
             std::optional<bool> useIgemmConvolution,
             std::optional<MlirAttribute> reorderWorkgroupsStrategy,
             MlirContext ctx) {
            return ireeGPUPipelineOptionsAttrGet(
                ctx,
                prefetchSharedMemory.has_value() ? &*prefetchSharedMemory
                                                 : nullptr,
                noReduceSharedMemoryBankConflicts.has_value()
                    ? &*noReduceSharedMemoryBankConflicts
                    : nullptr,
                useIgemmConvolution.has_value() ? &*useIgemmConvolution
                                                : nullptr,
                reorderWorkgroupsStrategy.has_value()
                    ? &*reorderWorkgroupsStrategy
                    : nullptr);
          },
          "cls"_a, "prefetch_shared_memory"_a = py::none(),
          "no_reduce_shared_memory_bank_conflicts"_a = py::none(),
          "use_igemm_convolution"_a = py::none(),
          "reorder_workgroups_strategy"_a = py::none(), py::kw_only(),
          "ctx"_a = py::none(),
          "Gets an #iree_gpu.pipeline_options from parameters.")
      .def_property_readonly(
          "prefetch_shared_memory",
          [](MlirAttribute self) -> std::optional<bool> {
            auto attr = ireeGPUPipelineOptionsAttrGetPrefetchSharedMemory(self);
            if (!mlirAttributeIsNull(attr))
              return mlirBoolAttrGetValue(attr);
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
                                   .attr("MMAIntrinsic")(rawValue);
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
                    .attr("VirtualMMAIntrinsic");

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
      .def_property_readonly("value",
                             [](MlirAttribute self) -> py::object {
                               uint32_t rawValue =
                                   ireeGPUVirtualMMAIntrinsicAttrGetValue(self);
                               return py::module_::import_(kGpuModuleImportPath)
                                   .attr("VirtualMMAIntrinsic")(rawValue);
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
          "subgroup_count_mn",
          [](MlirAttribute self) -> py::tuple {
            ireeGPUSubgroupCountInfo info =
                ireeGPULoweringConfigAttrGetSubgroupCount(self);
            MlirAttribute mCountAttr = info.subgroupMCountAttr;
            MlirAttribute nCountAttr = info.subgroupNCountAttr;
            std::optional<int64_t> mCount;
            if (!mlirAttributeIsNull(mCountAttr)) {
              mCount = mlirIntegerAttrGetValueInt(mCountAttr);
            }

            std::optional<int64_t> nCount;
            if (!mlirAttributeIsNull(nCountAttr)) {
              nCount = mlirIntegerAttrGetValueInt(nCountAttr);
            }
            return py::make_tuple(mCount, nCount);
          })
      .def_property_readonly(
          "mma_kind", [](MlirAttribute self) -> std::optional<MlirAttribute> {
            auto attr = ireeGPULoweringConfigAttrGetMmaKind(self);
            if (!mlirAttributeIsNull(attr))
              return attr;
            return std::nullopt;
          });

  //===-------------------------------------------------------------------===//
  // GPUComputeBitwidthsAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "ComputeBitwidthsAttr",
                          ireeAttributeIsAGPUComputeBitwidthsAttr,
                          ireeGPUComputeBitwidthsAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, uint32_t value, MlirContext ctx) {
            return ireeGPUComputeBitwidthsAttrGet(ctx, value);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets an #iree_gpu.compute_bitwidths from parameters.")
      .def_property_readonly("raw_value", ireeGPUComputeBitwidthsAttrGetValue)
      .def_property_readonly("value", [](MlirAttribute self) -> py::object {
        uint32_t rawValue = ireeGPUComputeBitwidthsAttrGetValue(self);
        return py::module_::import_(kGpuModuleImportPath)
            .attr("ComputeBitwidths")(rawValue);
      });

  //===-------------------------------------------------------------------===//
  // GPUStorageBitwidthsAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "StorageBitwidthsAttr",
                          ireeAttributeIsAGPUStorageBitwidthsAttr,
                          ireeGPUStorageBitwidthsAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, uint32_t value, MlirContext ctx) {
            return ireeGPUStorageBitwidthsAttrGet(ctx, value);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets an #iree_gpu.storage_bitwidths from a value.")
      .def_property_readonly("raw_value", ireeGPUStorageBitwidthsAttrGetValue)
      .def_property_readonly("value", [](MlirAttribute self) -> py::object {
        uint32_t rawValue = ireeGPUStorageBitwidthsAttrGetValue(self);
        return py::module_::import_(kGpuModuleImportPath)
            .attr("StorageBitwidths")(rawValue);
      });

  //===-------------------------------------------------------------------===//
  // GPUSubgroupOpsAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "SubgroupOpsAttr",
                          ireeAttributeIsAGPUSubgroupOpsAttr,
                          ireeGPUSubgroupOpsAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, uint32_t value, MlirContext ctx) {
            return ireeGPUSubgroupOpsAttrGet(ctx, value);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets an #iree_gpu.subgroup_ops from a value.")
      .def_property_readonly("raw_value", ireeGPUSubgroupOpsAttrGetValue)
      .def_property_readonly("value", [](MlirAttribute self) -> py::object {
        uint32_t rawValue = ireeGPUSubgroupOpsAttrGetValue(self);
        return py::module_::import_(kGpuModuleImportPath)
            .attr("SubgroupOps")(rawValue);
      });

  //===-------------------------------------------------------------------===//
  // GPUDotProductOpsAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "DotProductOpsAttr",
                          ireeAttributeIsAGPUDotProductOpsAttr,
                          ireeGPUDotProductOpsAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, uint32_t value, MlirContext ctx) {
            return ireeGPUDotProductOpsAttrGet(ctx, value);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets an #iree_gpu.dotproduct_ops from a value.")
      .def_property_readonly("raw_value", ireeGPUDotProductOpsAttrGetValue)
      .def_property_readonly("value", [](MlirAttribute self) -> py::object {
        uint32_t rawValue = ireeGPUDotProductOpsAttrGetValue(self);
        return py::module_::import_(kGpuModuleImportPath)
            .attr("DotProductOps")(rawValue);
      });

  //===-------------------------------------------------------------------===//
  // GPUMMAOpsArrayAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "MMAOpsArrayAttr",
                          ireeAttributeIsAGPUMMAOpsArrayAttr,
                          ireeGPUMMAOpsArrayAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, const std::vector<MlirAttribute> &mmaAttrs,
             MlirContext ctx) {
            return ireeGPUMMAOpsArrayAttrGet(ctx, mmaAttrs.data(),
                                             mmaAttrs.size());
          },
          "cls"_a, "mma_attrs"_a, "ctx"_a = py::none(),
          "Gets an #iree_gpu.mma_ops array from MMA attributes.")
      .def_property_readonly(
          "value", ireeGPUMMAOpsArrayAttrGetValue,
          "Gets the underlying MMA attributes as a generic ArrayAttr")
      .def("__getitem__",
           [](MlirAttribute self, size_t index) -> MlirAttribute {
             return ireeGPUMMAOpsArrayAttrGetElement(self, index);
           })
      .def("__len__",
           [](MlirAttribute self) -> size_t {
             return ireeGPUMMAOpsArrayAttrGetSize(self);
           })
      .def("__iter__", [](MlirAttribute self) {
        py::list result;
        size_t size = ireeGPUMMAOpsArrayAttrGetSize(self);
        for (size_t i = 0; i < size; ++i) {
          result.append(ireeGPUMMAOpsArrayAttrGetElement(self, i));
        }
        return result.attr("__iter__")();
      });

  //===-------------------------------------------------------------------===//
  // GPUTargetWgpAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(iree_gpu_module, "TargetWgpAttr",
                          ireeAttributeIsAGPUTargetWgpAttr,
                          ireeGPUTargetWgpAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, MlirAttribute compute, MlirAttribute storage,
             MlirAttribute subgroup, MlirAttribute dot, MlirAttribute mma,
             MlirAttribute subgroup_size_choices,
             MlirAttribute max_workgroup_sizes,
             int32_t max_thread_count_per_workgroup,
             int32_t max_workgroup_memory_bytes,
             MlirAttribute max_workgroup_counts,
             std::optional<int32_t> max_load_instruction_bits,
             std::optional<int32_t> simds_per_wgp,
             std::optional<int32_t> vgpr_space_bits,
             std::optional<MlirAttribute> extra, MlirContext ctx) {
            ireeGPUTargetWgpInfo targetInfo = {};
            targetInfo.compute = compute;
            targetInfo.storage = storage;
            targetInfo.subgroup = subgroup;
            targetInfo.dot = dot;
            targetInfo.mma = mma;
            targetInfo.subgroup_size_choices = subgroup_size_choices;
            targetInfo.max_workgroup_sizes = max_workgroup_sizes;
            targetInfo.max_thread_count_per_workgroup =
                max_thread_count_per_workgroup;
            targetInfo.max_workgroup_memory_bytes = max_workgroup_memory_bytes;
            targetInfo.max_workgroup_counts = max_workgroup_counts;

            // Convert std::optional<int32_t> to int32_t with -1 sentinel
            targetInfo.max_load_instruction_bits =
                max_load_instruction_bits.value_or(-1);
            targetInfo.simds_per_wgp = simds_per_wgp.value_or(-1);
            targetInfo.vgpr_space_bits = vgpr_space_bits.value_or(-1);
            targetInfo.extra = extra.value_or(mlirAttributeGetNull());

            return ireeGPUTargetWgpAttrGet(ctx, targetInfo);
          },
          "cls"_a, "compute"_a, "storage"_a, "subgroup"_a, "dot"_a, "mma"_a,
          "subgroup_size_choices"_a, "max_workgroup_sizes"_a,
          "max_thread_count_per_workgroup"_a, "max_workgroup_memory_bytes"_a,
          "max_workgroup_counts"_a, "max_load_instruction_bits"_a = py::none(),
          "simds_per_wgp"_a = py::none(), "vgpr_space_bits"_a = py::none(),
          "extra"_a = py::none(), py::kw_only(), "ctx"_a = py::none(),
          "Gets an #iree_gpu.target_wgp from parameters.")
      .def_property_readonly("compute",
                             [](MlirAttribute self) -> MlirAttribute {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               return targetInfo.compute;
                             })
      .def_property_readonly("storage",
                             [](MlirAttribute self) -> MlirAttribute {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               return targetInfo.storage;
                             })
      .def_property_readonly("subgroup",
                             [](MlirAttribute self) -> MlirAttribute {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               return targetInfo.subgroup;
                             })
      .def_property_readonly("dot",
                             [](MlirAttribute self) -> MlirAttribute {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               return targetInfo.dot;
                             })
      .def_property_readonly("mma",
                             [](MlirAttribute self) -> MlirAttribute {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               return targetInfo.mma;
                             })
      .def_property_readonly("subgroup_size_choices",
                             [](MlirAttribute self) -> MlirAttribute {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               return targetInfo.subgroup_size_choices;
                             })
      .def_property_readonly("max_workgroup_sizes",
                             [](MlirAttribute self) -> MlirAttribute {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               return targetInfo.max_workgroup_sizes;
                             })
      .def_property_readonly("max_thread_count_per_workgroup",
                             [](MlirAttribute self) -> int32_t {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               return targetInfo.max_thread_count_per_workgroup;
                             })
      .def_property_readonly("max_workgroup_memory_bytes",
                             [](MlirAttribute self) -> int32_t {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               return targetInfo.max_workgroup_memory_bytes;
                             })
      .def_property_readonly("max_workgroup_counts",
                             [](MlirAttribute self) -> MlirAttribute {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               return targetInfo.max_workgroup_counts;
                             })
      .def_property_readonly("max_load_instruction_bits",
                             [](MlirAttribute self) -> std::optional<int32_t> {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               if (targetInfo.max_load_instruction_bits == -1) {
                                 return std::nullopt;
                               }
                               return targetInfo.max_load_instruction_bits;
                             })
      .def_property_readonly("simds_per_wgp",
                             [](MlirAttribute self) -> std::optional<int32_t> {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               if (targetInfo.simds_per_wgp == -1) {
                                 return std::nullopt;
                               }
                               return targetInfo.simds_per_wgp;
                             })
      .def_property_readonly("vgpr_space_bits",
                             [](MlirAttribute self) -> std::optional<int32_t> {
                               auto targetInfo =
                                   ireeGPUTargetWgpAttrGetInfo(self);
                               if (targetInfo.simds_per_wgp == -1) {
                                 return std::nullopt;
                               }
                               return targetInfo.vgpr_space_bits;
                             })
      .def_property_readonly("extra", [](MlirAttribute self) -> MlirAttribute {
        auto targetInfo = ireeGPUTargetWgpAttrGetInfo(self);
        return targetInfo.extra;
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
  // Binding to utility function queryMMAIntrinsics
  //===-------------------------------------------------------------------===//

  iree_codegen_module.def(
      "query_mma_intrinsics", &ireeCodegenQueryMMAIntrinsicsBinding,
      "Queries the MMA intrinsics from an executable variant op.",
      py::arg("op"));

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
      "get_attention_op_detail",
      [](MlirAffineMap q, MlirAffineMap k, MlirAffineMap v, MlirAffineMap o) {
        ireeCodegenAttentionOpDetail result =
            ireeCodegenGetAttentionOpDetail(q, k, v, o);
        return result;
      },
      "Infers the structure of an attention operation from affine indexing "
      "maps.",
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("o"));

  iree_codegen_module.def(
      "isa_attention_op", &ireeCodegenMlirOperationIsACodegenAttentionOp,
      "Checks if the given operation is an IREE LinalgExt attention op.",
      py::arg("op"));
}
