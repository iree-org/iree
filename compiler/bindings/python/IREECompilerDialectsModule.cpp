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
#include "mlir/Bindings/Python/PybindAdaptors.h"

static const char *kCodegenModuleImportPath =
    MAKE_MLIR_PYTHON_QUALNAME("dialects.iree_codegen");
static const char *kGpuModuleImportPath =
    MAKE_MLIR_PYTHON_QUALNAME("dialects.iree_gpu");

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_ireeCompilerDialects, m) {
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
        return py::module_::import(kCodegenModuleImportPath)
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
        return py::module_::import(kGpuModuleImportPath)
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
                               return py::module_::import(kGpuModuleImportPath)
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
                             ireeGPULoweringConfigAttrGetAttributes);
}
