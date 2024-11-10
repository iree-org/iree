// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/dialects/iree_gpu.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_ireeCompilerDialects, m) {
  m.doc() = "iree-compiler dialects python extension";

  auto iree_gpu_module =
      m.def_submodule("iree_gpu", "iree_gpu dialect bindings");

  //===-------------------------------------------------------------------===//
  // GPUReorderWorkgroupsStrategyAttr
  //===-------------------------------------------------------------------===//

  auto strategyEnum =
      py::enum_<ireeGPUReorderWorkgroupsStrategyEnum>(
          iree_gpu_module, "ReorderWorkgroupsStrategy", py::module_local())
          .value("None_", ireeGPUReorderWorkgroupsStrategyEnumNone)
          .value("Transpose", ireeGPUReorderWorkgroupsStrategyEnumTranspose)
          .def(
              "__str__",
              [](ireeGPUReorderWorkgroupsStrategyEnum &self) {
                switch (self) {
                case ireeGPUReorderWorkgroupsStrategyEnumNone:
                  return "None";
                case ireeGPUReorderWorkgroupsStrategyEnumTranspose:
                  return "Transpose";
                default:
                  llvm::report_fatal_error(
                      "unknown ReorderWorkgroupsStrategy variant");
                }
              },
              // pybind overloads are tried in the order they were registered.
              // As a result, enums used the default __str__ method instead of
              // the custom one. Adding py::prepend() fixes this issue.
              py::prepend());

  mlir_attribute_subclass(iree_gpu_module, "ReorderWorkgroupsStrategyAttr",
                          ireeAttributeIsAGPUReorderWorkgroupsStrategyAttr,
                          ireeGPUReorderWorkgroupsStrategyAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, ireeGPUReorderWorkgroupsStrategyEnum value,
             MlirContext ctx) {
            return ireeGPUReorderWorkgroupsStrategyAttrGet(ctx, value);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets a gpu.reorder_workgroups_strategy from parameters.")
      .def_property_readonly("value",
                             ireeGPUReorderWorkgroupsStrategyAttrGetValue);

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
          "ctx"_a = py::none(), "Gets a gpu.pipeline_options from parameters.")
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
  auto mmaIntrinsicEnum =
      py::enum_<ireeGPUMMAIntrinsicEnum>(iree_gpu_module, "MMAIntrinsic",
                                         py::module_local())

#define X_DO(EnumName, EnumValue)                                              \
  .value(#EnumName, ireeGPUMMAIntrinsicEnum##EnumName)

          IREE_GPU_FOR_ALL_MMA_INTRINSIC_VALUES

#undef X_DO

              .def(
                  "__str__",
                  [](ireeGPUMMAIntrinsicEnum &self) {
                    switch (self) {
#define X_DO(EnumName, EnumValue)                                              \
  case ireeGPUMMAIntrinsicEnum##EnumName:                                      \
    return #EnumName;

                      IREE_GPU_FOR_ALL_MMA_INTRINSIC_VALUES

#undef X_DO
                    default:
                      llvm::report_fatal_error("unknown MMAIntrinsic variant");
                    }
                  },
                  py::prepend());

  mlir_attribute_subclass(iree_gpu_module, "MMAIntrinsicAttr",
                          ireeAttributeIsAGPUMMAIntrinsicAttr,
                          ireeGPUMMAIntrinsicAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, ireeGPUMMAIntrinsicEnum value,
             MlirContext ctx) {
            return ireeGPUMMAIntrinsicAttrGet(ctx, value);
          },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets a gpu.mma_intrinsic from parameters.")
      .def_property_readonly("value", ireeGPUMMAIntrinsicAttrGetValue)
      .def_property_readonly("mma", [](MlirAttribute self) -> MlirAttribute {
        ireeGPUMMAIntrinsicEnum value = ireeGPUMMAIntrinsicAttrGetValue(self);
        return ireeGPUMMAAttrGet(mlirAttributeGetContext(self), value);
      });

  mlir_attribute_subclass(iree_gpu_module, "MMAAttr",
                          ireeAttributeIsAGPUMMAAttr, ireeGPUMMAAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, ireeGPUMMAIntrinsicEnum value,
             MlirContext ctx) { return ireeGPUMMAAttrGet(ctx, value); },
          "cls"_a, "value"_a, "ctx"_a = py::none(),
          "Gets a gpu.mma from parameters.")
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
}
