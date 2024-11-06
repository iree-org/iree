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
      .def_property_readonly(
          "value",
          [](MlirAttribute self) -> ireeGPUReorderWorkgroupsStrategyEnum {
            return ireeGPUReorderWorkgroupsStrategyAttrGetValue(self);
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
}
