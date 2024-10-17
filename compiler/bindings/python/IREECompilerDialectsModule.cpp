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

  //===-------------------------------------------------------------------===//
  // GPUPipelineOptionsAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(m, "GPUPipelineOptionsAttr",
                          ireeAttributeIsAGPUPipelineOptionsAttr,
                          ireeGPUPipelineOptionsAttrGetTypeID)
      .def_classmethod(
          "get",
          [](const py::object &, std::optional<bool> prefetchSharedMemory,
             std::optional<bool> noReduceSharedMemoryBankConflicts,
             std::optional<MlirAttribute> reorderWorkgroupsStrategy,
             MlirContext ctx) {
            return ireeGPUPipelineOptionsAttrGet(
                ctx,
                prefetchSharedMemory.has_value() ? &*prefetchSharedMemory
                                                 : nullptr,
                noReduceSharedMemoryBankConflicts.has_value()
                    ? &*noReduceSharedMemoryBankConflicts
                    : nullptr,
                reorderWorkgroupsStrategy.has_value()
                    ? &*reorderWorkgroupsStrategy
                    : nullptr);
          },
          "cls"_a, "prefetch_shared_memory"_a = py::none(),
          "no_reduce_shared_memory_bank_conflicts"_a = py::none(),
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
          "reorder_workgroups_strategy",
          [](MlirAttribute self) -> std::optional<MlirAttribute> {
            auto attr =
                ireeGPUPipelineOptionsAttrGetReorderWorkgroupsStrategy(self);
            if (!mlirAttributeIsNull(attr))
              return attr;
            return std::nullopt;
          });
}
