// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/dialects/iree_gpu.h"
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
          [](const py::object &, bool prefetchSharedMemory,
             bool noReduceSharedMemoryBankConflicts,
             MlirAttribute reorderWorkgroupsStrategy, MlirContext ctx) {
            return ireeGPUPipelineOptionsAttrGet(
                ctx, prefetchSharedMemory, noReduceSharedMemoryBankConflicts,
                reorderWorkgroupsStrategy);
          },
          "cls"_a, "prefetch_shared_memory"_a,
          "no_reduce_shared_memory_bank_conflicts"_a,
          "reorder_work_groups_strategy"_a, "ctx"_a = py::none(),
          "Gets a gpu.pipeline_options from parameters.")
      .def_property_readonly(
          "prefetch_shared_memory",
          [](MlirAttribute self) {
            return ireeGPUPipelineOptionsAttrGetPrefetchSharedMemory(self);
          })
      .def_property_readonly(
          "no_reduce_shared_memory_bank_conflicts",
          [](MlirAttribute self) {
            return ireeGPUPipelineOptionsAttrGetNoReduceSharedMemoryBankConflicts(
                self);
          })
      .def_property_readonly(
          "reorder_work_groups_strategy", [](MlirAttribute self) {
            return ireeGPUPipelineOptionsAttrGetReorderWorkgroupsStrategy(self);
          });
}
