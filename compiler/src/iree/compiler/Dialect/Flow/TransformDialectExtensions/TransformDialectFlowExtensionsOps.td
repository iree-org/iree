// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTFLOWEXTENSIONS
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTFLOWEXTENSIONS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ForeachThreadToFlowDispatchWorkgroupsOp : Op<Transform_Dialect, "iree.foreach_thread_to_flow",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Rewrite an scf.foreach_thread to Flow::DispatchWorkgroups.}];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}
 
#endif // IREE_COMPILER_DIALECT_FLOW_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTFLOWEXTENSIONS
