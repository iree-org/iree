// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORMDIALECTEXTENSIONS
#define IREE_COMPILER_CODEGEN_TRANSFORMDIALECTEXTENSIONS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def FunctionalStyleMultiOperandMultiResultTransformOpTrait
    : NativeOpTrait<"FunctionalStyleMultiOperandMultiResultTransformOpTrait"> {
  let cppNamespace = "::mlir::transform";
}

def ForeachThreadToFlowDispatchWorkgroupsOp : Op<Transform_Dialect, "iree.foreach_thread_to_flow",
    [FunctionalStyleMultiOperandMultiResultTransformOpTrait, 
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Rewrite an scf.foreach_thread to Flow::DispatchWorkgroups.}];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

def IREEBufferizeOp : Op<Transform_Dialect, "iree.bufferize",
    [FunctionalStyleMultiOperandMultiResultTransformOpTrait, 
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Calls upstream comprehensive bufferize with extra IREE hooks.}];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

def IREESetNumWorkgroupToOneOp : Op<Transform_Dialect, "iree.set_num_workgroups_to_one",
    [FunctionalStyleMultiOperandMultiResultTransformOpTrait, 
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Given a top-level HAL::ExecutableVariantOp, populates the
   HAL::EntryPointOp with all 1s.
  }];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

#endif // IREE_COMPILER_CODEGEN_TRANSFORMDIALECTEXTENSIONS
