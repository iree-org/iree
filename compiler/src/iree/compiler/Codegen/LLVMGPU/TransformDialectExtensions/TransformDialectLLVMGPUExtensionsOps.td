// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMGPUEXTENSIONS
#define IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMGPUEXTENSIONS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ForeachThreadToGpuAndTranslationInfo : 
  Op<Transform_Dialect, "iree.foreach_thread_to_gpu_and_translation_info",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Target the whole hal.executable_variant op and rewrite all scf.foreach_thread
    to distributed gpu.thread_id and translation_info attribute.

    The mapping of threads to gpu.thread_id is currently one-to-one and in order.
    Only **bufferized** scf.foreach_thread are currently supported.
    Only scf.foreach_thread distributed to *8at most 3 dimensions** are currently
    supported.

    Multiple scf.foreach_thread are supported per function in which case, the
    max of all the threads is computed and taken for the global gpu.thread_id.
    If necessary, scf.foreach_thread that do not use the whole thread range 
    result in predicated computations.
   
    Barriers are inserted after each scf.foreach_thread op for now.

    Example:

    ```
    hal.executable {
      hal.executable.variant {
        hal.executable.export {
          func @foo() {
            scf.foreach_thread (%i, %j) in (7, 9) {
              ... // body 1
            }
            scf.foreach_thread (%i) in (12) {
              ... // body 2
            }
          }
    ```

    is translated to:

    ```
    hal.executable {
      hal.executable.variant {
        hal.executable.export ... workgroup_size = [12 : index, 9 : index, 1 : index] {
          func @foo() {
            if (threadIdx.x < 7) {
              ... // body 1
            }
            if (threadIdx.y < 1) {
              ... // body 2
            }
          }
    ```
  }];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMGPUEXTENSIONS
