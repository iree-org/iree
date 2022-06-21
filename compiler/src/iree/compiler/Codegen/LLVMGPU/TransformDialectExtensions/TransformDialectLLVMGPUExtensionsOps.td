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

def ForeachThreadToGpuAndTranslationInfo : Op<Transform_Dialect, "iree.foreach_thread_to_gpu_and_translation_info",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Rewrite an scf.foreach_thread to distributed gpu.thread_id and translation_info attribute}];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

def VectorDistributionOp : Op<Transform_Dialect, "iree.vector_distribution",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let description = [{
    Given a target that is isolated from above, rewrite ops predicated by 
    `if (threadIdx.x == 0)` to distributed form running **on a single warp**.

    The warp size is determined by the `warp_size` attribute (it is generally 
    32 but we do not want to hardcode it).

    This rewrite only applies if it can be determined from the IR (i.e. from 
    the surrounding IREE::HAL::ExecutableExportOp) that the number of threads
    along the warp dimension is a multiple of the warp size. The transformation
    bails on non-perfect multiples of the warp size that would not properly 
    distribute.

    Example:

    ```
    hal.executable.export public @foo ... { workgroup_size = [64: index, 1: index, 1: index] }
    builtin.module {
      func.func @foo() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<128xf32>
        %1 = gpu.thread_id  x
        %2 = arith.cmpi ult, %1, %c1 : index
        scf.if %2 {
          %3 = arith.constant dense<1.0> : vector<128xf32>
          vector.transfer_write %3, %0[%c0] : vector<128xf32>, memref<128xf32>
        }
      }
    }
    ```

    distributes to:

    ```
    hal.executable.export public @foo ... { workgroup_size = [64: index, 1: index, 1: index] }
    builtin.module {
      func.func @foo() {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant dense<1.000000e+00> : vector<128xf32>
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<128xf32>
        %1 = gpu.thread_id  x
        %2 = arith.cmpi ult, %1, %c32 : index
        // Single-warp guard filters out threads 32-63.
        scf.if %2 {
          %3 = arith.cmpi eq, %1, %c0 : index
          %4 = memref.alloc() : memref<128xf32, 3>
          // Single-thread guard runs on thread 0 only.
          scf.if %3 {
            vector.store %cst, %4[%c0] : memref<128xf32, 3>, vector<128xf32>
          }
          %5 = arith.muli %1, %c4 : index
          %6 = vector.load %4[%5] : memref<128xf32, 3>, vector<4xf32>
          %7 = affine.apply #map()[%1]
          vector.transfer_write %6, %0[%7] {in_bounds = [true]} : vector<4xf32>, memref<128xf32>
        }
      }
    }
    ```
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64Attr, "{}">:$warp_size);
  let results = (outs);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::LogicalResult applyToOne(::mlir::Operation *target);
  }];
}

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_TRANSFORMDIALECTEXTENSIONS_TRANSFORMDIALECTLLVMGPUEXTENSIONS
