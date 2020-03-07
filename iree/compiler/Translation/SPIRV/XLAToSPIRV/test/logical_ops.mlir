// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

func @or(%arg0: memref<4xi1>, %arg1: memref<4xi1>, %arg2: memref<4xi1>)
attributes {iree.executable.export, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xi1>) : tensor<4xi1>
  %1 = iree.load_input(%arg1 : memref<4xi1>) : tensor<4xi1>
  // CHECK: spv.LogicalOr
  %2 = xla_hlo.or %0, %1 : tensor<4xi1>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  return
}
