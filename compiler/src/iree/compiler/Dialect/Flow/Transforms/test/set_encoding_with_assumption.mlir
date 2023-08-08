// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-flow-set-encoding{assume-tile-sizes-divisors-of=16}))" --split-input-file %s | FileCheck %s

func.func @matmul_f32f32f32(%arg0 : tensor<128x256xf32>, %arg1 : tensor<256x512xf32>,
    %arg2 : tensor<128x512xf32>) -> tensor<128x512xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%arg2 : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %0 : tensor<128x512xf32>
}
//      CHECK: func @matmul_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<128x256xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<256x512xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<128x512xf32>
// CHECK-NOT:    iree_linalg_ext.upper_bound_tile_size
// CHECK-DAG:    iree_linalg_ext.set_encoding %[[ARG0]] : tensor<128x256xf32> -> tensor<128x256xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>>
// CHECK-DAG:    iree_linalg_ext.set_encoding %[[ARG1]] : tensor<256x512xf32> -> tensor<256x512xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS>>
// CHECK-DAG:    iree_linalg_ext.set_encoding %[[ARG2]] : tensor<128x256xf32> -> tensor<128x512xf32> -> tensor<128x512xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>>
