// RUN: iree-opt %s -pass-pipeline='builtin.module(func.func(iree-vector-ext-vectorize-ops, iree-codegen-generic-vectorization))' | FileCheck %s

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [64, 64],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

func.func @vectorize_matmul_layout(%A: tensor<64x64xf32>,
                                   %B: tensor<64x64xf32>,
                                   %C: tensor<64x64xf32>)
                                   -> tensor<64x64xf32> {
  %AL = iree_vector_ext.to_layout %A to #layout : tensor<64x64xf32>
  %BL = iree_vector_ext.to_layout %B to #layout : tensor<64x64xf32>
  %CL = iree_vector_ext.to_layout %C to #layout : tensor<64x64xf32>
  %matmul = linalg.matmul ins(%AL, %BL : tensor<64x64xf32>, tensor<64x64xf32>)
                          outs(%CL: tensor<64x64xf32>) -> tensor<64x64xf32>
  return %matmul : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @vectorize_matmul_layout
// CHECK-SAME: %[[AT:.+]]: tensor<64x64xf32>, %[[BT:.+]]: tensor<64x64xf32>, %[[CT:.+]]: tensor<64x64xf32>

// CHECK: %[[AV:.+]] = vector.transfer_read %[[AT]]
// CHECK: %[[A:.+]] = iree_vector_ext.to_layout %[[AV]]
// CHECK: %[[BV:.+]] = vector.transfer_read %[[BT]]
// CHECK: %[[B:.+]] = iree_vector_ext.to_layout %[[BV]]
// CHECK: %[[CV:.+]] = vector.transfer_read %[[CT]]
// CHECK: %[[C:.+]] = iree_vector_ext.to_layout %[[CV]]

// CHECK: vector.contract
// CHECK-SAME: %[[A]], %[[B]], %[[C]]
