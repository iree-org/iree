// RUN: iree-opt %s -pass-pipeline='builtin.module(func.func(iree-vector-ext-vectorize-ops, iree-codegen-generic-vectorization))' | FileCheck %s

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup = [1, 1],
  outers_per_batch = [1, 1],
  threads_per_outer = [1, 1],
  elements_per_thread = [64, 64],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

// CHECK-LABEL: func.func @vectorize_matmul_layout
func.func @vectorize_matmul_layout(%A: tensor<64x64xf32>,
                                   %B: tensor<64x64xf32>,
                                   %C: tensor<64x64xf32>)
                                   -> tensor<64x64xf32> {
  %AL = iree_vector_ext.to_layout %A to #layout : tensor<64x64xf32>
  %BL = iree_vector_ext.to_layout %B to #layout : tensor<64x64xf32>
  %CL = iree_vector_ext.to_layout %C to #layout : tensor<64x64xf32>
  // CHECK: %[[A:.+]] = iree_vector_ext.to_layout
  // CHECK-SAME: vector<64x64xf32>
  // CHECK: %[[B:.+]] = iree_vector_ext.to_layout
  // CHECK-SAME: vector<64x64xf32>
  // CHECK: %[[C:.+]] = iree_vector_ext.to_layout
  // CHECK-SAME: vector<64x64xf32>
  %matmul = linalg.matmul ins(%AL, %BL : tensor<64x64xf32>, tensor<64x64xf32>)
                          outs(%CL: tensor<64x64xf32>) -> tensor<64x64xf32>
  // CHECK: vector.contract
  // CHECK-SAME: %[[A]], %[[B]], %[[C]]
  return %matmul : tensor<64x64xf32>
}
