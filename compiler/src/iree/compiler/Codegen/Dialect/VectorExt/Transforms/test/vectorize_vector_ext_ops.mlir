// RUN: iree-opt %s -pass-pipeline='builtin.module(func.func(iree-vector-ext-vectorize-ops, iree-codegen-generic-vectorization{enable-vector-masking=true}),canonicalize,cse,canonicalize)' --split-input-file --mlir-print-local-scope | FileCheck %s

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
  %AL = iree_vector_ext.to_layout %A to layout(#layout) : tensor<64x64xf32>
  %BL = iree_vector_ext.to_layout %B to layout(#layout) : tensor<64x64xf32>
  %CL = iree_vector_ext.to_layout %C to layout(#layout) : tensor<64x64xf32>
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


// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [64, 64],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

func.func @vectorize_matmul_dyn_parallel(%A: tensor<?x64xf32>,
                                         %B: tensor<64x?xf32>,
                                         %C: tensor<?x?xf32>)
                                   -> tensor<?x?xf32> {
  %AL = iree_vector_ext.to_layout %A to layout(#layout) : tensor<?x64xf32>
  %BL = iree_vector_ext.to_layout %B to layout(#layout) : tensor<64x?xf32>
  %CL = iree_vector_ext.to_layout %C to layout(#layout) : tensor<?x?xf32>
  %matmul = linalg.matmul ins(%AL, %BL : tensor<?x64xf32>, tensor<64x?xf32>)
                          outs(%CL: tensor<?x?xf32>) {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0], [64, 64, 0], [0, 0, 64]]>}
                          -> tensor<?x?xf32>
  return %matmul : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @vectorize_matmul_dyn_parallel
// CHECK-SAME: %[[AT:.+]]: tensor<?x64xf32>, %[[BT:.+]]: tensor<64x?xf32>, %[[CT:.+]]: tensor<?x?xf32>
// CHECK-DAG: %[[PAD:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[ADIM:.+]] = tensor.dim %arg0, %c0 : tensor<?x64xf32>
// CHECK-DAG: %[[BDIM:.+]] = tensor.dim %arg1, %c1 : tensor<64x?xf32>
// CHECK-DAG: %[[AMASK:.+]] = vector.create_mask %[[ADIM]], %c64 : vector<64x64xi1>
// CHECK-DAG: %[[AV:.+]] = vector.transfer_read %arg0[%c0, %c0], %[[PAD]], %[[AMASK]]
// CHECK-DAG: %[[A:.+]]  = iree_vector_ext.to_layout %[[AV]] to layout(#iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [1, 1], element_tile = [64, 64], subgroup_strides = [0, 0], thread_strides = [0, 0]>) : vector<64x64xf32>

// CHECK-DAG: %[[OPMASK:.+]]  = vector.create_mask %[[ADIM]], %[[BDIM]], %c64 : vector<64x64x64xi1>
// CHECK-DAG: vector.mask %[[OPMASK]] { vector.contract {{.*}} %[[A]]
