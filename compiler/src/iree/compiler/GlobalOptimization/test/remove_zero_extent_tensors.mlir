// RUN: iree-opt --split-input-file --allow-unregistered-dialect \
// RUN:          --pass-pipeline="builtin.module(func.func(iree-global-opt-remove-zero-extent-tensors))" \
// RUN:          %s | FileCheck %s

func.func @zero_sized_operands(%arg0 : tensor<?x0xf32>, %arg1 : index) -> tensor<?x?xf32> {
  %0 = tensor.empty(%arg1): tensor<0x?xf32>
  %1 = "some_op"(%arg0, %0) : (tensor<?x0xf32>, tensor<0x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func @zero_sized_operands
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x0xf32>
// CHECK-SAME:     %[[ARG1:.+]]: index
//      CHECK:   %[[EMPTY0:.+]] = tensor.empty(%[[ARG1]])
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG0]]
//      CHECK:   %[[EMPTY1:.+]] = tensor.empty(%[[DIM]])
//      CHECK:   %[[RESULT:.+]] = "some_op"(%[[EMPTY1]], %[[EMPTY0]]
//      CHECK:   return %[[RESULT]]

// -----

func.func @zero_sized_tensor_insert(%arg0 : tensor<?x?xf32>, %arg1 : tensor<0x?xf32>,
    %arg2 : index) -> tensor<?x?xf32> {
  %1 = tensor.insert_slice %arg1 into %arg0[0, 0] [0, %arg2] [1, 1] : tensor<0x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK: func @zero_sized_tensor_insert(%[[ARG0:.+]]: tensor<?x?xf32>
// CHECK:   return %[[ARG0]]

// -----

func.func @zero_sizes_tensor_insert_dest(%arg0 : tensor<0x?xf32>, %arg1 : index) -> tensor<0x?xf32> {
  %0 = tensor.empty(%arg1) : tensor<0x?xf32>
  %1 = tensor.insert_slice %arg0 into %0[0, 0] [0, %arg1] [1, 1] : tensor<0x?xf32> into tensor<0x?xf32>
  return %1 : tensor<0x?xf32>
}
// CHECK: func @zero_sizes_tensor_insert_dest(%[[ARG0:.+]]: tensor<0x?xf32>, %[[ARG1:.+]]: index)
// CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[ARG1]])
// CHECK:   return %[[EMPTY]]
