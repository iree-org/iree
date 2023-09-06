// RUN: iree-opt %s | iree-opt | FileCheck %s

// Tests simple round-trip of IR with sparse tensor type.

#CSR = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  crdWidth = 32,
  posWidth = 32
}>

// CHECK-LABEL: func.func @hello_sparse_world(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<3xf64>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<3xi32>,
// CHECK-SAME:    %[[ARG2:.*]]: tensor<11xi32>) -> index {
// CHECK:         %[[T:.*]] = sparse_tensor.pack %[[ARG0]], %[[ARG2]], %[[ARG1]] : tensor<3xf64>, tensor<11xi32>, tensor<3xi32> to tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:         %[[NSE:.*]] = sparse_tensor.number_of_entries %[[T]] : tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:         return %[[NSE]] : index
// CHECK:       }
func.func @hello_sparse_world(%arg0: tensor<3xf64>,
                              %arg1: tensor<3xi32>,
			      %arg2: tensor<11xi32>) -> (index) {
    %t = sparse_tensor.pack %arg0, %arg2, %arg1
      : tensor<3xf64>,
        tensor<11xi32>,
        tensor<3xi32> to tensor<10x10xf64, #CSR>
    %nse = sparse_tensor.number_of_entries %t : tensor<10x10xf64, #CSR>
    return %nse : index
}
