// RUN: iree-opt %s | iree-opt | FileCheck %s

// Tests simple round-trip of sparse tensor type.

#CSR = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ]
}>

// CHECK-LABEL: func.func @hello_sparse_world(
//  CHECK-SAME:   %[[ARG.*]]: tensor<?x?xf32, #sparse_tensor.encoding<{{{.*}}}>>)
func.func @hello_sparse_world(%arg0: tensor<?x?xf32, #CSR>) -> tensor<?x?xf32, #CSR> {
  return %arg0 : tensor<?x?xf32, #CSR>
}
