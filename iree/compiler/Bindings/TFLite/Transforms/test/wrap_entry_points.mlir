// RUN: iree-opt -iree-tflite-wrap-entry-points -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: func @_tflite_main(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x8x8x3xf32> {iree.identifier = "input0"},
//  CHECK-SAME:   %[[ARG1:.+]]: tensor<?x8x8x3xf32> {iree.identifier = "input1"})
//  CHECK-SAME: -> (
//  CHECK-SAME:   tensor<?x8x8x3xf32> {iree.identifier = "output0"},
//  CHECK-SAME:   tensor<?x8x8x3xf32> {iree.identifier = "output1"}
//  CHECK-SAME: ) attributes {
//  CHECK-SAME:   iree.abi.stub,
//  CHECK-SAME:   iree.module.export,
//  CHECK-SAME:   iree.reflection = {
//  CHECK-SAME:     tfl.io.names = "input0;input1;output0;output1"
//  CHECK-SAME:   }
//  CHECK-SAME: } {
// CHECK-NEXT:   %[[RET:.+]]:2 = call @dynamicEntry(%[[ARG0]], %[[ARG1]])
// CHECK-NEXT:   return %[[RET]]#0, %[[RET]]#1
// CHECK-NEXT: }

// CHECK-LABEL: func private @dynamicEntry(
func @dynamicEntry(
  %arg0: tensor<?x8x8x3xf32> {iree.identifier = "input0"},
  %arg1: tensor<?x8x8x3xf32> {iree.identifier = "input1"}
) -> (
  tensor<?x8x8x3xf32> {iree.identifier = "output0"},
  tensor<?x8x8x3xf32> {iree.identifier = "output1"}
) {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>) -> tensor<?x8x8x3xf32>
  %1 = "mhlo.add"(%0, %arg0) : (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>) -> tensor<?x8x8x3xf32>
  return %0, %1 : tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>
}
