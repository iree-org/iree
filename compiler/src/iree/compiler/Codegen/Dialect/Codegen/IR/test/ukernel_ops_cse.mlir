// RUN: iree-opt --split-input-file --cse %s | FileCheck %s

func.func @ukernel_generic(
    %in0: tensor<?x?xf32>, %in1: tensor<?xf32>,
    %out0: tensor<?xf32>, %out1 : tensor<?x?xf32>,
    %b0: f32, %b1: i64) -> (tensor<?xf32>, tensor<?x?xf32>) {
  %0:2 = iree_codegen.ukernel.generic "foo"
      ins(%in0, %in1: tensor<?x?xf32>, tensor<?xf32>)
      outs(%out0, %out1 : tensor<?xf32>, tensor<?x?xf32>)
      (%b0, %b1 : f32, i64) -> tensor<?xf32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func.func @ukernel_generic
// CHECK-SAME:     %[[IN0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[IN1:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:     %[[OUT0:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:     %[[OUT1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[B0:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:     %[[B1:[a-zA-Z0-9]+]]: i64
//      CHECK:   %[[RESULT:.+]]:2 = iree_codegen.ukernel.generic
// CHECK-SAME:       "foo"
// CHECK-SAME:       ins(%[[IN0]], %[[IN1]] :
// CHECK-SAME:       outs(%[[OUT0]], %[[OUT1]] :
// CHECK-SAME:       (%[[B0]], %[[B1]] :
//      CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1


// -----

func.func @unused_ukernel_generic(
    %in0: tensor<?x?xf32>, %in1: tensor<?xf32>,
    %out0: tensor<?xf32>, %out1 : tensor<?x?xf32>,
    %b0: f32, %b1: i64) -> (tensor<?xf32>, tensor<?x?xf32>) {
  %0:2 = iree_codegen.ukernel.generic "foo"
      ins(%in0, %in1: tensor<?x?xf32>, tensor<?xf32>)
      outs(%out0, %out1 : tensor<?xf32>, tensor<?x?xf32>)
      (%b0, %b1 : f32, i64) -> tensor<?xf32>, tensor<?x?xf32>
  return %out0, %out1 : tensor<?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: func.func @unused_ukernel_generic
// CHECK-SAME:     %[[IN0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[IN1:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:     %[[OUT0:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:     %[[OUT1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[B0:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:     %[[B1:[a-zA-Z0-9]+]]: i64
// CHECK-NOT:      iree_codegen.ukernel.generic
// CHECK:          return %[[OUT0]], %[[OUT1]]

// -----

func.func @ukernel_generic_memref(
    %in0: memref<?x?xf32>, %in1: memref<?xf32>,
    %out0: memref<?xf32>, %out1 : memref<?x?xf32>,
    %b0: f32, %b1: i64) {
  iree_codegen.ukernel.generic "foo"
      ins(%in0, %in1: memref<?x?xf32>, memref<?xf32>)
      outs(%out0, %out1 : memref<?xf32>, memref<?x?xf32>)
      (%b0, %b1 : f32, i64)
  return
}
//      CHECK: func @ukernel_generic_memref(
// CHECK-SAME:     %[[IN0:[a-zA-Z0-9]+]]: memref<?x?xf32>
// CHECK-SAME:     %[[IN1:[a-zA-Z0-9]+]]: memref<?xf32>
// CHECK-SAME:     %[[OUT0:[a-zA-Z0-9]+]]: memref<?xf32>
// CHECK-SAME:     %[[OUT1:[a-zA-Z0-9]+]]: memref<?x?xf32>
// CHECK-SAME:     %[[B0:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:     %[[B1:[a-zA-Z0-9]+]]: i64
//      CHECK:   iree_codegen.ukernel.generic
// CHECK-SAME:       "foo"
// CHECK-SAME:       ins(%[[IN0]], %[[IN1]] :
// CHECK-SAME:       outs(%[[OUT0]], %[[OUT1]] :
// CHECK-SAME:       (%[[B0]], %[[B1]] :
