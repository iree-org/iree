// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @inlineConstant
func @inlineConstant() -> index {
  %cst = constant 4 : index
  // CHECK: flow.ex.stream.fragment()
  %0 = flow.ex.stream.fragment(%cst) : (index) -> index =
      (%arg0: index) -> index {
    // CHECK: %[[C:.+]] = constant 4 : index
    // CHECK-NEXT: return %[[C]]
    flow.return %arg0 : index
  }
  return %0 : index
}

// -----

// CHECK-LABEL: func @removeUnusedCapture
// CHECK-SAME: (%[[ARG:.+]]: index)
func @removeUnusedCapture(%arg: index) -> index {
  %unused = constant 5 : index
  // CHECK: flow.ex.stream.fragment(%[[ARG]])
  %0 = flow.ex.stream.fragment(%arg, %unused) : (index, index) -> index =
      // CHECK-NEXT: (%[[INNER_ARG:.+]]: index) -> index {
      (%arg0: index, %arg1: index) -> index {
    // CHECK-NEXT: %[[T:.+]] = addi %[[INNER_ARG]], %[[INNER_ARG]]
    %t = addi %arg0, %arg0 : index
    // CHECK-NEXT: flow.return %[[T]]
    flow.return %t : index
  }
  return %0 : index
}

// -----

// CHECK-LABEL: func @removeUnusedDupCapture
// CHECK-SAME: (%[[ARG:.+]]: index)
func @removeUnusedDupCapture(%arg: index) -> index {
  // CHECK: flow.ex.stream.fragment(%[[ARG]])
  %0 = flow.ex.stream.fragment(%arg, %arg) : (index, index) -> index =
      (%arg0: index, %arg1: index) -> index {
    %t = addi %arg0, %arg0 : index
    flow.return %t : index
  }
  return %0 : index
}

// -----

// CHECK-LABEL: func @removeUnusedResult
// CHECK-SAME: (%[[ARG0:.+]]: index, %[[ARG1:.+]]: index)
func @removeUnusedResult(%arg0: index, %arg1: index) -> index {
  // CHECK: flow.ex.stream.fragment(%[[ARG1]])
  %0:2 = flow.ex.stream.fragment(%arg0, %arg1) : (index, index) -> (index, index) =
      (%unused: index, %arg1: index) -> (index, index) {
    %t = addi %arg1, %arg1 : index
    flow.return %t, %unused : index, index
  }
  return %0#0 : index
}

// -----

// CHECK-LABEL: func @removeUnusedDynamicResult
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x?xf32>, %[[DIM0:.+]]: index,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<8x?xf32>, %[[DIM1:.+]]: index)
func @removeUnusedDynamicResult(%arg0: tensor<4x?xf32>, %dim0: index,
                                %arg1: tensor<8x?xf32>, %dim1: index) -> tensor<8x?xf32> {
  // CHECK: flow.ex.stream.fragment(%[[ARG1]]) :
  %0:2 = flow.ex.stream.fragment(%arg0, %arg1) :
      // CHECK-SAME: (tensor<8x?xf32>{%[[DIM1]]}) -> %[[ARG1]] =
      (tensor<4x?xf32>{%dim0}, tensor<8x?xf32>{%dim1}) -> (%arg0, %arg1) =
      // CHECK-NEXT: (%[[INNER_ARG:.+]]: tensor<8x?xf32>) -> tensor<8x?xf32>
      (%unused: tensor<4x?xf32>, %arg1: tensor<8x?xf32>) -> (tensor<4x?xf32>, tensor<8x?xf32>) {
    // CHECK-NEXT: flow.return %[[INNER_ARG]] : tensor<8x?xf32>
    flow.return %unused, %arg1 : tensor<4x?xf32>, tensor<8x?xf32>
  }
  return %0#1 : tensor<8x?xf32>
}

// -----

// Testing inserted clones: a clone here is required as %stream_target is used
// after it is updated.

// CHECK-LABEL: @dynamicUpdateSliceImmutability
func @dynamicUpdateSliceImmutability(
    %target: tensor<2x4xi32>, %update: tensor<1x1xi32>) -> tensor<2x4xi32> {
  // CHECK: %[[RET:.+]] = flow.ex.stream.fragment
  %ret = flow.ex.stream.fragment(%target, %update) :
      (tensor<2x4xi32>, tensor<1x1xi32>) -> tensor<2x4xi32> =
      // CHECK-NEXT: (%[[TARGET:.+]]: tensor<2x4xi32>, %[[UPDATE:.+]]: tensor<1x1xi32>)
      (%stream_target: tensor<2x4xi32>, %stream_update: tensor<1x1xi32>) -> tensor<2x4xi32> {
    %start0 = constant 0 : index
    %start1 = constant 1 : index
    %workload = constant 8 : index
    //      CHECK: %[[TARGET_CLONE:.+]] = flow.tensor.clone %[[TARGET]] : tensor<2x4xi32>
    //      CHECK: %[[UPDATED:.+]] = flow.tensor.update %[[UPDATE]], %[[TARGET]]
    %t0 = flow.tensor.update %stream_update, %stream_target[%start0, %start1] : tensor<1x1xi32> -> tensor<2x4xi32>
    // CHECK-NEXT: %[[RETURN:.+]] = flow.dispatch @ex::@entry[%c8](%[[TARGET_CLONE]], %[[UPDATED]])
    %t1 = flow.dispatch @ex::@entry[%workload](%stream_target, %t0) : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
    // CHECK-NEXT: flow.return %[[RETURN]]
    flow.return %t1 : tensor<2x4xi32>
  }
  // CHECK: return %[[RET]]
  return %ret : tensor<2x4xi32>
}
