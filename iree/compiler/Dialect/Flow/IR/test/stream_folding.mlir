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
      // CHECK-SAME: (tensor<8x?xf32>{%[[DIM1]]}) -> %[[ARG1]]{%[[DIM1]]} =
      (tensor<4x?xf32>{%dim0}, tensor<8x?xf32>{%dim1}) -> (%arg0{%dim0}, %arg1{%dim1}) =
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

// -----

// CHECK-LABEL: @dagImmutability
// CHECK:         %{{.+}} = flow.ex.stream.fragment
// CHECK:           %[[SRC:.+]] = flow.dispatch @_run_dispatch_1::@_run_dispatch_1[%c1, %c1, %c1]() : () -> tensor<i32>
// CHECK:           %[[RET0:.+]] = flow.tensor.clone %[[SRC]] : tensor<i32>
// CHECK:           %[[RET1:.+]] = flow.tensor.reshape %[[SRC]] : tensor<i32> -> tensor<1xi32>
// CHECK:           %[[RET2:.+]] = flow.tensor.slice
// CHECK:           flow.return %[[RET0]], %[[RET1]], %[[RET2]]
func @dagImmutability(%arg0: tensor<1xi32>) -> (tensor<i32>, tensor<1xi32>, tensor<3xi32>) {
  %0:3 = flow.ex.stream.fragment(%arg0) : (tensor<1xi32>) -> (tensor<i32>, tensor<1xi32>, tensor<3xi32>) =
      (%arg1: tensor<1xi32>) -> (tensor<i32>, tensor<1xi32>, tensor<3xi32>) {
    %c9 = constant 9 : index
    %c1 = constant 1 : index
    %c18 = constant 18 : index
    %c0 = constant 0 : index
    %c3 = constant 3 : index
    %1 = flow.dispatch @_run_dispatch_1::@_run_dispatch_1[%c1, %c1, %c1]() : () -> tensor<i32>
    %2 = flow.dispatch @_run_dispatch_2::@_run_dispatch_2[%c9, %c1, %c1](%1) : (tensor<i32>) -> tensor<9xi32>
    %3 = flow.tensor.reshape %1 : tensor<i32> -> tensor<1xi32>
    %4 = flow.tensor.slice %2[%c0 for %c3] : tensor<9xi32> -> tensor<3xi32>
    flow.return %1, %3, %4 : tensor<i32>, tensor<1xi32>, tensor<3xi32>
  }
  return %0#0, %0#1, %0#2 : tensor<i32>, tensor<1xi32>, tensor<3xi32>
}

// -----

// Testing inserted clones: a clone here is required as we cannot update %_large_const in-place.

// CHECK-LABEL: func @insertCloneForUpdatedConstant
func @insertCloneForUpdatedConstant(%input: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %4 = flow.ex.stream.fragment(%input) : (tensor<2x2xi32>) -> tensor<4x4xi32> =
      (%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
    %c4 = constant 4 : index
    %c1 = constant 1 : index
    // CHECK: %[[LOAD:.+]] = util.global.load @_large_const
    %5 = util.global.load @_large_const : tensor<4x4xi32>
    // CHECK: %[[CLONE:.+]] = flow.tensor.clone %[[LOAD]]
    // CHECK: flow.dispatch @pad_dispatch::@pad_dispatch[{{.+}}](%{{.+}}, %[[CLONE]])
    %6 = flow.dispatch @pad_dispatch::@pad_dispatch[%c4, %c4, %c1](%arg0, %5) : (tensor<2x2xi32>, tensor<4x4xi32>) -> %5
    flow.return %6 : tensor<4x4xi32>
  }
  return %4 : tensor<4x4xi32>
}

util.global private @_large_const {noinline} = dense<0> : tensor<4x4xi32>

// -----

// CHECK-LABEL: func @insertCloneForUpdatedConstant
func @insertCloneForUpdatedConstant(%input: tensor<2xi32>) -> tensor<7xi32> {
  %4 = flow.ex.stream.fragment(%input) : (tensor<2xi32>) -> tensor<7xi32> =
      (%arg0: tensor<2xi32>) -> tensor<7xi32> {
    %c3 = constant 3 : index
    // CHECK: %[[LOAD:.+]] = util.global.load @_large_const
    %5 = util.global.load @_large_const : tensor<7xi32>
    // CHECK: %[[CLONE:.+]] = flow.tensor.clone %[[LOAD]]
    // CHECK: flow.tensor.update %{{.+}}, %[[CLONE]]
    %6 = flow.tensor.update %arg0, %5[%c3] : tensor<2xi32> -> tensor<7xi32>
    flow.return %6 : tensor<7xi32>
  }
  return %4 : tensor<7xi32>
}

util.global private @_large_const {noinline} = dense<0> : tensor<7xi32>
