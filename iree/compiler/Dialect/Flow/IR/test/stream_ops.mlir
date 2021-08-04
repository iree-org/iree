// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

flow.executable @dispatch_0 {
  flow.dispatch.entry @rgn_dispatch_0
  module {
    func @rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.multiply %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// CHECK-LABEL: func @fragment
func @fragment(%arg0 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK: %[[WORKLOAD:.+]] = constant
  %cst = constant 4 : index
  //      CHECK: %0:2 = flow.ex.stream.fragment(%[[WORKLOAD]], %arg0) : (index, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) =
  // CHECK-NEXT: (%arg1: index, %arg2: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0:2 = flow.ex.stream.fragment(%cst, %arg0) : (index, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) =
      (%arg1 : index, %arg2 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    // CHECK-NEXT: flow.dispatch
    %1 = flow.dispatch @dispatch_0::@rgn_dispatch_0[%arg1] (%arg2) : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-NEXT: flow.return
    flow.return %1, %1 : tensor<4xf32>, tensor<4xf32>
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: return
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}

// -----

// CHECK-LABEL: func @typeChange
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x?xf32>, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
func @typeChange(%arg0: tensor<?x?xf32>, %dim0: index, %dim1: index) -> (tensor<4x?xf32>) {
  //      CHECK: %[[RET:.+]] = flow.ex.stream.fragment(%[[ARG0]], %[[DIM0]], %[[DIM1]]) :
  // CHECK-SAME:     (tensor<?x?xf32>{%[[DIM0]], %[[DIM1]]}, index, index) -> %[[ARG0]] as tensor<4x?xf32>{%[[DIM1]]} =
  // CHECK-NEXT: (%[[STREAM_ARG0:.+]]: tensor<?x?xf32>, %[[STREAM_DIM0:.+]]: index, %[[STREAM_DIM1:.+]]: index) -> tensor<4x?xf32> {
  %0 = flow.ex.stream.fragment(%arg0, %dim0, %dim1) : (tensor<?x?xf32>{%dim0, %dim1}, index, index) -> %arg0 as tensor<4x?xf32>{%dim1} =
      (%stream_arg0: tensor<?x?xf32>, %stream_dim0: index, %stream_dim1: index) -> tensor<4x?xf32> {
    // CHECK-NEXT: %[[STREAM_RET:.+]] = flow.tensor.reshape %[[STREAM_ARG0:.+]] : tensor<?x?xf32>{%[[STREAM_DIM0]], %[[STREAM_DIM1]]} -> tensor<4x?xf32>{%[[STREAM_DIM1]]}
    %1 = flow.tensor.reshape %stream_arg0 : tensor<?x?xf32>{%stream_dim0, %stream_dim1} -> tensor<4x?xf32>{%stream_dim1}
    // CHECK-NEXT: flow.return %[[STREAM_RET]] : tensor<4x?xf32>
    flow.return %1 : tensor<4x?xf32>
    // CHECK-NEXT: }
  }
  // CHECK-NEXT: return %[[RET]] : tensor<4x?xf32>
  return %0 : tensor<4x?xf32>
}
