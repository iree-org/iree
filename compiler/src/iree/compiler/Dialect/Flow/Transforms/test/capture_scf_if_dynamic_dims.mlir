// RUN: iree-opt --split-input-file --iree-flow-capture-dynamic-dims %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-flow-capture-dynamic-dims --iree-flow-capture-dynamic-dims %s | FileCheck %s

// The second RUN line captures twice: capturing already captured dimensions
// must change nothing.

// CHECK-LABEL: @captureIfDims
// CHECK-SAME: %[[COND:[[:alnum:]]+]]
// CHECK-SAME: %[[A:[[:alnum:]]+]]
// CHECK-SAME: %[[A_DIM0:[[:alnum:]]+]]
// CHECK-SAME: %[[A_DIM2:[[:alnum:]]+]]
// CHECK-SAME: %[[B:[[:alnum:]]+]]
// CHECK-SAME: %[[B_DIM0:[[:alnum:]]+]]
// CHECK-SAME: %[[B_DIM2:[[:alnum:]]+]]
// CHECK-SAME: %[[C:[[:alnum:]]+]]
util.func public @captureIfDims(%cond: i1, %a: tensor<?x7x?xf32>, %a_dim0: index, %a_dim2: index, %b: tensor<?x7x?xf32>, %b_dim0: index, %b_dim2: index, %c: tensor<3x5xf32>) -> (tensor<?x7x?xf32>, tensor<3x5xf32>) {
  // CHECK: %[[RET:.*]]:4 = scf.if %[[COND]] -> (tensor<?x7x?xf32>, tensor<3x5xf32>, index, index)
  %ret:2 = scf.if %cond -> (tensor<?x7x?xf32>, tensor<3x5xf32>) {
    // CHECK: %[[A_TIED:.*]] = flow.tensor.tie_shape %[[A]]
    %a_tied = flow.tensor.tie_shape %a : tensor<?x7x?xf32>{%a_dim0, %a_dim2}
    // CHECK: scf.yield %[[A_TIED]], %[[C]], %[[A_DIM0]], %[[A_DIM2]]
    scf.yield %a_tied, %c : tensor<?x7x?xf32>, tensor<3x5xf32>
  } else {
    // CHECK: %[[B_TIED:.*]] = flow.tensor.tie_shape %[[B]]
    %b_tied = flow.tensor.tie_shape %b : tensor<?x7x?xf32>{%b_dim0, %b_dim2}
    // CHECK: scf.yield %[[B_TIED]], %[[C]], %[[B_DIM0]], %[[B_DIM2]]
    scf.yield %b_tied, %c : tensor<?x7x?xf32>, tensor<3x5xf32>
  }
  // CHECK: %[[RET_TIED:.*]] = flow.tensor.tie_shape %[[RET]]#0 : tensor<?x7x?xf32>{%[[RET]]#2, %[[RET]]#3}
  // CHECK-NOT: flow.tensor.tie_shape
  // CHECK: util.return %[[RET_TIED]], %[[RET]]#1
  util.return %ret#0, %ret#1 : tensor<?x7x?xf32>, tensor<3x5xf32>
}

// -----

// A branch whose dims cannot be found leaves the scf.if as it was.

// CHECK-LABEL: @unknownBranchDims
// CHECK-SAME: %[[COND:[[:alnum:]]+]]
// CHECK-SAME: %[[A:[[:alnum:]]+]]
util.func public @unknownBranchDims(%cond: i1, %a: tensor<?x6xf32>, %a_dim0: index, %b: tensor<?x6xf32>) -> tensor<?x6xf32> {
  // CHECK: %[[RET:.*]] = scf.if %[[COND]] -> (tensor<?x6xf32>)
  %ret = scf.if %cond -> tensor<?x6xf32> {
    %a_tied = flow.tensor.tie_shape %a : tensor<?x6xf32>{%a_dim0}
    scf.yield %a_tied : tensor<?x6xf32>
  } else {
    scf.yield %b : tensor<?x6xf32>
  }
  // CHECK-NOT: flow.tensor.tie_shape %[[RET]]
  // CHECK: util.return %[[RET]]
  util.return %ret : tensor<?x6xf32>
}
