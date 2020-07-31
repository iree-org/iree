// RUN: iree-opt -split-input-file -iree-flow-form-streams %s | IreeFileCheck %s

// CHECK-LABEL: func @outsideTieShape
func @outsideTieShape(%arg0: tensor<?xi32> {iree.reflection = {}}, %arg1: !shapex.ranked_shape<[?]> {iree.reflection = {}}) -> (tensor<?xi32> {iree.reflection = {}}) attributes {iree.module.export} {
  // CHECK: %[[WORKLOAD0:.+]] = constant 0 : index
  %c0 = constant 0 : index
  // CHECK-NEXT: %0 = shapex.tie_shape %arg0, %arg1 : tensor<?xi32>, !shapex.ranked_shape<[?]>
  %2 = shapex.tie_shape %arg0, %arg1 : tensor<?xi32>, !shapex.ranked_shape<[?]>
  // CHECK-NEXT: %[[WORKLOAD1:.+]] = constant 1 : index
  %c1 = constant 1 : index
  // CHECK-NEXT: %1 = shapex.ranked_dim %arg1[0] : !shapex.ranked_shape<[?]> -> index
  // CHECK-NEXT: %2 = flow.ex.stream.fragment(%arg2 = %arg0 : tensor<?xi32>, %arg3 = %1 : index, %arg4 = %[[WORKLOAD0]] : index, %arg5 = %0 : tensor<?xi32>) -> tensor<?xi32> {
  // CHECK-NEXT:   %3 = shapex.make_ranked_shape %arg3 : (index) -> !shapex.ranked_shape<[?]>
  // CHECK-NEXT:   %4 = shapex.tie_shape %arg2, %3 : tensor<?xi32>, !shapex.ranked_shape<[?]>
  // CHECK-NEXT:   %5 = flow.dispatch @main_ex_dispatch_1::@main_ex_dispatch_1[%arg4 : index](%arg4, %4) : (index, tensor<?xi32>) -> tensor<?xi32>
  // CHECK-NEXT:   flow.return %5 : tensor<?xi32>
  // CHECK-NEXT: }
  %15 = flow.dispatch @main_ex_dispatch_1::@main_ex_dispatch_1[%c0 : index](%c0, %2) : (index, tensor<?xi32>) -> tensor<?xi32>
  // CHECK-NEXT: return %2 : tensor<?xi32>
  return %15 : tensor<?xi32>
}

// -----

flow.executable @outerOps_ex_dispatch_0 {
  flow.dispatch.entry @outerOps_rgn_dispatch_0 attributes {
    workload = 4 : index
  }
  module {
    func @outerOps_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func @outerOps
func @outerOps(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[WORKLOAD0:.+]] = constant 4 : index
  %cst = constant 4 : index
  // CHECK-NEXT: %0 = addf %arg0, %arg0 : tensor<4xf32>
  %0 = addf %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT: %1 = flow.ex.stream.fragment(%arg1 = %[[WORKLOAD0]] : index, %arg2 = %0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %3 = flow.dispatch @outerOps_ex_dispatch_0::@outerOps_rgn_dispatch_0[%arg1 : index](%arg2) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  %1 = flow.dispatch @outerOps_ex_dispatch_0::@outerOps_rgn_dispatch_0[%cst : index](%0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %2 = addf %1, %1 : tensor<4xf32>
  %2 = addf %1, %1 : tensor<4xf32>
  // CHECK-NEXT: return %2 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @nondependentOuterOps(
func @nondependentOuterOps(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4 : index
  %cst = constant 4 : index
  // CHECK-NEXT: %[[ADD1:.+]] = addf %arg0, %arg0 : tensor<4xf32>
  %add1 = addf %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT: %[[S:.+]] = flow.ex.stream.fragment(%arg1 = %[[WORKLOAD]] : index, %arg2 = %arg0 : tensor<4xf32>, %arg3 = %[[ADD1]] : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %[[D1:.+]] = flow.dispatch @dispatch_1::@dispatch_1[%arg1 : index](%arg2, %arg2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %d1 = flow.dispatch @dispatch_1::@dispatch_1[%cst : index](%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %[[D2:.+]] = flow.dispatch @dispatch_2::@dispatch_2[%arg1 : index](%[[D1]], %arg3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %d2 = flow.dispatch @dispatch_2::@dispatch_2[%cst : index](%d1, %add1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %[[D2]] : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[ADD2:.+]] = addf %[[S]], %arg0 : tensor<4xf32>
  %add2 = addf %d2, %arg0 : tensor<4xf32>
  // CHECK-NEXT: return %[[ADD2]] : tensor<4xf32>
  return %add2 : tensor<4xf32>
}

// -----

flow.executable @interleavedOuterOps_ex_dispatch_0 {
  flow.dispatch.entry @interleavedOuterOps_rgn_dispatch_0 attributes {
    workload = 4 : index
  }
  module {
    func @interleavedOuterOps_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func @interleavedOuterOps(
func @interleavedOuterOps(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4 : index
  %cst = constant 4 : index
  // CHECK-NEXT: %0 = flow.ex.stream.fragment(%arg1 = %[[WORKLOAD]] : index, %arg2 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %3 = flow.dispatch @interleavedOuterOps_ex_dispatch_0::@interleavedOuterOps_rgn_dispatch_0[%arg1 : index](%arg2) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  %0 = flow.dispatch @interleavedOuterOps_ex_dispatch_0::@interleavedOuterOps_rgn_dispatch_0[%cst : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %1 = addf %0, %0 : tensor<4xf32>
  %1 = addf %0, %0 : tensor<4xf32>
  // CHECK-NEXT: %2 = flow.ex.stream.fragment(%arg1 = %[[WORKLOAD]] : index, %arg2 = %1 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %3 = flow.dispatch @interleavedOuterOps_ex_dispatch_0::@interleavedOuterOps_rgn_dispatch_0[%arg1 : index](%arg2) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  %2 = flow.dispatch @interleavedOuterOps_ex_dispatch_0::@interleavedOuterOps_rgn_dispatch_0[%cst : index](%1) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: return %2 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

flow.executable @independentOps_ex_dispatch_0 {
  flow.dispatch.entry @independentOps_rgn_dispatch_0 attributes {
    workload = 4 : index
  }
  flow.dispatch.entry @independentOps_rgn_dispatch_1 attributes {
    workload = 4 : index
  }
}
// CHECK-LABEL: func @independentOps(
func @independentOps(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4 : index
  %cst = constant 4 : index
  // CHECK-NEXT: %0:2 = flow.ex.stream.fragment(%arg1 = %[[WORKLOAD]] : index, %arg2 = %arg0 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK-DAG:    = flow.dispatch @independentOps_ex_dispatch_0::@independentOps_rgn_dispatch_0[%arg1 : index](%arg2)
  %0 = flow.dispatch @independentOps_ex_dispatch_0::@independentOps_rgn_dispatch_0[%cst : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-DAG:    = flow.dispatch @independentOps_ex_dispatch_0::@independentOps_rgn_dispatch_1[%arg1 : index](%arg2)
  %1 = flow.dispatch @independentOps_ex_dispatch_0::@independentOps_rgn_dispatch_1[%cst : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %{{.+}}, %{{.+}}
  // CHECK-NEXT: }
  // CHECK-NEXT: return %{{.+}}, %{{.+}}
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// -----

flow.executable @interleavedDot_ex_dispatch_0 {
  flow.dispatch.entry @interleavedDot_rgn_dispatch_0 attributes {
    workload = 16 : index
  }
  module {
    func @interleavedDot_rgn_dispatch_0(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4x4xf32>
      return %0 : tensor<4x4xf32>
    }
  }
}
flow.executable @interleavedDot_ex_dispatch_1 {
  flow.dispatch.entry @interleavedDot_rgn_dispatch_1 attributes {
    workload = 16 : index
  }
  module {
    func @interleavedDot_rgn_dispatch_1(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
      %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      return %0 : tensor<4x4xf32>
    }
  }
}
flow.executable @interleavedDot_ex_dispatch_2 {
  flow.dispatch.entry @interleavedDot_rgn_dispatch_2 attributes {
    workload = 16 : index
  }
  module {
    func @interleavedDot_rgn_dispatch_2(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
      %0 = mhlo.multiply %arg0, %arg1 : tensor<4x4xf32>
      return %0 : tensor<4x4xf32>
    }
  }
}
// CHECK-LABEL: func @interleavedDot(
func @interleavedDot(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 16 : index
  %cst = constant 16 : index
  // CHECK-NEXT: %0 = flow.ex.stream.fragment(%arg1 = %[[WORKLOAD]] : index, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT:   %1 = flow.dispatch @interleavedDot_ex_dispatch_0::@interleavedDot_rgn_dispatch_0[%arg1 : index](%arg2) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT:   %2 = flow.dispatch @interleavedDot_ex_dispatch_1::@interleavedDot_rgn_dispatch_1[%arg1 : index](%1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT:   %3 = flow.dispatch @interleavedDot_ex_dispatch_2::@interleavedDot_rgn_dispatch_2[%arg1 : index](%2, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
  // CHECK-NEXT: }
  %0 = flow.dispatch @interleavedDot_ex_dispatch_0::@interleavedDot_rgn_dispatch_0[%cst : index](%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = flow.dispatch @interleavedDot_ex_dispatch_1::@interleavedDot_rgn_dispatch_1[%cst : index](%0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = flow.dispatch @interleavedDot_ex_dispatch_2::@interleavedDot_rgn_dispatch_2[%cst : index](%1, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT: return %0 : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// -----

flow.executable @caller_ex_dispatch_0 {
  flow.dispatch.entry @caller_rgn_dispatch_0 attributes {
    workload = 4 : index
  }
  module {
    func @caller_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
flow.executable @caller_ex_dispatch_1 {
  flow.dispatch.entry @caller_rgn_dispatch_1 attributes {
    workload = 4 : index
  }
  module {
    func @caller_rgn_dispatch_1(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.multiply %arg0, %arg1 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func @caller(
func @caller(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4 : index
  %cst = constant 4 : index
  // CHECK-NEXT: %0 = flow.ex.stream.fragment(%arg1 = %[[WORKLOAD]] : index, %arg2 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %3 = flow.dispatch @caller_ex_dispatch_0::@caller_rgn_dispatch_0[%arg1 : index](%arg2) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  %0 = flow.dispatch @caller_ex_dispatch_0::@caller_rgn_dispatch_0[%cst : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %1 = call @callee(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = call @callee(%0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %2 = flow.ex.stream.fragment(%arg1 = %[[WORKLOAD]] : index, %arg2 = %arg0 : tensor<4xf32>, %arg3 = %1 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %3 = flow.dispatch @caller_ex_dispatch_1::@caller_rgn_dispatch_1[%arg1 : index](%arg2, %arg3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  %2 = flow.dispatch @caller_ex_dispatch_1::@caller_rgn_dispatch_1[%cst : index](%arg0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: return %2 : tensor<4xf32>
  return %2 : tensor<4xf32>
}
flow.executable @callee_ex_dispatch_0 {
  flow.dispatch.entry @callee_rgn_dispatch_0
  module {
    func @callee_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.multiply %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func @callee(
func @callee(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4 : index
  %cst = constant 4 : index
  // CHECK-NEXT: %0 = flow.ex.stream.fragment(%arg1 = %[[WORKLOAD]] : index, %arg2 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %1 = flow.dispatch @callee_ex_dispatch_0::@callee_rgn_dispatch_0[%arg1 : index](%arg2) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %1 : tensor<4xf32>
  // CHECK-NEXT: }
  %0 = flow.dispatch @callee_ex_dispatch_0::@callee_rgn_dispatch_0[%cst : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: return %0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @simple_unary
// CHECK-SAME: %[[A0:[^:[:space:]]+]]: tensor<?x?xf32>
// CHECK-SAME: %[[A1:[^:[:space:]]+]]: !shapex.ranked_shape<[?,?]>
func @simple_unary(%arg0: tensor<?x?xf32>, %arg1: !shapex.ranked_shape<[?,?]>) -> (tensor<?x?xf32>, !shapex.ranked_shape<[?,?]>) {
  %0 = shapex.ranked_dim %arg1[0] : !shapex.ranked_shape<[?,?]> -> index
  %1 = shapex.ranked_dim %arg1[1] : !shapex.ranked_shape<[?,?]> -> index
  %2 = muli %0, %1 : index
  %4 = shapex.ranked_dim %arg1[0] : !shapex.ranked_shape<[?,?]> -> index
  %5 = shapex.ranked_dim %arg1[1] : !shapex.ranked_shape<[?,?]> -> index
  %3 = shapex.tie_shape %arg0, %arg1 : tensor<?x?xf32>, !shapex.ranked_shape<[?,?]>
  // Verify that the fragment captures the tie_shapes and marshals the indices
  // in as loose index values (not as ranked_shape types).
  // CHECK: %[[S:.+]] = flow.ex.stream.fragment
  // CHECK-SAME: %[[STREAM_A0:[^:[:space:]]+]] = %[[A0]] : tensor<?x?xf32>,
  // CHECK-SAME: %[[STREAM_A1:[^:[:space:]]+]] = %[[UNUSED0:[^:[:space:]]+]] : index,
  // CHECK-SAME: %[[STREAM_A2:[^:[:space:]]+]] = %[[UNUSED1:[^:[:space:]]+]] : index,
  // CHECK-SAME: {
    // CHECK: %[[STREAM_RS0:.+]] = shapex.make_ranked_shape %[[STREAM_A1]], %[[STREAM_A2]]
    // CHECK: %[[STREAM_R0:.+]] = shapex.tie_shape %[[STREAM_A0]], %[[STREAM_RS0]]
    // CHECK: %[[STREAM_R1:.+]] = flow.dispatch @simple_unary_ex_dispatch_0
    // CHECK: %[[STREAM_R2:.+]] = shapex.tie_shape %[[STREAM_R1]], %[[STREAM_RS0]]
    // CHECK: return %[[STREAM_R2]]
  // CHECK: }
  %6 = flow.dispatch @simple_unary_ex_dispatch_0::@simple_unary_ex_dispatch_0[%2 : index](%3, %4, %5) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  %7 = shapex.tie_shape %6, %arg1 : tensor<?x?xf32>, !shapex.ranked_shape<[?,?]>
  return %7, %arg1 : tensor<?x?xf32>, !shapex.ranked_shape<[?,?]>
}


// -----

// CHECK-LABEL: @bad_input_ordering
func @bad_input_ordering() -> (tensor<i32>, tensor<f32>) {
  //      CHECK: %[[W:.+]] = constant 1 : index
  %workload = constant 1 : index
  //      CHECK: %[[S:.+]] = flow.ex.stream.fragment
  // CHECK-SAME: {
  //  CHECK-DAG:   %[[D1:.+]] = flow.dispatch @dispatch_1::@dispatch_1
  //      CHECK:   flow.return
  // CHECK-NEXT: }
  %0 = flow.dispatch @dispatch_1::@dispatch_1[%workload : index]() : () -> tensor<i32>
  //      CHECK: %[[C2:.+]] = constant 2 : i32
  %c2 = constant 2 : i32
  //  CHECK-DAG:   %[[D2:.+]] = flow.dispatch @dispatch_2::@dispatch_2
  %1 = flow.dispatch @dispatch_2::@dispatch_2[%workload : index](%c2) : (i32) -> tensor<f32>
  return %0, %1 : tensor<i32>, tensor<f32>
}

// -----

// CHECK-LABEL: @interstream_readback
func @interstream_readback() -> (tensor<i32>, tensor<f32>, tensor<2xf32>) {
  //      CHECK: %[[W:.+]] = constant 1 : index
  %w = constant 1 : index
  //      CHECK: %[[S1:.+]]:2 = flow.ex.stream.fragment
  // CHECK-SAME: {
  //      CHECK:    %[[D1:.+]] = flow.dispatch @dispatch_1::@dispatch_1
  //  CHECK-DAG:    %[[D2:.+]] = flow.dispatch @dispatch_2::@dispatch_2
                    // Could be returned in either order
  // CHECK-NEXT:    flow.return
  // CHECK-NEXT: }
  %d1 = flow.dispatch @dispatch_1::@dispatch_1[%w : index]() : () -> tensor<i32>
  %d2 = flow.dispatch @dispatch_2::@dispatch_2[%w : index]() : () -> tensor<f32>
  //      CHECK: %[[READBACK:.+]] = flow.tensor.load %[[S1]]
  %readback = flow.tensor.load %d1 : tensor<i32>
  //      CHECK: %[[S2:.+]] = flow.ex.stream.fragment
  // CHECK-SAME: {
  //  CHECK-DAG:    %[[D3:.+]] = flow.dispatch @dispatch_3::@dispatch_3
  //      CHECK:    flow.return %[[D3]]
  // CHECK-NEXT: }
  %d3 = flow.dispatch @dispatch_3::@dispatch_3[%w : index](%readback) : (i32) -> tensor<2xf32>
  //      CHECK: return %[[S1]]#
  // CHECK-SAME:   %[[S1]]#
  // CHECK-SAME:   %[[S2]]
  // CHECK-SAME:   tensor<i32>, tensor<f32>, tensor<2xf32>
  return %d1, %d2, %d3 : tensor<i32>, tensor<f32>, tensor<2xf32>
}

// -----
// CHECK-LABEL: @ordering
func @ordering(%w : index) -> (tensor<i32>, tensor<f32>, tensor<i32>) {
  // CHECK: %[[C1:.+]] = constant 1
  %c1 = constant 1 : i32
  //      CHECK: %[[S1:.+]] = flow.ex.stream.fragment
  // CHECK-SAME: {
  //  CHECK-DAG:    %[[D1:.+]] = flow.dispatch @dispatch_1::@dispatch_1
  // CHECK-NEXT:    flow.return %[[D1]]
  // CHECK-NEXT: }
  %d1 = flow.dispatch @dispatch_1::@dispatch_1[%w : index](%c1) : (i32) -> (tensor<i32>)
  // CHECK: %[[SE_USER:.+]] = iree.do_not_optimize(%[[S1]])
  %side_effecting_user = iree.do_not_optimize(%d1) : tensor<i32>
  // CHECK: %[[C2:.+]] = constant 2
  %c2 = constant 2 : i32
  //      CHECK: %[[S2:.+]] = flow.ex.stream.fragment
  // CHECK-SAME: {
  //  CHECK-DAG:    %[[D2:.+]] = flow.dispatch @dispatch_2::@dispatch_2
  // CHECK-NEXT:    flow.return %[[D2]]
  // CHECK-NEXT: }
  %d2 = flow.dispatch @dispatch_2::@dispatch_2[%w : index](%c2) : (i32) -> (tensor<f32>)
  //      CHECK: return %[[S1]], %[[S2]], %[[SE_USER]]
  return %d1, %d2, %side_effecting_user : tensor<i32>, tensor<f32>, tensor<i32>
}

// -----
// CHECK-LABEL: @metadata_only
func @metadata_only(%t: tensor<?xf32>) -> (tensor<?xf32>, !shapex.ranked_shape<[?]>) {
  // CHECK-NOT: flow.ex.stream.fragment
  %c0 = constant 0 : index
  %4 = dim %t, %c0 : tensor<?xf32>
  %5 = shapex.make_ranked_shape %4 : (index) -> !shapex.ranked_shape<[?]>
  %6 = shapex.tie_shape %t, %5 : tensor<?xf32>, !shapex.ranked_shape<[?]>
  return %6, %5 : tensor<?xf32>, !shapex.ranked_shape<[?]>
}
