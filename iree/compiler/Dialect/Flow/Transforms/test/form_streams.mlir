// RUN: iree-opt -split-input-file -iree-flow-form-streams -canonicalize -cse %s | IreeFileCheck %s

// CHECK-LABEL: func @outsideTieShape
func @outsideTieShape(%arg0: tensor<?xi32> {iree.reflection = {}}, %arg1: !shapex.ranked_shape<[?]> {iree.reflection = {}}) -> (tensor<?xi32> {iree.reflection = {}}) {
  %c0 = constant 0 : index
  // CHECK-DAG: %[[DIM:.+]] = shapex.ranked_dim %arg1[0]
  %dim = shapex.ranked_dim %arg1[0] : !shapex.ranked_shape<[?]> -> index
  // CHECK-NEXT: %[[RET:.+]] = flow.ex.stream.fragment(%[[DIM]], %arg0) : (index, tensor<?xi32>{%[[DIM]]}) -> tensor<?xi32>{%[[DIM]]} =
  // CHECK-NEXT:     (%[[INNER_DIM:.+]]: index, %[[CAPTURE:.+]]: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK-NEXT:   %[[WORKLOAD0:.+]] = constant 0 : index
  // CHECK-NEXT:   %[[INNER_RET:.+]] = flow.dispatch @main_ex_dispatch_1::@main_ex_dispatch_1[%[[WORKLOAD0]]](%[[INNER_DIM]], %[[CAPTURE]]) : (index, tensor<?xi32>{%[[INNER_DIM]]}) -> tensor<?xi32>{%[[INNER_DIM]]}
  // CHECK-NEXT:   flow.return %[[INNER_RET]] : tensor<?xi32>
  // CHECK-NEXT: }
  %15 = flow.dispatch @main_ex_dispatch_1::@main_ex_dispatch_1[%c0](%dim, %arg0) : (index, tensor<?xi32>{%dim}) -> (tensor<?xi32>{%dim})
  // CHECK-NEXT: return %[[RET]] : tensor<?xi32>
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
  // CHECK: %0 = addf %arg0, %arg0 : tensor<4xf32>
  %0 = addf %arg0, %arg0 : tensor<4xf32>
  %cst = constant 4 : index
  // CHECK-NEXT: %1 = flow.ex.stream.fragment(%0) : (tensor<4xf32>) -> tensor<4xf32> =
  // CHECK-NEXT:     (%[[INNER_ARG:.+]]: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %[[WORKLOAD:.+]] = constant 4 : index
  // CHECK-NEXT:   %3 = flow.dispatch @outerOps_ex_dispatch_0::@outerOps_rgn_dispatch_0[%[[WORKLOAD]]](%[[INNER_ARG]]) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  %1 = flow.dispatch @outerOps_ex_dispatch_0::@outerOps_rgn_dispatch_0[%cst](%0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %2 = addf %1, %1 : tensor<4xf32>
  %2 = addf %1, %1 : tensor<4xf32>
  // CHECK-NEXT: return %2 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @nondependentOuterOps(
func @nondependentOuterOps(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %cst = constant 4 : index
  // CHECK-NEXT: %[[ADD1:.+]] = addf %arg0, %arg0 : tensor<4xf32>
  %add1 = addf %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT: %[[S:.+]] = flow.ex.stream.fragment(%arg0, %[[ADD1]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> =
  // CHECK-NEXT:     (%arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %[[WORKLOAD:.+]] = constant 4 : index
  // CHECK-NEXT:   %[[D1:.+]] = flow.dispatch @dispatch_1::@dispatch_1[%[[WORKLOAD]]](%arg1, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %d1 = flow.dispatch @dispatch_1::@dispatch_1[%cst](%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %[[D2:.+]] = flow.dispatch @dispatch_2::@dispatch_2[%[[WORKLOAD]]](%[[D1]], %arg2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %d2 = flow.dispatch @dispatch_2::@dispatch_2[%cst](%d1, %add1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
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
  %cst = constant 4 : index
  // CHECK-NEXT: %0 = flow.ex.stream.fragment(%arg0) : (tensor<4xf32>) -> tensor<4xf32> =
  // CHECK-NEXT:     (%arg1: tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %[[WORKLOAD1:.+]] = constant 4 : index
  // CHECK-NEXT:   %3 = flow.dispatch @interleavedOuterOps_ex_dispatch_0::@interleavedOuterOps_rgn_dispatch_0[%[[WORKLOAD1]]](%arg1) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  %0 = flow.dispatch @interleavedOuterOps_ex_dispatch_0::@interleavedOuterOps_rgn_dispatch_0[%cst](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %1 = addf %0, %0 : tensor<4xf32>
  %1 = addf %0, %0 : tensor<4xf32>
  // CHECK-NEXT: %2 = flow.ex.stream.fragment(%1) : (tensor<4xf32>) -> tensor<4xf32> =
  // CHECK-NEXT:     (%arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %[[WORKLOAD2:.+]] = constant 4 : index
  // CHECK-NEXT:   %3 = flow.dispatch @interleavedOuterOps_ex_dispatch_0::@interleavedOuterOps_rgn_dispatch_0[%[[WORKLOAD2]]](%arg1) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  %2 = flow.dispatch @interleavedOuterOps_ex_dispatch_0::@interleavedOuterOps_rgn_dispatch_0[%cst](%1) : (tensor<4xf32>) -> tensor<4xf32>
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
  %cst = constant 4 : index
  // CHECK-NEXT: %0:2 = flow.ex.stream.fragment(%arg0) : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) =
  // CHECK-NEXT:     (%arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK-NEXT:   %[[WORKLOAD:.+]] = constant 4 : index
  // CHECK-DAG:    = flow.dispatch @independentOps_ex_dispatch_0::@independentOps_rgn_dispatch_0[%[[WORKLOAD]]](%arg1)
  %0 = flow.dispatch @independentOps_ex_dispatch_0::@independentOps_rgn_dispatch_0[%cst](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-DAG:    = flow.dispatch @independentOps_ex_dispatch_0::@independentOps_rgn_dispatch_1[%[[WORKLOAD]]](%arg1)
  %1 = flow.dispatch @independentOps_ex_dispatch_0::@independentOps_rgn_dispatch_1[%cst](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
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
  %cst = constant 16 : index
  // CHECK-NEXT: %0 = flow.ex.stream.fragment(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32> =
  // CHECK-NEXT:     (%arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT:   %[[WORKLOAD:.+]] = constant 16 : index
  // CHECK-NEXT:   %1 = flow.dispatch @interleavedDot_ex_dispatch_0::@interleavedDot_rgn_dispatch_0[%[[WORKLOAD]]](%arg1) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT:   %2 = flow.dispatch @interleavedDot_ex_dispatch_1::@interleavedDot_rgn_dispatch_1[%[[WORKLOAD]]](%1, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT:   %3 = flow.dispatch @interleavedDot_ex_dispatch_2::@interleavedDot_rgn_dispatch_2[%[[WORKLOAD]]](%2, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
  // CHECK-NEXT: }
  %0 = flow.dispatch @interleavedDot_ex_dispatch_0::@interleavedDot_rgn_dispatch_0[%cst](%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = flow.dispatch @interleavedDot_ex_dispatch_1::@interleavedDot_rgn_dispatch_1[%cst](%0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = flow.dispatch @interleavedDot_ex_dispatch_2::@interleavedDot_rgn_dispatch_2[%cst](%1, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
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
  %cst = constant 4 : index
  // CHECK-NEXT: %0 = flow.ex.stream.fragment(%arg0) : (tensor<4xf32>) -> tensor<4xf32> =
  // CHECK-NEXT:     (%arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %[[WORKLOAD1:.+]] = constant 4 : index
  // CHECK-NEXT:   %3 = flow.dispatch @caller_ex_dispatch_0::@caller_rgn_dispatch_0[%[[WORKLOAD1]]](%arg1) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  %0 = flow.dispatch @caller_ex_dispatch_0::@caller_rgn_dispatch_0[%cst](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %1 = call @callee(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = call @callee(%0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %2 = flow.ex.stream.fragment(%arg0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> =
  // CHECK-NEXT:     (%arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %[[WORKLOAD2:.+]] = constant 4 : index
  // CHECK-NEXT:   %3 = flow.dispatch @caller_ex_dispatch_1::@caller_rgn_dispatch_1[%[[WORKLOAD2]]](%arg1, %arg2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  %2 = flow.dispatch @caller_ex_dispatch_1::@caller_rgn_dispatch_1[%cst](%arg0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
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
  %cst = constant 4 : index
  // CHECK-NEXT: %0 = flow.ex.stream.fragment(%arg0) : (tensor<4xf32>) -> tensor<4xf32> =
  // CHECK-NEXT:     (%arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %[[WORKLOAD:.+]] = constant 4 : index
  // CHECK-NEXT:   %1 = flow.dispatch @callee_ex_dispatch_0::@callee_rgn_dispatch_0[%[[WORKLOAD]]](%arg1) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %1 : tensor<4xf32>
  // CHECK-NEXT: }
  %0 = flow.dispatch @callee_ex_dispatch_0::@callee_rgn_dispatch_0[%cst](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: return %0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @simple_unary
// CHECK-SAME: %[[A0:.+]]: tensor<?x?xf32>
// CHECK-SAME: %[[A1:.+]]: !shapex.ranked_shape<[?,?]>
func @simple_unary(%arg0: tensor<?x?xf32>, %arg1: !shapex.ranked_shape<[?,?]>) -> (tensor<?x?xf32>, !shapex.ranked_shape<[?,?]>) {
  // CHECK-DAG: %[[DIM0:.+]] = shapex.ranked_dim %arg1[0]
  %dim0 = shapex.ranked_dim %arg1[0] : !shapex.ranked_shape<[?,?]> -> index
  // CHECK-DAG: %[[DIM1:.+]] = shapex.ranked_dim %arg1[1]
  %dim1 = shapex.ranked_dim %arg1[1] : !shapex.ranked_shape<[?,?]> -> index
  // CHECK: %[[SZ:.+]] = muli
  %2 = muli %dim0, %dim1 : index
  // Verify that the fragment captures the tie_shapes and marshals the indices
  // in as loose index values (not as ranked_shape types).
  // CHECK: %[[S:.+]] = flow.ex.stream.fragment(%[[SZ]], %[[A0]], %[[DIM0]], %[[DIM1]]) : (index, tensor<?x?xf32>{%[[DIM0]], %[[DIM1]]}, index, index) -> tensor<?x?xf32>{%[[DIM0]], %[[DIM1]]} =
  // CHECK:     (%arg2: index, %arg3: tensor<?x?xf32>, %arg4: index, %arg5: index) -> tensor<?x?xf32> {
  // CHECK:   %[[STREAM_RET:.+]] = flow.dispatch @simple_unary_ex_dispatch_0{{.+}}[%arg2](%arg3, %arg4, %arg5) : (tensor<?x?xf32>{%arg4, %arg5}, index, index) -> tensor<?x?xf32>{%arg4, %arg5}
  // CHECK:   return %[[STREAM_RET]]
  // CHECK: }
  %3 = flow.dispatch @simple_unary_ex_dispatch_0::@simple_unary_ex_dispatch_0[%2](%arg0, %dim0, %dim1) : (tensor<?x?xf32>{%dim0, %dim1}, index, index) -> tensor<?x?xf32>{%dim0, %dim1}
  return %3, %arg1 : tensor<?x?xf32>, !shapex.ranked_shape<[?,?]>
}

// -----

// CHECK-LABEL: @bad_input_ordering
func @bad_input_ordering() -> (tensor<i32>, tensor<f32>) {
  //      CHECK: %[[S:.+]] = flow.ex.stream.fragment
  //      CHECK:   = constant 1 : index
  //      CHECK:   %[[D1:.+]] = flow.dispatch @dispatch_1::@dispatch_1
  %workload = constant 1 : index
  %0 = flow.dispatch @dispatch_1::@dispatch_1[%workload]() : () -> tensor<i32>
  //      CHECK:   %[[C2:.+]] = constant 2 : i32
  //  CHECK-DAG:   %[[D2:.+]] = flow.dispatch @dispatch_2::@dispatch_2
  %c2 = constant 2 : i32
  %1 = flow.dispatch @dispatch_2::@dispatch_2[%workload](%c2) : (i32) -> tensor<f32>
  //      CHECK:   flow.return
  return %0, %1 : tensor<i32>, tensor<f32>
}

// -----

// CHECK-LABEL: @interstream_readback
func @interstream_readback() -> (tensor<i32>, tensor<f32>, tensor<2xf32>) {
  %w = constant 1 : index
  //      CHECK: %[[S1:.+]]:2 = flow.ex.stream.fragment
  //      CHECK:   %[[W:.+]] = constant 1 : index
  //      CHECK:   %[[D1:.+]] = flow.dispatch @dispatch_1::@dispatch_1
  //  CHECK-DAG:   %[[D2:.+]] = flow.dispatch @dispatch_2::@dispatch_2
                   // Could be returned in either order
  // CHECK-NEXT:   flow.return
  %d1 = flow.dispatch @dispatch_1::@dispatch_1[%w]() : () -> tensor<i32>
  %d2 = flow.dispatch @dispatch_2::@dispatch_2[%w]() : () -> tensor<f32>
  //      CHECK: %[[READBACK:.+]] = flow.tensor.load %[[S1]]
  %readback = flow.tensor.load %d1 : tensor<i32>
  //      CHECK: %[[S2:.+]] = flow.ex.stream.fragment
  //  CHECK-DAG:    %[[D3:.+]] = flow.dispatch @dispatch_3::@dispatch_3
  //      CHECK:    flow.return %[[D3]]
  %d3 = flow.dispatch @dispatch_3::@dispatch_3[%w] (%readback) : (i32) -> tensor<2xf32>
  //      CHECK: return %[[S1]]#
  // CHECK-SAME:   %[[S1]]#
  // CHECK-SAME:   %[[S2]]
  // CHECK-SAME:   tensor<i32>, tensor<f32>, tensor<2xf32>
  return %d1, %d2, %d3 : tensor<i32>, tensor<f32>, tensor<2xf32>
}

// -----

// CHECK-LABEL: @ordering
func @ordering(%w : index) -> (tensor<i32>, tensor<f32>, tensor<i32>) {
  %c1 = constant 1 : i32
  //      CHECK: %[[S1:.+]] = flow.ex.stream.fragment
  //      CHECK:   %[[C1:.+]] = constant 1
  //  CHECK-DAG:   %[[D1:.+]] = flow.dispatch @dispatch_1::@dispatch_1
  // CHECK-NEXT:   flow.return %[[D1]]
  %d1 = flow.dispatch @dispatch_1::@dispatch_1[%w](%c1) : (i32) -> tensor<i32>
  // CHECK: %[[SE_USER:.+]] = iree.do_not_optimize(%[[S1]])
  %side_effecting_user = iree.do_not_optimize(%d1) : tensor<i32>
  %c2 = constant 2 : i32
  //      CHECK: %[[S2:.+]] = flow.ex.stream.fragment
  //      CHECK:   %[[C2:.+]] = constant 2
  //  CHECK-DAG:   %[[D2:.+]] = flow.dispatch @dispatch_2::@dispatch_2
  // CHECK-NEXT:   flow.return %[[D2]]
  %d2 = flow.dispatch @dispatch_2::@dispatch_2[%w](%c2) : (i32) -> tensor<f32>
  //      CHECK: return %[[S1]], %[[S2]], %[[SE_USER]]
  return %d1, %d2, %side_effecting_user : tensor<i32>, tensor<f32>, tensor<i32>
}

// -----

// CHECK-LABEL: @metadata_only
func @metadata_only(%t: tensor<?xf32>) -> (tensor<?xf32>, !shapex.ranked_shape<[?]>) {
  // CHECK-NOT: flow.ex.stream.fragment
  %c0 = constant 0 : index
  %4 = memref.dim %t, %c0 : tensor<?xf32>
  %5 = shapex.make_ranked_shape %4 : (index) -> !shapex.ranked_shape<[?]>
  %6 = shapex.tie_shape %t, %5 : tensor<?xf32>, !shapex.ranked_shape<[?]>
  return %6, %5 : tensor<?xf32>, !shapex.ranked_shape<[?]>
}
