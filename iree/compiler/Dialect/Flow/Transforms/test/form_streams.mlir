// RUN: iree-opt -split-input-file -iree-flow-form-streams %s | IreeFileCheck %s

flow.executable @outerOps_ex_dispatch_0 {
  flow.dispatch.entry @outerOps_rgn_dispatch_0 attributes {
    workload = 4 : index
  }
  module {
    func @outerOps_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
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

flow.executable @nondependentOuterOps_ex_dispatch_0 {
  flow.dispatch.entry @nondependentOuterOps_rgn_dispatch_0 attributes {
    workload = 4 : index
  }
  module {
    func @nondependentOuterOps_rgn_dispatch_0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
      %0 = xla_hlo.add %arg0, %arg1 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func @nondependentOuterOps(
func @nondependentOuterOps(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4 : index
  %cst = constant 4 : index
  %0 = flow.dispatch @nondependentOuterOps_ex_dispatch_0::@nondependentOuterOps_rgn_dispatch_0[%cst : index](%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %0 = addf %arg0, %arg0 : tensor<4xf32>
  %1 = addf %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT: %1 = flow.ex.stream.fragment(%arg1 = %[[WORKLOAD]] : index, %arg2 = %arg0 : tensor<4xf32>, %arg3 = %0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %3 = flow.dispatch @nondependentOuterOps_ex_dispatch_0::@nondependentOuterOps_rgn_dispatch_0[%arg1 : index](%arg2, %arg2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %4 = flow.dispatch @nondependentOuterOps_ex_dispatch_0::@nondependentOuterOps_rgn_dispatch_0[%arg1 : index](%3, %arg3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   flow.return %4 : tensor<4xf32>
  // CHECK-NEXT: }
  %2 = flow.dispatch @nondependentOuterOps_ex_dispatch_0::@nondependentOuterOps_rgn_dispatch_0[%cst : index](%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %2 = addf %1, %arg0 : tensor<4xf32>
  %3 = addf %2, %arg0 : tensor<4xf32>
  // CHECK-NEXT: return %2 : tensor<4xf32>
  return %3 : tensor<4xf32>
}

// -----

flow.executable @interleavedOuterOps_ex_dispatch_0 {
  flow.dispatch.entry @interleavedOuterOps_rgn_dispatch_0 attributes {
    workload = 4 : index
  }
  module {
    func @interleavedOuterOps_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
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
  // CHECK-NEXT:   flow.return {{%.+}}, {{%.+}}
  // CHECK-NEXT: }
  // CHECK-NEXT: return {{%.+}}, {{%.+}}
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// -----

flow.executable @interleavedDot_ex_dispatch_0 {
  flow.dispatch.entry @interleavedDot_rgn_dispatch_0 attributes {
    workload = 16 : index
  }
  module {
    func @interleavedDot_rgn_dispatch_0(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
      %0 = xla_hlo.add %arg0, %arg0 : tensor<4x4xf32>
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
      %0 = "xla_hlo.dot"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
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
      %0 = xla_hlo.multiply %arg0, %arg1 : tensor<4x4xf32>
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
      %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
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
      %0 = xla_hlo.multiply %arg0, %arg1 : tensor<4xf32>
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
      %0 = xla_hlo.multiply %arg0, %arg0 : tensor<4xf32>
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
  %3 = shapex.tie_shape %arg0, %arg1 : tensor<?x?xf32>, !shapex.ranked_shape<[?,?]>
  %4 = shapex.ranked_dim %arg1[0] : !shapex.ranked_shape<[?,?]> -> index
  %5 = shapex.ranked_dim %arg1[1] : !shapex.ranked_shape<[?,?]> -> index
  // Verify that the fragment captures the tie_shapes and marshals the indices
  // in as loose index values (not as ranked_shape types).
  // CHECK: %7 = flow.ex.stream.fragment
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
