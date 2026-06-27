// Tests that two scatter ops consuming the same fill-initialized buffer each
// see their own private zero-initialized buffer.
//
// This pattern can arise in MoE (Mixture of Experts) models where a single
// zeros tensor feeds two independent ScatterElements operations:
//   zeros = Expand(0.0, [B, S, num_experts])
//   layer1_mask = ScatterElements(zeros, top_k_indices_1, routing_weights_1)
//   layer2_mask = ScatterElements(zeros, top_k_indices_2, routing_weights_2)
//
// We use flow.dispatch.region to prevent IREE from fusing the fill kernel into
// each scatter kernel (which it would otherwise do, sidestepping the bug).
// With explicit dispatch boundaries the fill becomes a standalone
// stream.async.dispatch with no resource inputs (preferCloneToConsumers() ==
// true), which is exactly the pattern that triggered a CSE aliasing bug:
//
//   1. MaterializeCopyOnWrite inserts clone(fill_dispatch) for each scatter.
//   2. PropagateCloneableOps replaces clone(fill_dispatch) with a fresh
//      fill_dispatch at each use site (because preferCloneToConsumers() == true).
//   3. Without the fix: CSE merges the two identical Pure fill_dispatch calls
//      into one -> both scatters alias the same buffer -> in-place mutations
//      corrupt each other's result.
//
// Without the fix the first check in each test fails:
//   result1 = [10, 20, 30, 40]   (scatter2 also wrote here)
// instead of the correct:
//   result1 = [10, 20,  0,  0]

// Test 1: 1-D integer variant.
func.func @scatter_multi_consumer_1d() {
  %zero_val = util.unfoldable_constant 0 : i32
  %zeros = flow.dispatch.region -> (tensor<4xi32>) {
    %empty = tensor.empty() : tensor<4xi32>
    %z = linalg.fill ins(%zero_val : i32) outs(%empty : tensor<4xi32>) -> tensor<4xi32>
    flow.return %z : tensor<4xi32>
  }

  %update1 = util.unfoldable_constant dense<[10, 20]> : tensor<2xi32>
  %indices1 = util.unfoldable_constant dense<[[0], [1]]> : tensor<2x1xi32>
  %result1 = flow.dispatch.region -> (tensor<4xi32>) {
    %r = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
      ins(%update1, %indices1 : tensor<2xi32>, tensor<2x1xi32>)
      outs(%zeros : tensor<4xi32>) {
        ^bb0(%arg0: i32, %arg1: i32): iree_linalg_ext.yield %arg0 : i32
    } -> tensor<4xi32>
    flow.return %r : tensor<4xi32>
  }

  %update2 = util.unfoldable_constant dense<[30, 40]> : tensor<2xi32>
  %indices2 = util.unfoldable_constant dense<[[2], [3]]> : tensor<2x1xi32>
  %result2 = flow.dispatch.region -> (tensor<4xi32>) {
    %r = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
      ins(%update2, %indices2 : tensor<2xi32>, tensor<2x1xi32>)
      outs(%zeros : tensor<4xi32>) {
        ^bb0(%arg0: i32, %arg1: i32): iree_linalg_ext.yield %arg0 : i32
    } -> tensor<4xi32>
    flow.return %r : tensor<4xi32>
  }

  check.expect_eq_const(%result1, dense<[10, 20, 0, 0]> : tensor<4xi32>) : tensor<4xi32>
  // Without the fix: [10, 20, 30, 40]
  check.expect_eq_const(%result2, dense<[0, 0, 30, 40]> : tensor<4xi32>) : tensor<4xi32>
  return
}

// Test 2: 2-D integer variant with disjoint positions.
func.func @scatter_multi_consumer_2d() {
  %zero_val = util.unfoldable_constant 0 : i32
  %zeros = flow.dispatch.region -> (tensor<2x2xi32>) {
    %empty = tensor.empty() : tensor<2x2xi32>
    %z = linalg.fill ins(%zero_val : i32) outs(%empty : tensor<2x2xi32>) -> tensor<2x2xi32>
    flow.return %z : tensor<2x2xi32>
  }

  %update1 = util.unfoldable_constant dense<[10, 20]> : tensor<2xi32>
  %indices1 = util.unfoldable_constant dense<[[0, 0], [1, 1]]> : tensor<2x2xi32>
  %result1 = flow.dispatch.region -> (tensor<2x2xi32>) {
    %r = iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
      ins(%update1, %indices1 : tensor<2xi32>, tensor<2x2xi32>)
      outs(%zeros : tensor<2x2xi32>) {
        ^bb0(%arg0: i32, %arg1: i32): iree_linalg_ext.yield %arg0 : i32
    } -> tensor<2x2xi32>
    flow.return %r : tensor<2x2xi32>
  }

  %update2 = util.unfoldable_constant dense<[30, 40]> : tensor<2xi32>
  %indices2 = util.unfoldable_constant dense<[[0, 1], [1, 0]]> : tensor<2x2xi32>
  %result2 = flow.dispatch.region -> (tensor<2x2xi32>) {
    %r = iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
      ins(%update2, %indices2 : tensor<2xi32>, tensor<2x2xi32>)
      outs(%zeros : tensor<2x2xi32>) {
        ^bb0(%arg0: i32, %arg1: i32): iree_linalg_ext.yield %arg0 : i32
    } -> tensor<2x2xi32>
    flow.return %r : tensor<2x2xi32>
  }

  check.expect_eq_const(%result1, dense<[[10, 0], [0, 20]]> : tensor<2x2xi32>) : tensor<2x2xi32>
  // Without the fix: [[10, 30], [40, 20]]
  check.expect_eq_const(%result2, dense<[[0, 30], [40, 0]]> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

// Test 3: Float variant matching MoE routing-weights pattern.
func.func @scatter_multi_consumer_f32() {
  %zero_val = util.unfoldable_constant 0.0 : f32
  %zeros = flow.dispatch.region -> (tensor<2x4xf32>) {
    %empty = tensor.empty() : tensor<2x4xf32>
    %z = linalg.fill ins(%zero_val : f32) outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>
    flow.return %z : tensor<2x4xf32>
  }

  %update1 = util.unfoldable_constant dense<[0.7, 0.3]> : tensor<2xf32>
  %indices1 = util.unfoldable_constant dense<[[0, 0], [0, 2]]> : tensor<2x2xi32>
  %result1 = flow.dispatch.region -> (tensor<2x4xf32>) {
    %r = iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
      ins(%update1, %indices1 : tensor<2xf32>, tensor<2x2xi32>)
      outs(%zeros : tensor<2x4xf32>) {
        ^bb0(%arg0: f32, %arg1: f32): iree_linalg_ext.yield %arg0 : f32
    } -> tensor<2x4xf32>
    flow.return %r : tensor<2x4xf32>
  }

  %update2 = util.unfoldable_constant dense<[0.6, 0.4]> : tensor<2xf32>
  %indices2 = util.unfoldable_constant dense<[[1, 1], [1, 3]]> : tensor<2x2xi32>
  %result2 = flow.dispatch.region -> (tensor<2x4xf32>) {
    %r = iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
      ins(%update2, %indices2 : tensor<2xf32>, tensor<2x2xi32>)
      outs(%zeros : tensor<2x4xf32>) {
        ^bb0(%arg0: f32, %arg1: f32): iree_linalg_ext.yield %arg0 : f32
    } -> tensor<2x4xf32>
    flow.return %r : tensor<2x4xf32>
  }

  check.expect_eq_const(%result1, dense<[
    [0.7, 0.0, 0.3, 0.0],
    [0.0, 0.0, 0.0, 0.0]
  ]> : tensor<2x4xf32>) : tensor<2x4xf32>
  // Without the fix: [[0.7, 0.0, 0.3, 0.0], [0.0, 0.6, 0.0, 0.4]]
  check.expect_eq_const(%result2, dense<[
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.6, 0.0, 0.4]
  ]> : tensor<2x4xf32>) : tensor<2x4xf32>
  return
}
