// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-set-split-reduction-sizes))" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @basic
util.func public @basic(%arg0: tensor<4096xf32>) -> tensor<1xf32> {
  // CHECK: iree_linalg_ext.split_reduction = [1024 : index]
  %1 = arith.constant dense<0.0> : tensor<1xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (0)>],
      iterator_types = ["reduction"]
  } ins(%arg0 : tensor<4096xf32>) outs(%1 : tensor<1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<1xf32>
  util.return %2 : tensor<1xf32>
}

// -----

// CHECK-LABEL: @basic_multi_dim
util.func public @basic_multi_dim(%arg0: tensor<4x512xf32>) -> tensor<f32> {
  // With multiple reduction dims, inner dims are tiled first.
  // CHECK: iree_linalg_ext.split_reduction = [2 : index, 512 : index]
  %1 = arith.constant dense<0.0> : tensor<f32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>],
      iterator_types = ["reduction", "reduction"]
  } ins(%arg0 : tensor<4x512xf32>) outs(%1 : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<f32>
  util.return %2 : tensor<f32>
}

// -----

// CHECK-LABEL: @basic_round_split_tile_size_up
util.func public @basic_round_split_tile_size_up(%arg0: tensor<255x255xf32>) -> tensor<f32> {
  // To get the tile size to divide the iteration domain evenly, we chose a tile
  // size (5x255=1275) that exceeds the specified target tile size (1024).
  // CHECK: iree_linalg_ext.split_reduction = [5 : index, 255 : index]
  %1 = arith.constant dense<0.0> : tensor<f32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>],
      iterator_types = ["reduction", "reduction"]
  } ins(%arg0 : tensor<255x255xf32>) outs(%1 : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<f32>
  util.return %2 : tensor<f32>
}

// -----

// CHECK-LABEL: @inner_dynamic_parallel
util.func public @inner_dynamic_parallel(%arg0: tensor<4096x?xf32>, %d0: index) -> tensor<?xf32> {
  // Dynamic parallel dimensions shouldn't prevent tiling.
  // CHECK: iree_linalg_ext.split_reduction = [1024 : index]
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%d0) : tensor<?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]
  } ins(%arg0 : tensor<4096x?xf32>) outs(%1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<?xf32>
  util.return %2 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @negative_outer_dynamic_reduction
util.func public @negative_outer_dynamic_reduction(%arg0: tensor<?x64xf32>, %d0: index) -> tensor<64xf32> {
  // We bail out on dynamic reduction dimensions.
  // CHECK-NOT: iree_linalg_ext.split_reduction
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]
  } ins(%arg0 : tensor<?x64xf32>) outs(%1 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<64xf32>
  util.return %2 : tensor<64xf32>
}

// -----

// CHECK-LABEL: @arg_compare_basic
util.func public @arg_compare_basic(%arg0: tensor<4096xf32>)
    -> (tensor<f32>, tensor<index>) {
  // CHECK: iree_linalg_ext.split_reduction = [1024 : index]
  %c0f = arith.constant 0.0 : f32
  %c0i = arith.constant 0 : index

  %init_val = tensor.empty() : tensor<f32>
  %init_idx = tensor.empty() : tensor<index>
  %filled_val = linalg.fill ins(%c0f : f32)
                outs(%init_val : tensor<f32>) -> tensor<f32>
  %filled_idx = linalg.fill ins(%c0i : index)
                outs(%init_idx : tensor<index>) -> tensor<index>

  %res:2 = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%arg0 : tensor<4096xf32>)
    outs(%filled_val, %filled_idx : tensor<f32>, tensor<index>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<f32>, tensor<index>

  util.return %res#0, %res#1 : tensor<f32>, tensor<index>
}

// -----

// CHECK-LABEL: @arg_compare_inner_dynamic_parallel
util.func public @arg_compare_inner_dynamic_parallel(%arg0: tensor<4096x?xf32>, %d0: index)
    -> (tensor<?xf32>, tensor<?xindex>) {
  // Dynamic parallel dimension shouldn't prevent tiling.
  // CHECK: iree_linalg_ext.split_reduction = [1024 : index]
  %c0f = arith.constant 0.0 : f32
  %c0i = arith.constant 0 : index

  %init_val = tensor.empty(%d0) : tensor<?xf32>
  %init_idx = tensor.empty(%d0) : tensor<?xindex>
  %filled_val = linalg.fill ins(%c0f : f32)
                 outs(%init_val : tensor<?xf32>) -> tensor<?xf32>
  %filled_idx = linalg.fill ins(%c0i : index)
                 outs(%init_idx : tensor<?xindex>) -> tensor<?xindex>

  %res_val, %res_idx = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%arg0 : tensor<4096x?xf32>)
    outs(%filled_val, %filled_idx : tensor<?xf32>, tensor<?xindex>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<?xf32>, tensor<?xindex>

  util.return %res_val, %res_idx : tensor<?xf32>, tensor<?xindex>
}

// -----

// CHECK-LABEL: @arg_compare_negative_outer_dynamic_reduction
util.func public @arg_compare_negative_outer_dynamic_reduction(
    %arg0: tensor<?x64xf32>, %d0: index)
    -> (tensor<64xf32>, tensor<64xindex>) {
  // We bail out on dynamic reduction dimensions.
  // CHECK-NOT: iree_linalg_ext.split_reduction
  %c0f = arith.constant 0.0 : f32
  %c0i = arith.constant 0 : index

  %init_val = tensor.empty() : tensor<64xf32>
  %init_idx = tensor.empty() : tensor<64xindex>
  %filled_val = linalg.fill ins(%c0f : f32)
                 outs(%init_val : tensor<64xf32>) -> tensor<64xf32>
  %filled_idx = linalg.fill ins(%c0i : index)
                 outs(%init_idx : tensor<64xindex>) -> tensor<64xindex>

  %res_val, %res_idx = iree_linalg_ext.arg_compare
    dimension(0)
    ins(%arg0 : tensor<?x64xf32>)
    outs(%filled_val, %filled_idx : tensor<64xf32>, tensor<64xindex>) {
  ^bb0(%a: f32, %b: f32):
    %cmp = arith.cmpf ogt, %a, %b : f32
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<64xf32>, tensor<64xindex>

  util.return %res_val, %res_idx : tensor<64xf32>, tensor<64xindex>
}

// -----

// CHECK-LABEL: @arg_compare_large_inner_reduction
util.func public @arg_compare_large_inner_reduction(%arg0: tensor<4x1x128256xf16>)
    -> tensor<4x1xi32> {
  // CHECK: iree_linalg_ext.split_reduction = [1336 : index]
  %init_val = tensor.empty() : tensor<4x1xf16>
  %init_idx = tensor.empty() : tensor<4x1xi32>

  %res:2 = iree_linalg_ext.arg_compare
    dimension(2)
    ins(%arg0 : tensor<4x1x128256xf16>)
    outs(%init_val, %init_idx : tensor<4x1xf16>, tensor<4x1xi32>) {
  ^bb0(%arg1: f16, %arg2: f16):
    %cmp = arith.cmpf ogt, %arg1, %arg2 : f16
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<4x1xf16>, tensor<4x1xi32>

  util.return %res#1 : tensor<4x1xi32>
}

// -----

// CHECK-LABEL: @arg_compare_small_inner_reduction
util.func public @arg_compare_small_inner_reduction(%arg0: tensor<4x1x512xf16>)
    -> tensor<4x1xi32> {
  // Reduction dimension (512) is below the threshold (1024), so no split.
  // CHECK-NOT: iree_linalg_ext.split_reduction
  %init_val = tensor.empty() : tensor<4x1xf16>
  %init_idx = tensor.empty() : tensor<4x1xi32>

  %res:2 = iree_linalg_ext.arg_compare
    dimension(2)
    ins(%arg0 : tensor<4x1x512xf16>)
    outs(%init_val, %init_idx : tensor<4x1xf16>, tensor<4x1xi32>) {
  ^bb0(%arg1: f16, %arg2: f16):
    %cmp = arith.cmpf ogt, %arg1, %arg2 : f16
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<4x1xf16>, tensor<4x1xi32>

  util.return %res#1 : tensor<4x1xi32>
}

// -----

// CHECK-LABEL: @arg_compare_dynamic_inner_reduction
util.func public @arg_compare_dynamic_inner_reduction(%arg0: tensor<4x1x?xf16>, %d0: index)
    -> tensor<4x1xi32> {
  // Dynamic reduction dimension should not be tiled.
  // CHECK-NOT: iree_linalg_ext.split_reduction
  %init_val = tensor.empty() : tensor<4x1xf16>
  %init_idx = tensor.empty() : tensor<4x1xi32>

  %res:2 = iree_linalg_ext.arg_compare
    dimension(2)
    ins(%arg0 : tensor<4x1x?xf16>)
    outs(%init_val, %init_idx : tensor<4x1xf16>, tensor<4x1xi32>) {
  ^bb0(%arg1: f16, %arg2: f16):
    %cmp = arith.cmpf ogt, %arg1, %arg2 : f16
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<4x1xf16>, tensor<4x1xi32>

  util.return %res#1 : tensor<4x1xi32>
}
