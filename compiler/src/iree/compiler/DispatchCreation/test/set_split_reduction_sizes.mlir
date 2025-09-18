// RUN: iree-opt %s --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-set-split-reduction-sizes))" --split-input-file > %t
// RUN: FileCheck %s < %t

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

// CHECK-LABEL: @negative_complex_reduction
util.func public @negative_complex_reduction(%arg0: tensor<256x4096xf32>, %arg1: tensor<4096x256xf32>) -> tensor<256x256xf32> {
  // More complex reductions (like matmul) shouldn't have split reduction applied.
  // CHECK-NOT: iree_linalg_ext.split_reduction
  %1 = arith.constant dense<0.0> : tensor<256x256xf32>
  %2 = linalg.generic {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, n)>,
        affine_map<(m, n, k) -> (n, k)>,
        affine_map<(m, n, k) -> (m, k)>
      ],
      iterator_types = ["parallel", "reduction", "parallel"]
  } ins(%arg0, %arg1 : tensor<256x4096xf32>, tensor<4096x256xf32>) outs(%1 : tensor<256x256xf32>) {
  ^bb0(%a: f32, %b: f32, %acc: f32):
    %3 = arith.mulf %a, %b : f32
    %4 = arith.addf %3, %acc : f32
    linalg.yield %4 : f32
  } -> tensor<256x256xf32>
  util.return %2 : tensor<256x256xf32>
}
