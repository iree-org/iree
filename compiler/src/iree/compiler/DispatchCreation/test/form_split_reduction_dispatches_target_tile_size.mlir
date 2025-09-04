// RUN: iree-opt %s --verify-diagnostics --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-split-reduction-dispatches{target-split-reduction-size=1024 emit-remarks=true}))" --split-input-file

util.func public @basic(%arg0: tensor<4096xf32>) -> tensor<f32> {
  %1 = arith.constant dense<0.0> : tensor<f32>
  // expected-remark@below {{forming split reduction dispatch with tile sizes: [1024 : index]}}
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
  } ins(%arg0 : tensor<4096xf32>) outs(%1 : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<f32>
  util.return %2 : tensor<f32>
}

// -----

util.func public @multi_dim(%arg0: tensor<4x512xf32>) -> tensor<f32> {
  // With multiple reduction dims, inner dims are tiled first.
  %1 = arith.constant dense<0.0> : tensor<f32>
  // expected-remark@below {{forming split reduction dispatch with tile sizes: [2 : index, 512 : index]}}
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

util.func public @round_split_tile_size_up(%arg0: tensor<255x255xf32>) -> tensor<f32> {
  // To get the tile size to divide the iteration domain evenly, we chose a tile
  // size (5x255=1275) that exceeds the specified target tile size (1024).
  %1 = arith.constant dense<0.0> : tensor<f32>
  // expected-remark@below {{forming split reduction dispatch with tile sizes: [5 : index, 255 : index]}}
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

util.func public @dynamic_parallel(%arg0: tensor<?x4096xf32>, %d0: index) -> tensor<?xf32> {
  // Dynamic parallel dimensions shouldn't prevent tiling.
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%d0) : tensor<?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  // expected-remark@below {{forming split reduction dispatch with tile sizes: [1024 : index]}}
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<?x4096xf32>) outs(%1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<?xf32>
  util.return %2 : tensor<?xf32>
}

// -----

util.func public @negative_dynamic_reduction(%arg0: tensor<64x?xf32>, %d0: index) -> tensor<64xf32> {
  // We bail out on dynamic reduction dimensions.
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  // expected-remark@below {{failed to infer split-reduction tile sizes due to a reduction dim (dim 1) having a dynamic size}}
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<64x?xf32>) outs(%1 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<64xf32>
  util.return %2 : tensor<64xf32>
}
