// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-verify-input-legality))" --verify-diagnostics %s -split-input-file

// expected-error@below {{illegal operations still remain}}
util.func public @check_no_stablehlo(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{illegal op still exists}}
  %0 = stablehlo.add %arg0, %arg1 : tensor<?x?xf32>
  // expected-error@+1 {{illegal op still exists}}
  %1 = chlo.broadcast_add %0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %1 : tensor<?x?xf32>
}

// -----

// expected-error@below {{illegal operations still remain}}
util.func public @check_no_tosa(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{illegal op still exists}}
  %0 = tosa.add %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}

// -----

// Note: checking that this is illegal even if the op could be folded. This pass
// shouldn't be modifying the IR.
// expected-error@below {{illegal operations still remain}}
util.func public @check_no_unrealized_cast(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error@+1 {{illegal op still exists}}
  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<?xf32> to memref<?xf32>
  // expected-error@+1 {{illegal op still exists}}
  %1 = builtin.unrealized_conversion_cast %0 : memref<?xf32> to tensor<?xf32>
  util.return %1 : tensor<?xf32>
}

// -----

util.func public @check_linalg_ok(%conv : tensor<1x112x112x16xf32>, %bias : tensor<16xf32>, %init : tensor<1x112x112x16xf32>) -> tensor<1x112x112x16xf32> {
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv, %bias : tensor<1x112x112x16xf32>, tensor<16xf32>)
      outs(%init : tensor<1x112x112x16xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
        %0 = arith.addf %arg0, %arg1 : f32
        linalg.yield %0 : f32
      } -> tensor<1x112x112x16xf32>
  util.return %result : tensor<1x112x112x16xf32>
}
