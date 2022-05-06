// RUN: iree-dialects-opt --linalg-interp-transforms %s | FileCheck %s

func @fun_to_benchmark(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) ->
    tensor<128x128xf32> attributes {passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  // With scalarization we expect vectorization to still work albeit with a leading
  // `1` dimension.
  // CHECK: vector.contract {{.*}} : vector<1x32xf32>, vector<32x16xf32> into vector<1x16xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                    outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

pdl.pattern @isa_linalg.matmul : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation "linalg.matmul"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  rewrite %2 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @isa_linalg.matmul
  %tiled_linalg_op, %loops:3 = tile %0 {interchange = [1, 0, 2], sizes = [6, 16, 32]}
  %1 = peel_loop %loops#0
  // This test checks the proper handling of the scalarize dims attribute.
  // The first dimension does not divide but we can always scalarize a `?` into `1`
  // and enable vectorization of a lower-rank op this way.
  %tiled_linalg_op_0 = scalarize %tiled_linalg_op
  vectorize {vectorize_padding = false}
}
