// RUN: iree-opt --iree-global-opt-raise-special-ops --iree-flow-canonicalize --split-input-file --mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: @softmax
//  CHECK-SAME: %[[ARG:.+]]: tensor<?x?x?xf32>
//       CHECK:   %[[E:.+]] = tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}) : tensor<?x?x?xf32>
//       CHECK:   %[[S:.+]] = linalg.softmax dimension(2) ins(%[[ARG]] : tensor<?x?x?xf32>) outs(%[[E]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
//       CHECK:   util.return %[[S]] : tensor<?x?x?xf32>

util.func public @softmax(%src : tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant -3.40282347E+38 : f32
  %c_0_index = arith.constant 0 : index
  %c_1_index = arith.constant 1 : index
  %c_2_index = arith.constant 2 : index
  %dim_0 = tensor.dim %src, %c_0_index : tensor<?x?x?xf32>
  %dim_1 = tensor.dim %src, %c_1_index : tensor<?x?x?xf32>
  %dim_2 = tensor.dim %src, %c_2_index : tensor<?x?x?xf32>
  %1 = tensor.empty(%dim_0, %dim_1) : tensor<?x?xf32>
  %2 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src : tensor<?x?x?xf32>) outs(%2 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.maximumf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?xf32>
  %4 = tensor.empty(%dim_0, %dim_1, %dim_2) : tensor<?x?x?xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%src, %3 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %11 = arith.subf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?x?xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<?x?x?xf32>) outs(%4 : tensor<?x?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = math.exp %arg0 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?x?xf32>
  %7 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6 : tensor<?x?x?xf32>) outs(%7 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.addf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<?x?xf32>) outs(%1 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.divf %cst, %arg0 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %9 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %11 = arith.mulf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?x?xf32>
  util.return %10 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: @softmax_no_rcp
//  CHECK-SAME: %[[ARG:.+]]: tensor<10x4096x4096xf16>
//       CHECK:   %[[E:.+]] = tensor.empty() : tensor<10x4096x4096xf16>
//       CHECK:   %[[S:.+]] = linalg.softmax dimension(2) ins(%[[ARG]] : tensor<10x4096x4096xf16>) outs(%[[E]] : tensor<10x4096x4096xf16>) -> tensor<10x4096x4096xf16>
//       CHECK:   util.return %[[S]] : tensor<10x4096x4096xf16>
util.func public @softmax_no_rcp(%src : tensor<10x4096x4096xf16>) -> (tensor<10x4096x4096xf16>) {
  %cst_158 = arith.constant -6.550400e+04 : f16
  %cst_121 = arith.constant 0.000000e+00 : f16
  %224 = tensor.empty() : tensor<10x4096xf16>
  %216 = tensor.empty() : tensor<10x4096x4096xf16>
  %225 = linalg.fill ins(%cst_158 : f16) outs(%224 : tensor<10x4096xf16>) -> tensor<10x4096xf16>
  %226 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src : tensor<10x4096x4096xf16>) outs(%225 : tensor<10x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5290 = arith.maximumf %in, %out : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096xf16>
  %227 = linalg.generic
  {indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%src, %226 : tensor<10x4096x4096xf16>, tensor<10x4096xf16>) outs(%216 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %in_1572: f16, %out: f16):
    %5290 = arith.subf %in, %in_1572 : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096x4096xf16>
  %228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%227 : tensor<10x4096x4096xf16>) outs(%216 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5290 = math.exp %in : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096x4096xf16>
  %229 = tensor.empty() : tensor<10x4096xf16>
  %230 = linalg.fill ins(%cst_121 : f16) outs(%229 : tensor<10x4096xf16>) -> tensor<10x4096xf16>
  %231 = linalg.generic
  {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel", "reduction"]}
  ins(%228 : tensor<10x4096x4096xf16>) outs(%230 : tensor<10x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5290 = arith.addf %in, %out : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096xf16>
  %232 = linalg.generic
  {indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%228, %231 : tensor<10x4096x4096xf16>, tensor<10x4096xf16>) outs(%216 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %in_1572: f16, %out: f16):
    %5290 = arith.divf %in, %in_1572 : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096x4096xf16>
  util.return %232 : tensor<10x4096x4096xf16>
}

// -----

// CHECK-LABEL: @softmax_broadcast
//  CHECK-SAME: %[[ARG:.+]]: tensor<12x128x128xf32>
//       CHECK:   %[[E:.+]] = tensor.empty() : tensor<12x128x128xf32>
//       CHECK:   %[[S:.+]] = linalg.softmax dimension(2) ins(%[[ARG]] : tensor<12x128x128xf32>) outs(%[[E]] : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
//       CHECK:   util.return %[[S]] : tensor<12x128x128xf32>
util.func public @softmax_broadcast(%93 : tensor<12x128x128xf32>) -> (tensor<12x128x128xf32>) {
  %cst_16 = arith.constant 0xFF800000 : f32
  %cst_18 = arith.constant -0.000000e+00 : f32
  %94 = tensor.empty() : tensor<12x128xf32>
  %95 = linalg.fill ins(%cst_16 : f32) outs(%94 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%93 : tensor<12x128x128xf32>) outs(%95 : tensor<12x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2460 = arith.maximumf %out, %in : f32
    linalg.yield %2460 : f32
  } -> tensor<12x128xf32>
  %97 = tensor.empty() : tensor<12x128x128xf32>
  %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%96 : tensor<12x128xf32>) outs(%97 : tensor<12x128x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<12x128x128xf32>
  %99 = tensor.empty() : tensor<12x128x128xf32>
  %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93, %98 : tensor<12x128x128xf32>, tensor<12x128x128xf32>) outs(%99 : tensor<12x128x128xf32>) {
  ^bb0(%in: f32, %in_261: f32, %out: f32):
    %2460 = arith.subf %in, %in_261 : f32
    linalg.yield %2460 : f32
  } -> tensor<12x128x128xf32>
  %101 = tensor.empty() : tensor<12x128x128xf32>
  %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100 : tensor<12x128x128xf32>) outs(%101 : tensor<12x128x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2460 = math.exp %in : f32
    linalg.yield %2460 : f32
  } -> tensor<12x128x128xf32>
  %103 = tensor.empty() : tensor<12x128xf32>
  %104 = linalg.fill ins(%cst_18 : f32) outs(%103 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%102 : tensor<12x128x128xf32>) outs(%104 : tensor<12x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2460 = arith.addf %out, %in : f32
    linalg.yield %2460 : f32
  } -> tensor<12x128xf32>
  %106 = tensor.empty() : tensor<12x128x128xf32>
  %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%105 : tensor<12x128xf32>) outs(%106 : tensor<12x128x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<12x128x128xf32>
  %108 = tensor.empty() : tensor<12x128x128xf32>
  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102, %107 : tensor<12x128x128xf32>, tensor<12x128x128xf32>) outs(%108 : tensor<12x128x128xf32>) {
  ^bb0(%in: f32, %in_261: f32, %out: f32):
    %2460 = arith.divf %in, %in_261 : f32
    linalg.yield %2460 : f32
  } -> tensor<12x128x128xf32>
  util.return %109 : tensor<12x128x128xf32>
}

// -----

util.func public @aTransposeBMatmul(%arg0 : tensor<10x20xf32>,
    %arg1 : tensor<40x20xf32>) -> tensor<10x40xf32> {
  %0 = tensor.empty() : tensor<20x40xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1 : tensor<40x20xf32>) outs(%0 : tensor<20x40xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      linalg.yield %b0 : f32
  } -> tensor<20x40xf32>
  %2 = tensor.empty() : tensor<10x40xf32>
  %3 = arith.constant 0.0 : f32
  %4 = linalg.fill ins(%3 : f32) outs(%2 : tensor<10x40xf32>) -> tensor<10x40xf32>
  %5 = linalg.matmul ins(%arg0, %1 : tensor<10x20xf32>, tensor<20x40xf32>)
      outs(%4 : tensor<10x40xf32>) -> tensor<10x40xf32>
  util.return %5 : tensor<10x40xf32>
}
// CHECK-LABEL: util.func public @aTransposeBMatmul
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<40x20xf32>
//       CHECK:   %[[RESULT:.+]] = linalg.matmul_transpose_b
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @aTransposeBBatchMatmul(%arg0 : tensor<5x10x20xf32>,
    %arg1 : tensor<5x40x20xf32>) -> tensor<5x10x40xf32> {
  %0 = tensor.empty() : tensor<5x20x40xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg1 : tensor<5x40x20xf32>) outs(%0 : tensor<5x20x40xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      linalg.yield %b0 : f32
  } -> tensor<5x20x40xf32>
  %2 = tensor.empty() : tensor<5x10x40xf32>
  %3 = arith.constant 0.0 : f32
  %4 = linalg.fill ins(%3 : f32) outs(%2 : tensor<5x10x40xf32>) -> tensor<5x10x40xf32>
  %5 = linalg.batch_matmul ins(%arg0, %1 : tensor<5x10x20xf32>, tensor<5x20x40xf32>)
      outs(%4 : tensor<5x10x40xf32>) -> tensor<5x10x40xf32>
  util.return %5 : tensor<5x10x40xf32>
}
// CHECK-LABEL: util.func public @aTransposeBBatchMatmul
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<5x10x20xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<5x40x20xf32>
//       CHECK:   %[[RESULT:.+]] = linalg.batch_matmul_transpose_b
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @generic_fill(%arg0: tensor<?x?xf32>) -> tensor<1x1x?x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<1x1x?x?xf32>
    %1 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        outs(%0 : tensor<1x1x?x?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
    } -> tensor<1x1x?x?xf32>
    util.return %1 : tensor<1x1x?x?xf32>
}
// CHECK-LABEL: util.func public @generic_fill
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//       CHECK:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[EMPTY:.+]] = tensor.empty
//  CHECK-SAME:   : tensor<1x1x?x?xf32>
//       CHECK:   %[[RESULT:.+]] = linalg.fill
//  CHECK-SAME:       ins(%[[CST]] : f32)
//  CHECK-SAME:       outs(%[[EMPTY]] : tensor<1x1x?x?xf32>)
//       CHECK:   util.return %[[RESULT]]

// -----

#map = affine_map<(d0) -> (d0)>
util.func public @test_rank_reduce(%A : tensor<1x1x5120xf32>, %B : tensor<5120xf32>) -> tensor<5120xf32> {
  %c0 = arith.constant 0 : index
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%B : tensor<5120xf32>) {
  ^bb0(%out: f32):
    %12 = linalg.index 0 : index
    %extracted = tensor.extract %A[%c0, %c0, %12] : tensor<1x1x5120xf32>
    linalg.yield %extracted : f32
  } -> tensor<5120xf32>
  util.return %0 : tensor<5120xf32>
}

// CHECK-LABEL: util.func public @test_rank_reduce
//       CHECK:   tensor.extract_slice %{{.*}}[0, 0, 0] [1, 1, 5120] [1, 1, 1]
//  CHECK-SAME:     tensor<1x1x5120xf32> to tensor<5120xf32>

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
util.func public @test_slice_middle(%A : tensor<64x64x64xf32>, %B : tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%B : tensor<64x64xf32>) {
  ^bb0(%out: f32):
    %i1 = linalg.index 0 : index
    %i2 = linalg.index 1 : index
    %extracted = tensor.extract %A[%i1, %c0, %i2] : tensor<64x64x64xf32>
    linalg.yield %extracted : f32
  } -> tensor<64x64xf32>
  util.return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: util.func public @test_slice_middle
//       CHECK:   tensor.extract_slice %{{.*}}[0, 0, 0] [64, 1, 64] [1, 1, 1]
//  CHECK-SAME:     tensor<64x64x64xf32> to tensor<64x64xf32>

// -----

util.func public @test_trailing_elementwise(%arg0: tensor<180x320x1xf32>) -> tensor<320xf32> {
  %c0 = arith.constant 0 : index
  %c179 = arith.constant 179 : index
  %70 = tensor.empty() : tensor<320xf32>
  %71 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%70 : tensor<320xf32>) {
  ^bb0(%out: f32):
    %76 = linalg.index 0 : index
    %extracted = tensor.extract %arg0[%c0, %76, %c0] : tensor<180x320x1xf32>
    linalg.yield %extracted : f32
  } -> tensor<320xf32>
  util.return %71 : tensor<320xf32>
}

// CHECK-LABEL: util.func public @test_trailing_elementwise
//       CHECK:   tensor.extract_slice %{{.*}}[0, 0, 0] [1, 320, 1] [1, 1, 1]
//  CHECK-SAME:     tensor<180x320x1xf32> to tensor<320xf32>

// -----

// This currently should not be raised as the operation does not remain
// elementwise after raising the tensor.extract to input.
#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: util.func public @test_non_slice
util.func public @test_non_slice(%A : tensor<128x128x128xf32>, %B : tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  // CHECK: linalg.generic
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%B : tensor<64x64xf32>) {
  ^bb0(%out: f32):
    %i1 = linalg.index 0 : index
    %i2 = linalg.index 1 : index
    %extracted = tensor.extract %A[%i1, %c0, %i2] : tensor<128x128x128xf32>
    linalg.yield %extracted : f32
  } -> tensor<64x64xf32>
  util.return %0 : tensor<64x64xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
util.func public @test_slice_negate_cat_peephole(%arg0: tensor<1x32x1x128xf16>) -> tensor<1x32x1x128xf16> {
  %1 = tensor.empty() : tensor<1x32x1x128xf16>
  %2 = tensor.empty() : tensor<32x64xf16>
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x128xf16> to tensor<32x64xf16>
  %extracted_slice_0 = tensor.extract_slice %arg0[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x128xf16> to tensor<32x64xf16>
  %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_0 : tensor<32x64xf16>) outs(%2 : tensor<32x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.negf %in : f16
    linalg.yield %5 : f16
  } -> tensor<32x64xf16>
  %inserted_slice = tensor.insert_slice %3 into %1[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<32x64xf16> into tensor<1x32x1x128xf16>
  %inserted_slice_1 = tensor.insert_slice %extracted_slice into %inserted_slice[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<32x64xf16> into tensor<1x32x1x128xf16>
  util.return %inserted_slice_1 : tensor<1x32x1x128xf16>
}

// CHECK-LABEL: util.func public @test_slice_negate_cat_peephole
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<1x32x1x128xf16>
//       CHECK:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[EXPIN:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [1, 32, 1, 2, 64] : tensor<1x32x1x128xf16> into tensor<1x32x1x2x64xf16>
//       CHECK:   %[[NREV:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]

//       CHECK:      %[[I0:.+]] = linalg.index 0 : index
//       CHECK:      %[[I1:.+]] = linalg.index 1 : index
//       CHECK:      %[[I2:.+]] = linalg.index 2 : index
//       CHECK:      %[[I3:.+]] = linalg.index 3 : index
//       CHECK:      %[[I4:.+]] = linalg.index 4 : index
//       CHECK:      %[[R3:.+]] = arith.subi %[[C1]], %[[I3]] : index
//       CHECK:      %[[EXTR:.+]] = tensor.extract %expanded[%[[I0]], %[[I1]], %[[I2]], %[[R3]], %[[I4]]] : tensor<1x32x1x2x64xf16>
//       CHECK:      %[[NEGF:.+]] = arith.negf %[[EXTR]] : f16
//       CHECK:      %[[CMP:.+]] = arith.cmpi eq, %[[R3]], %[[C1]] : index
//       CHECK:      %[[SEL:.+]] = arith.select %[[CMP]], %[[NEGF]], %[[EXTR]] : f16
//       CHECK:      linalg.yield %[[SEL]] : f16

//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[NREV]] {{\[\[}}0], [1], [2], [3, 4]] : tensor<1x32x1x2x64xf16> into tensor<1x32x1x128xf16>
//       CHECK:   util.return %[[COLLAPSE]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
util.func public @test_slice_negate_cat_peephole_dynamic(%arg0: tensor<1x32x?x128xf16>) -> tensor<1x32x?x128xf16> {
  %c2 = arith.constant 2 : index
  %d2 = tensor.dim %arg0, %c2 : tensor<1x32x?x128xf16>
  %1 = tensor.empty(%d2) : tensor<1x32x?x128xf16>
  %2 = tensor.empty(%d2) : tensor<32x?x64xf16>
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 32, %d2, 64] [1, 1, 1, 1] : tensor<1x32x?x128xf16> to tensor<32x?x64xf16>
  %extracted_slice_0 = tensor.extract_slice %arg0[0, 0, 0, 64] [1, 32, %d2, 64] [1, 1, 1, 1] : tensor<1x32x?x128xf16> to tensor<32x?x64xf16>
  %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_0 : tensor<32x?x64xf16>) outs(%2 : tensor<32x?x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.negf %in : f16
    linalg.yield %5 : f16
  } -> tensor<32x?x64xf16>
  %inserted_slice = tensor.insert_slice %3 into %1[0, 0, 0, 0] [1, 32, %d2, 64] [1, 1, 1, 1] : tensor<32x?x64xf16> into tensor<1x32x?x128xf16>
  %inserted_slice_1 = tensor.insert_slice %extracted_slice into %inserted_slice[0, 0, 0, 64] [1, 32, %d2, 64] [1, 1, 1, 1] : tensor<32x?x64xf16> into tensor<1x32x?x128xf16>
  util.return %inserted_slice_1 : tensor<1x32x?x128xf16>
}

/// Verify that the pattern kicks in for a simple dynamic example.
// CHECK-LABEL: util.func public @test_slice_negate_cat_peephole_dynamic
//       CHECK:    tensor.expand_shape
//       CHECK:    linalg.generic
//       CHECK:      tensor.extract
//       CHECK:    %[[COL:.+]] = tensor.collapse_shape
//       CHECK:    util.return %[[COL]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
util.func public @test_slice_negate_cat_peephole_dynamic(%arg0: tensor<32x?x128xf16>) -> tensor<32x?x128xf16> {
  %c2 = arith.constant 2 : index
  %d2 = tensor.dim %arg0, %c2 : tensor<32x?x128xf16>
  %1 = tensor.empty(%d2) : tensor<32x?x128xf16>
  %2 = tensor.empty(%d2) : tensor<32x?x64xf16>
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [32, %d2, 64] [1, 1, 1] : tensor<32x?x128xf16> to tensor<32x?x64xf16>
  %extracted_slice_0 = tensor.extract_slice %arg0[0, 0, 64] [32, %d2, 64] [1, 1, 1] : tensor<32x?x128xf16> to tensor<32x?x64xf16>
  %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_0 : tensor<32x?x64xf16>) outs(%2 : tensor<32x?x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5 = arith.negf %in : f16
    linalg.yield %5 : f16
  } -> tensor<32x?x64xf16>
  %concat = tensor.concat dim(2) %3, %extracted_slice : (tensor<32x?x64xf16>, tensor<32x?x64xf16>) -> tensor<32x?x128xf16>
  util.return %concat : tensor<32x?x128xf16>
}

/// Verify that the pattern kicks in for tensor.concat as well.
// CHECK-LABEL: util.func public @test_slice_negate_cat_peephole_dynamic
//       CHECK:    tensor.expand_shape
//       CHECK:    linalg.generic
//       CHECK:      tensor.extract
//       CHECK:    %[[COL:.+]] = tensor.collapse_shape
//       CHECK:    util.return %[[COL]]

// -----

util.func public @matmul_extf(%arg0 : tensor<10x20xf32>,
                              %arg1 : tensor<20x40xf16>) -> tensor<10x40xf32> {
  %0 = tensor.empty() : tensor<20x40xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1 : tensor<20x40xf16>) outs(%0 : tensor<20x40xf32>) {
    ^bb0(%b0 : f16, %b1 : f32):
      %e = arith.extf %b0 : f16 to f32
      linalg.yield %e : f32
  } -> tensor<20x40xf32>
  %2 = tensor.empty() : tensor<10x40xf32>
  %3 = arith.constant 0.0 : f32
  %4 = linalg.fill ins(%3 : f32) outs(%2 : tensor<10x40xf32>) -> tensor<10x40xf32>
  %5 = linalg.matmul ins(%arg0, %1 : tensor<10x20xf32>, tensor<20x40xf32>)
      outs(%4 : tensor<10x40xf32>) -> tensor<10x40xf32>
  util.return %5 : tensor<10x40xf32>
}
// CHECK-LABEL: util.func public @matmul_extf
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<20x40xf16>
//       CHECK:   %[[RESULT:.+]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]]
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_extf_a(%arg0 : tensor<10x20xf16>,
                                %arg1 : tensor<20x40xf32>) -> tensor<10x40xf32> {
  %0 = tensor.empty() : tensor<10x20xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<10x20xf16>) outs(%0 : tensor<10x20xf32>) {
    ^bb0(%b0 : f16, %b1 : f32):
      %e = arith.extf %b0 : f16 to f32
      linalg.yield %e : f32
  } -> tensor<10x20xf32>
  %2 = tensor.empty() : tensor<10x40xf32>
  %3 = arith.constant 0.0 : f32
  %4 = linalg.fill ins(%3 : f32) outs(%2 : tensor<10x40xf32>) -> tensor<10x40xf32>
  %5 = linalg.matmul ins(%1, %arg1 : tensor<10x20xf32>, tensor<20x40xf32>)
      outs(%4 : tensor<10x40xf32>) -> tensor<10x40xf32>
  util.return %5 : tensor<10x40xf32>
}
// CHECK-LABEL: util.func public @matmul_extf_a
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xf16>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<20x40xf32>
//       CHECK:   %[[RESULT:.+]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]]
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_extf_both(%arg0 : tensor<10x20xf16>,
                                   %arg1 : tensor<20x40xf16>) -> tensor<10x40xf32> {
  %0 = tensor.empty() : tensor<10x20xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<10x20xf16>) outs(%0 : tensor<10x20xf32>) {
    ^bb0(%b0 : f16, %b1 : f32):
      %e = arith.extf %b0 : f16 to f32
      linalg.yield %e : f32
  } -> tensor<10x20xf32>
  %2 = tensor.empty() : tensor<20x40xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1 : tensor<20x40xf16>) outs(%2 : tensor<20x40xf32>) {
    ^bb0(%b2 : f16, %b3 : f32):
      %e1 = arith.extf %b2 : f16 to f32
      linalg.yield %e1 : f32
  } -> tensor<20x40xf32>
  %4 = tensor.empty() : tensor<10x40xf32>
  %5 = arith.constant 0.0 : f32
  %6 = linalg.fill ins(%5 : f32) outs(%4 : tensor<10x40xf32>) -> tensor<10x40xf32>
  %7 = linalg.matmul ins(%1, %3 : tensor<10x20xf32>, tensor<20x40xf32>)
      outs(%6 : tensor<10x40xf32>) -> tensor<10x40xf32>
  util.return %7 : tensor<10x40xf32>
}
// CHECK-LABEL: util.func public @matmul_extf_both
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xf16>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<20x40xf16>
//       CHECK:   %[[RESULT:.+]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]]
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @conv_nchw_extf_both(%arg0 : tensor<1x5x10x10xf16>,
                                      %arg1 : tensor<5x5x3x3xf16>) -> tensor<1x5x8x8xf32> {
  %0 = tensor.empty() : tensor<1x5x10x10xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<1x5x10x10xf16>) outs(%0 : tensor<1x5x10x10xf32>) {
    ^bb0(%b0 : f16, %b1 : f32):
      %e = arith.extf %b0 : f16 to f32
      linalg.yield %e : f32
  } -> tensor<1x5x10x10xf32>
  %2 = tensor.empty() : tensor<5x5x3x3xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg1 : tensor<5x5x3x3xf16>) outs(%2 : tensor<5x5x3x3xf32>) {
    ^bb0(%b2 : f16, %b3 : f32):
      %e1 = arith.extf %b2 : f16 to f32
      linalg.yield %e1 : f32
  } -> tensor<5x5x3x3xf32>
  %4 = tensor.empty() : tensor<1x5x8x8xf32>
  %5 = arith.constant 0.0 : f32
  %6 = linalg.fill ins(%5 : f32) outs(%4 : tensor<1x5x8x8xf32>) -> tensor<1x5x8x8xf32>
  %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%1, %3 : tensor<1x5x10x10xf32>, tensor<5x5x3x3xf32>)
      outs(%6 : tensor<1x5x8x8xf32>) -> tensor<1x5x8x8xf32>
  util.return %7 : tensor<1x5x8x8xf32>
}
// CHECK-LABEL: util.func public @conv_nchw_extf_both
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<1x5x10x10xf16>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<5x5x3x3xf16>
//       CHECK:   %[[RESULT:.+]] = linalg.conv_2d_nchw_fchw {{.*}} ins(%[[ARG0]], %[[ARG1]]
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_extsi(%arg0 : tensor<10x20xi32>,
                               %arg1 : tensor<20x40xi16>) -> tensor<10x40xi32> {
  %0 = tensor.empty() : tensor<20x40xi32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1 : tensor<20x40xi16>) outs(%0 : tensor<20x40xi32>) {
    ^bb0(%b0 : i16, %b1 : i32):
      %e = arith.extsi %b0 : i16 to i32
      linalg.yield %e : i32
  } -> tensor<20x40xi32>
  %2 = tensor.empty() : tensor<10x40xi32>
  %3 = arith.constant 0 : i32
  %4 = linalg.fill ins(%3 : i32) outs(%2 : tensor<10x40xi32>) -> tensor<10x40xi32>
  %5 = linalg.matmul ins(%arg0, %1 : tensor<10x20xi32>, tensor<20x40xi32>)
      outs(%4 : tensor<10x40xi32>) -> tensor<10x40xi32>
  util.return %5 : tensor<10x40xi32>
}
// CHECK-LABEL: util.func public @matmul_extsi
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xi32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<20x40xi16>
//       CHECK:   %[[RESULT:.+]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]]
//       CHECK:   util.return %[[RESULT]]
// -----

// Regression test. extsi is transposed, dont't fuse into matmul.
util.func public @matmul_extsi_transposed(%arg0 : tensor<10x20xi32>,
                                          %arg1 : tensor<40x20xi16>) -> tensor<10x40xi32> {
  %0 = tensor.empty() : tensor<20x40xi32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1 : tensor<40x20xi16>) outs(%0 : tensor<20x40xi32>) {
    ^bb0(%b0 : i16, %b1 : i32):
      %e = arith.extsi %b0 : i16 to i32
      linalg.yield %e : i32
  } -> tensor<20x40xi32>
  %2 = tensor.empty() : tensor<10x40xi32>
  %3 = arith.constant 0 : i32
  %4 = linalg.fill ins(%3 : i32) outs(%2 : tensor<10x40xi32>) -> tensor<10x40xi32>
  %5 = linalg.matmul ins(%arg0, %1 : tensor<10x20xi32>, tensor<20x40xi32>)
      outs(%4 : tensor<10x40xi32>) -> tensor<10x40xi32>
  util.return %5 : tensor<10x40xi32>
}
// CHECK-LABEL: util.func public @matmul_extsi_transposed
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xi32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<40x20xi16>
//       CHECK:   %[[GEN:.+]] = linalg.generic
//       CHECK:   %[[RESULT:.+]] = linalg.matmul ins(%[[ARG0]], %[[GEN]]
//       CHECK:   util.return %[[RESULT]]
// -----

util.func public @matmul_extsi_a(%arg0 : tensor<10x20xi16>,
                                 %arg1 : tensor<20x40xi32>) -> tensor<10x40xi32> {
  %0 = tensor.empty() : tensor<10x20xi32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<10x20xi16>) outs(%0 : tensor<10x20xi32>) {
    ^bb0(%b0 : i16, %b1 : i32):
      %e = arith.extsi %b0 : i16 to i32
      linalg.yield %e : i32
  } -> tensor<10x20xi32>
  %2 = tensor.empty() : tensor<10x40xi32>
  %3 = arith.constant 0 : i32
  %4 = linalg.fill ins(%3 : i32) outs(%2 : tensor<10x40xi32>) -> tensor<10x40xi32>
  %5 = linalg.matmul ins(%1, %arg1 : tensor<10x20xi32>, tensor<20x40xi32>)
      outs(%4 : tensor<10x40xi32>) -> tensor<10x40xi32>
  util.return %5 : tensor<10x40xi32>
}
// CHECK-LABEL: util.func public @matmul_extsi_a
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xi16>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<20x40xi32>
//       CHECK:   %[[RESULT:.+]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]]
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_extsi_both(%arg0 : tensor<10x20xi16>,
                                    %arg1 : tensor<20x40xi16>) -> tensor<10x40xi32> {
  %0 = tensor.empty() : tensor<10x20xi32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<10x20xi16>) outs(%0 : tensor<10x20xi32>) {
    ^bb0(%b0 : i16, %b1 : i32):
      %e = arith.extsi %b0 : i16 to i32
      linalg.yield %e : i32
  } -> tensor<10x20xi32>
  %2 = tensor.empty() : tensor<20x40xi32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1 : tensor<20x40xi16>) outs(%2 : tensor<20x40xi32>) {
    ^bb0(%b2 : i16, %b3 : i32):
      %e1 = arith.extsi %b2 : i16 to i32
      linalg.yield %e1 : i32
  } -> tensor<20x40xi32>
  %4 = tensor.empty() : tensor<10x40xi32>
  %5 = arith.constant 0 : i32
  %6 = linalg.fill ins(%5 : i32) outs(%4 : tensor<10x40xi32>) -> tensor<10x40xi32>
  %7 = linalg.matmul ins(%1, %3 : tensor<10x20xi32>, tensor<20x40xi32>)
      outs(%6 : tensor<10x40xi32>) -> tensor<10x40xi32>
  util.return %7 : tensor<10x40xi32>
}
// CHECK-LABEL: util.func public @matmul_extsi_both
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xi16>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<20x40xi16>
//       CHECK:   %[[RESULT:.+]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]]
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @conv_nchw_extsi_both(%arg0 : tensor<1x5x10x10xi16>,
                                       %arg1 : tensor<5x5x3x3xi16>) -> tensor<1x5x8x8xi32> {
  %0 = tensor.empty() : tensor<1x5x10x10xi32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<1x5x10x10xi16>) outs(%0 : tensor<1x5x10x10xi32>) {
    ^bb0(%b0 : i16, %b1 : i32):
      %e = arith.extsi %b0 : i16 to i32
      linalg.yield %e : i32
  } -> tensor<1x5x10x10xi32>
  %2 = tensor.empty() : tensor<5x5x3x3xi32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg1 : tensor<5x5x3x3xi16>) outs(%2 : tensor<5x5x3x3xi32>) {
    ^bb0(%b2 : i16, %b3 : i32):
      %e1 = arith.extsi %b2 : i16 to i32
      linalg.yield %e1 : i32
  } -> tensor<5x5x3x3xi32>
  %4 = tensor.empty() : tensor<1x5x8x8xi32>
  %5 = arith.constant 0 : i32
  %6 = linalg.fill ins(%5 : i32) outs(%4 : tensor<1x5x8x8xi32>) -> tensor<1x5x8x8xi32>
  %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%1, %3 : tensor<1x5x10x10xi32>, tensor<5x5x3x3xi32>)
      outs(%6 : tensor<1x5x8x8xi32>) -> tensor<1x5x8x8xi32>
  util.return %7 : tensor<1x5x8x8xi32>
}
// CHECK-LABEL: util.func public @conv_nchw_extsi_both
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<1x5x10x10xi16>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<5x5x3x3xi16>
//       CHECK:   %[[RESULT:.+]] = linalg.conv_2d_nchw_fchw {{.*}} ins(%[[ARG0]], %[[ARG1]]
//       CHECK:   util.return %[[RESULT]]

// -----

// Can't fuse exti with unsigned ops because internally they use extui
util.func public @unsigned_matmul_extsi(%arg0 : tensor<10x20xi32>,
                               %arg1 : tensor<20x40xi16>) -> tensor<10x40xi32> {
  %0 = tensor.empty() : tensor<20x40xi32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1 : tensor<20x40xi16>) outs(%0 : tensor<20x40xi32>) {
    ^bb0(%b0 : i16, %b1 : i32):
      %e = arith.extsi %b0 : i16 to i32
      linalg.yield %e : i32
  } -> tensor<20x40xi32>
  %2 = tensor.empty() : tensor<10x40xi32>
  %3 = arith.constant 0 : i32
  %4 = linalg.fill ins(%3 : i32) outs(%2 : tensor<10x40xi32>) -> tensor<10x40xi32>
  %5 = linalg.matmul {cast = #linalg.type_fn<cast_unsigned>} ins(%arg0, %1 : tensor<10x20xi32>, tensor<20x40xi32>)
      outs(%4 : tensor<10x40xi32>) -> tensor<10x40xi32>
  util.return %5 : tensor<10x40xi32>
}
// CHECK-LABEL: util.func public @unsigned_matmul_extsi
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xi32>
//       CHECK:   %[[GEN:.+]] = linalg.generic
//       CHECK:   %[[RESULT:.+]] = linalg.matmul {cast = #linalg.type_fn<cast_unsigned>} ins(%[[ARG0]], %[[GEN]]
//       CHECK:   util.return %[[RESULT]]

// -----

util.func public @constant_pad_i8(%arg0: tensor<10x20xi8>) -> tensor<13x23xi8> {
  %cst = arith.constant dense<1> : tensor<13x23xi8>
  %1 = tensor.insert_slice %arg0 into %cst[1, 2] [10, 20] [1, 1] : tensor<10x20xi8> into tensor<13x23xi8>
  util.return %1 : tensor<13x23xi8>
}
// CHECK-LABEL: util.func public @constant_pad_i8
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xi8>
//       CHECK:   %[[C1:.+]] = arith.constant 1 : i8
//       CHECK:   %[[PAD:.+]] = tensor.pad %[[ARG0]] low[1, 2] high[2, 1]
//       CHECK:     tensor.yield %[[C1]]
//       CHECK:   util.return %[[PAD]]

// -----

util.func public @constant_pad_f32(%arg0: tensor<?x?xf32>, %x: index, %y: index) -> tensor<13x23xf32> {
  %cst = arith.constant dense<1.0> : tensor<13x23xf32>
  %1 = tensor.insert_slice %arg0 into %cst[1, 2] [%x, %y] [1, 1] : tensor<?x?xf32> into tensor<13x23xf32>
  util.return %1 : tensor<13x23xf32>
}
// CHECK-LABEL: util.func public @constant_pad_f32
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[X:[A-Za-z0-9]+]]: index
//  CHECK-SAME:     %[[Y:[A-Za-z0-9]+]]: index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1.000000e+00 : f32
//   CHECK-DAG:   %[[H0:.+]] = affine.apply affine_map<()[s0] -> (-s0 + 12)>()[%[[X]]]
//   CHECK-DAG:   %[[H1:.+]] = affine.apply affine_map<()[s0] -> (-s0 + 21)>()[%[[Y]]]
//       CHECK:   %[[PAD:.+]] = tensor.pad %[[ARG0]] low[1, 2] high[%[[H0]], %[[H1]]]
//       CHECK:     tensor.yield %[[C1]]
//       CHECK:   util.return %[[PAD]]
