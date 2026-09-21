// RUN: iree-opt --split-input-file --mlir-print-local-scope -iree-global-opt-convert-strided-contraction-to-contraction %s | FileCheck %s

util.func public @strided_from_output_static(%input: tensor<2x118x182x448xbf16>, %filter: tensor<896x448xbf16>) -> tensor<2x59x91x896xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x59x91x896xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x59x91x896xf32>) -> tensor<2x59x91x896xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, 2 * d1, d2 * 2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%input, %filter : tensor<2x118x182x448xbf16>, tensor<896x448xbf16>) outs(%1 : tensor<2x59x91x896xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %3 = arith.extf %in : bf16 to f32
    %4 = arith.extf %in_0 : bf16 to f32
    %5 = arith.mulf %3, %4 : f32
    %6 = arith.addf %out, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<2x59x91x896xf32>
  util.return %2 : tensor<2x59x91x896xf32>
}

// CHECK-LABEL: @strided_from_output_static(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<2x118x182x448xbf16>
// CHECK-SAME:      %[[FILTER:.*]]: tensor<896x448xbf16>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[INPUT]][0, 0, 0, 0] [2, 59, 91, 448] [1, 2, 2, 1]
// CHECK-SAME:      tensor<2x118x182x448xbf16> to tensor<2x59x91x448xbf16>
// CHECK: %[[GEN:.*]] = linalg.generic
// CHECK-SAME:      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// CHECK-SAME:      affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
// CHECK-SAME:      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-SAME:      ins(%[[SLICE]], %[[FILTER]]
// CHECK: util.return %[[GEN]]


// -----

util.func public @strided_from_output_dynamic_batch(%input: tensor<?x118x182x448xbf16>, %filter: tensor<896x448xbf16>) -> tensor<?x59x91x896xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %input, %c0 : tensor<?x118x182x448xbf16>
  %0 = tensor.empty(%dim) : tensor<?x59x91x896xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x59x91x896xf32>) -> tensor<?x59x91x896xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2, d2 * 2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%input, %filter : tensor<?x118x182x448xbf16>, tensor<896x448xbf16>) outs(%1 : tensor<?x59x91x896xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %3 = arith.extf %in : bf16 to f32
    %4 = arith.extf %in_0 : bf16 to f32
    %5 = arith.mulf %3, %4 : f32
    %6 = arith.addf %out, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<?x59x91x896xf32>
  util.return %2 : tensor<?x59x91x896xf32>
}

// CHECK-LABEL: @strided_from_output_dynamic_batch(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<?x118x182x448xbf16>
// CHECK-SAME:      %[[FILTER:.*]]: tensor<896x448xbf16>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[INPUT]][0, 0, 0, 0] [%[[DIM:.*]], 59, 91, 448] [1, 2, 2, 1]
// CHECK-SAME:      tensor<?x118x182x448xbf16> to tensor<?x59x91x448xbf16>
// CHECK: %[[GEN:.*]] = linalg.generic
// CHECK-SAME:      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// CHECK-SAME:      affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
// CHECK-SAME:      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-SAME:      ins(%[[SLICE]], %[[FILTER]]
// CHECK: util.return %[[GEN]]

// -----

util.func public @strided_from_output_partial_conv(%input: tensor<2x118x182x448xbf16>, %filter: tensor<896x2x448xbf16>) -> tensor<2x59x91x896xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x59x91x896xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x59x91x896xf32>) -> tensor<2x59x91x896xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2, d2 * 2 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%input, %filter : tensor<2x118x182x448xbf16>, tensor<896x2x448xbf16>) outs(%1 : tensor<2x59x91x896xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %3 = arith.extf %in : bf16 to f32
    %4 = arith.extf %in_0 : bf16 to f32
    %5 = arith.mulf %3, %4 : f32
    %6 = arith.addf %out, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<2x59x91x896xf32>
  util.return %2 : tensor<2x59x91x896xf32>
}

// CHECK-LABEL: @strided_from_output_partial_conv
// CHECK-SAME:      %[[INPUT:.*]]: tensor<2x118x182x448xbf16>
// CHECK-SAME:      %[[FILTER:.*]]: tensor<896x2x448xbf16>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[INPUT]][0, 0, 0, 0] [2, 59, 182, 448] [1, 2, 1, 1]
// CHECK-SAME:     tensor<2x118x182x448xbf16> to tensor<2x59x182x448xbf16>
// CHECK: linalg.generic
// CHECK-SAME:      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d5)>
// CHECK-SAME:      affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
// CHECK-SAME:      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-SAME:      ins(%[[SLICE]], %[[FILTER]]

// -----

util.func public @strided_from_filter_static(%input: tensor<896x118x16xbf16>, %filter: tensor<448x59x16xbf16>) -> tensor<896x448xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<896x448xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<896x448xf32>) -> tensor<896x448xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2 * 2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%input, %filter : tensor<896x118x16xbf16>, tensor<448x59x16xbf16>) outs(%1 : tensor<896x448xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %3 = arith.extf %in : bf16 to f32
    %4 = arith.extf %in_0 : bf16 to f32
    %5 = arith.mulf %3, %4 : f32
    %6 = arith.addf %out, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<896x448xf32>
  util.return %2 : tensor<896x448xf32>
}

// CHECK-LABEL: @strided_from_filter_static(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<896x118x16xbf16>
// CHECK-SAME:      %[[FILTER:.*]]: tensor<448x59x16xbf16>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[INPUT]][0, 0, 0] [896, 59, 16] [1, 2, 1]
// CHECK-SAME:      tensor<896x118x16xbf16> to tensor<896x59x16xbf16>
// CHECK: %[[GEN:.*]] = linalg.generic
// CHECK-SAME:      affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-SAME:      affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-SAME:      affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK-SAME:      ins(%[[SLICE]], %[[FILTER]]
// CHECK: util.return %[[GEN]]

// -----

// The strided input is produced by an elementwise generic (dequantization
// pattern). Factoring the strides into an extract_slice would separate the
// producer from the contraction and block elementwise fusion during codegen,
// which then materializes the producer's whole result as a dispatch-local
// buffer (see https://github.com/iree-org/iree/issues/24752). The strided
// indexing map is kept so that the producer can fuse.
util.func public @strided_from_elementwise_producer(%input: tensor<2x118x182x448xi8>) -> tensor<2x59x91x896xf32> {
  %cst = arith.constant 1.250000e-01 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x118x182x448xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%input : tensor<2x118x182x448xi8>) outs(%0 : tensor<2x118x182x448xf32>) {
  ^bb0(%in: i8, %out: f32):
    %2 = arith.uitofp %in : i8 to f32
    %3 = arith.mulf %2, %cst : f32
    linalg.yield %3 : f32
  } -> tensor<2x118x182x448xf32>
  %filter = arith.constant dense<0.000000e+00> : tensor<896x448xf32>
  %4 = tensor.empty() : tensor<2x59x91x896xf32>
  %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<2x59x91x896xf32>) -> tensor<2x59x91x896xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2, d2 * 2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1, %filter : tensor<2x118x182x448xf32>, tensor<896x448xf32>) outs(%5 : tensor<2x59x91x896xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.mulf %in, %in_0 : f32
    %8 = arith.addf %out, %7 : f32
    linalg.yield %8 : f32
  } -> tensor<2x59x91x896xf32>
  util.return %6 : tensor<2x59x91x896xf32>
}

// CHECK-LABEL: @strided_from_elementwise_producer(
// CHECK-NOT: tensor.extract_slice
// CHECK: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2, d2 * 2, d4)>
// CHECK-NOT: tensor.extract_slice
// CHECK: util.return

// -----

// A producer with a reduction loop cannot fuse elementwise into the
// contraction, so the stride factoring still applies.
util.func public @strided_from_reduction_producer(%input: tensor<2x64x118x182xf32>) -> tensor<2x59x91x896xf32> {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x118x182x448xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<2x118x182x448xf32>) -> tensor<2x118x182x448xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%input : tensor<2x64x118x182xf32>) outs(%1 : tensor<2x118x182x448xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %out, %in : f32
    linalg.yield %3 : f32
  } -> tensor<2x118x182x448xf32>
  %filter = arith.constant dense<0.000000e+00> : tensor<896x448xf32>
  %4 = tensor.empty() : tensor<2x59x91x896xf32>
  %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<2x59x91x896xf32>) -> tensor<2x59x91x896xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2, d2 * 2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%2, %filter : tensor<2x118x182x448xf32>, tensor<896x448xf32>) outs(%5 : tensor<2x59x91x896xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.mulf %in, %in_0 : f32
    %8 = arith.addf %out, %7 : f32
    linalg.yield %8 : f32
  } -> tensor<2x59x91x896xf32>
  util.return %6 : tensor<2x59x91x896xf32>
}

// CHECK-LABEL: @strided_from_reduction_producer(
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[PROD:.*]][0, 0, 0, 0] [2, 59, 91, 448] [1, 2, 2, 1]
// CHECK: linalg.generic
// CHECK-SAME: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// CHECK-SAME: ins(%[[SLICE]]
// CHECK: util.return

// -----

// The producer result has a second use that cannot fuse elementwise (the
// function return), so it is materialized regardless: the stride factoring
// is kept to preserve the projected-permutation input map.
util.func public @strided_from_multi_use_producer(%input: tensor<2x118x182x448xi8>) -> (tensor<2x59x91x896xf32>, tensor<2x118x182x448xf32>) {
  %cst = arith.constant 1.250000e-01 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x118x182x448xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%input : tensor<2x118x182x448xi8>) outs(%0 : tensor<2x118x182x448xf32>) {
  ^bb0(%in: i8, %out: f32):
    %2 = arith.uitofp %in : i8 to f32
    %3 = arith.mulf %2, %cst : f32
    linalg.yield %3 : f32
  } -> tensor<2x118x182x448xf32>
  %filter = arith.constant dense<0.000000e+00> : tensor<896x448xf32>
  %4 = tensor.empty() : tensor<2x59x91x896xf32>
  %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<2x59x91x896xf32>) -> tensor<2x59x91x896xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2, d2 * 2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1, %filter : tensor<2x118x182x448xf32>, tensor<896x448xf32>) outs(%5 : tensor<2x59x91x896xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.mulf %in, %in_0 : f32
    %8 = arith.addf %out, %7 : f32
    linalg.yield %8 : f32
  } -> tensor<2x59x91x896xf32>
  util.return %6, %1 : tensor<2x59x91x896xf32>, tensor<2x118x182x448xf32>
}

// CHECK-LABEL: @strided_from_multi_use_producer(
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[DEQ:.*]][0, 0, 0, 0] [2, 59, 91, 448] [1, 2, 2, 1]
// CHECK: linalg.generic
// CHECK-SAME: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// CHECK-SAME: ins(%[[SLICE]]
// CHECK: util.return

// -----

// Keep the strided map when all uses of the producer result can fuse
// elementwise, even when the result has multiple uses.
util.func public @strided_from_multiple_fusable_uses(%input: tensor<2x118x182x448xi8>) -> (tensor<2x59x91x896xf32>, tensor<2x118x182x448xf32>) {
  %cst = arith.constant 1.250000e-01 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x118x182x448xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%input : tensor<2x118x182x448xi8>) outs(%0 : tensor<2x118x182x448xf32>) {
  ^bb0(%in: i8, %out: f32):
    %2 = arith.uitofp %in : i8 to f32
    %3 = arith.mulf %2, %cst : f32
    linalg.yield %3 : f32
  } -> tensor<2x118x182x448xf32>
  %filter = arith.constant dense<0.000000e+00> : tensor<896x448xf32>
  %4 = tensor.empty() : tensor<2x59x91x896xf32>
  %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<2x59x91x896xf32>) -> tensor<2x59x91x896xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2, d2 * 2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1, %filter : tensor<2x118x182x448xf32>, tensor<896x448xf32>) outs(%5 : tensor<2x59x91x896xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.mulf %in, %in_0 : f32
    %8 = arith.addf %out, %7 : f32
    linalg.yield %8 : f32
  } -> tensor<2x59x91x896xf32>
  %init2 = tensor.empty() : tensor<2x118x182x448xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<2x118x182x448xf32>) outs(%init2 : tensor<2x118x182x448xf32>) {
  ^bb0(%in: f32, %out: f32):
    %10 = arith.negf %in : f32
    linalg.yield %10 : f32
  } -> tensor<2x118x182x448xf32>
  util.return %6, %9 : tensor<2x59x91x896xf32>, tensor<2x118x182x448xf32>
}

// CHECK-LABEL: @strided_from_multiple_fusable_uses(
// CHECK-NOT: tensor.extract_slice
// CHECK: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2, d2 * 2, d4)>
// CHECK-NOT: tensor.extract_slice
// CHECK: util.return
