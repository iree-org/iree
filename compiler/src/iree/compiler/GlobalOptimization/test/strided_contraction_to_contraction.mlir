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
