// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-insert-batch-dim-for-batchless-conv))" --split-input-file %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>

func.func @batchless_conv_2d_nhwc_hwcf(%input: tensor<16x16x4xf32>, %filter: tensor<3x3x4x16xf32>, %output: tensor<14x14x16xf32>) -> tensor<14x14x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%input, %filter : tensor<16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%output : tensor<14x14x16xf32>) {
  ^bb0(%in: f32, %f: f32, %out: f32):
    %mul = arith.mulf %in, %f : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<14x14x16xf32>
  return %0 : tensor<14x14x16xf32>
}

// CHECK-LABEL: func.func @batchless_conv_2d_nhwc_hwcf
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<16x16x4xf32>
// CHECK-SAME:    %[[FILTER:[a-zA-Z0-9]+]]: tensor<3x3x4x16xf32>
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]+]]: tensor<14x14x16xf32>
// CHECK:       %[[EXPANDED_INPUT:.+]] = tensor.expand_shape %[[INPUT]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [1, 16, 16, 4] : tensor<16x16x4xf32> into tensor<1x16x16x4xf32>
// CHECK:       %[[EXPANDED_OUTPUT:.+]] = tensor.expand_shape %[[OUTPUT]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [1, 14, 14, 16] : tensor<14x14x16xf32> into tensor<1x14x14x16xf32>
// CHECK:       %[[CONV:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK-SAME:    ins(%[[EXPANDED_INPUT]], %[[FILTER]] : tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
// CHECK-SAME:    outs(%[[EXPANDED_OUTPUT]] : tensor<1x14x14x16xf32>)
// CHECK:       %[[RESULT:.+]] = tensor.collapse_shape %[[CONV]] {{\[}}[0, 1], [2], [3]{{\]}} : tensor<1x14x14x16xf32> into tensor<14x14x16xf32>
// CHECK:       return %[[RESULT]] : tensor<14x14x16xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0 + d4, d1 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d1)>

func.func @batchless_conv_2d_nchw_fchw(%input: tensor<16x18x18xf32>, %filter: tensor<32x16x3x3xf32>, %output: tensor<32x16x16xf32>) -> tensor<32x16x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%input, %filter : tensor<16x18x18xf32>, tensor<32x16x3x3xf32>)
    outs(%output : tensor<32x16x16xf32>) {
  ^bb0(%in: f32, %f: f32, %out: f32):
    %mul = arith.mulf %in, %f : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<32x16x16xf32>
  return %0 : tensor<32x16x16xf32>
}

// CHECK-LABEL: func.func @batchless_conv_2d_nchw_fchw
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<16x18x18xf32>
// CHECK-SAME:    %[[FILTER:[a-zA-Z0-9]+]]: tensor<32x16x3x3xf32>
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]+]]: tensor<32x16x16xf32>
// CHECK:       %[[EXPANDED_INPUT:.+]] = tensor.expand_shape %[[INPUT]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [1, 16, 18, 18] : tensor<16x18x18xf32> into tensor<1x16x18x18xf32>
// CHECK:       %[[EXPANDED_OUTPUT:.+]] = tensor.expand_shape %[[OUTPUT]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [1, 32, 16, 16] : tensor<32x16x16xf32> into tensor<1x32x16x16xf32>
// CHECK:       %[[CONV:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK-SAME:    ins(%[[EXPANDED_INPUT]], %[[FILTER]] : tensor<1x16x18x18xf32>, tensor<32x16x3x3xf32>)
// CHECK-SAME:    outs(%[[EXPANDED_OUTPUT]] : tensor<1x32x16x16xf32>)
// CHECK:       %[[RESULT:.+]] = tensor.collapse_shape %[[CONV]] {{\[}}[0, 1], [2], [3]{{\]}} : tensor<1x32x16x16xf32> into tensor<32x16x16xf32>
// CHECK:       return %[[RESULT]] : tensor<32x16x16xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0 + d3, d1 + d4, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

func.func @batchless_pooling_nhwc_sum(%input: tensor<18x18x16xf32>, %filter: tensor<3x3xf32>, %output: tensor<16x16x16xf32>) -> tensor<16x16x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<18x18x16xf32>, tensor<3x3xf32>)
    outs(%output : tensor<16x16x16xf32>) {
  ^bb0(%in: f32, %f: f32, %out: f32):
    %add = arith.addf %out, %in : f32
    linalg.yield %add : f32
  } -> tensor<16x16x16xf32>
  return %0 : tensor<16x16x16xf32>
}

// CHECK-LABEL: func.func @batchless_pooling_nhwc_sum
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<18x18x16xf32>
// CHECK-SAME:    %[[FILTER:[a-zA-Z0-9]+]]: tensor<3x3xf32>
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]+]]: tensor<16x16x16xf32>
// CHECK:       %[[EXPANDED_INPUT:.+]] = tensor.expand_shape %[[INPUT]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [1, 18, 18, 16] : tensor<18x18x16xf32> into tensor<1x18x18x16xf32>
// CHECK:       %[[EXPANDED_OUTPUT:.+]] = tensor.expand_shape %[[OUTPUT]] {{\[}}[0, 1], [2], [3]{{\]}} output_shape [1, 16, 16, 16] : tensor<16x16x16xf32> into tensor<1x16x16x16xf32>
// CHECK:       %[[POOL:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:    ins(%[[EXPANDED_INPUT]], %[[FILTER]] : tensor<1x18x18x16xf32>, tensor<3x3xf32>)
// CHECK-SAME:    outs(%[[EXPANDED_OUTPUT]] : tensor<1x16x16x16xf32>)
// CHECK:       %[[RESULT:.+]] = tensor.collapse_shape %[[POOL]] {{\[}}[0, 1], [2], [3]{{\]}} : tensor<1x16x16x16xf32> into tensor<16x16x16xf32>
// CHECK:       return %[[RESULT]] : tensor<16x16x16xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0 + d3, d1 + d4, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

func.func @batchless_pooling_nhwc_max(%input: tensor<18x18x16xf32>, %filter: tensor<3x3xf32>, %output: tensor<16x16x16xf32>) -> tensor<16x16x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<18x18x16xf32>, tensor<3x3xf32>)
    outs(%output : tensor<16x16x16xf32>) {
  ^bb0(%in: f32, %f: f32, %out: f32):
    %max = arith.maximumf %out, %in : f32
    linalg.yield %max : f32
  } -> tensor<16x16x16xf32>
  return %0 : tensor<16x16x16xf32>
}

// CHECK-LABEL: func.func @batchless_pooling_nhwc_max
// CHECK:       tensor.expand_shape {{.*}} : tensor<18x18x16xf32> into tensor<1x18x18x16xf32>
// CHECK:       tensor.expand_shape {{.*}} : tensor<16x16x16xf32> into tensor<1x16x16x16xf32>
// CHECK:       %[[POOL:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK:       tensor.collapse_shape %[[POOL]] {{.*}} : tensor<1x16x16x16xf32> into tensor<16x16x16xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0 + d3, d1 + d4, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

func.func @batchless_pooling_nhwc_max_unsigned(%input: tensor<18x18x16xi32>, %filter: tensor<3x3xi32>, %output: tensor<16x16x16xi32>) -> tensor<16x16x16xi32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<18x18x16xi32>, tensor<3x3xi32>)
    outs(%output : tensor<16x16x16xi32>) {
  ^bb0(%in: i32, %f: i32, %out: i32):
    %max = arith.maxui %out, %in : i32
    linalg.yield %max : i32
  } -> tensor<16x16x16xi32>
  return %0 : tensor<16x16x16xi32>
}

// CHECK-LABEL: func.func @batchless_pooling_nhwc_max_unsigned
// CHECK:       tensor.expand_shape {{.*}} : tensor<18x18x16xi32> into tensor<1x18x18x16xi32>
// CHECK:       tensor.expand_shape {{.*}} : tensor<16x16x16xi32> into tensor<1x16x16x16xi32>
// CHECK:       %[[POOL:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK:       tensor.collapse_shape %[[POOL]] {{.*}} : tensor<1x16x16x16xi32> into tensor<16x16x16xi32>

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0 + d3, d1 + d4, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

func.func @batchless_pooling_nhwc_min(%input: tensor<18x18x16xf32>, %filter: tensor<3x3xf32>, %output: tensor<16x16x16xf32>) -> tensor<16x16x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<18x18x16xf32>, tensor<3x3xf32>)
    outs(%output : tensor<16x16x16xf32>) {
  ^bb0(%in: f32, %f: f32, %out: f32):
    %min = arith.minimumf %out, %in : f32
    linalg.yield %min : f32
  } -> tensor<16x16x16xf32>
  return %0 : tensor<16x16x16xf32>
}

// CHECK-LABEL: func.func @batchless_pooling_nhwc_min
// CHECK:       tensor.expand_shape {{.*}} : tensor<18x18x16xf32> into tensor<1x18x18x16xf32>
// CHECK:       tensor.expand_shape {{.*}} : tensor<16x16x16xf32> into tensor<1x16x16x16xf32>
// CHECK:       %[[POOL:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK:       tensor.collapse_shape %[[POOL]] {{.*}} : tensor<1x16x16x16xf32> into tensor<16x16x16xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0 + d3, d1 + d4, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

func.func @batchless_pooling_nhwc_min_unsigned(%input: tensor<18x18x16xi32>, %filter: tensor<3x3xi32>, %output: tensor<16x16x16xi32>) -> tensor<16x16x16xi32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<18x18x16xi32>, tensor<3x3xi32>)
    outs(%output : tensor<16x16x16xi32>) {
  ^bb0(%in: i32, %f: i32, %out: i32):
    %min = arith.minui %out, %in : i32
    linalg.yield %min : i32
  } -> tensor<16x16x16xi32>
  return %0 : tensor<16x16x16xi32>
}

// CHECK-LABEL: func.func @batchless_pooling_nhwc_min_unsigned
// CHECK:       tensor.expand_shape {{.*}} : tensor<18x18x16xi32> into tensor<1x18x18x16xi32>
// CHECK:       tensor.expand_shape {{.*}} : tensor<16x16x16xi32> into tensor<1x16x16x16xi32>
// CHECK:       %[[POOL:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK:       tensor.collapse_shape %[[POOL]] {{.*}} : tensor<1x16x16x16xi32> into tensor<16x16x16xi32>

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0 + d3, d1 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d1)>

func.func @batchless_pooling_nchw_sum(%input: tensor<16x18x18xf32>, %filter: tensor<3x3xf32>, %output: tensor<16x16x16xf32>) -> tensor<16x16x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<16x18x18xf32>, tensor<3x3xf32>)
    outs(%output : tensor<16x16x16xf32>) {
  ^bb0(%in: f32, %f: f32, %out: f32):
    %add = arith.addf %out, %in : f32
    linalg.yield %add : f32
  } -> tensor<16x16x16xf32>
  return %0 : tensor<16x16x16xf32>
}

// CHECK-LABEL: func.func @batchless_pooling_nchw_sum
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<16x18x18xf32>
// CHECK-SAME:    %[[FILTER:[a-zA-Z0-9]+]]: tensor<3x3xf32>
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]+]]: tensor<16x16x16xf32>
// CHECK:       tensor.expand_shape {{.*}} output_shape [1, 16, 18, 18] : tensor<16x18x18xf32> into tensor<1x16x18x18xf32>
// CHECK:       tensor.expand_shape {{.*}} output_shape [1, 16, 16, 16] : tensor<16x16x16xf32> into tensor<1x16x16x16xf32>
// CHECK:       %[[POOL:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK:       tensor.collapse_shape %[[POOL]] {{.*}} : tensor<1x16x16x16xf32> into tensor<16x16x16xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0 + d3, d1 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d1)>

func.func @batchless_pooling_nchw_max(%input: tensor<16x18x18xf32>, %filter: tensor<3x3xf32>, %output: tensor<16x16x16xf32>) -> tensor<16x16x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<16x18x18xf32>, tensor<3x3xf32>)
    outs(%output : tensor<16x16x16xf32>) {
  ^bb0(%in: f32, %f: f32, %out: f32):
    %max = arith.maximumf %out, %in : f32
    linalg.yield %max : f32
  } -> tensor<16x16x16xf32>
  return %0 : tensor<16x16x16xf32>
}

// CHECK-LABEL: func.func @batchless_pooling_nchw_max
// CHECK:       tensor.expand_shape {{.*}} : tensor<16x18x18xf32> into tensor<1x16x18x18xf32>
// CHECK:       tensor.expand_shape {{.*}} : tensor<16x16x16xf32> into tensor<1x16x16x16xf32>
// CHECK:       %[[POOL:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK:       tensor.collapse_shape %[[POOL]] {{.*}} : tensor<1x16x16x16xf32> into tensor<16x16x16xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0 + d3, d1 + d4, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

func.func @batchless_depthwise_conv_2d_nhwc_hwc(%input: tensor<18x18x16xf32>, %filter: tensor<3x3x16xf32>, %output: tensor<16x16x16xf32>) -> tensor<16x16x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%input, %filter : tensor<18x18x16xf32>, tensor<3x3x16xf32>)
    outs(%output : tensor<16x16x16xf32>) {
  ^bb0(%in: f32, %f: f32, %out: f32):
    %mul = arith.mulf %in, %f : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<16x16x16xf32>
  return %0 : tensor<16x16x16xf32>
}

// CHECK-LABEL: func.func @batchless_depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<18x18x16xf32>
// CHECK-SAME:    %[[FILTER:[a-zA-Z0-9]+]]: tensor<3x3x16xf32>
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]+]]: tensor<16x16x16xf32>
// CHECK:       tensor.expand_shape {{.*}} output_shape [1, 18, 18, 16] : tensor<18x18x16xf32> into tensor<1x18x18x16xf32>
// CHECK:       tensor.expand_shape {{.*}} output_shape [1, 16, 16, 16] : tensor<16x16x16xf32> into tensor<1x16x16x16xf32>
// CHECK:       %[[CONV:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK:       tensor.collapse_shape %[[CONV]] {{.*}} : tensor<1x16x16x16xf32> into tensor<16x16x16xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

func.func @conv_with_batch_not_transformed(%input: tensor<1x16x16x4xf32>, %filter: tensor<3x3x4x16xf32>, %output: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%input, %filter : tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%output : tensor<1x14x14x16xf32>) {
  ^bb0(%in: f32, %f: f32, %out: f32):
    %mul = arith.mulf %in, %f : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<1x14x14x16xf32>
  return %0 : tensor<1x14x14x16xf32>
}

// CHECK-LABEL: func.func @conv_with_batch_not_transformed
// CHECK-NOT:   tensor.expand_shape
// CHECK-NOT:   tensor.collapse_shape
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK:       return %[[RESULT]]

// -----

// Test reshape propagation.

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @reshape_propagation_fill_conv_elem(%input: tensor<16x16x4xf32>, %filter: tensor<3x3x4x16xf32>) -> tensor<14x14x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<14x14x16xf32>

  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<14x14x16xf32>) -> tensor<14x14x16xf32>

  %conv = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%input, %filter : tensor<16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%fill : tensor<14x14x16xf32>) {
  ^bb0(%in: f32, %f: f32, %out: f32):
    %mul = arith.mulf %in, %f : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<14x14x16xf32>

  %elem = linalg.generic {
    indexing_maps = [#map3, #map3],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%conv : tensor<14x14x16xf32>)
    outs(%empty : tensor<14x14x16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %zero = arith.constant 0.000000e+00 : f32
    %relu = arith.maximumf %in, %zero : f32
    linalg.yield %relu : f32
  } -> tensor<14x14x16xf32>

  return %elem : tensor<14x14x16xf32>
}

// CHECK-LABEL: func.func @reshape_propagation_fill_conv_elem
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<16x16x4xf32>
// CHECK-SAME:    %[[FILTER:[a-zA-Z0-9]+]]: tensor<3x3x4x16xf32>
// CHECK:       %[[FILL:.+]] = linalg.fill
// CHECK-SAME:    outs({{.*}} : tensor<1x14x14x16xf32>)
// CHECK:       %[[EXPANDED_INPUT:.+]] = tensor.expand_shape %[[INPUT]] {{\[}}[0, 1], [2], [3]{{\]}}
// CHECK-SAME:    : tensor<16x16x4xf32> into tensor<1x16x16x4xf32>
// CHECK:       %[[CONV:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK-SAME:    ins(%[[EXPANDED_INPUT]], %[[FILTER]] : tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<1x14x14x16xf32>)
// CHECK:       %[[ELEM:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:    ins(%[[CONV]] : tensor<1x14x14x16xf32>)
// CHECK:       %[[RESULT:.+]] = tensor.collapse_shape %[[ELEM]] {{\[}}[0, 1], [2], [3]{{\]}}
// CHECK-SAME:    : tensor<1x14x14x16xf32> into tensor<14x14x16xf32>
// CHECK:       return %[[RESULT]] : tensor<14x14x16xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0 + d4, d1 + d5, d2 + d6, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3)>

func.func @batchless_conv_3d_ndhwc_dhwcf(%input: tensor<10x10x10x4xf32>, %filter: tensor<3x3x3x4x8xf32>, %output: tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]
  } ins(%input, %filter : tensor<10x10x10x4xf32>, tensor<3x3x3x4x8xf32>)
    outs(%output : tensor<8x8x8x8xf32>) {
  ^bb0(%in: f32, %f: f32, %out: f32):
    %mul = arith.mulf %in, %f : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: func.func @batchless_conv_3d_ndhwc_dhwcf
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<10x10x10x4xf32>
// CHECK-SAME:    %[[FILTER:[a-zA-Z0-9]+]]: tensor<3x3x3x4x8xf32>
// CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]+]]: tensor<8x8x8x8xf32>
// CHECK:       %[[EXPANDED_INPUT:.+]] = tensor.expand_shape %[[INPUT]] {{\[}}[0, 1], [2], [3], [4]{{\]}} output_shape [1, 10, 10, 10, 4] : tensor<10x10x10x4xf32> into tensor<1x10x10x10x4xf32>
// CHECK:       %[[EXPANDED_OUTPUT:.+]] = tensor.expand_shape %[[OUTPUT]] {{\[}}[0, 1], [2], [3], [4]{{\]}} output_shape [1, 8, 8, 8, 8] : tensor<8x8x8x8xf32> into tensor<1x8x8x8x8xf32>
// CHECK:       %[[CONV:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]
// CHECK-SAME:    ins(%[[EXPANDED_INPUT]], %[[FILTER]] : tensor<1x10x10x10x4xf32>, tensor<3x3x3x4x8xf32>)
// CHECK-SAME:    outs(%[[EXPANDED_OUTPUT]] : tensor<1x8x8x8x8xf32>)
// CHECK:       %[[RESULT:.+]] = tensor.collapse_shape %[[CONV]] {{\[}}[0, 1], [2], [3], [4]{{\]}} : tensor<1x8x8x8x8xf32> into tensor<8x8x8x8xf32>
// CHECK:       return %[[RESULT]] : tensor<8x8x8x8xf32>
