// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-filter-to-channels-last{filter-layout=hwfc}))" %s | FileCheck --check-prefixes=CHECK-HWFC,CHECK-ALL %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-filter-to-channels-last))" %s | FileCheck --check-prefixes=CHECK-FHWC,CHECK-ALL %s

// CHECK-HWFC: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d3, d6)>
// CHECK-FHWC: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
// CHECK-ALL-LABEL: @conv_i8
util.func @conv_i8(%arg0: tensor<2x130x130x16xi8>, %arg1: tensor<3x3x16x320xi8>,
%arg2: tensor<2x128x128x320xi32>)
    -> tensor<2x128x128x320xi32> {

// CHECK-HWFC:      %[[EMPTY:.*]] = tensor.empty() : tensor<3x3x320x16xi8>
// CHECK-HWFC:      %[[TRANSPOSE:.*]] = linalg.transpose ins(%arg1 : tensor<3x3x16x320xi8>) outs(%[[EMPTY]] : tensor<3x3x320x16xi8>) permutation = [0, 1, 3, 2]
// CHECK-HWFC:      %[[GENERIC:.*]] = linalg.generic
// CHECK-HWFC-SAME: indexing_maps = [#map, #[[$MAP]], #map2],
// CHECK-HWFC-SAME: ins(%arg0, %[[TRANSPOSE]] : tensor<2x130x130x16xi8>, tensor<3x3x320x16xi8>)

// CHECK-FHWC:      %[[EMPTY:.*]] = tensor.empty() : tensor<320x3x3x16xi8>
// CHECK-FHWC:      %[[TRANSPOSE:.*]] = linalg.transpose ins(%arg1 : tensor<3x3x16x320xi8>) outs(%[[EMPTY]] : tensor<320x3x3x16xi8>) permutation = [3, 0, 1, 2]
// CHECK-FHWC:      %[[GENERIC:.*]] = linalg.generic
// CHECK-FHWC-SAME: indexing_maps = [#map, #[[$MAP]], #map2],
// CHECK-FHWC-SAME: ins(%arg0, %[[TRANSPOSE]] : tensor<2x130x130x16xi8>, tensor<320x3x3x16xi8>)
  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<2x130x130x16xi8>, tensor<3x3x16x320xi8>)
             outs(%arg2 : tensor<2x128x128x320xi32>) -> tensor<2x128x128x320xi32>
  util.return %conv0 : tensor<2x128x128x320xi32>
}

// -----

// CHECK-ALL-LABEL: @conv_dyn_input
util.func @conv_dyn_input(%arg0: tensor<?x?x?x16xf16>, %arg1: tensor<3x3x16x320xf16>,
%arg2: tensor<?x?x?x320xf32>)
    -> tensor<?x?x?x320xf32> {
// CHECK-ALL:       %[[GENERIC:.*]] = linalg.generic

// CHECK-HWFC-SAME: ins(%arg0, %[[TRANSPOSED:.*]]: tensor<?x?x?x16xf16>, tensor<3x3x320x16xf16>)
// CHECK-FHWC-SAME: ins(%arg0, %[[TRANSPOSED:.*]]: tensor<?x?x?x16xf16>, tensor<320x3x3x16xf16>)

// CHECK-ALL-SAME:  outs(%arg2 : tensor<?x?x?x320xf32>)
// CHECK-ALL:       util.return %[[GENERIC]] : tensor<?x?x?x320xf32>
  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<?x?x?x16xf16>, tensor<3x3x16x320xf16>)
             outs(%arg2 : tensor<?x?x?x320xf32>) -> tensor<?x?x?x320xf32>
  util.return %conv0 : tensor<?x?x?x320xf32>
}

// -----

// CHECK-ALL-LABEL: @conv_dyn_filter
util.func @conv_dyn_filter(%arg0: tensor<2x130x130x16xf16>, %arg1: tensor<?x?x16x320xf16>,
%arg2: tensor<2x128x128x320xf32>)
    -> tensor<2x128x128x320xf32> {

// CHECK-ALL-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-ALL-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-ALL-DAG:   %[[DIM0:.*]] = tensor.dim %arg1, %[[C0]]
// CHECK-ALL-DAG:   %[[DIM1:.*]] = tensor.dim %arg1, %[[C1]]
// CHECK-ALL:       %[[EMPTY:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]])

// CHECK-HWFC:      %[[TRANSPOSE:.*]] = linalg.transpose ins(%arg1 : tensor<?x?x16x320xf16>) outs(%[[EMPTY]] : tensor<?x?x320x16xf16>) permutation = [0, 1, 3, 2]
// CHECK-FHWC:      %[[TRANSPOSE:.*]] = linalg.transpose ins(%arg1 : tensor<?x?x16x320xf16>) outs(%[[EMPTY]] : tensor<320x?x?x16xf16>) permutation = [3, 0, 1, 2]

// CHECK-ALL:       linalg.generic

// CHECK-HWFC-SAME: ins(%arg0, %[[TRANSPOSE]] : tensor<2x130x130x16xf16>, tensor<?x?x320x16xf16>) outs(%arg2 : tensor<2x128x128x320xf32>)
// CHECK-FHWC-SAME: ins(%arg0, %[[TRANSPOSE]] : tensor<2x130x130x16xf16>, tensor<320x?x?x16xf16>) outs(%arg2 : tensor<2x128x128x320xf32>)

  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<2x130x130x16xf16>, tensor<?x?x16x320xf16>)
             outs(%arg2 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
  util.return %conv0 : tensor<2x128x128x320xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @conv_2d_nhwc_chwf(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<4x3x3x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<4x3x3x16xf32>) outs(%arg2 : tensor<1x14x14x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}

// CHECK-FHWC: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
// CHECK-FHWC: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>

// CHECK-FHWC-LABEL:  @conv_2d_nhwc_chwf
// CHECK-FHWC:        %[[EMPTY:.*]] = tensor.empty() : tensor<16x3x3x4xf32>
// CHECK-FHWC:        %[[TRANSPOSE:.*]] = linalg.transpose ins({{.*}} : tensor<4x3x3x16xf32>) outs(%[[EMPTY]] : tensor<16x3x3x4xf32>)
// CHECK-FHWC-SAME:   permutation = [3, 1, 2, 0]
// CHECK-FHWC:        %[[START:.*]] = iree_tensor_ext.compute_barrier.start %[[TRANSPOSE]]
// CHECK-FHWC:        %[[GENERIC:.*]] = linalg.generic
// CHECK-FHWC-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]], #map2],
// CHECK-FHWC-SAME:   ins({{.*}}, %[[START]] : tensor<1x16x16x4xf32>, tensor<16x3x3x4xf32>)

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d7, d5, d6, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
util.func public @conv_2d_nhwgc_gchwf(%arg0: tensor<2x10x10x7x4xf32>, %arg1: tensor<7x4x3x3x16xf32>, %arg2: tensor<2x8x8x7x16xf32>) -> tensor<2x8x8x7x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<2x10x10x7x4xf32>, tensor<7x4x3x3x16xf32>) outs(%arg2 : tensor<2x8x8x7x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x8x8x7x16xf32>
  util.return %0 : tensor<2x8x8x7x16xf32>
}

// CHECK-FHWC: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>
// CHECK-FHWC: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>

// CHECK-FHWC-LABEL:  @conv_2d_nhwgc_gchwf
// CHECK-FHWC:        %[[EMPTY:.*]] = tensor.empty() : tensor<7x16x3x3x4xf32>
// CHECK-FHWC:        %[[TRANSPOSE:.*]] = linalg.transpose ins({{.*}} : tensor<7x4x3x3x16xf32>) outs(%[[EMPTY]] : tensor<7x16x3x3x4xf32>)
// CHECK-FHWC-SAME:   permutation = [0, 4, 2, 3, 1]
// CHECK-FHWC:        %[[START:.*]] = iree_tensor_ext.compute_barrier.start %[[TRANSPOSE]]
// CHECK-FHWC:        %[[GENERIC:.*]] = linalg.generic
// CHECK-FHWC-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]], #map2],
// CHECK-FHWC-SAME:   ins({{.*}}, %[[START]] : tensor<2x10x10x7x4xf32>, tensor<7x16x3x3x4xf32>)

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @conv_2d_nhwc_hwcf_no_transpose(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>) outs(%arg2 : tensor<1x14x14x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}

// CHECK-FHWC-LABEL:  @conv_2d_nhwc_hwcf_no_transpose
// CHECK-FHWC-NOT:    linalg.transpose

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5, d2 + d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @conv_2d_chwn_chwf_no_transpose(%arg0: tensor<16x26x18x288xf32>, %arg1: tensor<16x24x16x288xf32>, %arg2: tensor<288x3x3x288xf32>) -> tensor<288x3x3x288xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x26x18x288xf32>, tensor<16x24x16x288xf32>) outs(%arg2 : tensor<288x3x3x288xf32>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %12 = arith.mulf %in, %in_3 : f32
    %13 = arith.addf %out, %12 : f32
    linalg.yield %13 : f32
  } -> tensor<288x3x3x288xf32>
  util.return %0 : tensor<288x3x3x288xf32>
}

// CHECK-FHWC-LABEL:  @conv_2d_chwn_chwf_no_transpose
// CHECK-FHWC-NOT:    linalg.transpose

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1, d2 + d5, d3 + d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d0, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @conv_2d_cnhw_cfhw_no_transpose(%arg0: tensor<16x288x26x18xf32>, %arg1: tensor<16x288x24x16xf32>, %arg2: tensor<288x288x3x3xf32>) -> tensor<288x288x3x3xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x288x26x18xf32>, tensor<16x288x24x16xf32>) outs(%arg2 : tensor<288x288x3x3xf32>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %12 = arith.mulf %in, %in_3 : f32
    %13 = arith.addf %out, %12 : f32
    linalg.yield %13 : f32
  } -> tensor<288x288x3x3xf32>
  util.return %0 : tensor<288x288x3x3xf32>
}

// CHECK-FHWC-LABEL:  @conv_2d_cnhw_cfhw_no_transpose
// CHECK-FHWC-NOT:    linalg.transpose

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @small_nhwc_chwf_filter_1x1(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<4x1x1x16xf32>, %arg2: tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<4x1x1x16xf32>) outs(%arg2 : tensor<1x16x16x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1x16x16x16xf32>
  util.return %0 : tensor<1x16x16x16xf32>
}

// CHECK-FHWC-LABEL:  @small_nhwc_chwf_filter_1x1
// CHECK-FHWC-NOT:    linalg.transpose

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d3, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
util.func public @conv_2d_nhwc_cfhw(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<4x16x3x3xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<4x16x3x3xf32>) outs(%arg2 : tensor<1x14x14x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1x14x14x16xf32>
  util.return %0 : tensor<1x14x14x16xf32>
}

// CHECK-FHWC: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
// CHECK-FHWC: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>

// CHECK-FHWC-LABEL:  @conv_2d_nhwc_cfhw
// CHECK-FHWC:        %[[EMPTY:.*]] = tensor.empty() : tensor<16x3x3x4xf32>
// CHECK-FHWC:        %[[TRANSPOSE:.*]] = linalg.transpose ins({{.*}} : tensor<4x16x3x3xf32>) outs(%[[EMPTY]] : tensor<16x3x3x4xf32>)
// CHECK-FHWC-SAME:   permutation = [1, 2, 3, 0]
// CHECK-FHWC:        %[[START:.*]] = iree_tensor_ext.compute_barrier.start %[[TRANSPOSE]]
// CHECK-FHWC:        %[[GENERIC:.*]] = linalg.generic
// CHECK-FHWC-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]], #map2],
// CHECK-FHWC-SAME:   ins({{.*}}, %[[START]] : tensor<1x16x16x4xf32>, tensor<16x3x3x4xf32>)

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d7, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
util.func public @conv_2d_nhwgc_gcfhw(%arg0: tensor<2x10x10x7x4xf32>, %arg1: tensor<7x4x16x3x3xf32>, %arg2: tensor<2x8x8x7x16xf32>) -> tensor<2x8x8x7x16xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<2x10x10x7x4xf32>, tensor<7x4x16x3x3xf32>) outs(%arg2 : tensor<2x8x8x7x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x8x8x7x16xf32>
  util.return %0 : tensor<2x8x8x7x16xf32>
}

// CHECK-FHWC: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>
// CHECK-FHWC: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>

// CHECK-FHWC-LABEL:  @conv_2d_nhwgc_gcfhw
// CHECK-FHWC:        %[[EMPTY:.*]] = tensor.empty() : tensor<7x16x3x3x4xf32>
// CHECK-FHWC:        %[[TRANSPOSE:.*]] = linalg.transpose ins({{.*}} : tensor<7x4x16x3x3xf32>) outs(%[[EMPTY]] : tensor<7x16x3x3x4xf32>)
// CHECK-FHWC-SAME:   permutation = [0, 2, 3, 4, 1]
// CHECK-FHWC:        %[[START:.*]] = iree_tensor_ext.compute_barrier.start %[[TRANSPOSE]]
// CHECK-FHWC:        %[[GENERIC:.*]] = linalg.generic
// CHECK-FHWC-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]], #map2],
// CHECK-FHWC-SAME:   ins({{.*}}, %[[START]] : tensor<2x10x10x7x4xf32>, tensor<7x16x3x3x4xf32>)
