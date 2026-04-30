// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-set-split-reduction-sizes))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
util.func public @split_matmul(%arg0: tensor<32x40960xf32>, %arg1: tensor<40960x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<32x40960xf32>, tensor<40960x32xf32>) outs(%arg2 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %1, %out : f32
    linalg.yield %2 : f32
  } -> tensor<32x32xf32>
  util.return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: @split_matmul
//       CHECK: iree_linalg_ext.split_reduction = [320 : index]

// -----

util.func public @split_very_large_k(%arg0: tensor<128x16800000xbf16>, %arg1: tensor<16800000x134xbf16>, %arg2: tensor<128x134xf32>) -> tensor<128x134xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x16800000xbf16>, tensor<16800000x134xbf16>) outs(%arg2 : tensor<128x134xf32>) -> tensor<128x134xf32>
  util.return %0 : tensor<128x134xf32>
}

// CHECK-LABEL: @split_very_large_k
//       CHECK: iree_linalg_ext.split_reduction = [8400 : index]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
util.func public @split_multi_dims_gemm(%arg0: tensor<16x96x48x288xf32>, %arg1: tensor<16x96x48x288xf32>, %arg2: tensor<288x288xf32>) -> tensor<288x288xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg0 : tensor<16x96x48x288xf32>, tensor<16x96x48x288xf32>) outs(%arg2 : tensor<288x288xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<288x288xf32>
  util.return %0 : tensor<288x288xf32>
}

// CHECK-LABEL: @split_multi_dims_gemm
//       CHECK: iree_linalg_ext.split_reduction = [1 : index, 96 : index, 48 : index]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
util.func public @split_multi_dims_gemm_2(%arg0: tensor<235x363x224xf32>, %arg1: tensor<235x363x224xf32>, %arg2: tensor<224x224xf32>) -> tensor<224x224xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<235x363x224xf32>, tensor<235x363x224xf32>) outs(%arg2 : tensor<224x224xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<224x224xf32>
  util.return %0 : tensor<224x224xf32>
}

// CHECK-LABEL: @split_multi_dims_gemm_2
//       CHECK: iree_linalg_ext.split_reduction = [47 : index, 363 : index]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
util.func public @split_multi_dims_gemm_3(%arg0: tensor<16x96x48x32xf32>, %arg1: tensor<16x96x48x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg0 : tensor<16x96x48x32xf32>, tensor<16x96x48x32xf32>) outs(%arg2 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<32x32xf32>
  util.return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: @split_multi_dims_gemm_3
//       CHECK: iree_linalg_ext.split_reduction = [1 : index, 12 : index, 48 : index]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
util.func public @no_split_dynamic_matmul(%arg0: tensor<?x40960xf32>, %arg1: tensor<40960x32xf32>, %arg2: tensor<?x32xf32>) -> tensor<?x32xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x40960xf32>, tensor<40960x32xf32>) outs(%arg2 : tensor<?x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %1, %out : f32
    linalg.yield %2 : f32
  } -> tensor<?x32xf32>
  util.return %0 : tensor<?x32xf32>
}

// CHECK-LABEL: @no_split_dynamic_matmul
//   CHECK-NOT: iree_linalg_ext.split_reduction

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
util.func public @no_split_matmul_large_mn(%arg0: tensor<4096x150000xf32>, %arg1: tensor<150000x2268xf32>, %arg2: tensor<4096x2268xf32>) -> tensor<4096x2268xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4096x150000xf32>, tensor<150000x2268xf32>) outs(%arg2 : tensor<4096x2268xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %1, %out : f32
    linalg.yield %2 : f32
  } -> tensor<4096x2268xf32>
  util.return %0 : tensor<4096x2268xf32>
}

// CHECK-LABEL: @no_split_matmul_large_mn
//   CHECK-NOT: iree_linalg_ext.split_reduction

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
util.func public @no_split_matmul_small_k(%arg0: tensor<256x4096xf32>, %arg1: tensor<4096x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<256x4096xf32>, tensor<4096x256xf32>) outs(%arg2 : tensor<256x256xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %1, %out : f32
    linalg.yield %2 : f32
  } -> tensor<256x256xf32>
  util.return %0 : tensor<256x256xf32>
}

// CHECK-LABEL: @no_split_matmul_small_k
//   CHECK-NOT: iree_linalg_ext.split_reduction

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
util.func public @no_split_multi_dims_small_k(%arg0: tensor<16x24x16x288xf32>, %arg1: tensor<16x24x16x288xf32>, %arg2: tensor<288x288xf32>) -> tensor<288x288xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg0 : tensor<16x24x16x288xf32>, tensor<16x24x16x288xf32>) outs(%arg2 : tensor<288x288xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<288x288xf32>
  util.return %0 : tensor<288x288xf32>
}

// CHECK-LABEL: @no_split_multi_dims_small_k
//   CHECK-NOT: iree_linalg_ext.split_reduction
