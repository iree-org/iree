// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-collapse-dimensions))" %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
util.func public @do_not_collapse_cst_in_place(%arg0: tensor<1x1x2304xf32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x1x2304xf32>
  %0 = tensor.empty() : tensor<1x1x2304xf32>
  %1 = flow.dispatch.region -> (tensor<1x1x2304xf32>) {
    %2 = tensor.empty() : tensor<1x1x2304xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %cst : tensor<1x1x2304xf32>, tensor<1x1x2304xf32>) outs(%2 : tensor<1x1x2304xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<1x1x2304xf32>
    flow.return %3 : tensor<1x1x2304xf32>
  }
  util.return
}
// CHECK-LABEL: util.func public @do_not_collapse_cst_in_place
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]+]]
// CHECK-DAG:     %[[CST:.+]] = arith.constant
// CHECK-DAG:     %[[COLLAPSED_ARG0:.+]] = tensor.collapse_shape %[[ARG0]]
// CHECK-DAG:     %[[COLLAPSED_CST:.+]] = tensor.collapse_shape %[[CST]]
// CHECK:         %{{.+}} = flow.dispatch.region
// CHECK:            %[[RES:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[COLLAPSED_ARG0]], %[[COLLAPSED_CST]]
// CHECK:            flow.return %[[RES]]


// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
util.func public @unpack_collapse(%arg0: tensor<2x320x128x128xf32>, %arg1: tensor<320xf32>, %arg2: tensor<320xf32>, %arg3: tensor<1x5x2x64xf32>) -> tensor<2x320x128x128xf16> {
  %dispatch = flow.dispatch.region -> (tensor<2x320x128x128xf16>) {
    %0 = tensor.empty() : tensor<2x320xf32>
    %unpack = tensor.unpack %arg3 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [2, 64] into %0 : tensor<1x5x2x64xf32> -> tensor<2x320xf32>
    %1 = tensor.empty() : tensor<2x320x128x128xf16>
    %2 = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map1, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }
    ins(%arg0, %arg1, %unpack, %arg2 : tensor<2x320x128x128xf32>, tensor<320xf32>, tensor<2x320xf32>, tensor<320xf32>)
    outs(%1 : tensor<2x320x128x128xf16>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f16):
      %3 = arith.addf %in_1, %in_2 : f32
      %4 = arith.addf %in, %in_0 : f32
      %5 = arith.truncf %3 : f32 to f16
      %6 = arith.truncf %4 : f32 to f16
      %7 = arith.addf %6, %5 : f16
      linalg.yield %7 : f16
    } -> tensor<2x320x128x128xf16>
    flow.return %2 : tensor<2x320x128x128xf16>
  }
  util.return %dispatch : tensor<2x320x128x128xf16>
}

// CHECK-LABEL:  util.func public @unpack_collapse
//  CHECK-SAME:    %[[ARG0:.*]]: tensor<2x320x128x128xf32>
//       CHECK:    %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]]
//       CHECK:    flow.dispatch.region
//       CHECK:    %[[GEN:.+]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins(%[[COLLAPSED]], {{.*}} : tensor<2x320x16384xf32>, tensor<320xf32>, tensor<2x320xf32>, tensor<320xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<2x320x16384xf16>)
//       CHECK:    flow.return %[[GEN]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
util.func public @unpack_elementwise_collapse(%arg0: tensor<2x320x128x128xf32>, %arg1: tensor<320xf32>, %arg2: tensor<320xf32>, %arg3: tensor<1x5x2x64xf32>) -> tensor<2x320x128x128xf16> {
  %0 = flow.dispatch.region -> (tensor<2x320x128x128xf16>) {
    %1 = tensor.empty() : tensor<2x320xf32>
    %2 = tensor.empty() : tensor<2x320x128x128xf16>
    %empty = tensor.empty() : tensor<2x320x128x128xf32>
    %cst = arith.constant 3.14 : f32

    %elementwise = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>

    %unpack = tensor.unpack %arg3 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [2, 64] into %1 : tensor<1x5x2x64xf32> -> tensor<2x320xf32>

    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise, %arg1, %unpack, %arg2 : tensor<2x320x128x128xf32>, tensor<320xf32>, tensor<2x320xf32>, tensor<320xf32>) outs(%2 : tensor<2x320x128x128xf16>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f16):
      %4 = arith.addf %in_1, %in_2 : f32
      %5 = arith.addf %in, %in_0 : f32
      %6 = arith.truncf %4 : f32 to f16
      %7 = arith.truncf %5 : f32 to f16
      %8 = arith.addf %7, %6 : f16
      linalg.yield %8 : f16
    } -> tensor<2x320x128x128xf16>
    flow.return %3 : tensor<2x320x128x128xf16>
  }
  util.return %0 : tensor<2x320x128x128xf16>
}

// CHECK-LABEL:  util.func public @unpack_elementwise_collapse
//  CHECK-SAME:    %[[ARG0:.*]]: tensor<2x320x128x128xf32>
//       CHECK:    %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]]
//       CHECK:    flow.dispatch.region
//       CHECK:    %[[ELEMENTWISE:.+]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins(%[[COLLAPSED]] : tensor<2x320x16384xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<2x320x16384xf32>)
//       CHECK:    %[[GEN:.+]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins({{.*}} : tensor<2x320x16384xf32>, tensor<320xf32>, tensor<2x320xf32>, tensor<320xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<2x320x16384xf16>)
//       CHECK:    flow.return %[[GEN]]


// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
util.func public @prevent_collapse(%arg0: tensor<2x320x128x128xf32>, %arg1: tensor<320xf32>, %arg2: tensor<320xf32>, %arg3: tensor<1x5x2x64xf32>) -> tensor<2x320x128x128xf16> {
  %0 = flow.dispatch.region -> (tensor<2x320x128x128xf16>) {
    %1 = tensor.empty() : tensor<2x320xf32>
    %2 = tensor.empty() : tensor<2x320x128x128xf16>
    %empty = tensor.empty() : tensor<2x320x128x128xf32>
    %cst = arith.constant 3.14 : f32

    %elementwise = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>

    %barrier = util.optimization_barrier %elementwise : tensor<2x320x128x128xf32>
    %unpack = tensor.unpack %arg3 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [2, 64] into %1 : tensor<1x5x2x64xf32> -> tensor<2x320xf32>

    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%barrier, %arg1, %unpack, %arg2 : tensor<2x320x128x128xf32>, tensor<320xf32>, tensor<2x320xf32>, tensor<320xf32>) outs(%2 : tensor<2x320x128x128xf16>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f16):
      %4 = arith.addf %in_1, %in_2 : f32
      %5 = arith.addf %in, %in_0 : f32
      %6 = arith.truncf %4 : f32 to f16
      %7 = arith.truncf %5 : f32 to f16
      %8 = arith.addf %7, %6 : f16
      linalg.yield %8 : f16
    } -> tensor<2x320x128x128xf16>
    flow.return %3 : tensor<2x320x128x128xf16>
  }
  util.return %0 : tensor<2x320x128x128xf16>
}

// CHECK-LABEL:  util.func public @prevent_collapse
//  CHECK-SAME:    %[[ARG0:.*]]: tensor<2x320x128x128xf32>
//       CHECK:    %[[ELEMENTWISE:.+]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins(%[[ARG0]] : tensor<2x320x128x128xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<2x320x128x128xf32>)
//       CHECK:    %[[GEN:.+]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins({{.*}} : tensor<2x320x128x128xf32>, tensor<320xf32>, tensor<2x320xf32>, tensor<320xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<2x320x128x128xf16>)

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
util.func public @quantized_matmul(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>) -> tensor<1x1x4096xf32> {
  %cst = arith.constant dense_resource<__elided__> : tensor<4096x32xf32>
  %cst_0 = arith.constant dense_resource<__elided__> : tensor<4096x32xf32>
  %0 = flow.dispatch.region -> (tensor<1x1x4096xf32>) {
    %cst_1 = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<1x1x4096xf32>
    %2 = tensor.empty() : tensor<4096x32x128xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %cst, %cst_0 : tensor<4096x32x128xi8>, tensor<4096x32xf32>, tensor<4096x32xf32>) outs(%2 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i8, %in_2: f32, %in_3: f32, %out: f32):
      %6 = arith.extui %in : i8 to i32
      %7 = arith.uitofp %6 : i32 to f32
      %8 = arith.subf %7, %in_3 : f32
      %9 = arith.mulf %8, %in_2 : f32
      linalg.yield %9 : f32
    } -> tensor<4096x32x128xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %4 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%3 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.mulf %in, %in_2 : f32
      %7 = arith.addf %6, %out : f32
      linalg.yield %7 : f32
    } -> tensor<1x1x4096xf32>
    flow.return %5 : tensor<1x1x4096xf32>
  }
  util.return %0 : tensor<1x1x4096xf32>
}

// CHECK-LABEL:  util.func public @quantized_matmul
//  CHECK-SAME:    %[[ARG0:.*]]: tensor<4096x32x128xi8>
//  CHECK-SAME:    %[[ARG1:.*]]: tensor<1x1x32x128xf32>
//       CHECK:    %[[CST:.*]] = arith.constant dense_resource<__elided__> : tensor<4096x32xf32>
//       CHECK:    %[[CST_0:.*]] = arith.constant dense_resource<__elided__> : tensor<4096x32xf32>
//       CHECK:    %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG1]]
//       CHECK:    flow.dispatch.region
//       CHECK:    %[[VAL0:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins(%[[ARG0]], %[[CST]], %[[CST_0]] : tensor<4096x32x128xi8>, tensor<4096x32xf32>, tensor<4096x32xf32>)
//  CHECK-SAME:      outs(%{{.*}} :  tensor<4096x32x128xf32>)
//       CHECK:    %[[VAL2:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
//       CHECK:      ins(%[[COLLAPSED]], %[[VAL0]] : tensor<1x32x128xf32>, tensor<4096x32x128xf32>)
//       CHECK:      outs(%{{.*}} : tensor<1x4096xf32>)
//       CHECK:    flow.return

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
util.func public @elementwise_chain(%arg0: tensor<2x320x128x128xf32>) -> tensor<2x320x128x128xf32> {
  %0 = flow.dispatch.region -> (tensor<2x320x128x128xf32>) {
    %empty = tensor.empty() : tensor<2x320x128x128xf32>
    %cst = arith.constant 3.14 : f32

    %elementwise1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>
    %elementwise2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise1 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>
    %elementwise3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise2 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>
    %elementwise4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise3 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>

    flow.return %elementwise4 : tensor<2x320x128x128xf32>
  }
  util.return %0 : tensor<2x320x128x128xf32>
}

// CHECK-LABEL:  util.func public @elementwise_chain
//       CHECK:    %[[COLLAPSED:.*]] = tensor.collapse_shape
//       CHECK:    flow.dispatch.region
//       CHECK:    %[[VAL0:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel"]
//  CHECK-SAME:      ins(%[[COLLAPSED]] : tensor<10485760xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<10485760xf32>)
//       CHECK:    %[[VAL1:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel"]
//  CHECK-SAME:      ins(%[[VAL0]] : tensor<10485760xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<10485760xf32>)
//       CHECK:    %[[VAL2:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel"]
//  CHECK-SAME:      ins(%[[VAL1]] : tensor<10485760xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<10485760xf32>)
//       CHECK:    %[[VAL3:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel"]
//  CHECK-SAME:      ins(%[[VAL2]] : tensor<10485760xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<10485760xf32>)
//       CHECK:    flow.return %[[VAL3]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
util.func public @elementwise_dag(%arg0: tensor<2x320x128x128xf32>) -> tensor<2x320x128x128xf32> {
  %0 = flow.dispatch.region -> (tensor<2x320x128x128xf32>) {
    %empty = tensor.empty() : tensor<2x320x128x128xf32>
    %cst = arith.constant 3.14 : f32

    %elementwise1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>
    %elementwise2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise1 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>
    %elementwise3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise1 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>
    %elementwise4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise3, %elementwise2 : tensor<2x320x128x128xf32>, tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %in_1 : f32, %out : f32):
      %22 = arith.mulf %in_1, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>

    flow.return %elementwise4 : tensor<2x320x128x128xf32>
  }
  util.return %0 : tensor<2x320x128x128xf32>
}

// CHECK-LABEL:  util.func public @elementwise_dag
//       CHECK:    %[[COLLAPSED:.*]] = tensor.collapse_shape
//       CHECK:    flow.dispatch.region
//       CHECK:    %[[VAL0:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel"]
//  CHECK-SAME:      ins(%[[COLLAPSED]] : tensor<10485760xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<10485760xf32>)
//       CHECK:    %[[VAL1:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel"]
//  CHECK-SAME:      ins(%[[VAL0]] : tensor<10485760xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<10485760xf32>)
//       CHECK:    %[[VAL2:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel"]
//  CHECK-SAME:      ins(%[[VAL0]] : tensor<10485760xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<10485760xf32>)
//       CHECK:    %[[VAL3:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel"]
//  CHECK-SAME:      ins(%[[VAL2]], %[[VAL1]] : tensor<10485760xf32>, tensor<10485760xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<10485760xf32>)
//       CHECK:    flow.return %[[VAL3]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
util.func public @elementwise_dag_transpose(%arg0: tensor<2x320x128x128xf32>) -> tensor<2x320x128x128xf32> {
  %0 = flow.dispatch.region -> (tensor<2x320x128x128xf32>) {
    %empty = tensor.empty() : tensor<2x320x128x128xf32>
    %cst = arith.constant 3.14 : f32

    // Check that reducing dims propagates more than 1 op away
    %elementwise0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>
    %elementwise1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise0: tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>
    %elementwise2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise1 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>
    %elementwise3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise1 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>
    %elementwise4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise3, %elementwise2 : tensor<2x320x128x128xf32>, tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %in_1 : f32, %out : f32):
      %22 = arith.mulf %in_1, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>

    // Check that reducing dims propagates more than 1 op away
    %elementwise5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise4 : tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>

    flow.return %elementwise5 : tensor<2x320x128x128xf32>
  }
  util.return %0 : tensor<2x320x128x128xf32>
}

// CHECK-LABEL:  util.func public @elementwise_dag_transpose
//       CHECK:    %[[COLLAPSED:.*]] = tensor.collapse_shape
//       CHECK:    flow.dispatch.region
//       CHECK:    %[[VAL0:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins(%[[COLLAPSED]] : tensor<640x128x128xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<640x128x128xf32>)
//       CHECK:    %[[VAL1:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins(%[[VAL0]] : tensor<640x128x128xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<640x128x128xf32>)
//       CHECK:    %[[VAL2:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins(%[[VAL1]] : tensor<640x128x128xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<640x128x128xf32>)
//       CHECK:    %[[VAL3:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins(%[[VAL1]] : tensor<640x128x128xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<640x128x128xf32>)
//       CHECK:    %[[VAL4:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins(%[[VAL3]], %[[VAL2]] : tensor<640x128x128xf32>, tensor<640x128x128xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<640x128x128xf32>)
//       CHECK:    %[[VAL5:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins(%[[VAL4]] : tensor<640x128x128xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<640x128x128xf32>)
//       CHECK:    flow.return %[[VAL5]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>

util.func public @quantized_matmul(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>) -> tensor<1x1x4096xf32> {
  %cst = arith.constant dense_resource<__elided__> : tensor<4096x32xf32>
  %cst_0 = arith.constant dense_resource<__elided__> : tensor<4096x32xf32>
  %0 = flow.dispatch.region -> (tensor<1x1x4096xf32>) {
    %cst_1 = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<1x1x4096xf32>
    %2 = tensor.empty() : tensor<4096x32x128xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0: tensor<4096x32x128xi8>) outs(%2 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i8, %out: f32):
      %6 = arith.extui %in : i8 to i32
      %7 = arith.uitofp %6 : i32 to f32
      linalg.yield %7 : f32
    } -> tensor<4096x32x128xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %4 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%3 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.mulf %in, %in_2 : f32
      %7 = arith.addf %6, %out : f32
      linalg.yield %7 : f32
    } -> tensor<1x1x4096xf32>
    flow.return %5 : tensor<1x1x4096xf32>
  }
  util.return %0 : tensor<1x1x4096xf32>
}

// CHECK-LABEL:  util.func public @quantized_matmul
//  CHECK-SAME:    %[[ARG0:.*]]: tensor<4096x32x128xi8>
//  CHECK-SAME:    %[[ARG1:.*]]: tensor<1x1x32x128xf32>
//   CHECK-DAG:    %[[COLLAPSED0:.*]] = tensor.collapse_shape %[[ARG0]]
//   CHECK-DAG:    %[[COLLAPSED1:.*]] = tensor.collapse_shape %[[ARG1]]
//       CHECK:    flow.dispatch.region
//       CHECK:    %[[VAL0:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:      ins(%[[COLLAPSED0]] : tensor<4096x4096xi8>)
//  CHECK-SAME:      outs(%{{.*}} :  tensor<4096x4096xf32>)
//       CHECK:    %[[VAL2:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
//       CHECK:      ins(%[[COLLAPSED1]], %[[VAL0]] : tensor<1x4096xf32>, tensor<4096x4096xf32>)
//       CHECK:      outs(%{{.*}} : tensor<1x4096xf32>)
//       CHECK:    flow.return

// -----

util.func public @uncollapsable_op(%arg0 : tensor<10x10xi64>) -> tensor<10x10xi64> {
  %0 = flow.dispatch.region -> (tensor<10x10xi64>) {
    %1 = tensor.empty() : tensor<10x10xi64>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<10x10xi64>) outs(%1 : tensor<10x10xi64>){
    ^bb0(%in : i64, %out : i64):
      %00 = linalg.index 0 : index
      %01 = arith.index_cast %00 : index to i64
      linalg.yield %01 : i64
    } -> tensor<10x10xi64>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2: tensor<10x10xi64>) outs(%1 : tensor<10x10xi64>){
    ^bb0(%in : i64, %out : i64):
      linalg.yield %in : i64
    } -> tensor<10x10xi64>
    flow.return %3 : tensor<10x10xi64>
  }
  util.return %0 : tensor<10x10xi64>
}

// CHECK-LABEL:  util.func public @uncollapsable_op
//  CHECK-SAME:    %[[ARG0:.*]]: tensor<10x10xi64>
//       CHECK:    flow.dispatch.region
//       CHECK:    %[[VAL0:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:      ins(%[[ARG0]] : tensor<10x10xi64>)
//  CHECK-SAME:      outs(%{{.*}} :  tensor<10x10xi64>)
//       CHECK:    %[[VAL1:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:      ins(%[[VAL0]] : tensor<10x10xi64>)
//  CHECK-SAME:      outs(%{{.*}} :  tensor<10x10xi64>)
//       CHECK:    flow.return

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
util.func public @propagate_uncollapsable(%arg0: tensor<2x320x128x128xf32>) -> tensor<2x320x128x128xf32> {
  %0 = flow.dispatch.region -> (tensor<2x320x128x128xf32>) {
    %empty = tensor.empty() : tensor<2x320x128x128xf32>
    %cst = arith.constant 3.14 : f32

    %elementwise2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0: tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %out : f32):
      %22 = arith.mulf %cst, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>
    %barrier = util.optimization_barrier %arg0: tensor<2x320x128x128xf32>
    %elementwise4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%barrier, %elementwise2 : tensor<2x320x128x128xf32>, tensor<2x320x128x128xf32>) outs(%empty : tensor<2x320x128x128xf32>) {
    ^bb0(%in : f32, %in_1 : f32, %out : f32):
      %22 = arith.mulf %in_1, %in : f32
      linalg.yield %22 : f32
    } -> tensor<2x320x128x128xf32>

    flow.return %elementwise4 : tensor<2x320x128x128xf32>
  }
  util.return %0 : tensor<2x320x128x128xf32>
}

// CHECK-LABEL:  util.func public @propagate_uncollapsable
//  CHECK-SAME:    %[[ARG0:.*]]: tensor<2x320x128x128xf32>
//       CHECK:    flow.dispatch.region
//       CHECK:    %[[VAL1:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins(%[[ARG0]] : tensor<2x320x128x128xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<2x320x128x128xf32>)
//       CHECK:    %[[VAL2:.*]] = util.optimization_barrier %[[ARG0]] : tensor<2x320x128x128xf32>
//       CHECK:    %[[VAL3:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:      ins(%[[VAL2]], %[[VAL1]] : tensor<2x320x128x128xf32>, tensor<2x320x128x128xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<2x320x128x128xf32>)
//       CHECK:    flow.return %[[VAL3]]

// -----

util.func public @dequant_contraction(%arg0: tensor<2x32xf32>, %arg1: tensor<2x32x10x16384xf16>) -> tensor<2x32xf32> {
  %0 = flow.dispatch.region -> (tensor<2x32xf32>) {
    %1 = tensor.empty() : tensor<2x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = tensor.empty() : tensor<2x32x10x16384xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<2x32x10x16384xf16>) outs(%2 : tensor<2x32x10x16384xf32>) {
    ^bb0(%in: f16, %out: f32):
      %6 = arith.extf %in : f16 to f32
      linalg.yield %6 : f32
    } -> tensor<2x32x10x16384xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x32xf32>) -> tensor<2x32xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%3, %arg0 : tensor<2x32x10x16384xf32>, tensor<2x32xf32>) outs(%4 : tensor<2x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.subf %in, %in_0 : f32
      %7 = arith.mulf %6, %6 : f32
      %8 = arith.addf %7, %out : f32
      linalg.yield %8 : f32
    } -> tensor<2x32xf32>
    flow.return %5 : tensor<2x32xf32>
  }
  util.return %0 : tensor<2x32xf32>
}

// CHECK-LABEL: util.func public @dequant_contraction
//  CHECK-SAME:    %[[ARG0:.*]]: tensor<2x32xf32>
//  CHECK-SAME:    %[[ARG1:.+]]: tensor<2x32x10x16384xf16>
//   CHECK-DAG:     %[[COLLAPSED_ARG0:.+]] = tensor.collapse_shape %[[ARG0]]
//   CHECK-DAG:     %[[COLLAPSED_ARG1:.+]] = tensor.collapse_shape %[[ARG1]]
//       CHECK:    flow.dispatch.region
//       CHECK:    %[[VAL0:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:      ins(%[[COLLAPSED_ARG1]] : tensor<64x163840xf16>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<64x163840xf32>)
//       CHECK:    %[[VAL1:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:      ins(%[[VAL0]], %[[COLLAPSED_ARG0]] : tensor<64x163840xf32>, tensor<64xf32>)
//  CHECK-SAME:      outs(%{{.*}} : tensor<64xf32>)
//       CHECK:    flow.return %[[VAL1]]

// -----

util.func public @collapse_attention(%arg0: tensor<20x4096x16xf16>, %arg1: tensor<20x1024x16xf16>, %arg2: tensor<20x1024x64xf16>, %arg3: f16) -> tensor<2x10x4096x64xf16> {
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2], [3]] output_shape [2, 10, 4096, 16] : tensor<20x4096x16xf16> into tensor<2x10x4096x16xf16>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1], [2], [3]] output_shape [2, 10, 1024, 16] : tensor<20x1024x16xf16> into tensor<2x10x1024x16xf16>
    %expanded_1 = tensor.expand_shape %arg2 [[0, 1], [2], [3]] output_shape [2, 10, 1024, 64] : tensor<20x1024x64xf16> into tensor<2x10x1024x64xf16>
  %0 = flow.dispatch.region -> (tensor<2x10x4096x64xf16>) {
    %0 = tensor.empty() : tensor<2x10x4096x64xf16>
    %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]} ins(%expanded, %expanded_0, %expanded_1, %arg3 : tensor<2x10x4096x16xf16>, tensor<2x10x1024x16xf16>, tensor<2x10x1024x64xf16>, f16) outs(%0 : tensor<2x10x4096x64xf16>) {
    ^bb0(%arg4: f16):
      iree_linalg_ext.yield %arg4 : f16
    } -> tensor<2x10x4096x64xf16>
    flow.return %1 : tensor<2x10x4096x64xf16>
  }
  util.return %0 : tensor<2x10x4096x64xf16>
}

// CHECK-LABEL: util.func public @collapse_attention
//       CHECK:   %[[ATTN:.*]] = iree_linalg_ext.attention
//  CHECK-SAME:      tensor<20x4096x16xf16>, tensor<20x1024x16xf16>, tensor<20x1024x64xf16>, f16
//  CHECK-SAME:      tensor<20x4096x64xf16>
//       CHECK:   flow.return %[[ATTN]] : tensor<20x4096x64xf16>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
util.func public @collapse_attention_with_truncf(%arg0: tensor<20x4096x16xf32>, %arg1: tensor<20x1024x16xf32>, %arg2: tensor<20x1024x64xf32>, %arg3: f32) -> tensor<2x10x4096x64xf16> {
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2], [3]] output_shape [2, 10, 4096, 16] : tensor<20x4096x16xf32> into tensor<2x10x4096x16xf32>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1], [2], [3]] output_shape [2, 10, 1024, 16] : tensor<20x1024x16xf32> into tensor<2x10x1024x16xf32>
    %expanded_1 = tensor.expand_shape %arg2 [[0, 1], [2], [3]] output_shape [2, 10, 1024, 64] : tensor<20x1024x64xf32> into tensor<2x10x1024x64xf32>
  %0 = flow.dispatch.region -> (tensor<2x10x4096x64xf16>) {
    %0 = tensor.empty() : tensor<2x10x4096x64xf32>
    %5 = tensor.empty() : tensor<2x10x4096x64xf16>
    %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> ()>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]} ins(%expanded, %expanded_0, %expanded_1, %arg3 : tensor<2x10x4096x16xf32>, tensor<2x10x1024x16xf32>, tensor<2x10x1024x64xf32>, f32) outs(%0 : tensor<2x10x4096x64xf32>) {
    ^bb0(%arg4: f32):
      iree_linalg_ext.yield %arg4 : f32
    } -> tensor<2x10x4096x64xf32>
    %2 = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }
    ins(%1 : tensor<2x10x4096x64xf32>)
    outs(%5 : tensor<2x10x4096x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %6 = arith.truncf %in : f32 to f16
      linalg.yield %6 : f16
    } -> tensor<2x10x4096x64xf16>
    flow.return %2 : tensor<2x10x4096x64xf16>
  }
  util.return %0 : tensor<2x10x4096x64xf16>
}

// CHECK-LABEL: util.func public @collapse_attention_with_truncf
//       CHECK:   %[[ATTN:.*]] = iree_linalg_ext.attention
//  CHECK-SAME:      tensor<20x4096x16xf32>, tensor<20x1024x16xf32>, tensor<20x1024x64xf32>, f32
//  CHECK-SAME:      tensor<20x4096x64xf32>
//       CHECK:   %[[TRUNC:.*]] = linalg.generic
//  CHECK-SAME:      ins(%[[ATTN]] : tensor<20x4096x64xf32>
//       CHECK:   flow.return %[[TRUNC]] : tensor<20x4096x64xf16>

// -----

util.func public @collapse(%10: tensor<2x32x32x1280xi8>, %11 : tensor<10240x1280xi8>, %12 : tensor<10240xi32>, %13 : tensor<10240xf32>) -> (tensor<2x32x32x10240xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %14 = tensor.empty() : tensor<2x32x32x10240xf16>
  %15 = tensor.empty() : tensor<2x32x32x10240xi32>
  %16 = linalg.fill ins(%c0_i32 : i32) outs(%15 : tensor<2x32x32x10240xi32>) -> tensor<2x32x32x10240xi32>
  %dispatch = flow.dispatch.region -> (tensor<2x32x32x10240xf16>) {
    %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%10, %11 : tensor<2x32x32x1280xi8>, tensor<10240x1280xi8>) outs(%16 : tensor<2x32x32x10240xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %19 = arith.extsi %in : i8 to i32
      %20 = arith.extsi %in_0 : i8 to i32
      %21 = arith.muli %19, %20 : i32
      %22 = arith.addi %out, %21 : i32
      linalg.yield %22 : i32
    } -> tensor<2x32x32x10240xi32>
    %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %12, %13 : tensor<2x32x32x10240xi32>, tensor<10240xi32>, tensor<10240xf32>) outs(%14 : tensor<2x32x32x10240xf16>) {
    ^bb0(%in: i32, %in_0: i32, %in_1: f32, %out: f16):
      %19 = arith.addi %in, %in_0 : i32
      %20 = arith.sitofp %19 : i32 to f32
      %21 = arith.mulf %20, %in_1 : f32
      %22 = arith.truncf %21 : f32 to f16
      linalg.yield %22 : f16
    } -> tensor<2x32x32x10240xf16>
    flow.return %18 : tensor<2x32x32x10240xf16>
  }
  util.return %dispatch  : tensor<2x32x32x10240xf16>
}

// CHECK-LABEL: util.func public @collapse
//       CHECK:   %[[GEN0:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
//       CHECK:   %[[GEN1:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel"]
//       CHECK:   flow.return %[[GEN1]] : tensor<2048x10240xf16>

// -----

util.func public @update_from_producer(%arg0: tensor<2x1x256x16x16xi8>, %arg1: tensor<2x1x256xf32>) -> tensor<1x256x16x16xi8> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = flow.dispatch.region -> (tensor<1x256x16x16xi8>) {
    %1 = tensor.empty() : tensor<1x256x16x16xi8>
    %2 = tensor.empty() : tensor<1x256x16x16xf32>
    %3 = tensor.empty() : tensor<2x1x256x16x16xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x1x256x16x16xi8>) outs(%3 : tensor<2x1x256x16x16xf32>) {
    ^bb0(%in: i8, %out: f32):
      %8 = arith.extsi %in : i8 to i32
      %9 = arith.sitofp %8 : i32 to f32
      linalg.yield %9 : f32
    } -> tensor<2x1x256x16x16xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x256x16x16xf32>) -> tensor<1x256x16x16xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%4, %arg1 : tensor<2x1x256x16x16xf32>, tensor<2x1x256xf32>) outs(%5 : tensor<1x256x16x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.mulf %in, %in_0 : f32
      %9 = arith.addf %8, %out : f32
      linalg.yield %9 : f32
    } -> tensor<1x256x16x16xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x256x16x16xf32>) outs(%1 : tensor<1x256x16x16xi8>) {
    ^bb0(%in: f32, %out: i8):
      %8 = arith.fptosi %in : f32 to i8
      linalg.yield %8 : i8
    } -> tensor<1x256x16x16xi8>
    flow.return %7 : tensor<1x256x16x16xi8>
  }
  util.return %0 : tensor<1x256x16x16xi8>
}

// CHECK-LABEL: util.func public @update_from_producer
//       CHECK:   %[[GEN0:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
//       CHECK:   %[[GEN1:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:      ins(%[[GEN0]]
//       CHECK:   %[[GEN2:.*]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:      ins(%[[GEN1]]
//       CHECK:   flow.return %[[GEN2]] : tensor<256x256xi8>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
util.func public @uncollapsable_consumer(%arg0: tensor<1x1x2304xf32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x1x2304xf32>
  %0 = tensor.empty() : tensor<1x1x2304xf32>
  %1 = flow.dispatch.region -> (tensor<1x1x2304xf32>) {
    %2 = tensor.empty() : tensor<1x1x2304xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %cst : tensor<1x1x2304xf32>, tensor<1x1x2304xf32>) outs(%2 : tensor<1x1x2304xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<1x1x2304xf32>
    %10 = util.optimization_barrier %3 : tensor<1x1x2304xf32>
    flow.return %3 : tensor<1x1x2304xf32>
  }
  util.return
}
// CHECK-LABEL: util.func public @uncollapsable_consumer
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]+]]
//  CHECK-DAG:    %[[CST:.+]] = arith.constant
//      CHECK:     %{{.+}} = flow.dispatch.region
//      CHECK:        %[[RES:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[ARG0]], %[[CST]]
//     CHECK:        %[[BARRIER:.+]] = util.optimization_barrier %[[RES]]
//     CHECK:        flow.return %[[RES]]

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
util.func public @uncollapsable_consumer_partial(%arg0: tensor<10x20x30x2304xf32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<10x20x30x2304xf32>
  %0 = tensor.empty() : tensor<30x2304xf32>
  %1 = flow.dispatch.region -> (tensor<30x2304xf32>) {
    %2 = tensor.empty() : tensor<30x2304xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %cst : tensor<10x20x30x2304xf32>, tensor<10x20x30x2304xf32>) outs(%2 : tensor<30x2304xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<30x2304xf32>
    %10 = util.optimization_barrier %3 : tensor<30x2304xf32>
    flow.return %3 : tensor<30x2304xf32>
  }
  util.return
}
// CHECK-LABEL: util.func public @uncollapsable_consumer_partial
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]+]]
//  CHECK-DAG:    %[[CST:.+]] = arith.constant
//      CHECK:     %{{.+}} = flow.dispatch.region
//      CHECK:        %[[RES:.+]] = linalg.generic
// CHECK-SAME:         iterator_types = ["parallel", "parallel", "reduction"]
//     CHECK:        %[[BARRIER:.+]] = util.optimization_barrier %[[RES]]
//     CHECK:        flow.return %[[RES]]

// -----

util.func @elementwise_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32>{
  %cst_0 = arith.constant 0 : index
  %cst_1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %cst_0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %cst_1 : tensor<?x?xf32>
  %3 = flow.dispatch.region -> (tensor<?x?xf32>{%0, %1}) {
    %5 = tensor.empty(%0, %1) : tensor<?x?xf32>
    %cst = arith.constant 1.000000e+02 : f32
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.addf %in, %cst : f32
      linalg.yield %7 : f32
    } -> tensor<?x?xf32>
    flow.return %6 : tensor<?x?xf32>
  }
  util.return %3 : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @elementwise_dynamic
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]+]]
//  CHECK-SAME:   %[[ARG1:[0-9a-zA-Z]+]]
//   CHECK-DAG:   %[[CST0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[CST1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[CST0]]
//   CHECK-DAG:   %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[CST1]]
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[VAL:.+]] = linalg.generic
//  CHECK-SAME:      iterator_types = ["parallel"]
//       CHECK:     flow.return %[[VAL]] : tensor<?xf32>
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[DISPATCH]]
//  CHECK-SAME:     {{.+}} output_shape [%[[DIM0]], %[[DIM1]]]
//       CHECK:   util.return %[[EXPAND]] : tensor<?x?xf32>
