// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-root-and-fuse-producer-consumer{tiling-level=0}), canonicalize)"  --split-input-file %s | FileCheck %s

#config1 = #iree_codegen.lowering_config<tile_sizes = [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 16, 16, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0]]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
func.func @mmt4d_bias_relu(%arg0: tensor<?x?x16x1xf32>, %arg1: tensor<?x?x16x1xf32>, %arg2: tensor<?x16xf32>) -> tensor<?x?x16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x16x1xf32>
  %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?x16x1xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?x16x16xf32>
  %1 = tensor.empty(%dim, %dim_0) : tensor<?x?x16x16xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  %3 = linalg.mmt4d {lowering_config = #config1} ins(%arg0, %arg1 : tensor<?x?x16x1xf32>, tensor<?x?x16x1xf32>) outs(%2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %arg2 : tensor<?x?x16x16xf32>, tensor<?x16xf32>) outs(%1 : tensor<?x?x16x16xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %5 = arith.addf %in, %in_1 : f32
    %6 = arith.maximumf %5, %cst : f32
    linalg.yield %6 : f32
  } -> tensor<?x?x16x16xf32>
  return %4 : tensor<?x?x16x16xf32>
}
//      CHECK: func.func @mmt4d_bias_relu(
//      CHECK:   scf.for
// CHECK-SAME:   {
//      CHECK:     linalg.fill
//      CHECK:     linalg.mmt4d
//      CHECK:     linalg.generic
//      CHECK:   }

// -----
#config = #iree_codegen.lowering_config<tile_sizes = [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 16, 16, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0]]>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
func.func @quantized_matmul(%arg0: tensor<2x4x128x16x1xi8>, %arg1: tensor<2x4x16xf32>, %arg2: tensor<2x4x16xf32>, %arg3: tensor<2x688x128x16x1xi8>, %arg4: tensor<2x688x16xf32>, %arg5: tensor<2x688x16xf32>) -> tensor<2x11008x64xf32> {
  %c2995200 = arith.constant 2995200 : index
  %c2994688 = arith.constant 2994688 : index
  %c2994176 = arith.constant 2994176 : index
  %c176128 = arith.constant 176128 : index
  %c88064 = arith.constant 88064 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x4x128x16x1xf32>
  %1 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<2x4x128x16x1xi8>, tensor<2x4x16xf32>, tensor<2x4x16xf32>) outs(%0 : tensor<2x4x128x16x1xf32>) {
  ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
    %8 = arith.extui %in : i8 to i32
    %9 = arith.uitofp %8 : i32 to f32
    %10 = arith.subf %9, %in_1 : f32
    %11 = arith.mulf %10, %in_0 : f32
    linalg.yield %11 : f32
  } -> tensor<2x4x128x16x1xf32>
  %2 = tensor.empty() : tensor<2x688x128x16x1xf32>
  %3 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg3, %arg4, %arg5 : tensor<2x688x128x16x1xi8>, tensor<2x688x16xf32>, tensor<2x688x16xf32>) outs(%2 : tensor<2x688x128x16x1xf32>) {
  ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
    %8 = arith.extui %in : i8 to i32
    %9 = arith.uitofp %8 : i32 to f32
    %10 = arith.subf %9, %in_1 : f32
    %11 = arith.mulf %10, %in_0 : f32
    linalg.yield %11 : f32
  } -> tensor<2x688x128x16x1xf32>
  %4 = tensor.empty() : tensor<2x4x688x16x16xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<2x4x688x16x16xf32>) -> tensor<2x4x688x16x16xf32>
  %6 = linalg.batch_mmt4d {lowering_config = #config} ins(%1, %3 : tensor<2x4x128x16x1xf32>, tensor<2x688x128x16x1xf32>) outs(%5 : tensor<2x4x688x16x16xf32>) -> tensor<2x4x688x16x16xf32>
  %7 = tensor.empty() : tensor<2x11008x64xf32>
  %unpack = tensor.unpack %6 outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [16, 16] into %7 : tensor<2x4x688x16x16xf32> -> tensor<2x11008x64xf32>
  return %unpack : tensor<2x11008x64xf32>
}
//      CHECK: func.func @quantized_matmul(
//      CHECK:  scf.for
// CHECK-SAME:   {
//      CHECK:       linalg.generic
//      CHECK:       linalg.generic
//      CHECK:       linalg.fill
//      CHECK:       linalg.batch_mmt4d
//      CHECK:       tensor.unpack
//      CHECK:   }
