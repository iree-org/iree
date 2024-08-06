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

#config2 = #iree_codegen.lowering_config<tile_sizes = [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 16, 16, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0]]>
func.func @quantized_matmul() {
  %c2995200 = arith.constant 2995200 : index
  %c2994688 = arith.constant 2994688 : index
  %c2994176 = arith.constant 2994176 : index
  %c176128 = arith.constant 176128 : index
  %c88064 = arith.constant 88064 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c2995200) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4x128x16x1xi8>>
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c2994688) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4x16xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c2994176) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4x16xf32>>
  %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c176128) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x688x128x16x1xi8>>
  %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c88064) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x688x16xf32>>
  %5 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x688x16xf32>>
  %6 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x11008x64xf32>>
  %7 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0], sizes = [2, 4, 128, 16, 1], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x128x16x1xi8>> -> tensor<2x4x128x16x1xi8>
  %8 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, 4, 16], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x16xf32>> -> tensor<2x4x16xf32>
  %9 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [2, 4, 16], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x16xf32>> -> tensor<2x4x16xf32>
  %10 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0, 0], sizes = [2, 688, 128, 16, 1], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x688x128x16x1xi8>> -> tensor<2x688x128x16x1xi8>
  %11 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [2, 688, 16], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x688x16xf32>> -> tensor<2x688x16xf32>
  %12 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [2, 688, 16], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x688x16xf32>> -> tensor<2x688x16xf32>
  %13 = tensor.empty() : tensor<2x4x128x16x1xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%7, %8, %9 : tensor<2x4x128x16x1xi8>, tensor<2x4x16xf32>, tensor<2x4x16xf32>) outs(%13 : tensor<2x4x128x16x1xf32>) {
  ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
    %21 = arith.extui %in : i8 to i32
    %22 = arith.uitofp %21 : i32 to f32
    %23 = arith.subf %22, %in_1 : f32
    %24 = arith.mulf %23, %in_0 : f32
    linalg.yield %24 : f32
  } -> tensor<2x4x128x16x1xf32>
  %15 = tensor.empty() : tensor<2x688x128x16x1xf32>
  %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%10, %11, %12 : tensor<2x688x128x16x1xi8>, tensor<2x688x16xf32>, tensor<2x688x16xf32>) outs(%15 : tensor<2x688x128x16x1xf32>) {
  ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
    %21 = arith.extui %in : i8 to i32
    %22 = arith.uitofp %21 : i32 to f32
    %23 = arith.subf %22, %in_1 : f32
    %24 = arith.mulf %23, %in_0 : f32
    linalg.yield %24 : f32
  } -> tensor<2x688x128x16x1xf32>
  %17 = tensor.empty() : tensor<2x4x688x16x16xf32>
  %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<2x4x688x16x16xf32>) -> tensor<2x4x688x16x16xf32>
  %19 = linalg.batch_mmt4d {lowering_config = #config2} ins(%14, %16 : tensor<2x4x128x16x1xf32>, tensor<2x688x128x16x1xf32>) outs(%18 : tensor<2x4x688x16x16xf32>) -> tensor<2x4x688x16x16xf32>
  %20 = tensor.empty() : tensor<2x11008x64xf32>
  %unpack = tensor.unpack %19 outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [16, 16] into %20 : tensor<2x4x688x16x16xf32> -> tensor<2x11008x64xf32>
  flow.dispatch.tensor.store %unpack, %6, offsets = [0, 0, 0], sizes = [2, 11008, 64], strides = [1, 1, 1] : tensor<2x11008x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x11008x64xf32>>
  return
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
