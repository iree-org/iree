// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-root-and-fuse-producer-consumer{tiling-level=0}), canonicalize)"  --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-root-and-fuse-producer-consumer{tiling-level=2 only-fuse-producer-input-operands=true}), canonicalize)"  --split-input-file %s | FileCheck %s --check-prefix=CHECK-REDUCTION

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 16, 16, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0]]>
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
  %3 = linalg.mmt4d {lowering_config = #config} ins(%arg0, %arg1 : tensor<?x?x16x1xf32>, tensor<?x?x16x1xf32>) outs(%2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %arg2 : tensor<?x?x16x16xf32>, tensor<?x16xf32>) outs(%1 : tensor<?x?x16x16xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %5 = arith.addf %in, %in_1 : f32
    %6 = arith.maximumf %5, %cst : f32
    linalg.yield %6 : f32
  } -> tensor<?x?x16x16xf32>
  return %4 : tensor<?x?x16x16xf32>
}
// CHECK-LABEL: func.func @mmt4d_bias_relu(
// CHECK:         scf.for
// CHECK:           linalg.fill
// CHECK-NEXT:      %[[MMT4D:.+]] = linalg.mmt4d
// CHECK:           %[[ELEM:.+]] = linalg.generic
// CHECK:           %[[RES0:.+]] =  tensor.insert_slice %[[MMT4D]]
// CHECK:           %[[RES1:.+]] =  tensor.insert_slice %[[ELEM]]
// CHECK:           scf.yield %[[RES0]], %[[RES1]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 16, 16, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0]]>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
func.func @quantized_matmul(%arg0: tensor<2x4x128x16x1xi8>, %arg1: tensor<2x4x16xf32>, %arg2: tensor<2x4x16xf32>, %arg3: tensor<2x688x128x16x1xi8>, %arg4: tensor<2x688x16xf32>, %arg5: tensor<2x688x16xf32>) -> tensor<2x11008x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x4x128x16x1xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<2x4x128x16x1xi8>, tensor<2x4x16xf32>, tensor<2x4x16xf32>) outs(%0 : tensor<2x4x128x16x1xf32>) {
  ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
    %8 = arith.extui %in : i8 to i32
    %9 = arith.uitofp %8 : i32 to f32
    %10 = arith.subf %9, %in_1 : f32
    %11 = arith.mulf %10, %in_0 : f32
    linalg.yield %11 : f32
  } -> tensor<2x4x128x16x1xf32>
  %2 = tensor.empty() : tensor<2x688x128x16x1xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg3, %arg4, %arg5 : tensor<2x688x128x16x1xi8>, tensor<2x688x16xf32>, tensor<2x688x16xf32>) outs(%2 : tensor<2x688x128x16x1xf32>) {
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
  %unpack = linalg.unpack %6 outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [16, 16] into %7 : tensor<2x4x688x16x16xf32> -> tensor<2x11008x64xf32>
  return %unpack : tensor<2x11008x64xf32>
}
// CHECK: func.func @quantized_matmul(
// CHECK:  scf.for
// CHECK:    linalg.generic
// CHECK:    linalg.generic
// CHECK:    linalg.fill
// CHECK:    %[[MMT4D:.+]] = linalg.batch_mmt4d
// CHECK:    %[[UNPACK:.+]] = linalg.unpack
// CHECK:    %[[RES0:.+]] =  tensor.insert_slice %[[MMT4D]]
// CHECK:    %[[RES1:.+]] =  tensor.insert_slice %[[UNPACK]]
// CHECK:    scf.yield %[[RES0]], %[[RES1]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 32, 0, 0, 0, 0], [1, 16, 1, 1, 0, 0], [0, 0, 0, 0, 1, 5], [0, 0, 0, 0, 0, 0]]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @dequant_avgpool(%arg0: tensor<1x320x65x65xi8>) -> tensor<1x320x1x1xf32> {
  %cst = arith.constant 1.250000e-01 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x320x1x1xf32>
  %1 = tensor.empty() : tensor<65x65xf32>
  %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<65x65xf32>) -> tensor<65x65xf32>
  %3 = tensor.empty() : tensor<1x320x65x65xf32>
  %4 = tensor.empty() : tensor<1x320x1x1xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x320x65x65xi8>) outs(%3 : tensor<1x320x65x65xf32>) {
  ^bb0(%in: i8, %out: f32):
    %7 = arith.extsi %in : i8 to i32
    %8 = arith.sitofp %7 : i32 to f32
    %9 = arith.mulf %8, %cst : f32
    linalg.yield %9 : f32
  } -> tensor<1x320x65x65xf32>
  %6 = linalg.pooling_nchw_sum {lowering_config = #config} ins(%5, %2 : tensor<1x320x65x65xf32>, tensor<65x65xf32>) outs(%4 : tensor<1x320x1x1xf32>) -> tensor<1x320x1x1xf32>
  return %6 : tensor<1x320x1x1xf32>
}
// CHECK-REDUCTION-LABEL: func.func @dequant_avgpool(
// CHECK-REDUCTION:         scf.for
// CHECK-REDUCTION:           scf.for
// CHECK-REDUCTION:             %[[DEQUANT:.+]] = linalg.generic
// CHECK-REDUCTION:             %[[POOL:.+]] = linalg.pooling_nchw_sum
// CHECK-REDUCTION-SAME:          ins(%[[DEQUANT]]
// CHECK-REDUCTION:             scf.yield %[[POOL]]

// -----

// Silently bail in the case of no root op.
func.func @silently_bail_no_root_op(%arg0: tensor<1x2x1x2xi8>, %arg1: tensor<1x2x4x2xi8>, %arg2: tensor<1x1x1x4xi32>) -> tensor<1x1x1x4xi32> {
  %c1794_i32 = arith.constant 1794 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %0:2 = iree_codegen.ukernel.generic "iree_uk_mmt4d" ins(%arg0, %arg1 : tensor<1x2x1x2xi8>, tensor<1x2x4x2xi8>) outs(%arg2 : tensor<1x1x1x4xi32>) (%c1, %c1, %c2, %c1_i32, %c4_i32, %c2_i32, %c1794_i32 : index, index, index, i32, i32, i32, i32) fn_def_attrs {hal.import.bitcode = true, hal.import.fields = ["processor_data"]} strided_outer_dims(1) -> tensor<1x1x1x4xi32>, i32
  return %0#0 : tensor<1x1x1x4xi32>
}
// CHECK-REDUCTION-LABEL: func.func @silently_bail_no_root_op(
// CHECK-REDUCTION-NOT:     scf.for
