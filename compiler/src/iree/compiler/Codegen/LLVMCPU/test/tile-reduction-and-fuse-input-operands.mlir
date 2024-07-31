// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-reduction-and-fuse-input-operands{tiling-level=2}), canonicalize)"  --split-input-file %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 32, 0, 0, 0, 0], [1, 16, 1, 1, 0, 0], [0, 0, 0, 0, 1, 5], [0, 0, 0, 0, 0, 0]]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @dequant_avgpool(%arg0: tensor<1x320x65x65xi8>) -> tensor<1x320x1x1xf32> {
  %cst = arith.constant 1.250000e-01 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c5408000 = arith.constant 5408000 : index
  %c0 = arith.constant 0 : index
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

// CHECK-LABEL:   func.func @dequant_avgpool(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<1x320x65x65xi8>) -> tensor<1x320x1x1xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 65 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1.250000e-01 : f32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<1x320x1x1xf32>
// CHECK:           %[[VAL_8:.*]] = scf.for %[[VAL_9:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_10:.*]] = %[[VAL_7]]) -> (tensor<1x320x1x1xf32>) {
// CHECK:             %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_1]] iter_args(%[[ITER_ARG:.*]] = %[[VAL_10]]) -> (tensor<1x320x1x1xf32>) {
// CHECK:               %[[VAL_14:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, %[[VAL_9]], %[[VAL_12]]] [1, 320, 1, 5] [1, 1, 1, 1] : tensor<1x320x65x65xi8> to tensor<1x320x1x5xi8>
// CHECK:               %[[VAL_15:.*]] = tensor.empty() : tensor<1x320x1x5xf32>
// CHECK:               %[[VAL_16:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_14]] : tensor<1x320x1x5xi8>) outs(%[[VAL_15]] : tensor<1x320x1x5xf32>) {
// CHECK:               ^bb0(%[[VAL_17:.*]]: i8, %[[VAL_18:.*]]: f32):
// CHECK:                 %[[VAL_19:.*]] = arith.extsi %[[VAL_17]] : i8 to i32
// CHECK:                 %[[VAL_20:.*]] = arith.sitofp %[[VAL_19]] : i32 to f32
// CHECK:                 %[[VAL_21:.*]] = arith.mulf %[[VAL_20]], %[[VAL_4]] : f32
// CHECK:                 linalg.yield %[[VAL_21]] : f32
// CHECK:               } -> tensor<1x320x1x5xf32>
// CHECK:               %[[VAL_22:.*]] = tensor.empty() : tensor<1x5xf32>
// CHECK:               %[[VAL_23:.*]] = linalg.fill ins(%[[VAL_5]] : f32) outs(%[[VAL_22]] : tensor<1x5xf32>) -> tensor<1x5xf32>
// CHECK:               %[[RED:.*]] = linalg.pooling_nchw_sum {lowering_config = #config} ins(%[[VAL_16]], %[[VAL_23]] : tensor<1x320x1x5xf32>, tensor<1x5xf32>) outs(%[[ITER_ARG]] : tensor<1x320x1x1xf32>) -> tensor<1x320x1x1xf32>
// CHECK:               scf.yield %[[RED]] : tensor<1x320x1x1xf32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_11]] : tensor<1x320x1x1xf32>
// CHECK:           }
// CHECK:           return %[[VAL_8]] : tensor<1x320x1x1xf32>
// CHECK:         }
