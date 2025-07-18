// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-root-and-fuse-producer-consumer{tiling-level=vector_common_parallel}), cse)"  --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-root-and-fuse-producer-consumer{tiling-level=vector_reduction only-fuse-producer-input-operands=true}), cse)"  --split-input-file %s | FileCheck %s --check-prefix=CHECK-REDUCTION

#config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0, 0, 0, 0, 0]>
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
// CHECK:         scf.forall
// CHECK:           linalg.fill
// CHECK-NEXT:      %[[MMT4D:.+]] = linalg.mmt4d
// CHECK:           %[[ELEM:.+]] = linalg.generic
// CHECK:           scf.forall.in_parallel
// CHECK:             tensor.parallel_insert_slice %[[ELEM]]

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0, 0, 0, 0, 0]>
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
// CHECK-LABEL: func.func @quantized_matmul(
// CHECK:         scf.forall
// CHECK:           linalg.generic
// CHECK:           linalg.generic
// CHECK:           linalg.fill
// CHECK:           %[[MMT4D:.+]] = linalg.batch_mmt4d
// CHECK:           %[[UNPACK:.+]] = linalg.unpack
// CHECK:           scf.forall.in_parallel
// CHECK:             tensor.parallel_insert_slice %[[UNPACK]]

// -----

#config = #iree_cpu.lowering_config<vector_reduction = [0, 0, 0, 0, 1, 5]>
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
  %0:2 = iree_codegen.ukernel.generic "iree_uk_mmt4d" ins(%arg0, %arg1 : tensor<1x2x1x2xi8>, tensor<1x2x4x2xi8>) outs(%arg2 : tensor<1x1x1x4xi32>) (%c1, %c1, %c2, %c1_i32, %c4_i32, %c2_i32, %c1794_i32 : index, index, index, i32, i32, i32, i32) fn_def_attrs {hal.import.bitcode = true, hal.import.fields = ["processor_data"]} strided_dims([[0], [0], [0]]) -> tensor<1x1x1x4xi32>, i32
  return %0#0 : tensor<1x1x1x4xi32>
}
// CHECK-REDUCTION-LABEL: func.func @silently_bail_no_root_op(
// CHECK-REDUCTION-NOT:     scf.for

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [4, 8, 0]>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @multi_use_producer_no_yield_replacement(%7: tensor<12x197x197xf32>) -> tensor<12x197x197xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant -3.40282347E+38 : f32
  %8 = tensor.empty() : tensor<12x197x197xf32>
  %9 = tensor.empty() : tensor<12x197xf32>
  %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<12x197xf32>) -> tensor<12x197xf32>
  %max = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%7 : tensor<12x197x197xf32>) outs(%10 : tensor<12x197xf32>) {
  ^bb0(%in: f32, %out: f32):
    %15 = arith.maxnumf %in, %out : f32
    linalg.yield %15 : f32
  } -> tensor<12x197xf32>
  %12 = linalg.fill ins(%cst : f32) outs(%9 : tensor<12x197xf32>) -> tensor<12x197xf32>
  %exp_sum = linalg.generic {
    indexing_maps = [#map, #map1, #map1],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%7, %max : tensor<12x197x197xf32>, tensor<12x197xf32>)
    outs(%12 : tensor<12x197xf32>) attrs =  { lowering_config = #config } {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %15 = arith.subf %in, %in_1 : f32
    %16 = math.exp %15 : f32
    %17 = arith.addf %16, %out : f32
    linalg.yield %17 : f32
  } -> tensor<12x197xf32>
  %softmax:2 = linalg.generic {
    indexing_maps = [#map, #map1, #map1, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%7, %max, %exp_sum : tensor<12x197x197xf32>, tensor<12x197xf32>, tensor<12x197xf32>)
    outs(%8, %8 : tensor<12x197x197xf32>, tensor<12x197x197xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
    %15 = arith.subf %in, %in_1 : f32
    %16 = math.exp %15 : f32
    %17 = arith.divf %16, %in_2 : f32
    linalg.yield %16, %17 : f32, f32
  } -> (tensor<12x197x197xf32>, tensor<12x197x197xf32>)
  return %softmax#1 : tensor<12x197x197xf32>
}
// CHECK-LABEL: func @multi_use_producer_no_yield_replacement(
//       CHECK:   %[[RESULT:.+]] = scf.forall
//       CHECK:     %[[MAX:.+]] = linalg.generic
//       CHECK:       arith.maxnumf
//       CHECK:     %[[EXPSUM:.+]] = linalg.generic
//  CHECK-SAME:       ins(%{{.*}}, %[[MAX]]
//       CHECK:       arith.subf
//       CHECK:       math.exp
//       CHECK:       arith.addf
//       CHECK:     %[[EXPDIV:.+]] = linalg.generic
//  CHECK-SAME:       ins(%{{.*}}, %[[MAX]], %[[EXPSUM]]
//       CHECK:       arith.subf
//       CHECK:       math.exp
//       CHECK:       arith.divf
//       CHECK:   return %[[RESULT]]

// -----

// The test case demonstrates that the rootOp can mismatch the result of
// `getRootOperation()` method. It prioritizes the operation that has workgroup
// tiling level, if only one op has such config.

#config = #iree_cpu.lowering_config<vector_common_parallel = [1, 4]>
#config1 = #iree_cpu.lowering_config<vector_common_parallel = [1, 4, 0]>
#config2 = #iree_cpu.lowering_config<distribution = [10, 32, 0], vector_common_parallel = [1, 4, 0]>
#config3 = #iree_cpu.lowering_config<vector_common_parallel = [1, 4, 0]>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @reordered_softmax(%arg0: tensor<1x8x4096xf32>, %arg1: tensor<1x8x4096xf32>, %arg2: tensor<1x8x4096xf32>, %arg3: tensor<1x8x4096xf32>, %arg4: tensor<1x8x4096xf32>) -> tensor<1x8x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFFC00000 : f32
  %0 = tensor.empty() : tensor<1x8xf32>
  %1 = linalg.fill {lowering_config = #config} ins(%cst_0 : f32) outs(%0 : tensor<1x8xf32>) -> tensor<1x8xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1 : tensor<1x8x4096xf32>) outs(%1 : tensor<1x8xf32>) attrs =  {lowering_config = #config1} {
  ^bb0(%in: f32, %out: f32):
    %7 = arith.maxnumf %in, %out : f32
    linalg.yield %7 : f32
  } -> tensor<1x8xf32>
  %3 = linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%0 : tensor<1x8xf32>) -> tensor<1x8xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %2 : tensor<1x8x4096xf32>, tensor<1x8xf32>) outs(%3 : tensor<1x8xf32>) attrs =  {lowering_config = #config2} {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %7 = arith.subf %in, %in_1 : f32
    %8 = math.exp %7 : f32
    %9 = arith.addf %8, %out : f32
    linalg.yield %9 : f32
  } -> tensor<1x8xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg2 : tensor<1x8x4096xf32>) outs(%1 : tensor<1x8xf32>) attrs =  {lowering_config = #config1} {
  ^bb0(%in: f32, %out: f32):
    %7 = arith.maxnumf %in, %out : f32
    linalg.yield %7 : f32
  } -> tensor<1x8xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg3, %5, %4 : tensor<1x8x4096xf32>, tensor<1x8xf32>, tensor<1x8xf32>) outs(%arg4 : tensor<1x8x4096xf32>) attrs =  {lowering_config = #config3} {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32):
    %7 = arith.subf %in, %in_1 : f32
    %8 = math.exp %7 : f32
    %9 = arith.divf %8, %in_2 : f32
    linalg.yield %9 : f32
  } -> tensor<1x8x4096xf32>
  return %6 : tensor<1x8x4096xf32>
}
// CHECK-LABEL: func @reordered_softmax(
//       CHECK:   %[[RESULT:.+]] = scf.forall
//       CHECK:     %[[MAX:.+]] = linalg.generic
//       CHECK:       arith.maxnumf
//       CHECK:     %[[EXPSUM:.+]] = linalg.generic
//  CHECK-SAME:       ins(%{{.*}}, %[[MAX]]
//       CHECK:       arith.subf
//       CHECK:       math.exp
//       CHECK:       arith.addf
//       CHECK:     %[[MAX2:.+]] = linalg.generic
//       CHECK:       arith.maxnumf
//       CHECK:     %[[EXPDIV:.+]] = linalg.generic
//  CHECK-SAME:       ins(%{{.*}}, %[[MAX2]], %[[EXPSUM]]
//       CHECK:       arith.subf
//       CHECK:       math.exp
//       CHECK:       arith.divf
//       CHECK:   return %[[RESULT]]

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [10, 20, 30]>
func.func @matmul_bias_add(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #config}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%1, %arg2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//      CHECK: func.func @matmul_bias_add(
//      CHECK:   scf.forall
//      CHECK:       linalg.fill
//      CHECK:       linalg.matmul
//      CHECK:       linalg.generic
//      CHECK:   scf.forall.in_parallel {

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [0, 0, 0]>
func.func @all_zeros(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #config}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%1, %arg2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//      CHECK: func.func @all_zeros(
//  CHECK-NOT:   scf.forall

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [1, [32], 0]>
func.func @scalable_matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32>{
  // Matrix multiplication (ijk) with scalable tiling in the j-th dimension.
  %1 = linalg.matmul {lowering_config = #config} ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @scalable_matmul(
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
//       CHECK: %[[VSCALE:.*]] = vector.vscale
//  CHECK-NEXT: %[[SCALABLE_TILE_SIZE:.*]] = arith.muli %[[VSCALE]], %[[C32]] : index
//       CHECK: scf.forall
//  CHECK-SAME:       step (1, %[[SCALABLE_TILE_SIZE]])

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [0, 20, 0]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @ukernel_generic(%arg0: tensor<1x192x1x16xf32>, %arg1: tensor<1x768x1x1xf32>, %arg2: tensor<192x768x16x1xf32>, %arg3: tensor<1x192x1x16xf32>) -> tensor<1x192x1x16xf32> {
  %c1 = arith.constant 1 : index
  %c192 = arith.constant 192 : index
  %c768 = arith.constant 768 : index
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1025_i32 = arith.constant 1025 : i32
  %0 = tensor.empty() : tensor<1x192x1x16xf32>
  %1 = iree_codegen.ukernel.generic "iree_uk_mmt4d" ins(%arg1, %arg2 : tensor<1x768x1x1xf32>, tensor<192x768x16x1xf32>) outs(%0 : tensor<1x192x1x16xf32>) (%c1, %c192, %c768, %c1_i32, %c16_i32, %c1_i32, %c1025_i32 : index, index, index, i32, i32, i32, i32) fn_def_attrs {hal.import.bitcode = true, hal.import.fields = ["processor_data"]} strided_dims([[0], [0], [0]]) -> tensor<1x192x1x16xf32>
  %2 = linalg.generic { indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"] }
    ins(%1, %arg3 : tensor<1x192x1x16xf32>, tensor<1x192x1x16xf32>)
    outs(%arg0 : tensor<1x192x1x16xf32>)
    attrs = {lowering_config = #config} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.addf %in, %in_0 : f32
    linalg.yield %3 : f32
  } -> tensor<1x192x1x16xf32>
  return %2 : tensor<1x192x1x16xf32>
}
// CHECK-LABEL: func.func @ukernel_generic
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]
// CHECK:         %[[UK:.+]] = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK:         scf.forall {{.+}} shared_outs(%[[ITER:.+]] = %[[ARG0]])
// CHECK:           %[[UK_SLICE:.+]] = tensor.extract_slice %[[UK]]
// CHECK:           %[[ARG3_SLICE:.+]] = tensor.extract_slice %[[ARG3]]
// CHECK:           %[[ITER_SLICE:.+]] = tensor.extract_slice %[[ITER]]
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%[[UK_SLICE]], %[[ARG3_SLICE]]
// CHECK-SAME:        outs(%[[ITER_SLICE]]

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [0, 1]>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @tile_linalg_ext_scan(%arg0: tensor<128x2xf32>) -> tensor<128x2xi64> {
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty() : tensor<2xi64>
  %1 = tensor.empty() : tensor<128x2xi64>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<128x2xf32>)
    outs(%1 : tensor<128x2xi64>) {
  ^bb0(%in: f32, %out: i64):
    %6 = arith.fptosi %in : f32 to i64
    linalg.yield %6 : i64
  } -> tensor<128x2xi64>
  %3 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<2xi64>) -> tensor<2xi64>
  %4 = linalg.fill ins(%c0_i64 : i64) outs(%1 : tensor<128x2xi64>) -> tensor<128x2xi64>
  %5:2 = iree_linalg_ext.scan {lowering_config = #config} dimension(0) inclusive(true)
    ins(%2 : tensor<128x2xi64>)
    outs(%4, %3 : tensor<128x2xi64>, tensor<2xi64>) {
  ^bb0(%arg1: i64, %arg2: i64):
    %6 = arith.addi %arg1, %arg2 : i64
    iree_linalg_ext.yield %6 : i64
  } -> tensor<128x2xi64>, tensor<2xi64>
  return %5#0 : tensor<128x2xi64>
}
// CHECK-LABEL: func.func @tile_linalg_ext_scan
// CHECK:         scf.forall
// CHECK:           linalg.generic
// CHECK:           linalg.fill
// CHECK:           linalg.fill
// CHECK:           iree_linalg_ext.scan
// CHECK:         scf.forall.in_parallel {

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [1, 16]>
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @infusible_pack(%arg0 : tensor<30xf32>) -> tensor<5x6xf32> {
  %empty = tensor.empty() : tensor<30xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<30xf32>) outs(%empty : tensor<30xf32>)
      attrs = {lowering_config = #config} {
    ^bb0(%b0 : f32, %b1 : f32) :
      %1 = arith.addf %b0, %b0 : f32
      linalg.yield %1 : f32
  } -> tensor<30xf32>
  %empty1 = tensor.empty() : tensor<5x6xf32>
  %pack = linalg.pack %0 outer_dims_perm = [0]
      inner_dims_pos = [0] inner_tiles = [6] into %empty1
      : tensor<30xf32> -> tensor<5x6xf32>
  return %pack : tensor<5x6xf32>
}
// CHECK-LABEL: func.func @infusible_pack
// CHECK:         scf.forall
// CHECK:           linalg.generic
// CHECK:         scf.forall.in_parallel
// CHECK:         linalg.pack
