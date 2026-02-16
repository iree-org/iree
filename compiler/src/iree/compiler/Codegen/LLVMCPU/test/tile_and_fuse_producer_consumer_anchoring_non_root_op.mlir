// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-and-fuse-producer-consumer{tiling-level=vector_common_parallel anchor-on-root-op=false}), cse)" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-and-fuse-producer-consumer{tiling-level=vector_inner_parallel anchor-on-root-op=false}, cse))" --split-input-file %s | FileCheck %s --check-prefix=INNER-PARALLEL

#config = #iree_cpu.lowering_config<vector_common_parallel = [10, 20, 0]>
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
//      CHECK:     linalg.fill
//      CHECK:     linalg.matmul
//      CHECK:     linalg.generic

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
//  CHECK-NOT:   scf.for
//      CHECK:   linalg.fill
//      CHECK:   linalg.matmul
//      CHECK:   linalg.generic

// -----

#config0 = #iree_cpu.lowering_config<vector_common_parallel = [0, 0]>
#config1 = #iree_cpu.lowering_config<vector_common_parallel = [10, 20]>
func.func @multi_config(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #config0}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%1, %arg2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%init : tensor<?x?xf32>) attrs = {lowering_config = #config1} {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
// Both linalg.matmul op and linalg.generic op have lowering_config. Test that
// the lowering_config of linalg.generic op is picked in the pass. In this case,
// an scf.forall op is created. If the lowering_config of linalg.matmul op is
// picked, the scf.forall is not generated. Because the tiling sizes are zeros.
//      CHECK: func.func @multi_config(
//      CHECK:   scf.forall
//      CHECK:       linalg.fill
//      CHECK:       linalg.matmul
//      CHECK:       linalg.generic

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [8, 128, 0]>
func.func @shared_out_operand(%arg0: tensor<391x384xf32>, %arg1: tensor<384x384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<391x384xf32>) -> tensor<391x384xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 6.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg3 : tensor<391x384xf32>) -> tensor<391x384xf32>
  %1 = linalg.matmul {lowering_config = #config}
    ins(%arg0, %arg1 : tensor<391x384xf32>, tensor<384x384xf32>)
    outs(%0 : tensor<391x384xf32>) -> tensor<391x384xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%arg2, %1 : tensor<384xf32>, tensor<391x384xf32>)
    outs(%arg3 : tensor<391x384xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.addf %in, %in_1 : f32
    %4 = arith.minimumf %3, %cst_0 : f32
    %5 = arith.maximumf %4, %cst : f32
    linalg.yield %5 : f32
  } -> tensor<391x384xf32>
  return %2 : tensor<391x384xf32>
}
//      CHECK: func.func @shared_out_operand(
// CHECK-SAME:    %{{[a-zA-Z0-9]+}}
// CHECK-SAME:    %{{[a-zA-Z0-9]+}}
// CHECK-SAME:    %{{[a-zA-Z0-9]+}}
// CHECK-SAME:    %[[DST:[a-zA-Z0-9]+]]
//  CHECK-DAG:   %[[CST0:.+]] = arith.constant 0.000000e+00 : f32
//      CHECK:   scf.forall {{.+}} shared_outs(%[[ITER:.+]] = %[[DST]])
//      CHECK:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[DST]]
//      CHECK:       %{{.+}} = linalg.fill ins(%[[CST0]] : f32) outs(%[[OUT_SLICE]]
//      CHECK:       %{{.+}} = linalg.matmul
//      CHECK:       %[[OUT_SLICE2:.+]] = tensor.extract_slice %[[ITER]]
//      CHECK:       %{{.+}} = linalg.generic
// CHECK-SAME:         outs(%[[OUT_SLICE2]]

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [1, [32], 0]>
func.func @scalable_matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32>{
  // Matrix multiplication (ijk) with scalable tiling in the j-th dimension.
  %1 = linalg.matmul {lowering_config = #config} ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @scalable_matmul(
//   CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
//       CHECK: %[[VSCALE:.*]] = vector.vscale
//  CHECK-NEXT: %[[SCALABLE_TILE_SIZE:.*]] = arith.muli %[[VSCALE]], %[[C32]] : index
//       CHECK: scf.forall
//  CHECK-SAME:   step (1, %[[SCALABLE_TILE_SIZE]])

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
// CHECK:         scf.forall.in_parallel

// -----

// Check that the tiling here does not create an SSA violation error. See #21828

func.func @inner_parallel_SSA_violation_error(%3 : tensor<123x456xf32>) -> (tensor<123x456xf32>, tensor<123x456xf32>)  {
  %c0 = arith.constant 0 : index
  %4 = tensor.empty() : tensor<123x456xf32>
  %5:2 = scf.forall (%arg0) = (0) to (123) step (3) shared_outs(%arg1 = %4, %arg2 = %4) -> (tensor<123x456xf32>, tensor<123x456xf32>) {
    %extracted_slice = tensor.extract_slice %3[%arg0, 0] [3, 456] [1, 1] : tensor<123x456xf32> to tensor<3x456xf32>
    %6 = tensor.empty() : tensor<3x456xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<3x456xf32>) outs(%6 : tensor<3x456xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [4, 0], vector_inner_parallel = [0, 16]>} {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.addf %in, %in : f32
      linalg.yield %14 : f32
    } -> tensor<3x456xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [3, 456] [1, 1] : tensor<123x456xf32> to tensor<3x456xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<3x456xf32>) outs(%extracted_slice_0 : tensor<3x456xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [4, 0], vector_inner_parallel = [0, 16]>} {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.addf %in, %in : f32
      linalg.yield %14 : f32
    } -> tensor<3x456xf32>
    %9 = tensor.empty() : tensor<3xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%8 : tensor<3x456xf32>) outs(%9 : tensor<3xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [4, 0], vector_reduction = [0, 16]>} {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.addf %in, %in : f32
      linalg.yield %14 : f32
    } -> tensor<3xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%7, %10 : tensor<3x456xf32>, tensor<3xf32>) outs(%9 : tensor<3xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<distribution = [3, 0], vector_common_parallel = [4, 0], vector_reduction = [0, 16]>} {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %14 = arith.addf %in, %in_2 : f32
      linalg.yield %14 : f32
    } -> tensor<3xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<3x456xf32>) outs(%9 : tensor<3xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [4, 0], vector_reduction = [0, 16]>} {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.addf %in, %in : f32
      linalg.yield %14 : f32
    } -> tensor<3xf32>
    %extracted_slice_1 = tensor.extract_slice %arg2[%arg0, 0] [3, 456] [1, 1] : tensor<123x456xf32> to tensor<3x456xf32>
    %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %12, %11 : tensor<3x456xf32>, tensor<3xf32>, tensor<3xf32>) outs(%extracted_slice_1 : tensor<3x456xf32>) attrs =  {lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [4, 0], vector_inner_parallel = [0, 16]>} {
    ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
      %14 = arith.addf %in, %in_2 : f32
      %15 = arith.mulf %14, %in_3 : f32
      linalg.yield %15 : f32
    } -> tensor<3x456xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %arg1[%arg0, 0] [3, 456] [1, 1] : tensor<3x456xf32> into tensor<123x456xf32>
      tensor.parallel_insert_slice %13 into %arg2[%arg0, 0] [3, 456] [1, 1] : tensor<3x456xf32> into tensor<123x456xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  return %5#0, %5#1 : tensor<123x456xf32>, tensor<123x456xf32>
}
// Just checking that there is no failure here.
// INNER-PARALLEL-LABEL: func @inner_parallel_SSA_violation_error

// -----

#config = #iree_cpu.lowering_config<vector_inner_parallel = [0, 20, 0]>
#config1 = #iree_cpu.lowering_config<vector_inner_parallel = [16, 0, 0]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
func.func @ukernel_generic_with_broadcast(%arg0: tensor<1x192x1x16xf32>, %arg1: tensor<1x768x1x1xf32>, %arg2: tensor<768x16x1xf32>, %arg3: tensor<1x192x1x16xf32>) -> tensor<1x192x1x16xf32> {
  %c1 = arith.constant 1 : index
  %c192 = arith.constant 192 : index
  %c768 = arith.constant 768 : index
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1025_i32 = arith.constant 1025 : i32
  %0 = tensor.empty() : tensor<192x768x16x1xf32>
  %1 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg2 : tensor<768x16x1xf32>)
    outs(%0 : tensor<192x768x16x1xf32>)
    attrs = {lowering_config = #config1} {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<192x768x16x1xf32>
  %2 = tensor.empty() : tensor<1x192x1x16xf32>
  %3 = iree_codegen.ukernel.generic "iree_uk_mmt4d" ins(%arg1, %1 : tensor<1x768x1x1xf32>, tensor<192x768x16x1xf32>) outs(%2 : tensor<1x192x1x16xf32>) (%c1, %c192, %c768, %c1_i32, %c16_i32, %c1_i32, %c1025_i32 : index, index, index, i32, i32, i32, i32) fn_def_attrs {hal.import.bitcode = true, hal.import.fields = ["processor_data"]} strided_dims([[0], [0], [0]]) -> tensor<1x192x1x16xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%3, %arg3 : tensor<1x192x1x16xf32>, tensor<1x192x1x16xf32>)
    outs(%arg0 : tensor<1x192x1x16xf32>)
    attrs = {lowering_config = #config} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %5 = arith.addf %in, %in_0 : f32
    linalg.yield %5 : f32
  } -> tensor<1x192x1x16xf32>
  return %4 : tensor<1x192x1x16xf32>
}
// INNER-PARALLEL-LABEL: func.func @ukernel_generic_with_broadcast
// INNER-PARALLEL-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// INNER-PARALLEL-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// INNER-PARALLEL-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// INNER-PARALLEL-SAME:    %[[ARG3:[a-zA-Z0-9]+]]
// INNER-PARALLEL:         scf.forall
// INNER-PARALLEL:           linalg.generic
// INNER-PARALLEL-SAME:        ins(%[[ARG2]]
// INNER-PARALLEL:           scf.forall.in_parallel
// INNER-PARALLEL:         %[[UK:.+]] = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// INNER-PARALLEL:         scf.forall {{.+}} shared_outs(%[[ITER:.+]] = %[[ARG0]])
// INNER-PARALLEL:           %[[UK_SLICE:.+]] = tensor.extract_slice %[[UK]]
// INNER-PARALLEL:           %[[ARG3_SLICE:.+]] = tensor.extract_slice %[[ARG3]]
// INNER-PARALLEL:           %[[ITER_SLICE:.+]] = tensor.extract_slice %[[ITER]]
// INNER-PARALLEL:           linalg.generic
// INNER-PARALLEL-SAME:        ins(%[[UK_SLICE]], %[[ARG3_SLICE]]
// INNER-PARALLEL-SAME:        outs(%[[ITER_SLICE]]

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [16, 0]>
#config1 = #iree_cpu.lowering_config<vector_inner_parallel = [16, 1]>
#config2 = #iree_cpu.lowering_config<vector_common_parallel = [1, 1, 16, 16]>
#config3 = #iree_cpu.lowering_config<distribution = [16, 1, 0, 0, 0, 0], vector_common_parallel = [1, 1, 0, 16, 16, 0], vector_reduction = [0, 0, 1, 0, 0, 1]>
#map = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
func.func @unpack_pack_fill_mmt4d_map_store(%arg0: tensor<1x128x16x1xf32>, %arg1: tensor<1x128x16x1xf32>) -> tensor<16x128xf32> {
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<16x128xf32>
  %1 = linalg.unpack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %0 {lowering_config = #config1} : tensor<1x128x16x1xf32> -> tensor<16x128xf32>
  %2 = tensor.empty() : tensor<16x128x1x1xf32>
  %pack = linalg.pack %1 padding_value(%cst : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %2 : tensor<16x128xf32> -> tensor<16x128x1x1xf32>
  %3 = tensor.empty() : tensor<16x128xf32>
  %4 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %3) -> (tensor<16x128xf32>) {
    %5 = tensor.empty() : tensor<1x1x1x16xf32>
    %6 = linalg.fill {lowering_config = #config2} ins(%cst : f32) outs(%5 : tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %extracted_slice = tensor.extract_slice %pack[%arg2, 0, 0, 0] [1, 128, 1, 1] [1, 1, 1, 1] : tensor<16x128x1x1xf32> to tensor<1x128x1x1xf32>
    %7 = linalg.mmt4d {lowering_config = #config3} ins(%extracted_slice, %arg1 : tensor<1x128x1x1xf32>, tensor<1x128x16x1xf32>) outs(%6 : tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
    %extracted_slice_0 = tensor.extract_slice %arg3[0, 0] [16, 128] [1, 1] : tensor<16x128xf32> to tensor<16x128xf32>
    %8 = iree_linalg_ext.map_store {lowering_config = #config} %7 into %extracted_slice_0 {
    ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
      %9 = affine.apply #map(%c0, %arg4, %arg2)
      %10 = arith.cmpi ult, %9, %c16 : index
      %11 = arith.cmpi ult, %arg5, %c128 : index
      %12 = arith.andi %10, %11 : i1
      iree_linalg_ext.yield %9, %arg5, %12 : index, index, i1
    } : tensor<1x1x1x16xf32> into tensor<16x128xf32> -> tensor<16x128xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %arg3[0, 0] [16, 128] [1, 1] : tensor<16x128xf32> into tensor<16x128xf32>
    }
  }
  return %4 : tensor<16x128xf32>
}
// INNER-PARALLEL-LABEL: func.func @unpack_pack_fill_mmt4d_map_store
// INNER-PARALLEL:         scf.forall
// INNER-PARALLEL:           linalg.unpack
// INNER-PARALLEL:           linalg.pack
// INNER-PARALLEL:           scf.forall.in_parallel
// INNER-PARALLEL:         scf.forall
// INNER-PARALLEL:           linalg.fill
// INNER-PARALLEL:           linalg.mmt4d
// INNER-PARALLEL:           iree_linalg_ext.map_store
