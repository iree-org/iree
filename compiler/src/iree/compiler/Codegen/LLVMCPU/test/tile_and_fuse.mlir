// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-and-fuse{tiling-level=0}))" --split-input-file %s | FileCheck %s

func.func @matmul_bias_add(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[10, 20, 30]]>}
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
//      CHECK:   scf.for
// CHECK-SAME:   {
//      CHECK:     scf.for
// CHECK-SAME:     {
//      CHECK:       linalg.fill
//      CHECK:       linalg.matmul
//      CHECK:       linalg.generic
//      CHECK:     }
//      CHECK:   }

// -----

func.func @all_zeros(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0]]>}
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

func.func @multi_config(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0]]>}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%1, %arg2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%init : tensor<?x?xf32>) attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[10, 20]]>} {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
// Both linalg.matmul op and linalg.generic op have lowering_config. Test that
// the lowering_config of linalg.generic op is picked in the pass. In this case,
// scf.for ops are created. If the lowering_config of linalg.matmul op is
// picked, there are no scf.for ops. Because the tiling sizes are zeros.
//      CHECK: func.func @multi_config(
//      CHECK:   scf.for
//      CHECK:     scf.for
//  CHECK-NOT:       scf.for
//      CHECK:       linalg.fill
//      CHECK:       linalg.matmul
//      CHECK:       linalg.generic

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[8, 128, 0]]>
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
//      CHECK:   scf.for
// CHECK-SAME:       iter_args(%[[ITER0:.+]] = %[[DST]])
//      CHECK:     scf.for
// CHECK-SAME:       iter_args(%[[ITER1:.+]] = %[[ITER0]])
//      CHECK:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[ITER1]]
//      CHECK:       %{{.+}} = linalg.fill ins(%[[CST0]] : f32) outs(%[[OUT_SLICE]]
//      CHECK:       %{{.+}} = linalg.matmul
//      CHECK:       %[[OUT_SLICE2:.+]] = tensor.extract_slice %[[ITER1]]
//      CHECK:       %{{.+}} = linalg.generic
// CHECK-SAME:         outs(%[[OUT_SLICE2]]

// -----

func.func @scalable_matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32>{
  // Matrix multiplication (ijk) with scalable tiling in the j-th dimension.
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, [32], 1]]>} ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @scalable_matmul(
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
//       CHECK: %[[VSCALE:.*]] = vector.vscale
//  CHECK-NEXT: %[[SCALABLE_TILE_SIZE:.*]] = arith.muli %[[VSCALE]], %[[C32]] : index
//       CHECK: scf.for
//  CHECK-SAME:     step %[[C1]]
//       CHECK:   scf.for
//  CHECK-SAME:       step %[[SCALABLE_TILE_SIZE]]
//       CHECK:     scf.for
//  CHECK-SAME:         step %[[C1]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @ukernel_generic(%arg0: tensor<1x192x1x16xf32>, %arg1: tensor<1x768x1x1xf32>, %arg2: tensor<192x768x16x1xf32>, %arg3: tensor<1x192x1x16xf32>) -> tensor<1x192x1x16xf32> {
  %c1 = arith.constant 1 : index
  %c192 = arith.constant 192 : index
  %c768 = arith.constant 768 : index
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1025_i32 = arith.constant 1025 : i32
  %0 = tensor.empty() : tensor<1x192x1x16xf32>
  %1 = iree_codegen.ukernel.generic "iree_uk_mmt4d" ins(%arg1, %arg2 : tensor<1x768x1x1xf32>, tensor<192x768x16x1xf32>) outs(%0 : tensor<1x192x1x16xf32>) (%c1, %c192, %c768, %c1_i32, %c16_i32, %c1_i32, %c1025_i32 : index, index, index, i32, i32, i32, i32) fn_def_attrs {hal.import.bitcode = true, hal.import.fields = ["processor_data"]} strided_outer_dims(1) -> tensor<1x192x1x16xf32>
  %2 = linalg.generic { indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"] }
    ins(%1, %arg3 : tensor<1x192x1x16xf32>, tensor<1x192x1x16xf32>)
    outs(%arg0 : tensor<1x192x1x16xf32>)
    attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 20, 0]]>} {
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
// CHECK:         scf.for {{.+}} iter_args(%[[ITER:.+]] = %[[ARG0]])
// CHECK:           %[[UK_SLICE:.+]] = tensor.extract_slice %[[UK]]
// CHECK:           %[[ARG3_SLICE:.+]] = tensor.extract_slice %[[ARG3]]
// CHECK:           %[[ITER_SLICE:.+]] = tensor.extract_slice %[[ITER]]
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%[[UK_SLICE]], %[[ARG3_SLICE]]
// CHECK-SAME:        outs(%[[ITER_SLICE]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1], [0, 1]]>
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
// CHECK:         scf.for
// CHECK-SAME:    {
// CHECK:           linalg.generic
// CHECK:           linalg.fill
// CHECK:           linalg.fill
// CHECK:           iree_linalg_ext.scan
// CHECK:           scf.yield
// CHECK:         }
