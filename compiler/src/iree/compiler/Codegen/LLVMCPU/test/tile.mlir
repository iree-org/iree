// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile{tiling-level=0}))" --split-input-file %s | FileCheck %s

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
    outs(%init : tensor<?x?xf32>) attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 16]]>} {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @matmul_bias_add(
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C20:.+]] = arith.constant 20 : index
// CHECK-DAG:     %[[C30:.+]] = arith.constant 30 : index
// CHECK:         scf.for {{.*}} step %[[C10]]
// CHECK:           scf.for {{.*}} step %[[C20]]
// CHECK:             linalg.fill
// CHECK:         scf.for {{.*}} step %[[C10]]
// CHECK:           scf.for {{.*}} step %[[C20]]
// CHECK:             scf.for {{.*}} step %[[C30]]
// CHECK:             linalg.matmul
// CHECK:         scf.for {{.*}} step %[[C1]]
// CHECK:           scf.for {{.*}} step %[[C16]]
// CHECK:             linalg.generic

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

// CHECK-LABEL: scalable_lowering_config_with_no_1s
// CHECK: vector.vscale
func.func @scalable_lowering_config_with_no_1s(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[8, [32], 0]]>} ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

func.func @do_not_tile_ukernel(%arg0: tensor<?x?x16x1xf32>, %arg1: tensor<?x?x16x1xf32>, %arg2: tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16_i32 = arith.constant 16 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1025_i32 = arith.constant 1025 : i32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x16x1xf32>
  %dim_0 = tensor.dim %arg1, %c0 : tensor<?x?x16x1xf32>
  %dim_1 = tensor.dim %arg1, %c1 : tensor<?x?x16x1xf32>
  %0 = iree_codegen.ukernel.generic {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[10, 20, 30]]>}
    "iree_uk_mmt4d"
    ins(%arg0, %arg1 : tensor<?x?x16x1xf32>, tensor<?x?x16x1xf32>)
    outs(%arg2 : tensor<?x?x16x16xf32>)
    (%dim, %dim_0, %dim_1, %c16_i32, %c16_i32, %c1_i32, %c1025_i32 : index, index, index, i32, i32, i32, i32)
    fn_def_attrs {hal.import.bitcode = true, hal.import.cconv = 1 : i32, hal.import.fields = ["processor_data"]}
    strided_outer_dims(1) -> tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}

// CHECK-LABEL: func.func @do_not_tile_ukernel
// CHECK-NOT:     scf.for
// CHECK:         iree_codegen.ukernel.generic
