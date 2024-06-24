// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op))" --mlir-print-local-scope --split-input-file %s | FileCheck %s

module {
  func.func @matmul_tensors() attributes {translation_info = #iree_codegen.translation_info<CPUDoubleTilingExpert>} {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
    %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
    %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
    %9 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    %10 = flow.dispatch.tensor.load %6, offsets = [0, 0], sizes = [%0, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
    %11 = flow.dispatch.tensor.load %7, offsets = [0, 0], sizes = [%2, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
    %12 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
    %13 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0]]>} ins(%10, %11 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.dispatch.tensor.store %13, %9, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    return
  }
}
// CHECK-LABEL: func @matmul_tensors()
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//   CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[INIT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
//   CHECK-DAG:   %[[OUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(3)
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[K]]{{\]}}, strides = [1, 1]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[K]], %[[N]]{{\]}}, strides = [1, 1]
//       CHECK:   %[[INIT:.+]] = flow.dispatch.tensor.load %[[INIT_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[N]]{{\]}}, strides = [1, 1]
//       CHECK:   %[[RESULT:.+]] = scf.forall (%[[ARG0:[a-zA-Z0-9]+]], %[[ARG1:[a-zA-Z0-9]+]])
//  CHECK-SAME:       (0, 0) to (%[[M]], %[[N]]) step (64, 64)
//  CHECK-SAME:       shared_outs(%[[OUTS:.+]] = %[[INIT]])
//       CHECK:     %[[TILESIZE_M:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%[[ARG0]])[%[[M]]{{\]}}
//       CHECK:     %[[TILESIZE_N:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%[[ARG1]])[%[[N]]{{\]}}
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][%[[ARG0]], 0] [%[[TILESIZE_M]], %[[K]]{{\]}}
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, %[[ARG1]]{{\]}} [%[[K]], %[[TILESIZE_N]]{{\]}}
//   CHECK-DAG:     %[[INIT_SLICE:.+]] = tensor.extract_slice %[[OUTS]][%[[ARG0]], %[[ARG1]]{{\]}} [%[[TILESIZE_M]], %[[TILESIZE_N]]{{\]}}
//       CHECK:     %[[MATMUL_SLICE:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[LHS_SLICE]], %[[RHS_SLICE]] :
//  CHECK-SAME:         outs(%[[INIT_SLICE]] :
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice %[[MATMUL_SLICE]] into %[[OUTS]]
//  CHECK-SAME:           [%[[ARG0]], %[[ARG1]]] [%[[TILESIZE_M]], %[[TILESIZE_N]]{{\]}} [1, 1]
//       CHECK:   flow.dispatch.tensor.store %[[RESULT]], %[[OUT_BINDING]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[M]], %[[N]]{{\]}}, strides = [1, 1]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

module {
  func.func @add() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
        : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
        : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%1}
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
        : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1]
        : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
    %6 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [%1], strides = [1]
        : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%1} -> tensor<?xf32>
    %7 = tensor.empty(%0, %1) : tensor<?x?xf32>
    %8 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%5, %6 : tensor<?x?xf32>, tensor<?xf32>) outs(%7 : tensor<?x?xf32>)
        attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>} {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
        %9 = arith.addf %arg0, %arg1 : f32
        linalg.yield %9 : f32
      } -> tensor<?x?xf32>
    flow.dispatch.tensor.store %8, %4, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1]
        : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    return
  }
}
//      CHECK: func.func @add()
//      CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:.+]]) =
// CHECK-SAME:       shared_outs(%[[ARG2:.+]] =
//      CHECK:     %[[TILE_RESULT:.+]] = linalg.generic
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[TILE_RESULT]] into %[[ARG2]][%[[IV0]], %[[IV1]]{{\]}}
//      CHECK:   flow.dispatch.tensor.store %[[RESULT]], %{{.+}}, offsets = [0, 0]

// -----

module {
  func.func @add4D() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
        : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32)
        : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32)
        : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
        : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
    %8 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
        : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
    %9 = tensor.empty(%0, %1, %2, %3) : tensor<?x?x?x?xf32>
    %10 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%7, %8 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
        outs(%9 : tensor<?x?x?x?xf32>)
        attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 64, 64, 64]]>} {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
        %11 = arith.addf %arg0, %arg1 : f32
        linalg.yield %11 : f32
      } -> tensor<?x?x?x?xf32>
    flow.dispatch.tensor.store %10, %6, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
        : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    return
  }
}
//      CHECK: func.func @add4D()
//      CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]], %[[IV2:[a-zA-Z0-9]+]]) =
//      CHECK:     %[[TILED_GENERIC:.+]] = linalg.generic
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[TILED_GENERIC]] into %{{.+}}[0, %[[IV0]], %[[IV1]], %[[IV2]]{{\]}}
//      CHECK:   flow.dispatch.tensor.store %[[RESULT]], %{{.+}}, offsets = [0, 0, 0, 0]

// -----

module {
  func.func @add_distribute4D() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
        : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32)
        : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32)
        : !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
        : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
    %8 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
        : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
    %9 = tensor.empty(%0, %1, %2, %3) : tensor<?x?x?x?xf32>
    %10 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, 
                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%7, %8 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
        outs(%9 : tensor<?x?x?x?xf32>)
        attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[2, 64, 64, 64]]>} {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
        %11 = arith.addf %arg0, %arg1 : f32
        linalg.yield %11 : f32
      } -> tensor<?x?x?x?xf32>
    flow.dispatch.tensor.store %10, %6, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, %3], strides = [1, 1, 1, 1]
        : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%0, %1, %2, %3}
    return
  }
}
//      CHECK: func.func @add_distribute4D()
//      CHECK:   %[[RESULT:.+]] = scf.forall
// CHECK-SAME:       (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]], %[[IV2:[a-zA-Z0-9]+]], %[[IV3:[a-zA-Z0-9]+]]) =
//      CHECK:     %[[TILED_GENERIC:.+]] = linalg.generic
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[TILED_GENERIC]] into %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]{{\]}}
//      CHECK:   flow.dispatch.tensor.store %[[RESULT]], %{{.+}}, offsets = [0, 0, 0, 0]
