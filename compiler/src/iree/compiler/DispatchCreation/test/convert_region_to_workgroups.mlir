// RUN: iree-opt %s --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-convert-dispatch-regions-to-workgroups, iree-flow-canonicalize, cse))" -split-input-file | FileCheck %s

util.global private @device : !hal.device

// CHECK-LABEL: util.func public @foo(
//       CHECK:   %[[argA:.*]]: tensor<?x?xf32>, %[[argB:.*]]: tensor<5x10xf32>, %[[argC:.*]]: tensor<10x11xf32>
util.func public @foo(%argA: tensor<?x?xf32>, %argB: tensor<5x10xf32>, %argC: tensor<10x11xf32>) -> (tensor<?x?xf32>, tensor<5x11xf32>) {
  //  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  //  CHECK-DAG: %[[dim_argA_0:.*]] = tensor.dim %[[argA]], %[[c0]]
  //  CHECK-DAG: %[[dim_argA_1:.*]] = tensor.dim %[[argA]], %[[c1]]
  //      CHECK: %[[r0:.*]] = flow.dispatch.workgroups(%[[argA]], %[[dim_argA_0]], %[[dim_argA_1]]) : (tensor<?x?xf32>{%[[dim_argA_0]], %[[dim_argA_1]]}, index, index) -> %[[argA]]{%[[dim_argA_0]], %[[dim_argA_1]]} =
  // CHECK-NEXT: (%[[arg1:.*]]: !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>, %[[arg2:.*]]: index, %[[arg3:.*]]: index) {
  //      CHECK:   %[[load:.*]] = flow.dispatch.tensor.load %[[arg1]], offsets = [0, 0], sizes = [%[[arg2]], %[[arg3]]], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%[[arg2]], %[[arg3]]} -> tensor<?x?xf32>
  //      CHECK:   flow.dispatch.tensor.store %[[load]], %[[arg1]], offsets = [0, 0], sizes = [%[[arg2]], %[[arg3]]], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%[[arg2]], %[[arg3]]}
  //      CHECK:   flow.return
  //      CHECK: }
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dimA0 = tensor.dim %argA, %c0 : tensor<?x?xf32>
  %dimA1 = tensor.dim %argA, %c1 : tensor<?x?xf32>
  %r0 = flow.dispatch.region -> (tensor<?x?xf32>{%dimA0, %dimA1}) {
    flow.return %argA : tensor<?x?xf32>
  }
  //      CHECK: %[[r1:.*]] = flow.dispatch.workgroups(%[[argB]], %[[argC]]) : (tensor<5x10xf32>, tensor<10x11xf32>) -> tensor<5x11xf32>
  // CHECK-SAME:   stream.affinity = #hal.device.affinity<@device>
  // CHECK-NEXT: (%[[arg3:.*]]: !flow.dispatch.tensor<readonly:tensor<5x10xf32>>, %[[arg4:.*]]: !flow.dispatch.tensor<readonly:tensor<10x11xf32>>, %[[arg5:.*]]: !flow.dispatch.tensor<writeonly:tensor<5x11xf32>>)
  //  CHECK-DAG:   %[[loadB:.*]] = flow.dispatch.tensor.load %[[arg3]], offsets = [0, 0], sizes = [5, 10], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<5x10xf32>> -> tensor<5x10xf32>
  //  CHECK-DAG:   %[[loadC:.*]] = flow.dispatch.tensor.load %[[arg4]], offsets = [0, 0], sizes = [10, 11], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10x11xf32>> -> tensor<10x11xf32>
  //      CHECK:   %[[empty:.*]] = tensor.empty() : tensor<5x11xf32>
  //      CHECK:   %[[fill:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[empty]] : tensor<5x11xf32>) -> tensor<5x11xf32>
  //      CHECK:   %[[matmul:.*]] = linalg.matmul ins(%[[loadB]], %[[loadC]] : tensor<5x10xf32>, tensor<10x11xf32>) outs(%[[fill]] : tensor<5x11xf32>) -> tensor<5x11xf32>
  //      CHECK:   flow.dispatch.tensor.store %[[matmul]], %[[arg5]], offsets = [0, 0], sizes = [5, 11], strides = [1, 1] : tensor<5x11xf32> -> !flow.dispatch.tensor<writeonly:tensor<5x11xf32>>
  //      CHECK:   flow.return
  //      CHECK: }
  %r1 = flow.dispatch.region -> (tensor<5x11xf32>) attributes {
    stream.affinity = #hal.device.affinity<@device>
  } {
    %zero = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<5x11xf32>
    %1 = linalg.fill ins(%zero : f32) outs(%0 : tensor<5x11xf32>) -> tensor<5x11xf32>
    %2 = linalg.matmul ins(%argB, %argC : tensor<5x10xf32>, tensor<10x11xf32>)
        outs(%1 : tensor<5x11xf32>) -> tensor<5x11xf32>
    flow.return %2 : tensor<5x11xf32>
  }

  //      CHECK: util.return %[[r0]], %[[r1]]
  util.return %r0, %r1 : tensor<?x?xf32>, tensor<5x11xf32>
}

// -----

// TODO(Max191): Remove this test once GPU data tiling stops using early
// materialization.
util.func public @multi_mma(
    %arg0: tensor<4x16x8x4x16x2x4xf16>,
    %arg1: tensor<4x16x4x2x4x16x2x4xf16>,
    %arg2: tensor<4x4x8x4x2x4x16x4xf32>) -> (tensor<4x4x8x4x2x4x16x4xf32>) {
  %9 = flow.dispatch.region -> (tensor<4x4x8x4x2x4x16x4xf32>) {
    %13 = iree_gpu.multi_mma %arg0, %arg1, %arg2 {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                         affine_map<(d0, d1, d2) -> (d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.data_tiled_mma_layout<intrinsic =  MFMA_F32_16x16x16_F16, unroll_m = 8, unroll_n = 2, unroll_n_to_subgroups = 4, unroll_k = 2>}
        : tensor<4x16x8x4x16x2x4xf16>, tensor<4x16x4x2x4x16x2x4xf16> into tensor<4x4x8x4x2x4x16x4xf32>
    flow.return %13 : tensor<4x4x8x4x2x4x16x4xf32>
  }
  util.return %9 : tensor<4x4x8x4x2x4x16x4xf32>
}

// CHECK-LABEL: util.func public @multi_mma(
//       CHECK:     %[[arg0:.*]]: tensor<4x16x8x4x16x2x4xf16>, %[[arg1:.*]]: tensor<4x16x4x2x4x16x2x4xf16>, %[[arg2:.*]]: tensor<4x4x8x4x2x4x16x4xf32>
//       CHECK:   %[[r0:.*]] = flow.dispatch.workgroups(%[[arg0]], %[[arg1]], %[[arg2]])
//  CHECK-SAME:       : (tensor<4x16x8x4x16x2x4xf16>, tensor<4x16x4x2x4x16x2x4xf16>, tensor<4x4x8x4x2x4x16x4xf32>)
//  CHECK-NEXT:       (%[[arg3:.*]]: !flow.dispatch.tensor<readonly:tensor<4x16x8x4x16x2x4xf16>>,
//  CHECK-SAME:        %[[arg4:.*]]: !flow.dispatch.tensor<readonly:tensor<4x16x4x2x4x16x2x4xf16>>,
//  CHECK-SAME:        %[[arg5:.*]]: !flow.dispatch.tensor<readwrite:tensor<4x4x8x4x2x4x16x4xf32>>)
//   CHECK-DAG:     %[[loadLHS:.*]] = flow.dispatch.tensor.load %[[arg3]]
//   CHECK-DAG:     %[[loadRHS:.*]] = flow.dispatch.tensor.load %[[arg4]]
//   CHECK-DAG:     %[[loadACC:.*]] = flow.dispatch.tensor.load %[[arg5]]
//       CHECK:     %[[MULTI_MMA:.*]] = iree_gpu.multi_mma %[[loadLHS]], %[[loadRHS]], %[[loadACC]]
//       CHECK:     flow.dispatch.tensor.store %[[MULTI_MMA]], %[[arg5]]
//       CHECK:     flow.return
//       CHECK:   }
//       CHECK:   util.return %[[r0]]
