// RUN: iree-opt -iree-codegen-tile-and-distribute-to-workgroups -cse -split-input-file %s | FileCheck %s

func @simple_gemm_dynamic() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %4 = arith.index_cast %0 : i32 to index
  %5 = arith.index_cast %1 : i32 to index
  %6 = arith.index_cast %2 : i32 to index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:?x?xf32>{%4, %5}
  %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:?x?xf32>{%5, %6}
  %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:?x?xf32>{%4, %6}
  %11 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [%4, %5], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32>{%4, %5} -> tensor<?x?xf32>
  %12 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [%5, %6], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32>{%5, %6} -> tensor<?x?xf32>
  %13 = linalg.init_tensor [%4, %6] : tensor<?x?xf32>
  %14 = linalg.fill(%cst, %13) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  %15 = linalg.matmul ins(%11, %12 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
  flow.dispatch.tensor.store %15, %10, offsets = [0, 0], sizes = [%4, %6], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%4, %6}
  return
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
//       CHECK: func @simple_gemm_dynamic()
//   CHECK-DAG:   %[[MVAL:.+]] = hal.interface.constant.load[0] : i32
//   CHECK-DAG:   %[[KVAL:.+]] = hal.interface.constant.load[1] : i32
//   CHECK-DAG:   %[[NVAL:.+]] = hal.interface.constant.load[2] : i32
//   CHECK-DAG:   %[[M:.+]] = arith.index_cast %[[MVAL]]
//   CHECK-DAG:   %[[K:.+]] = arith.index_cast %[[KVAL]]
//   CHECK-DAG:   %[[N:.+]] = arith.index_cast %[[NVAL]]
//   CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[OUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
//   CHECK-DAG:   %[[WG_SIZE_X:.+]] = hal.interface.workgroup.size[0]
//   CHECK-DAG:   %[[WG_SIZE_Y:.+]] = hal.interface.workgroup.size[1]
//   CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//   CHECK-DAG:   %[[WG_COUNT_X:.+]] = hal.interface.workgroup.count[0]
//   CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//   CHECK-DAG:   %[[WG_COUNT_Y:.+]] = hal.interface.workgroup.count[1]
//   CHECK-DAG:   %[[LB_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_Y]], %[[WG_SIZE_Y]]]
//   CHECK-DAG:   %[[STEP_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_COUNT_Y]], %[[WG_SIZE_Y]]]
//       CHECK:   scf.for %[[IV0:.+]] = %[[LB_Y]] to %[[M]] step %[[STEP_Y]]
//   CHECK-DAG:     %[[LB_X:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_X]], %[[WG_SIZE_X]]]
//   CHECK-DAG:     %[[STEP_X:.+]] = affine.apply #[[MAP0]]()[%[[WG_COUNT_X]], %[[WG_SIZE_X]]]
//       CHECK:     scf.for %[[IV1:.+]] = %[[LB_X]] to %[[N]] step %[[STEP_X]]
//   CHECK-DAG:       %[[M_TILE:.+]] = affine.min #[[MAP1]](%[[IV0]])[%[[M]], %[[WG_SIZE_Y]]]
//   CHECK-DAG:       %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//  CHECK-SAME:           offsets = [%[[IV0]], 0], sizes = [%[[M_TILE]], %[[K]]]
//   CHECK-DAG:       %[[N_TILE:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[N]], %[[WG_SIZE_X]]]
//   CHECK-DAG:       %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//  CHECK-SAME:           offsets = [0, %[[IV1]]], sizes = [%[[K]], %[[N_TILE]]]
//   CHECK-DAG:       %[[M_TILE2:.+]] = affine.min #[[MAP2]](%[[IV0]])[%[[M]], %[[WG_SIZE_Y]]]
//   CHECK-DAG:       %[[N_TILE2:.+]] = affine.min #[[MAP2]](%[[IV1]])[%[[N]], %[[WG_SIZE_X]]]
//   CHECK-DAG:       %[[INIT:.+]] = linalg.init_tensor [%[[M_TILE2]], %[[N_TILE2]]]
//   CHECK-DAG:       %[[FILL:.+]] = linalg.fill(%{{.+}}, %[[INIT]])
//   CHECK-DAG:       %[[GEMM:.+]] = linalg.matmul ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:           outs(%[[FILL]] :
//       CHECK:       flow.dispatch.tensor.store %[[GEMM]], %[[OUT_BINDING]]
//  CHECK-SAME:           offsets = [%[[IV0]], %[[IV1]]], sizes = [%[[M_TILE]], %[[N_TILE]]]

// -----

func @generic_op_alone() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %4 = arith.index_cast %0 : i32 to index
  %5 = arith.index_cast %1 : i32 to index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:?x?xf32>{%4, %5}
  %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:?xf32>{%5}
  %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:?x?xf32>{%4, %5}
  %11 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [%4, %5], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32>{%4, %5} -> tensor<?x?xf32>
  %12 = flow.dispatch.tensor.load %9, offsets = [0], sizes = [%5], strides = [1] : !flow.dispatch.tensor<readonly:?xf32>{%5} -> tensor<?xf32>
  %13 = linalg.init_tensor [%4, %5] : tensor<?x?xf32>
  %15 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins (%11, %12: tensor<?x?xf32>, tensor<?xf32>)
    outs (%13 : tensor<?x?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
        %2 = arith.addf %arg0, %arg1 : f32
        linalg.yield %2 : f32
    } -> tensor<?x?xf32>
  flow.dispatch.tensor.store %15, %10, offsets = [0, 0], sizes = [%4, %5], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%4, %5}
  return
}
// CHECK-LABEL: func @generic_op_alone()
//   CHECK-DAG:   %[[INPUT1:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[INPUT2:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(2)
//   CHECK-DAG:   %[[INIT:.+]] = linalg.init_tensor
//       CHECK:   scf.for
//       CHECK:     scf.for
//   CHECK-DAG:       %[[LHS:.+]] = flow.dispatch.tensor.load %[[INPUT1]]
//   CHECK-DAG:       %[[RHS:.+]] = flow.dispatch.tensor.load %[[INPUT2]]
//   CHECK-DAG:       %[[SLICE:.+]] = tensor.extract_slice %[[INIT]]
//       CHECK:       %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:           ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:           outs(%[[SLICE]] :
//       CHECK:       flow.dispatch.tensor.store %[[GENERIC]], %[[OUTPUT]]

// -----

func @generic_op_4D() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = arith.index_cast %0 : i32 to index
  %5 = arith.index_cast %1 : i32 to index
  %6 = arith.index_cast %2 : i32 to index
  %7 = arith.index_cast %3 : i32 to index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%4, %5, %6, %7}
  %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%4, %5, %6, %7}
  %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:?x?x?x?xf32>{%4, %5, %6, %7}
  %11 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [%4, %5, %6, %7], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%4, %5, %6, %7} -> tensor<?x?x?x?xf32>
  %12 = flow.dispatch.tensor.load %9, offsets = [0, 0, 0, 0], sizes = [%4, %5, %6, %7], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%4, %5, %6, %7} -> tensor<?x?x?x?xf32>
  %13 = linalg.init_tensor [%4, %5, %6, %7] : tensor<?x?x?x?xf32>
  %15 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins (%11, %12: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs (%13 : tensor<?x?x?x?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
        %14 = arith.addf %arg0, %arg1 : f32
        linalg.yield %14 : f32
    } -> tensor<?x?x?x?xf32>
  flow.dispatch.tensor.store %15, %10, offsets = [0, 0, 0, 0], sizes = [%4, %5, %6, %7], strides = [1, 1, 1, 1] : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?x?x?xf32>{%4, %5, %6, %7}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>
//      CHECK: func @generic_op_4D()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[D0VAL:.+]] = hal.interface.constant.load[0] : i32
//  CHECK-DAG:   %[[D1VAL:.+]] = hal.interface.constant.load[1] : i32
//  CHECK-DAG:   %[[D2VAL:.+]] = hal.interface.constant.load[2] : i32
//  CHECK-DAG:   %[[D3VAL:.+]] = hal.interface.constant.load[3] : i32
//  CHECK-DAG:   %[[D0:.+]] = arith.index_cast %[[D0VAL]]
//  CHECK-DAG:   %[[D1:.+]] = arith.index_cast %[[D1VAL]]
//  CHECK-DAG:   %[[D2:.+]] = arith.index_cast %[[D2VAL]]
//  CHECK-DAG:   %[[D3:.+]] = arith.index_cast %[[D3VAL]]
//  CHECK-DAG:   %[[INPUT1:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-DAG:   %[[INPUT2:.+]] = hal.interface.binding.subspan set(0) binding(1)
//  CHECK-DAG:   %[[OUTPUT1:.+]] = hal.interface.binding.subspan set(0) binding(2)
//  CHECK-DAG:   %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]], %[[D2]], %[[D3]]]
//  CHECK-DAG:   %[[WG_SIZE_X:.+]] = hal.interface.workgroup.size[0] : index
//  CHECK-DAG:   %[[WG_SIZE_Y:.+]] = hal.interface.workgroup.size[1] : index
//  CHECK-DAG:   %[[WG_SIZE_Z:.+]] = hal.interface.workgroup.size[2] : index
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0] : index
//  CHECK-DAG:   %[[WG_COUNT_X:.+]] = hal.interface.workgroup.count[0] : index
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
//  CHECK-DAG:   %[[WG_COUNT_Y:.+]] = hal.interface.workgroup.count[1] : index
//  CHECK-DAG:   %[[WG_ID_Z:.+]] = hal.interface.workgroup.id[2] : index
//  CHECK-DAG:   %[[WG_COUNT_Z:.+]] = hal.interface.workgroup.count[2] : index
//  CHECK-DAG:   %[[LB_Z:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_Z]], %[[WG_SIZE_Z]]]
//  CHECK-DAG:   %[[STEP_Z:.+]] = affine.apply #[[MAP0]]()[%[[WG_COUNT_Z]], %[[WG_SIZE_Z]]]
//      CHECK:   scf.for %[[IV0:.+]] = %[[LB_Z]] to %[[D1]] step %[[STEP_Z]]
//  CHECK-DAG:     %[[LB_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_Y]], %[[WG_SIZE_Y]]]
//  CHECK-DAG:     %[[STEP_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_COUNT_Y]], %[[WG_SIZE_Y]]]
//      CHECK:     scf.for %[[IV1:.+]] = %[[LB_Y]] to %[[D2]] step %[[STEP_Y]]
//  CHECK-DAG:       %[[LB_X:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_X]], %[[WG_SIZE_X]]]
//  CHECK-DAG:       %[[STEP_X:.+]] = affine.apply #[[MAP0]]()[%[[WG_COUNT_X]], %[[WG_SIZE_X]]]
//      CHECK:       scf.for %[[IV2:.+]] = %[[LB_X]] to %[[D3]] step %[[STEP_X]]
//  CHECK-DAG:         %[[TILE_Z:.+]] = affine.min #[[MAP1]](%[[IV0]])[%[[D1]], %[[WG_SIZE_Z]]]
//  CHECK-DAG:         %[[TILE_Y:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[D2]], %[[WG_SIZE_Y]]]
//  CHECK-DAG:         %[[TILE_X:.+]] = affine.min #[[MAP1]](%[[IV2]])[%[[D3]], %[[WG_SIZE_X]]]
//      CHECK:         flow.dispatch.tensor.load %[[INPUT1]]
// CHECK-SAME:             offsets = [0, %[[IV0]], %[[IV1]], %[[IV2]]]
// CHECK-SAME:             sizes = [%[[D0]], %[[TILE_Z]], %[[TILE_Y]], %[[TILE_X]]]
//      CHECK:         flow.dispatch.tensor.load %[[INPUT2]]
// CHECK-SAME:             offsets = [0, %[[IV0]], %[[IV1]], %[[IV2]]]
// CHECK-SAME:             sizes = [%[[D0]], %[[TILE_Z]], %[[TILE_Y]], %[[TILE_X]]]
//      CHECK:         %[[D02:[a-zA-Z0-9]+]] = tensor.dim %[[INIT]], %[[C0]] : tensor<?x?x?x?xf32>
//      CHECK:         tensor.extract_slice %[[INIT]][0, %[[IV0]], %[[IV1]], %[[IV2]]]
// CHECK-SAME:             [%[[D02]], %[[TILE_Z]], %[[TILE_Y]], %[[TILE_X]]]
//      CHECK:         flow.dispatch.tensor.store
// CHECK-SAME:             offsets = [0, %[[IV0]], %[[IV1]], %[[IV2]]]
// CHECK-SAME:             sizes = [%[[D02]], %[[TILE_Z]], %[[TILE_Y]], %[[TILE_X]]]


// -----

func @conv2d() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:1x225x225x16xf32>
  %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:3x3x16x32xf32>
  %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
  %11 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x225x225x16xf32> -> tensor<1x225x225x16xf32>
  %12 = flow.dispatch.tensor.load %9, offsets = [0, 0, 0, 0], sizes = [3, 3, 16, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x16x32xf32> -> tensor<3x3x16x32xf32>
  %13 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %15 = linalg.conv_2d_nhwc_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
         ins(%11, %12 : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>)
         outs(%13 : tensor<1x112x112x32xf32>)
         -> tensor<1x112x112x32xf32>
  flow.dispatch.tensor.store %15, %10, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 32], strides = [1, 1, 1, 1] : tensor<1x112x112x32xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
  return
}
// CHECK-LABEL: func @conv2d()
//   CHECK-DAG:   %[[C112:.+]] = arith.constant 112 : index
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//       CHECK:   scf.for %{{.+}} = %{{.+}} to %[[C112]]
//       CHECK:     scf.for %{{.+}} = %{{.+}} to %[[C112]]
//       CHECK;       scf.for %{{.+}} = %{{.+}} to %[[C32]]

// -----

func @depthwise_conv2d() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:1x113x113x96xf32>
  %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:3x3x96xf32>
  %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:1x56x56x96xf32>
  %11 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [1, 113, 113, 96], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x113x113x96xf32> -> tensor<1x113x113x96xf32>
  %12 = flow.dispatch.tensor.load %9, offsets = [0, 0, 0], sizes = [3, 3, 96], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x96xf32> -> tensor<3x3x96xf32>
  %13 = linalg.init_tensor [1, 56, 56, 96] : tensor<1x56x56x96xf32>
  %15 = linalg.depthwise_conv_2d_nhwc_hwc
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%11, %12 : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>)
      outs(%13 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
  flow.dispatch.tensor.store %15, %10, offsets = [0, 0, 0, 0], sizes = [1, 56, 56, 96], strides = [1, 1, 1, 1] : tensor<1x56x56x96xf32> -> !flow.dispatch.tensor<writeonly:1x56x56x96xf32>
  return
}
// CHECK-LABEL: func @depthwise_conv2d()
//   CHECK-DAG:   %[[C56:.+]] = arith.constant 56 : index
//   CHECK-DAG:   %[[C96:.+]] = arith.constant 96 : index
//       CHECK:   scf.for %{{.+}} = %{{.+}} to %[[C56]]
//       CHECK:     scf.for %{{.+}} = %{{.+}} to %[[C56]]
//       CHECK;       scf.for %{{.+}} = %{{.+}} to %[[C96]]

// -----

func @subtensor_insert() {
  %c0 = arith.constant 0 : index
  %offset_y_i32 = hal.interface.constant.load[0] : i32
  %offset_x_i32 = hal.interface.constant.load[1] : i32
  %size_y_i32 = hal.interface.constant.load[2] : i32
  %size_x_i32 = hal.interface.constant.load[3] : i32
  %dest_size_y_i32 = hal.interface.constant.load[4] : i32
  %dest_size_x_i32 = hal.interface.constant.load[5] : i32
  %offset_y = arith.index_cast %offset_y_i32 : i32 to index
  %offset_x = arith.index_cast %offset_x_i32 : i32 to index
  %size_y = arith.index_cast %size_y_i32 : i32 to index
  %size_x = arith.index_cast %size_x_i32 : i32 to index
  %dest_size_y = arith.index_cast %dest_size_y_i32 : i32 to index
  %dest_size_x = arith.index_cast %dest_size_x_i32 : i32 to index
  %source_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<readonly:?x?xf32>{%size_y, %size_x}
  %dest_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<readwrite:?x?xf32>{%dest_size_y, %dest_size_x}
  %source = flow.dispatch.tensor.load %source_binding, offsets = [0, 0], sizes = [%size_y, %size_x], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:?x?xf32>{%size_y, %size_x} -> tensor<?x?xf32>
  %dest = flow.dispatch.tensor.load %dest_binding, offsets = [0, 0], sizes = [%dest_size_y, %dest_size_x], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:?x?xf32>{%dest_size_y, %dest_size_x} -> tensor<?x?xf32>
  %insert = tensor.insert_slice %source into %dest[%offset_y, %offset_x] [%size_y, %size_x] [1, 1]
      : tensor<?x?xf32> into tensor<?x?xf32>
  flow.dispatch.tensor.store %insert, %dest_binding, offsets = [0, 0], sizes = [%dest_size_y, %dest_size_x], strides = [1, 1]
      : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>{%dest_size_y, %dest_size_x}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s1 * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//      CHECK: func @subtensor_insert()
//  CHECK-DAG:   %[[OFFSET_Y_VAL:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[OFFSET_X_VAL:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[SIZE_Y_VAL:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[SIZE_X_VAL:.+]] = hal.interface.constant.load[3]
//  CHECK-DAG:   %[[DEST_SIZE_Y_VAL:.+]] = hal.interface.constant.load[4]
//  CHECK-DAG:   %[[DEST_SIZE_X_VAL:.+]] = hal.interface.constant.load[5]
//  CHECK-DAG:   %[[OFFSET_Y:.+]] = arith.index_cast %[[OFFSET_Y_VAL]]
//  CHECK-DAG:   %[[OFFSET_X:.+]] = arith.index_cast %[[OFFSET_X_VAL]]
//  CHECK-DAG:   %[[SIZE_Y:.+]] = arith.index_cast %[[SIZE_Y_VAL]]
//  CHECK-DAG:   %[[SIZE_X:.+]] = arith.index_cast %[[SIZE_X_VAL]]
//  CHECK-DAG:   %[[DEST_SIZE_Y:.+]] = arith.index_cast %[[DEST_SIZE_Y_VAL]]
//  CHECK-DAG:   %[[DEST_SIZE_X:.+]] = arith.index_cast %[[DEST_SIZE_X_VAL]]
//  CHECK-DAG:   %[[SOURCE:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-DAG:   %[[DEST:.+]] = hal.interface.binding.subspan set(0) binding(1)
//  CHECK-DAG:   %[[WG_SIZE_X:.+]] = hal.interface.workgroup.size[0]
//  CHECK-DAG:   %[[WG_SIZE_Y:.+]] = hal.interface.workgroup.size[1]
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[WG_COUNT_X:.+]] = hal.interface.workgroup.count[0]
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[WG_COUNT_Y:.+]] = hal.interface.workgroup.count[1]
//  CHECK-DAG:   %[[LB_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_SIZE_Y]], %[[WG_ID_Y]]]
//  CHECK-DAG:   %[[STEP_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_SIZE_Y]], %[[WG_COUNT_Y]]]
//      CHECK:   scf.for %[[IV0:.+]] = %[[LB_Y]] to %[[SIZE_Y]] step %[[STEP_Y]]
//  CHECK-DAG:     %[[TILE_Y:.+]] = affine.min #[[MAP1]](%[[IV0]])[%[[WG_SIZE_Y]], %[[SIZE_Y]]]
//  CHECK-DAG:     %[[LB_X:.+]] = affine.apply #[[MAP0]]()[%[[WG_SIZE_X]], %[[WG_ID_X]]]
//  CHECK-DAG:     %[[STEP_X:.+]] = affine.apply #[[MAP0]]()[%[[WG_SIZE_X]], %[[WG_COUNT_X]]]
//      CHECK:     scf.for %[[IV1:.+]] = %[[LB_X]] to %[[SIZE_X]] step %[[STEP_X]]
//  CHECK-DAG:       %[[TILE_X:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[WG_SIZE_X]], %[[SIZE_X]]]
//  CHECK-DAG:       %[[SOURCE_TILE:.+]] = flow.dispatch.tensor.load %[[SOURCE]]
// CHECK-SAME:           offsets = [%[[IV0]], %[[IV1]]], sizes = [%[[TILE_Y]], %[[TILE_X]]]
//  CHECK-DAG:       %[[DEST_OFFSET_Y:.+]] = affine.apply #[[MAP2]](%[[IV0]])[%[[OFFSET_Y]]]
//  CHECK-DAG:       %[[DEST_OFFSET_X:.+]] = affine.apply #[[MAP2]](%[[IV1]])[%[[OFFSET_X]]]
//      CHECK:       flow.dispatch.tensor.store %[[SOURCE_TILE]], %[[DEST]]
// CHECK-SAME:           offsets = [%[[DEST_OFFSET_Y]], %[[DEST_OFFSET_X]]]
// CHECK-SAME:           sizes = [%[[TILE_Y]], %[[TILE_X]]]


// -----

func @non_tiled_reduction_fill() {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %input_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<readonly:1000xf32>
  %output_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<writeonly:f32>
  %input = flow.dispatch.tensor.load %input_binding, offsets = [0], sizes = [1000], strides = [1]
      : !flow.dispatch.tensor<readonly:1000xf32> -> tensor<1000xf32>
  %init = linalg.init_tensor [] : tensor<f32>
  %fill = linalg.fill(%zero, %init) : f32, tensor<f32> -> tensor<f32>
  %reduce = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]}
        ins(%input : tensor<1000xf32>) outs(%fill : tensor<f32>) {
          ^bb0(%b0 : f32, %b1 : f32):
            %update = arith.addf %b0, %b1 : f32
            linalg.yield %update : f32
        } -> tensor<f32>
  flow.dispatch.tensor.store %reduce, %output_binding, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:f32>
  return
}
// CHECK-LABEL: func @non_tiled_reduction_fill()
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]]
//  CHECK-SAME:       offsets = [0], sizes = [1000]
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor []
//       CHECK:   %[[FILL:.+]] = linalg.fill(%{{.+}}, %[[INIT]])
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[INPUT]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[OUTPUT_BINDING]]
//  CHECK-SAME:       offsets = [], sizes = []

// -----

func @multi_result() {
  %cmin = arith.constant -2147483648 : i32
  %czero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %d0_i32 = hal.interface.constant.load[0] : i32
  %d1_i32 = hal.interface.constant.load[1] : i32
  %d0 = arith.index_cast %d0_i32 : i32 to index
  %d1 = arith.index_cast %d1_i32 : i32 to index
  %input1_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<readonly:?x?xf32>{%d0, %d1}
  %input2_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<readonly:?x?xi32>{%d0, %d1}
  %output1_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<writeonly:?xf32>{%d0}
  %output2_binding = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<writeonly:?xi32>{%d0}
  %input1 = flow.dispatch.tensor.load %input1_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:?x?xf32>{%d0, %d1} -> tensor<?x?xf32>
  %input2 = flow.dispatch.tensor.load %input2_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:?x?xi32>{%d0, %d1} -> tensor<?x?xi32>
  %init1 = linalg.init_tensor [%d0] : tensor<?xf32>
  %init2 = linalg.init_tensor [%d0] : tensor<?xi32>
  %fill1 = linalg.fill(%czero, %init1) : f32, tensor<?xf32> -> tensor<?xf32>
  %fill2 = linalg.fill(%cmin, %init2) : i32, tensor<?xi32> -> tensor<?xi32>
  %generic:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input1, %input2 : tensor<?x?xf32>, tensor<?x?xi32>)
      outs(%fill1, %fill2 : tensor<?xf32>, tensor<?xi32>) {
      ^bb0(%arg2: f32, %arg3: i32, %arg4: f32, %arg5: i32):  // no predecessors
        %5 = arith.cmpf oge, %arg2, %arg4 : f32
        %6 = arith.select %5, %arg2, %arg4 : f32
        %7 = arith.cmpf oeq, %arg2, %arg4 : f32
        %8 = arith.cmpi slt, %arg3, %arg5 : i32
        %9 = arith.select %8, %arg3, %arg5 : i32
        %10 = arith.select %5, %arg3, %arg5 : i32
        %11 = arith.select %7, %9, %10 : i32
        linalg.yield %6, %11 : f32, i32
    } -> (tensor<?xf32>, tensor<?xi32>)
  flow.dispatch.tensor.store %generic#0, %output1_binding, offsets = [0], sizes = [%d0], strides = [1]
      : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:?xf32>{%d0}
  flow.dispatch.tensor.store %generic#1, %output2_binding, offsets = [0], sizes = [%d0], strides = [1]
      : tensor<?xi32> -> !flow.dispatch.tensor<writeonly:?xi32>{%d0}
  return
}
// CHECK-LABEL: func @multi_result()
//   CHECK-DAG:   %[[OUT1:.+]] = hal.interface.binding.subspan set(0) binding(2)
//   CHECK-DAG:   %[[OUT2:.+]] = hal.interface.binding.subspan set(0) binding(3)
//       CHECK:   scf.for
//   CHECK-NOT:     scf.for
//       CHECK:       %[[GENERIC:.+]]:2 = linalg.generic
//   CHECK-DAG:       flow.dispatch.tensor.store %[[GENERIC]]#0, %[[OUT1]]
//   CHECK-DAG:       flow.dispatch.tensor.store %[[GENERIC]]#1, %[[OUT2]]

// -----

func @scatter() {
  %c0 = arith.constant 0 : index
  %d0_i32 = hal.interface.constant.load[0] : i32
  %d1_i32 = hal.interface.constant.load[1] : i32
  %d2_i32 = hal.interface.constant.load[2] : i32
  %d0 = arith.index_cast %d0_i32 : i32 to index
  %d1 = arith.index_cast %d1_i32 : i32 to index
  %d2 = arith.index_cast %d2_i32 : i32 to index
  %original_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<readonly:?x?xf32>{%d0, %d1}
  %indices_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<readonly:?x1xi32>{%d0}
  %update_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<writeonly:?x?xf32>{%d2, %d1}
  %original = flow.dispatch.tensor.load %original_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:?x?xf32>{%d0, %d1} -> tensor<?x?xf32>
  %indices = flow.dispatch.tensor.load %indices_binding, offsets = [0, 0], sizes = [%d2, 1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:?x1xi32>{%d0} -> tensor<?x1xi32>
  %update = flow.dispatch.tensor.load %update_binding, offsets = [0, 0], sizes = [%d2, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<writeonly:?x?xf32>{%d2, %d1} -> tensor<?x?xf32>
  %result = iree_linalg_ext.scatter
      unique_indices(true)
      ins(%original, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%update : tensor<?x?xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %1 = arith.addf %arg0, %arg1 : f32
        iree_linalg_ext.yield %1 : f32
  } -> tensor<?x?xf32>
  flow.dispatch.tensor.store %result, %update_binding, offsets = [0, 0], sizes = [%d2, %d1], strides = [1, 1]
      : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%d2, %d1}
  return
}
// CHECK-LABEL: func @scatter()
//   CHECK-DAG:   %[[D0_VAL:.+]] = hal.interface.constant.load[0]
//   CHECK-DAG:   %[[D1_VAL:.+]] = hal.interface.constant.load[1]
//   CHECK-DAG:   %[[D2_VAL:.+]] = hal.interface.constant.load[2]
//   CHECK-DAG:   %[[D0:.+]] = arith.index_cast %[[D0_VAL]]
//   CHECK-DAG:   %[[D1:.+]] = arith.index_cast %[[D1_VAL]]
//   CHECK-DAG:   %[[D2:.+]] = arith.index_cast %[[D2_VAL]]
//   CHECK-DAG:   %[[ORIGINAL:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[INDICES:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[UPDATE:.+]] = hal.interface.binding.subspan set(0) binding(2)
//       CHECK:   scf.for %[[IV0:.+]] = %{{.+}} to %[[D0]]
//       CHECK:     scf.for %[[IV1:.+]] = %{{.+}} to %[[D1]]
//   CHECK-DAG:       %[[ORIGINAL_TILE:.+]] = flow.dispatch.tensor.load %[[ORIGINAL]], offsets = [%[[IV0]], %[[IV1]]]
//   CHECK-DAG:       %[[INDICES_TILE:.+]] = flow.dispatch.tensor.load %[[INDICES]], offsets = [%[[IV0]], 0]
//   CHECK-DAG:       %[[UPDATE_TILE:.+]] = flow.dispatch.tensor.load %[[UPDATE]], offsets = [0, %[[IV1]]], sizes = [%[[D2]],
//       CHECK:       %[[SCATTER_TILE:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:           ins(%[[ORIGINAL_TILE]], %[[INDICES_TILE]] :
//  CHECK-SAME:           outs(%[[UPDATE_TILE]] :
//       CHECK:       flow.dispatch.tensor.store %[[SCATTER_TILE]], %[[UPDATE]], offsets = [0, %[[IV1]]], sizes = [%[[D2]],

// -----

func @sort_3d() {
  %c0 = arith.constant 0 : index
  %d0_i32 = hal.interface.constant.load[0] : i32
  %d1_i32 = hal.interface.constant.load[1] : i32
  %d2_i32 = hal.interface.constant.load[2] : i32
  %d0 = arith.index_cast %d0_i32 : i32 to index
  %d1 = arith.index_cast %d1_i32 : i32 to index
  %d2 = arith.index_cast %d2_i32 : i32 to index
  %output1_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<readwrite:?x?x?xf32>{%d0, %d1, %d2}
  %output2_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<readwrite:?x?x?xi32>{%d0, %d1, %d2}
  %output1 = flow.dispatch.tensor.load %output1_binding, offsets = [0, 0, 0], sizes = [%d0, %d1, %d2], strides = [1, 1, 1]
      : !flow.dispatch.tensor<readwrite:?x?x?xf32>{%d0, %d1, %d2} -> tensor<?x?x?xf32>
  %output2 = flow.dispatch.tensor.load %output2_binding, offsets = [0, 0, 0], sizes = [%d0, %d1, %d2], strides = [1, 1, 1]
      : !flow.dispatch.tensor<readwrite:?x?x?xi32>{%d0, %d1, %d2} -> tensor<?x?x?xi32>
  %result:2 = iree_linalg_ext.sort dimension(0)
      outs(%output1, %output2 : tensor<?x?x?xf32>, tensor<?x?x?xi32>) {
        ^bb0(%b0: f32, %b1: f32, %b2 : i32, %b3 : i32):  // no predecessors
          %2 = arith.cmpf ogt, %b0, %b1 : f32
          iree_linalg_ext.yield %2 : i1
      } -> tensor<?x?x?xf32>, tensor<?x?x?xi32>
  flow.dispatch.tensor.store %result#0, %output1_binding, offsets = [0, 0, 0], sizes = [%d0, %d1, %d2], strides = [1, 1, 1]
      : tensor<?x?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?x?xf32>{%d0, %d1, %d2}
  flow.dispatch.tensor.store %result#1, %output2_binding, offsets = [0, 0, 0], sizes = [%d0, %d1, %d2], strides = [1, 1, 1]
      : tensor<?x?x?xi32> -> !flow.dispatch.tensor<readwrite:?x?x?xi32>{%d0, %d1, %d2}
  return
}
// CHECK-LABEL: func @sort_3d()
//   CHECK-DAG:   %[[OUTPUT1:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUTPUT2:.+]] = hal.interface.binding.subspan set(0) binding(1)
//       CHECK:   scf.for %[[IV0:.+]] =
//       CHECK:     scf.for %[[IV1:.+]] =
//   CHECK-DAG:       %[[OUTPUT1_TILE:.+]] = flow.dispatch.tensor.load %[[OUTPUT1]], offsets = [0, %[[IV0]], %[[IV1]]]
//   CHECK-DAG:       %[[OUTPUT2_TILE:.+]] = flow.dispatch.tensor.load %[[OUTPUT2]], offsets = [0, %[[IV0]], %[[IV1]]]
//       CHECK:       %[[SORT_TILE:.+]]:2 = iree_linalg_ext.sort dimension(0)
//  CHECK-SAME:           outs(%[[OUTPUT1_TILE]], %[[OUTPUT2_TILE]] :
//   CHECK-DAG:       flow.dispatch.tensor.store %[[SORT_TILE]]#0, %[[OUTPUT1]], offsets = [0, %[[IV0]], %[[IV1]]]
//   CHECK-DAG:       flow.dispatch.tensor.store %[[SORT_TILE]]#1, %[[OUTPUT2]], offsets = [0, %[[IV0]], %[[IV1]]]

// -----

func @extract_slice() {
  %c0 = arith.constant 0 : index
  %offset_y_i32 = hal.interface.constant.load[0] : i32
  %offset_x_i32 = hal.interface.constant.load[1] : i32
  %size_y_i32 = hal.interface.constant.load[2] : i32
  %size_x_i32 = hal.interface.constant.load[3] : i32
  %source_size_y_i32 = hal.interface.constant.load[4] : i32
  %source_size_x_i32 = hal.interface.constant.load[5] : i32
  %offset_y = arith.index_cast %offset_y_i32 : i32 to index
  %offset_x = arith.index_cast %offset_x_i32 : i32 to index
  %size_y = arith.index_cast %size_y_i32 : i32 to index
  %size_x = arith.index_cast %size_x_i32 : i32 to index
  %source_size_y = arith.index_cast %source_size_y_i32 : i32 to index
  %source_size_x = arith.index_cast %source_size_x_i32 : i32 to index
  %source_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<readonly:?x?xf32>{%source_size_y, %source_size_x}
  %result_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<writeonly:?x?xf32>{%size_y, %size_x}
  %source = flow.dispatch.tensor.load %source_binding, offsets = [0, 0], sizes = [%source_size_y, %source_size_x], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:?x?xf32>{%source_size_y, %source_size_x} -> tensor<?x?xf32>
  %slice = tensor.extract_slice %source[%offset_y, %offset_x] [%size_y, %size_x] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  flow.dispatch.tensor.store %slice, %result_binding, offsets = [0, 0], sizes = [%size_y, %size_x], strides = [1, 1]
      : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%size_y, %size_x}
  return
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s1 * s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//      CHECK: func @extract_slice()
//  CHECK-DAG:   %[[OFFSET_Y_VAL:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[OFFSET_X_VAL:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[OFFSET_Y:.+]] = arith.index_cast %[[OFFSET_Y_VAL]]
//  CHECK-DAG:   %[[OFFSET_X:.+]] = arith.index_cast %[[OFFSET_X_VAL]]
//  CHECK-DAG:   %[[SOURCE:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//  CHECK-DAG:       %[[TILE_OFFSET_Y:.+]] = affine.apply #[[MAP2]](%[[IV0]])[%[[OFFSET_Y]]]
//  CHECK-DAG:       %[[TILE_OFFSET_X:.+]] = affine.apply #[[MAP2]](%[[IV1]])[%[[OFFSET_X]]]
//      CHECK:       %[[TILE_SLICE:.+]] = flow.dispatch.tensor.load %[[SOURCE]], offsets = [%[[TILE_OFFSET_Y]], %[[TILE_OFFSET_X]]]
//      CHECK:       flow.dispatch.tensor.store %[[TILE_SLICE]], %[[RESULT]], offsets = [%[[IV0]], %[[IV1]]]

// -----

func @gemm_unitN() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %4 = arith.index_cast %0 : i32 to index
  %5 = arith.index_cast %1 : i32 to index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:?x?xf32>{%4, %5}
  %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:?x1xf32>{%5}
  %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:?x1xf32>{%4}
  %11 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [%4, %5], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32>{%4, %5} -> tensor<?x?xf32>
  %12 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [%5, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x1xf32>{%5} -> tensor<?x1xf32>
  %13 = linalg.init_tensor [%4, 1] : tensor<?x1xf32>
  %14 = linalg.fill(%cst, %13) : f32, tensor<?x1xf32> -> tensor<?x1xf32>
  %15 = linalg.matmul ins(%11, %12 : tensor<?x?xf32>, tensor<?x1xf32>) outs(%14 : tensor<?x1xf32>) -> tensor<?x1xf32>
  flow.dispatch.tensor.store %15, %10, offsets = [0, 0], sizes = [%4, 1], strides = [1, 1] : tensor<?x1xf32> -> !flow.dispatch.tensor<writeonly:?x1xf32>{%4}
  return
}
// CHECK-LABEL: func @gemm_unitN()
//   CHECK-DAG:   %[[M_VAL:.+]] = hal.interface.constant.load[0]
//   CHECK-DAG:   %[[M:.+]] = arith.index_cast %[[M_VAL]] : i32 to index
//       CHECK:   scf.for %[[IV0:.+]] = %{{.+}} to %[[M]]
//   CHECK-NOT:   scf.for
//       CHECK:     linalg.fill
//       CHECK:     linalg.matmul
//       CHECK:     flow.dispatch.tensor.store

// -----

func @gemm_unitM_unitN() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %1 = hal.interface.constant.load[0] : i32
  %5 = arith.index_cast %1 : i32 to index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:1x?xf32>{%5}
  %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:?x1xf32>{%5}
  %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:1x1xf32>
  %11 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [1, %5], strides = [1, 1] : !flow.dispatch.tensor<readonly:1x?xf32>{%5} -> tensor<1x?xf32>
  %12 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [%5, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x1xf32>{%5} -> tensor<?x1xf32>
  %13 = linalg.init_tensor [1, 1] : tensor<1x1xf32>
  %14 = linalg.fill(%cst, %13) : f32, tensor<1x1xf32> -> tensor<1x1xf32>
  %15 = linalg.matmul ins(%11, %12 : tensor<1x?xf32>, tensor<?x1xf32>) outs(%14 : tensor<1x1xf32>) -> tensor<1x1xf32>
  flow.dispatch.tensor.store %15, %10, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32> -> !flow.dispatch.tensor<writeonly:1x1xf32>
  return
}
// CHECK-LABEL: func @gemm_unitM_unitN()
//   CHECK-NOT:   scf.for
//       CHECK:   linalg.fill
//       CHECK:   linalg.matmul
//       CHECK:   flow.dispatch.tensor.store
//       CHECK:   return

// -----

func @gemm_unitM() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %1 = hal.interface.constant.load[0] : i32
  %2 = hal.interface.constant.load[1] : i32
  %5 = arith.index_cast %1 : i32 to index
  %6 = arith.index_cast %2 : i32 to index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:1x?xf32>{%5}
  %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:?x?xf32>{%5, %6}
  %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:1x?xf32>{%6}
  %11 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [1, %5], strides = [1, 1] : !flow.dispatch.tensor<readonly:1x?xf32>{%5} -> tensor<1x?xf32>
  %12 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [%5, %6], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32>{%5, %6} -> tensor<?x?xf32>
  %13 = linalg.init_tensor [1, %6] : tensor<1x?xf32>
  %14 = linalg.fill(%cst, %13) : f32, tensor<1x?xf32> -> tensor<1x?xf32>
  %15 = linalg.matmul ins(%11, %12 : tensor<1x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<1x?xf32>) -> tensor<1x?xf32>
  flow.dispatch.tensor.store %15, %10, offsets = [0, 0], sizes = [1, %6], strides = [1, 1] : tensor<1x?xf32> -> !flow.dispatch.tensor<writeonly:1x?xf32>{%6}
  return
}
// CHECK-LABEL: func @gemm_unitM()
//   CHECK-DAG:   %[[N_VAL:.+]] = hal.interface.constant.load[1]
//   CHECK-DAG:   %[[N:.+]] = arith.index_cast %[[N_VAL]] : i32 to index
//       CHECK:   scf.for %[[IV0:.+]] = %{{.+}} to %[[N]]
//   CHECK-NOT:   scf.for
//       CHECK:     linalg.fill
//       CHECK:     linalg.matmul
//       CHECK:     flow.dispatch.tensor.store

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
func @unit_dim_generic_op() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %d0 = arith.index_cast %0 : i32 to index
  %d1 = arith.index_cast %1 : i32 to index
  %d2 = arith.index_cast %2 : i32 to index
  %d3 = arith.index_cast %3 : i32 to index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<readonly:1x?x1x1x?x?x1x?xf32>{%d0, %d1, %d2, %d3}
  %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32)
      : !flow.dispatch.tensor<writeonly:1x?x1x1x?x?x1x?xf32>{%d0, %d1, %d2, %d3}
  %10 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [1, %d0, 1, 1, %d1, %d2, 1, %d3],
      strides = [1, 1, 1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x?x1x1x?x?x1x?xf32>{%d0, %d1, %d2, %d3} -> tensor<1x?x1x1x?x?x1x?xf32>
  %13 = linalg.init_tensor [1, %d0, 1, 1, %d1, %d2, 1, %d3] : tensor<1x?x1x1x?x?x1x?xf32>
  %15 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins (%10: tensor<1x?x1x1x?x?x1x?xf32>)
    outs (%13 : tensor<1x?x1x1x?x?x1x?xf32>) {
      ^bb0(%arg0 : f32, %arg2 : f32):
        %16 = arith.addf %arg0, %arg0 : f32
        linalg.yield %16 : f32
    } -> tensor<1x?x1x1x?x?x1x?xf32>
  flow.dispatch.tensor.store %15, %9, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [1, %d0, 1, 1, %d1, %d2, 1, %d3],
      strides = [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x?x1x1x?x?x1x?xf32> -> !flow.dispatch.tensor<writeonly:1x?x1x1x?x?x1x?xf32>{%d0, %d1, %d2, %d3}
  return
}
// CHECK-LABEL: func @unit_dim_generic_op()
//   CHECK-DAG:   %[[D0_VAL:.+]] = hal.interface.constant.load[0]
//   CHECK-DAG:   %[[D1_VAL:.+]] = hal.interface.constant.load[1]
//   CHECK-DAG:   %[[D2_VAL:.+]] = hal.interface.constant.load[2]
//   CHECK-DAG:   %[[D3_VAL:.+]] = hal.interface.constant.load[3]
//   CHECK-DAG:   %[[D0:.+]] = arith.index_cast %[[D0_VAL]]
//   CHECK-DAG:   %[[D1:.+]] = arith.index_cast %[[D1_VAL]]
//   CHECK-DAG:   %[[D2:.+]] = arith.index_cast %[[D2_VAL]]
//   CHECK-DAG:   %[[D3:.+]] = arith.index_cast %[[D3_VAL]]
//   CHECK-DAG:   %[[INPUT:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//       CHECK:   scf.for %[[IV0:.+]] = %{{.+}} to %[[D1]]
//       CHECK:     scf.for %[[IV1:.+]] = %{{.+}} to %[[D2]]
//       CHECK:       scf.for %[[IV2:.+]] = %{{.+}} to %[[D3]]
//       CHECK:         %[[INPUT_TILE:.+]] = flow.dispatch.tensor.load %[[INPUT]]
//  CHECK-SAME:             offsets = [0, 0, 0, 0, %[[IV0]], %[[IV1]], 0, %[[IV2]]]
//       CHECK:         %[[GENERIC_TILE:.+]] = linalg.generic
//  CHECK-SAME:             ins(%[[INPUT_TILE]] :
//       CHECK:         flow.dispatch.tensor.store %[[GENERIC_TILE]], %[[OUTPUT]]

// -----

func @repeated_indices_scatter_update_slice_2D() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:2x3xi32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:2x1xi32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readwrite:6x3xi32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:2x3xi32> -> tensor<2x3xi32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:2x1xi32> -> tensor<2x1xi32>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [6, 3], strides = [1, 1] : !flow.dispatch.tensor<readwrite:6x3xi32> -> tensor<6x3xi32>
  %6 = iree_linalg_ext.scatter
    unique_indices(false)
    ins(%3, %4 : tensor<2x3xi32>, tensor<2x1xi32>)
    outs(%5 : tensor<6x3xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    iree_linalg_ext.yield %arg0 : i32
  } -> tensor<6x3xi32>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [6, 3], strides = [1, 1] : tensor<6x3xi32> -> !flow.dispatch.tensor<readwrite:6x3xi32>
  return
}
//      CHECK: func @repeated_indices_scatter_update_slice_2D
//  CHECK-DAG:   %[[ARG0_CAPTURE:[a-zA-Z0-9_]+]] = {{.+}} !flow.dispatch.tensor<readonly:2x3xi32>
//  CHECK-DAG:   %[[ARG1_CAPTURE:[a-zA-Z0-9_]+]] = {{.+}} !flow.dispatch.tensor<readonly:2x1xi32>
//  CHECK-DAG:   %[[ARG2_CAPTURE:[a-zA-Z0-9_]+]] = {{.+}} !flow.dispatch.tensor<readwrite:6x3xi32>
//  CHECK-DAG:   %[[UPDATE:.+]] = flow.dispatch.tensor.load %[[ARG0_CAPTURE]], offsets = [0, 0]
//  CHECK-DAG:   %[[INDICES:.+]] = flow.dispatch.tensor.load %[[ARG1_CAPTURE]], offsets = [0, 0]
//  CHECK-DAG:   %[[ORIGINAL:.+]] = flow.dispatch.tensor.load %[[ARG2_CAPTURE]], offsets = [0, 0]
//  CHECK-DAG:   %[[SCATTER:.+]] = iree_linalg_ext.scatter
// CHECK-SAME:       unique_indices(false)
// CHECK-SAME:       ins(%[[UPDATE]], %[[INDICES]] : tensor<2x3xi32>, tensor<2x1xi32>)
// CHECK-SAME:       outs(%[[ORIGINAL]] : tensor<6x3xi32>)
//      CHECK:   flow.dispatch.tensor.store %[[SCATTER]], %[[ARG2_CAPTURE]], offsets = [0, 0]
