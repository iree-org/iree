// RUN: iree-opt %s --iree-codegen-iree-comprehensive-bufferize --canonicalize -cse --canonicalize --split-input-file | FileCheck %s

func.func @matmul() {
  %c0 = arith.constant 0 : index
  %m = hal.interface.constant.load[0] : index
  %n = hal.interface.constant.load[1] : index
  %k = hal.interface.constant.load[2] : index
  %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %k}
  %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:?x?xf32>{%k, %n}
  %init = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %n}
  %result = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?x?xf32>{%m, %n}
  %wg_id_y = hal.interface.workgroup.id[1] : index
  %wg_count_y = hal.interface.workgroup.count[1] : index
  %wg_size_y = hal.interface.workgroup.size[1] : index
  %wg_id_x = hal.interface.workgroup.id[0] : index
  %wg_count_x = hal.interface.workgroup.count[0] : index
  %wg_size_x = hal.interface.workgroup.size[0] : index
  %offset_y = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%wg_id_y, %wg_size_y]
  %step_y = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%wg_count_y, %wg_size_y]
  %offset_x = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%wg_id_x, %wg_size_x]
  %step_x = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%wg_count_x, %wg_size_x]
  scf.for %iv0 = %offset_y to %m step %step_y {
    %tilesize_y = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%iv0)[%wg_size_y, %m]
    scf.for %iv1 = %offset_x to %n step %step_x {
      %tilesize_x = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%iv1)[%wg_size_x, %n]
      %lhs_tile = flow.dispatch.tensor.load %lhs, offsets = [%iv0, 0], sizes = [%tilesize_y, %k], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %k} -> tensor<?x?xf32>
      %rhs_tile = flow.dispatch.tensor.load %rhs, offsets = [0, %iv1], sizes = [%k, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32>{%k, %n} -> tensor<?x?xf32>
      %init_tile = flow.dispatch.tensor.load %init, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %n} -> tensor<?x?xf32>
      %matmul_tile = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%m, %n}
    }
  }
  return
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//      CHECK: func.func @matmul()
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//  CHECK-DAG:   %[[INIT:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//  CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[WG_COUNT_Y:.+]] = hal.interface.workgroup.count[1]
//  CHECK-DAG:   %[[WG_SIZE_Y:.+]] = hal.interface.workgroup.size[1]
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[WG_COUNT_X:.+]] = hal.interface.workgroup.count[0]
//  CHECK-DAG:   %[[WG_SIZE_X:.+]] = hal.interface.workgroup.size[0]
//  CHECK-DAG:   %[[OFFSET_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_Y]], %[[WG_SIZE_Y]]]
//  CHECK-DAG:   %[[STEP_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_COUNT_Y]], %[[WG_SIZE_Y]]]
//  CHECK-DAG:   %[[OFFSET_X:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_X]], %[[WG_SIZE_X]]]
//  CHECK-DAG:   %[[STEP_X:.+]] = affine.apply #[[MAP0]]()[%[[WG_COUNT_X]], %[[WG_SIZE_X]]]
//      CHECK:   scf.for %[[IV0:.+]] = %[[OFFSET_Y]] to %[[M]] step %[[STEP_Y]]
//      CHECK:     %[[TILESIZE_Y:.+]] = affine.min #[[MAP1]](%[[IV0]])[%[[WG_SIZE_Y]], %[[M]]]
//      CHECK:     scf.for %[[IV1:.+]] = %[[OFFSET_X]] to %[[N]] step %[[STEP_X]]
//      CHECK:       %[[TILESIZE_X:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[WG_SIZE_X]], %[[N]]]
//  CHECK-DAG:       %[[LHS_TILE:.+]] = memref.subview %[[LHS]][%[[IV0]], 0] [%[[TILESIZE_Y]], %[[K]]]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = memref.subview %[[RHS]][0, %[[IV1]]] [%[[K]], %[[TILESIZE_X]]]
//  CHECK-DAG:       %[[INIT_TILE:.+]] = memref.subview %[[INIT]][%[[IV0]], %[[IV1]]] [%[[TILESIZE_Y]], %[[TILESIZE_X]]]
//      CHECK:       %[[ALLOC:.+]] = memref.alloc(%[[TILESIZE_Y]], %[[TILESIZE_X]])
//      CHECK:       linalg.generic {{.*}} ins(%[[INIT_TILE]] {{.*}} outs(%[[ALLOC]]
//      CHECK:       linalg.matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]]
// CHECK-SAME:           outs(%[[ALLOC]]
//      CHECK:       %[[RESULT_TILE:.+]] = memref.subview %[[RESULT]][%[[IV0]], %[[IV1]]] [%[[TILESIZE_Y]], %[[TILESIZE_X]]]
//      CHECK:       linalg.generic {{.*}} ins(%[[ALLOC]] {{.*}} outs(%[[RESULT_TILE]]
//      CHECK:       memref.dealloc %[[ALLOC]]


// -----

func.func @matmul_fill() {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %m = hal.interface.constant.load[0] : index
  %n = hal.interface.constant.load[1] : index
  %k = hal.interface.constant.load[2] : index
  %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %k}
  %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:?x?xf32>{%k, %n}
  %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:?x?xf32>{%m, %n}
  %wg_id_y = hal.interface.workgroup.id[1] : index
  %wg_count_y = hal.interface.workgroup.count[1] : index
  %wg_size_y = hal.interface.workgroup.size[1] : index
  %wg_id_x = hal.interface.workgroup.id[0] : index
  %wg_count_x = hal.interface.workgroup.count[0] : index
  %wg_size_x = hal.interface.workgroup.size[0] : index
  %offset_y = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%wg_id_y, %wg_size_y]
  %step_y = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%wg_count_y, %wg_size_y]
  %offset_x = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%wg_id_x, %wg_size_x]
  %step_x = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%wg_count_x, %wg_size_x]
  scf.for %iv0 = %offset_y to %m step %step_y {
    %tilesize_y = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%iv0)[%wg_size_y, %m]
    scf.for %iv1 = %offset_x to %n step %step_x {
      %tilesize_x = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%iv1)[%wg_size_x, %n]
      %lhs_tile = flow.dispatch.tensor.load %lhs, offsets = [%iv0, 0], sizes = [%tilesize_y, %k], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %k} -> tensor<?x?xf32>
      %rhs_tile = flow.dispatch.tensor.load %rhs, offsets = [0, %iv1], sizes = [%k, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32>{%k, %n} -> tensor<?x?xf32>
      %init_tile = flow.dispatch.tensor.load %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readwrite:?x?xf32>{%m, %n} -> tensor<?x?xf32>
      %fill_tile = linalg.fill ins(%cst : f32) outs(%init_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      %matmul_tile = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>{%m, %n}
    }
  }
  return
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
//      CHECK: func.func @matmul_fill()
//  CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
//  CHECK-DAG:   memref.assume_alignment %[[LHS]], 32
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//  CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[WG_COUNT_Y:.+]] = hal.interface.workgroup.count[1]
//  CHECK-DAG:   %[[WG_SIZE_Y:.+]] = hal.interface.workgroup.size[1]
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[WG_COUNT_X:.+]] = hal.interface.workgroup.count[0]
//  CHECK-DAG:   %[[WG_SIZE_X:.+]] = hal.interface.workgroup.size[0]
//  CHECK-DAG:   %[[OFFSET_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_Y]], %[[WG_SIZE_Y]]]
//  CHECK-DAG:   %[[STEP_Y:.+]] = affine.apply #[[MAP0]]()[%[[WG_COUNT_Y]], %[[WG_SIZE_Y]]]
//  CHECK-DAG:   %[[OFFSET_X:.+]] = affine.apply #[[MAP0]]()[%[[WG_ID_X]], %[[WG_SIZE_X]]]
//  CHECK-DAG:   %[[STEP_X:.+]] = affine.apply #[[MAP0]]()[%[[WG_COUNT_X]], %[[WG_SIZE_X]]]
//      CHECK:   scf.for %[[IV0:.+]] = %[[OFFSET_Y]] to %[[M]] step %[[STEP_Y]]
//      CHECK:     %[[TILESIZE_Y:.+]] = affine.min #[[MAP1]](%[[IV0]])[%[[WG_SIZE_Y]], %[[M]]]
//      CHECK:     scf.for %[[IV1:.+]] = %[[OFFSET_X]] to %[[N]] step %[[STEP_X]]
//  CHECK-NOT:       linalg.generic
//      CHECK:       %[[TILESIZE_X:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[WG_SIZE_X]], %[[N]]]
//  CHECK-DAG:       %[[LHS_TILE:.+]] = memref.subview %[[LHS]][%[[IV0]], 0] [%[[TILESIZE_Y]], %[[K]]]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = memref.subview %[[RHS]][0, %[[IV1]]] [%[[K]], %[[TILESIZE_X]]]
//  CHECK-DAG:       %[[RESULT_TILE:.+]] = memref.subview %[[RESULT]][%[[IV0]], %[[IV1]]] [%[[TILESIZE_Y]], %[[TILESIZE_X]]]
//      CHECK:       linalg.fill
// CHECK-SAME:           ins(%[[CST]] :
// CHECK-SAME:           outs(%[[RESULT_TILE]] :
//      CHECK:       linalg.matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]]
// CHECK-SAME:           outs(%[[RESULT_TILE]]

// -----

func.func @elementwise() {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant opaque<"elided_large_const", "0xDEADBEEF"> : tensor<1x10xf32>
  %c512 = arith.constant 512 : index
  %c64 = arith.constant 64 : index
  %c10 = arith.constant 10 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c512) alignment(64) : !flow.dispatch.tensor<readonly:1x10xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c64) alignment(64) : !flow.dispatch.tensor<writeonly:1x10xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %2 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_x]
  %3 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_x]
  scf.for %arg0 = %2 to %c10 step %3 {
    %4 = affine.min affine_map<(d0) -> (4, -d0 + 10)>(%arg0)
    %5 = flow.dispatch.tensor.load %0, offsets = [0, %arg0], sizes = [1, %4], strides = [1, 1] : !flow.dispatch.tensor<readonly:1x10xf32> -> tensor<1x?xf32>
    %6 = flow.dispatch.tensor.load %1, offsets = [0, %arg0], sizes = [1, %4], strides = [1, 1] : !flow.dispatch.tensor<writeonly:1x10xf32> -> tensor<1x?xf32>
    %7 = scf.for %arg1 = %c0 to %4 step %c4 iter_args(%arg2 = %6) -> (tensor<1x?xf32>) {
      %8 = affine.min affine_map<(d0, d1) -> (4, -d0 + d1)>(%arg1, %4)
      %9 = tensor.extract_slice %5[0, %arg1] [1, %8] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
      %10 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg1, %arg0)
      %11 = tensor.extract_slice %cst[0, %10] [1, %8] [1, 1] : tensor<1x10xf32> to tensor<1x?xf32>
      %12 = tensor.extract_slice %arg2[0, %arg1] [1, %8] [1, 1] : tensor<1x?xf32> to tensor<1x?xf32>
      %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                             affine_map<(d0, d1) -> (d0, d1)>,
                                             affine_map<(d0, d1) -> (d0, d1)>],
                            iterator_types = ["parallel", "parallel"]}
        ins(%9, %11 : tensor<1x?xf32>, tensor<1x?xf32>)
        outs(%12 : tensor<1x?xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %15 = arith.addf %arg3, %arg4 : f32
        linalg.yield %15 : f32
      } -> tensor<1x?xf32>
      %14 = tensor.insert_slice %13 into %arg2[0, %arg1] [1, %8] [1, 1] : tensor<1x?xf32> into tensor<1x?xf32>
      scf.yield %14 : tensor<1x?xf32>
    }
    flow.dispatch.tensor.store %7, %1, offsets = [0, %arg0], sizes = [1, %4], strides = [1, 1] : tensor<1x?xf32> -> !flow.dispatch.tensor<writeonly:1x10xf32>
  }
  return
}
//      CHECK: func.func @elementwise()
//  CHECK-DAG:   %[[CST_TENSOR:.+]] = arith.constant opaque<"elided_large_const", "0xDEADBEEF"> : tensor<1x10xf32>
//  CHECK-DAG:   %[[CST_BUF:.+]] = bufferization.to_memref %[[CST_TENSOR]]
//  CHECK-DAG:   %[[IN_BUF:.+]] = hal.interface.binding.subspan set(0)  binding(0) {{.+}} : memref<1x10xf32>
//  CHECK-DAG:   %[[OUT_BUF:.+]] = hal.interface.binding.subspan set(0)  binding(1) {{.+}} : memref<1x10xf32>
//      CHECK:   scf.for
//  CHECK-DAG:     %[[SUB_IN1:.+]] = memref.subview %[[IN_BUF]]
//  CHECK-DAG:     %[[SUB_OUT1:.+]] = memref.subview %[[OUT_BUF]]
//      CHECK:     scf.for
//  CHECK-DAG:       %[[SUB_IN2:.+]] = memref.subview %[[SUB_IN1]]
//  CHECK-DAG:       %[[SUB_CST:.+]] = memref.subview %[[CST_BUF]]
//  CHECK-DAG:       %[[SUB_OUT2:.+]] = memref.subview %[[SUB_OUT1]]
//      CHECK:       linalg.generic
// CHECK-SAME:         ins(%[[SUB_IN2]], %[[SUB_CST]]
// CHECK-SAME:         outs(%[[SUB_OUT2]]

// -----

#map0 = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<(d0) -> (d0)>
func.func @rank_reduced_slice() {
  %c10 = arith.constant 10 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:1x40xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:10xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %3 = affine.apply #map0()[%workgroup_id_x]
  %4 = affine.apply #map0()[%workgroup_count_x]
  scf.for %arg0 = %3 to %c10 step %4 {
    %5 = flow.dispatch.tensor.load %0, offsets = [0, %arg0], sizes = [1, 2], strides = [1, 1] : !flow.dispatch.tensor<readonly:1x40xf32> -> tensor<2xf32>
    %2 = flow.dispatch.tensor.load %1, offsets = [%arg0], sizes = [2], strides = [1] : !flow.dispatch.tensor<writeonly:10xf32> -> tensor<2xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%5 : tensor<2xf32>) outs(%2 : tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %8 = arith.addf %arg1, %arg1 : f32
      linalg.yield %8 : f32
    } -> tensor<2xf32>
    flow.dispatch.tensor.store %7, %1, offsets = [%arg0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:10xf32>
  }
  return
}
//      CHECK: func.func @rank_reduced_slice()
//  CHECK-DAG:   %[[SRC_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1x40xf32>
//  CHECK-DAG:   %[[DST_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<10xf32>
//      CHECK:   scf.for %[[IV0:.+]] =
//  CHECK-DAG:     %[[SRC_SUBVIEW:.+]] = memref.subview %[[SRC_BINDING]][0, %[[IV0]]] [1, 2] [1, 1] : memref<1x40xf32> to memref<2xf32
//  CHECK-DAG:     %[[DST_SUBVIEW:.+]] = memref.subview %[[DST_BINDING]][%[[IV0]]] [2] [1] : memref<10xf32> to memref<2xf32
//      CHECK:     linalg.generic
// CHECK-SAME:         ins(%[[SRC_SUBVIEW]] :
// CHECK-SAME:         outs(%[[DST_SUBVIEW]] :

// -----

// Check that there are no errors in early bufferized copy ops. The
// bufferization pass should make it as it is.
func.func @early_bufferized_copy_cst_ops() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant dense<0> : tensor<2x3xi32>
  %0 = bufferization.to_memref %cst : memref<2x3xi32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>>
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<2x5xi32>
  memref.assume_alignment %1, 64 : memref<2x5xi32>
  %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:2x5xi32>
  %3 = memref.subview %1[%c0, %c2] [2, 3] [%c1, %c1] : memref<2x5xi32> to memref<2x3xi32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<2x3xi32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>>) outs(%3 : memref<2x3xi32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>>) {
  ^bb0(%arg0: i32, %arg1: i32):
    linalg.yield %arg0 : i32
  }
  return
}
// CHECK: func.func @early_bufferized_copy_cst_ops
// CHECK:   %[[CST:.+]] = arith.constant dense<0> : tensor<2x3xi32>
// CHECK:   %{{.+}} = bufferization.to_memref %[[CST]]

// -----

// CHECK-LABEL: func.func @reverse_dim(
//   CHECK-DAG:   %[[alloc:.*]] = memref.alloc()
//   CHECK-DAG:   %[[cst:.*]] = bufferization.to_memref
//       CHECK:   iree_linalg_ext.reverse dimensions(dense<0> : tensor<1xi64>)
//  CHECK-SAME:       ins(%[[cst]] :
//  CHECK-SAME:       outs(%[[alloc]] :
//       CHECK:   %[[load:.*]] = memref.load %[[alloc]]
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   return %[[load]]
func.func @reverse_dim(%pos: index) -> f32 {
  %input = arith.constant dense<[[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]]> : tensor<2x3xf32>

  %init = bufferization.alloc_tensor() : tensor<2x3xf32>
  %0 = iree_linalg_ext.reverse
         dimensions(dense<0> : tensor<1xi64>)
         ins(%input : tensor<2x3xf32>)
         outs(%init : tensor<2x3xf32>) : tensor<2x3xf32>

  %1 = tensor.extract %0[%pos, %pos] : tensor<2x3xf32>
  return %1 : f32
}

// -----

// CHECK-LABEL: func.func @fft_tensor(
//       CHECK:   memref.alloc
//       CHECK:   memref.alloc
//       CHECK:   iree_linalg_ext.fft ins(%{{.*}} : index) outs(%{{.*}}, %{{.*}} : memref<1024xf32>, memref<1024xf32>)
func.func @fft_tensor(%idx: index) -> (tensor<1024xf32>, tensor<1024xf32>) {
  %t0 = bufferization.alloc_tensor() : tensor<1024xf32>
  %t1 = bufferization.alloc_tensor() : tensor<1024xf32>
  %0:2 = iree_linalg_ext.fft
    ins(%idx: index)
    outs(%t0, %t1: tensor<1024xf32>, tensor<1024xf32>)
  : tensor<1024xf32>, tensor<1024xf32>
  return %0#0, %0#1 : tensor<1024xf32>, tensor<1024xf32>
}

// -----

func.func @scan_1d_dim0_inclusive_sum() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:6xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:f32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:6xf32>
  %3 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [6], strides = [1] : !flow.dispatch.tensor<writeonly:6xf32> -> tensor<6xf32>
  %4 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [6], strides = [1] : !flow.dispatch.tensor<readonly:6xf32> -> tensor<6xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:f32> -> tensor<f32>
  %6:2 = iree_linalg_ext.scan dimension(0) inclusive(true) ins(%4 : tensor<6xf32>) outs(%3, %5 : tensor<6xf32>, tensor<f32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %7 = arith.addf %arg0, %arg1 : f32
    iree_linalg_ext.yield %7 : f32
  } -> tensor<6xf32>, tensor<f32>
  flow.dispatch.tensor.store %6#0, %2, offsets = [0], sizes = [6], strides = [1] : tensor<6xf32> -> !flow.dispatch.tensor<writeonly:6xf32>
  flow.dispatch.tensor.store %6#1, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<readwrite:f32>
  return
}
// CHECK:      func.func @scan_1d_dim0_inclusive_sum
// CHECK-NOT:    memref.alloca
// CHECK:        iree_linalg_ext.scan
// CHECK-SAME:     ins(%{{.*}} : memref<6xf32>)
// CHECK-SAME:      outs(%{{.*}}, %{{.*}} : memref<6xf32>, memref<f32>)

// -----

func.func @sort1D() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:4xi32>
  %1 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readwrite:4xi32> -> tensor<4xi32>
  %2 = iree_linalg_ext.sort dimension(0) outs(%1 : tensor<4xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %3 = arith.cmpi slt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %3 : i1
  } -> tensor<4xi32>
  flow.dispatch.tensor.store %2, %0, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !flow.dispatch.tensor<readwrite:4xi32>
  return
}
// CHECK:      func.func @sort1D
// CHECK:        %[[BUF:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<4xi32>
// CHECK:        iree_linalg_ext.sort
// CHECK-SAME:     outs(%[[BUF]] : memref<4xi32>)

// -----

func.func @scatter_update_scalar_1D() {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:4xi32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:4x1xi32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:8xi32>
  %3 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [8], strides = [1] : !flow.dispatch.tensor<readwrite:8xi32> -> tensor<8xi32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
  scf.for %arg0 = %4 to %c4 step %5 {
    %6 = flow.dispatch.tensor.load %0, offsets = [%arg0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:4xi32> -> tensor<4xi32>
    %7 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0], sizes = [4, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:4x1xi32> -> tensor<4x1xi32>
    %8 = iree_linalg_ext.scatter unique_indices(true) ins(%6, %7 : tensor<4xi32>, tensor<4x1xi32>) outs(%3 : tensor<8xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      iree_linalg_ext.yield %arg1 : i32
    } -> tensor<8xi32>
    flow.dispatch.tensor.store %8, %2, offsets = [0], sizes = [8], strides = [1] : tensor<8xi32> -> !flow.dispatch.tensor<readwrite:8xi32>
  }
  return
}
// CHECK:      func.func @scatter_update_scalar_1D
// CHECK-DAG:    %[[UPDATE:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<4xi32>
// CHECK-DAG:    %[[INDICES:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<4x1xi32>
// CHECK-DAG:    %[[ORIGINAL:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : memref<8xi32>
// CHECK:        scf.for %[[I:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK-DAG:      %[[SUB_UPDATE:.+]] = memref.subview %[[UPDATE]][%[[I]]]
// CHECK-DAG:      %[[SUB_INDICES:.+]] = memref.subview %[[INDICES]][%[[I]], 0]
// CHECK:          iree_linalg_ext.scatter
// CHECK-SAME:     ins(%[[SUB_UPDATE]], %[[SUB_INDICES]]
// CHECK-SAME:     outs(%[[ORIGINAL:.+]]

// -----

func.func @topk() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:200x8xf32>
  %input_values = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [200, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:200x8xf32> -> tensor<200x8xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:200x8xi32>
  %input_indices = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [200, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:200x8xi32> -> tensor<200x8xi32>
  %out_values = bufferization.alloc_tensor() : tensor<200x3xf32>
  %out_indices = bufferization.alloc_tensor() : tensor<200x3xi32>
  %2:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : tensor<200x8xf32> , tensor<200x8xi32>)
        outs(%out_values, %out_indices : tensor<200x3xf32>, tensor<200x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %2 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %2 : i1
        } -> tensor<200x3xf32>, tensor<200x3xi32>
  return
}

// XXX(hanchung): I don't know why there are memref.cast ops, might be a bug?
// Since we don't have e2e top-k tests, I can't figure out how it works today.
// CHECK:      func.func @topk
// CHECK-DAG:    %[[INPUT_VALUES:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<200x8xf32>
// CHECK-DAG:    %[[XXX_VALUES:.+]] = memref.cast %[[INPUT_VALUES]] : memref<200x8xf32> to memref<200x8xf32,
// CHECK-DAG:    %[[INPUT_INDICES:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<200x8xi32>
// CHECK-DAG:    %[[XXX_INDICES:.+]] = memref.cast %[[INPUT_INDICES]] : memref<200x8xi32> to memref<200x8xi32,
// CHECK-DAG:    %[[OUTPUT_VALUES:.+]] = memref.alloc() : memref<200x3xf32>
// CHECK-DAG:    %[[OUTPUT_INDICES:.+]] = memref.alloc() : memref<200x3xi32>
// CHECK:        iree_linalg_ext.topk
// CHECK-SAME:     ins(%[[XXX_VALUES]], %[[XXX_INDICES]]
// CHECK-SAME:     outs(%[[OUTPUT_VALUES]], %[[OUTPUT_INDICES]]
