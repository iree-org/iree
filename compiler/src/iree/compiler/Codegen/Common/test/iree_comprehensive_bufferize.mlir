// RUN: iree-opt %s --iree-codegen-iree-comprehensive-bufferize --canonicalize -cse --canonicalize --split-input-file | FileCheck %s

func.func @matmul() {
  %c0 = arith.constant 0 : index
  %m = hal.interface.constant.load[0] : index
  %n = hal.interface.constant.load[1] : index
  %k = hal.interface.constant.load[2] : index
  %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %k}
  %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%k, %n}
  %init = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %n}
  %result = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%m, %n}
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
      %lhs_tile = flow.dispatch.tensor.load %lhs, offsets = [%iv0, 0], sizes = [%tilesize_y, %k], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %k} -> tensor<?x?xf32>
      %rhs_tile = flow.dispatch.tensor.load %rhs, offsets = [0, %iv1], sizes = [%k, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%k, %n} -> tensor<?x?xf32>
      %init_tile = flow.dispatch.tensor.load %init, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %n} -> tensor<?x?xf32>
      %matmul_tile = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%m, %n}
    }
  }
  return
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
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
  %c1024 = arith.constant 1024 : index
  %m = hal.interface.constant.load[0] : index
  %n = hal.interface.constant.load[1] : index
  %k = hal.interface.constant.load[2] : index
  %base_offset_i32 = hal.interface.constant.load[3] alignment(8) : i32
  %base_offset = arith.index_castui %base_offset_i32 : i32 to index
  %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %k}
  %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%base_offset) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%k, %n}
  %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c1024) : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%m, %n}
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
      %lhs_tile = flow.dispatch.tensor.load %lhs, offsets = [%iv0, 0], sizes = [%tilesize_y, %k], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %k} -> tensor<?x?xf32>
      %rhs_tile = flow.dispatch.tensor.load %rhs, offsets = [0, %iv1], sizes = [%k, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%k, %n} -> tensor<?x?xf32>
      %init_tile = flow.dispatch.tensor.load %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%m, %n} -> tensor<?x?xf32>
      %fill_tile = linalg.fill ins(%cst : f32) outs(%init_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      %matmul_tile = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%m, %n}
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
//  CHECK-DAG:   %[[BASE_OFFSET_I32:.+]] = hal.interface.constant.load[3]
//  CHECK-DAG:   %[[BASE_OFFSET:.+]] = arith.index_castui %[[BASE_OFFSET_I32]]
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32)
//  CHECK-DAG:   memref.assume_alignment %[[LHS]], 32
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[BASE_OFFSET]])
//  CHECK-DAG:   memref.assume_alignment %[[RHS]], 8
//  CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c1024)
//  CHECK-DAG:   memref.assume_alignment %[[RESULT]], 64
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
  %cst = arith.constant dense_resource<__elided__> : tensor<1x10xf32>
  %c512 = arith.constant 512 : index
  %c64 = arith.constant 64 : index
  %c10 = arith.constant 10 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c512) : !flow.dispatch.tensor<readonly:tensor<1x10xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c64) : !flow.dispatch.tensor<writeonly:tensor<1x10xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %2 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_x]
  %3 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_x]
  scf.for %arg0 = %2 to %c10 step %3 {
    %4 = affine.min affine_map<(d0) -> (4, -d0 + 10)>(%arg0)
    %5 = flow.dispatch.tensor.load %0, offsets = [0, %arg0], sizes = [1, %4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x10xf32>> -> tensor<1x?xf32>
    %6 = flow.dispatch.tensor.load %1, offsets = [0, %arg0], sizes = [1, %4], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x10xf32>> -> tensor<1x?xf32>
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
    flow.dispatch.tensor.store %7, %1, offsets = [0, %arg0], sizes = [1, %4], strides = [1, 1] : tensor<1x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x10xf32>>
  }
  return
}
//      CHECK: func.func @elementwise()
//  CHECK-DAG:   %[[CST_TENSOR:.+]] = arith.constant dense_resource<__elided__> : tensor<1x10xf32>
//  CHECK-DAG:   %[[CST_BUF:.+]] = bufferization.to_memref %[[CST_TENSOR]]
//  CHECK-DAG:   %[[IN_BUF:.+]] = hal.interface.binding.subspan set(0)  binding(0) {{.+}} : memref<1x10xf32, strided<[10, 1], offset: 128>, #hal.descriptor_type<storage_buffer>>
//  CHECK-DAG:   %[[OUT_BUF:.+]] = hal.interface.binding.subspan set(0)  binding(1) {{.+}} : memref<1x10xf32, strided<[10, 1], offset: 16>, #hal.descriptor_type<storage_buffer>>
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x40xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<10xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %3 = affine.apply #map0()[%workgroup_id_x]
  %4 = affine.apply #map0()[%workgroup_count_x]
  scf.for %arg0 = %3 to %c10 step %4 {
    %5 = flow.dispatch.tensor.load %0, offsets = [0, %arg0], sizes = [1, 2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x40xf32>> -> tensor<2xf32>
    %2 = flow.dispatch.tensor.load %1, offsets = [%arg0], sizes = [2], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<10xf32>> -> tensor<2xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%5 : tensor<2xf32>) outs(%2 : tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %8 = arith.addf %arg1, %arg1 : f32
      linalg.yield %8 : f32
    } -> tensor<2xf32>
    flow.dispatch.tensor.store %7, %1, offsets = [%arg0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:tensor<10xf32>>
  }
  return
}
//      CHECK: func.func @rank_reduced_slice()
//  CHECK-DAG:   %[[SRC_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1x40xf32, #hal.descriptor_type<storage_buffer>>
//  CHECK-DAG:   %[[DST_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<10xf32, #hal.descriptor_type<storage_buffer>>
//      CHECK:   scf.for %[[IV0:.+]] =
//  CHECK-DAG:     %[[SRC_SUBVIEW:.+]] = memref.subview %[[SRC_BINDING]][0, %[[IV0]]] [1, 2] [1, 1] : memref<1x40xf32{{.+}}> to memref<2xf32
//  CHECK-DAG:     %[[DST_SUBVIEW:.+]] = memref.subview %[[DST_BINDING]][%[[IV0]]] [2] [1] : memref<10xf32{{.+}}> to memref<2xf32
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
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<2x5xi32>
  memref.assume_alignment %1, 64 : memref<2x5xi32>
  %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<2x5xi32>>
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

module {
  func.func @tile_from_tensor_load_inplace() {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1}
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    scf.for %arg0 = %workgroup_id_y to %c2 step %c2 {
      scf.for %arg1 = %workgroup_id_x to %c4 step %c4 {
        %6 = flow.dispatch.tensor.load %3, offsets = [%arg0, 0], sizes = [1, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<1x3xf32>
        %7 = flow.dispatch.tensor.load %4, offsets = [0, %arg1], sizes = [3, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<3x1xf32>
        %8 = flow.dispatch.tensor.load %5, offsets = [%arg0, %arg1], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1} -> tensor<1x1xf32>
        %9 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%8 : tensor<1x1xf32>) -> tensor<1x1xf32>
        flow.dispatch.tensor.store %9, %5, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1}
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @tile_from_tensor_load_inplace()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS:.+]] = memref.subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = memref.subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//   CHECK-DAG:       %[[RESULT:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS]], %[[RHS]]
//  CHECK-SAME:         outs(%[[RESULT]]

// -----

module {
  func.func @tile_from_tensor_load_inplace_and_copy() {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1}
    %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    scf.for %arg0 = %workgroup_id_y to %c2 step %c2 {
      scf.for %arg1 = %workgroup_id_x to %c4 step %c4 {
        %7 = flow.dispatch.tensor.load %3, offsets = [%arg0, 0], sizes = [1, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<1x3xf32>
        %8 = flow.dispatch.tensor.load %4, offsets = [0, %arg1], sizes = [3, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<3x1xf32>
        %9 = flow.dispatch.tensor.load %5, offsets = [%arg0, %arg1], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1} -> tensor<1x1xf32>
        %10 = linalg.matmul ins(%7, %8 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%9 : tensor<1x1xf32>) -> tensor<1x1xf32>
        flow.dispatch.tensor.store %10, %5, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1}
        flow.dispatch.tensor.store %10, %6, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @tile_from_tensor_load_inplace_and_copy()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG:   %[[RETURN1:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//   CHECK-DAG:   %[[RETURN2:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS:.+]] = memref.subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = memref.subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//   CHECK-DAG:       %[[RESULT1:.+]] = memref.subview %[[RETURN1]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS]], %[[RHS]]
//  CHECK-SAME:         outs(%[[RESULT1]]
//       CHECK:       %[[RESULT2:.+]] = memref.subview %[[RETURN2]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.generic {{.*}} ins(%[[RESULT1]] {{.*}} outs(%[[RESULT2]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @tile_from_pointwise_lhs_inplace() {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1}
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    scf.for %arg0 = %workgroup_id_y to %c2 step %c2 {
      scf.for %arg1 = %workgroup_id_x to %c4 step %c4 {
        %6 = flow.dispatch.tensor.load %3, offsets = [%arg0, 0], sizes = [1, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<1x3xf32>
        %7 = flow.dispatch.tensor.load %4, offsets = [0, %arg1], sizes = [3, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<3x1xf32>
        %8 = bufferization.alloc_tensor() : tensor<1x3xf32>
        %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<1x3xf32>) outs(%8 : tensor<1x3xf32>) {
        ^bb0(%arg2: f32, %arg3: f32):
          %12 = arith.addf %arg2, %arg2 : f32
          linalg.yield %12 : f32
        } -> tensor<1x3xf32>
        %10 = flow.dispatch.tensor.load %5, offsets = [%arg0, %arg1], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1} -> tensor<1x1xf32>
        %11 = linalg.matmul ins(%9, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%10 : tensor<1x1xf32>) -> tensor<1x1xf32>
        flow.dispatch.tensor.store %11, %5, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1}
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @tile_from_pointwise_lhs_inplace()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS:.+]] = memref.subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = memref.subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//   CHECK-DAG:       %[[ALLOC:.+]] = memref.alloc() : memref<1x3xf32>
//       CHECK:       linalg.generic
//  CHECK-SAME:         ins(%[[LHS]] :
//  CHECK-SAME:         outs(%[[ALLOC]]
//   CHECK-DAG:       %[[RESULT:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[ALLOC]], %[[RHS]]
//  CHECK-SAME:         outs(%[[RESULT]]
//       CHECK:       memref.dealloc %[[ALLOC]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @tile_from_pointwise_outs() {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
    %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    scf.for %arg0 = %workgroup_id_y to %c2 step %c2 {
      scf.for %arg1 = %workgroup_id_x to %c4 step %c4 {
        %7 = flow.dispatch.tensor.load %6, offsets = [%arg0, %arg1], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1} -> tensor<1x1xf32>
        %8 = flow.dispatch.tensor.load %3, offsets = [%arg0, 0], sizes = [1, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<1x3xf32>
        %9 = flow.dispatch.tensor.load %4, offsets = [0, %arg1], sizes = [3, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<3x1xf32>
        %10 = flow.dispatch.tensor.load %5, offsets = [%arg0, %arg1], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<1x1xf32>
        %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<1x1xf32>) outs(%7 : tensor<1x1xf32>) {
        ^bb0(%arg2: f32, %arg3: f32):
          %13 = arith.addf %arg2, %arg2 : f32
          linalg.yield %13 : f32
        } -> tensor<1x1xf32>
        %12 = linalg.matmul ins(%8, %9 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%11 : tensor<1x1xf32>) -> tensor<1x1xf32>
        flow.dispatch.tensor.store %12, %6, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @tile_from_pointwise_outs()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG:   %[[TENSOR_INIT:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[RESULT:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//   CHECK-DAG:       %[[LHS:.+]] = memref.subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = memref.subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//   CHECK-DAG:       %[[INIT:.+]] = memref.subview %[[TENSOR_INIT]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.generic
//  CHECK-SAME:         ins(%[[INIT]] :
//  CHECK-SAME:         outs(%[[RESULT]]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS]], %[[RHS]]
//  CHECK-SAME:         outs(%[[RESULT]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @tile_from_pointwise_outs_inplace() {
    %cst = arith.constant 1.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1}
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    scf.for %arg0 = %workgroup_id_y to %c2 step %c2 {
      scf.for %arg1 = %workgroup_id_x to %c4 step %c4 {
        %6 = flow.dispatch.tensor.load %3, offsets = [%arg0, 0], sizes = [1, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<1x3xf32>
        %7 = flow.dispatch.tensor.load %4, offsets = [0, %arg1], sizes = [3, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<3x1xf32>
        %8 = flow.dispatch.tensor.load %5, offsets = [%arg0, %arg1], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1} -> tensor<1x1xf32>
        %9 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%8 : tensor<1x1xf32>) {
        ^bb0(%arg2: f32):
          %11 = arith.addf %arg2, %cst : f32
          linalg.yield %11 : f32
        } -> tensor<1x1xf32>
        %10 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%9 : tensor<1x1xf32>) -> tensor<1x1xf32>
        flow.dispatch.tensor.store %10, %5, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1}
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @tile_from_pointwise_outs_inplace()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[RESULT:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//   CHECK-DAG:       %[[LHS:.+]] = memref.subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = memref.subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//       CHECK:       linalg.generic
//  CHECK-SAME:         outs(%[[RESULT]]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS]], %[[RHS]]
//  CHECK-SAME:         outs(%[[RESULT]]

// -----

module {
  func.func @tile_from_matmul_outs_inplace() {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1}
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    scf.for %arg0 = %workgroup_id_y to %c2 step %c2 {
      scf.for %arg1 = %workgroup_id_x to %c4 step %c4 {
        %6 = flow.dispatch.tensor.load %3, offsets = [%arg0, 0], sizes = [1, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<1x3xf32>
        %7 = flow.dispatch.tensor.load %4, offsets = [0, %arg1], sizes = [3, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<3x1xf32>
        %8 = flow.dispatch.tensor.load %5, offsets = [%arg0, %arg1], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1} -> tensor<1x1xf32>
        %9 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%8 : tensor<1x1xf32>) -> tensor<1x1xf32>
        %10 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%9 : tensor<1x1xf32>) -> tensor<1x1xf32>
        flow.dispatch.tensor.store %10, %5, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%0, %1}
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @tile_from_matmul_outs_inplace()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[RESULT:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//   CHECK-DAG:       %[[LHS:.+]] = memref.subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = memref.subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         outs(%[[RESULT]]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         outs(%[[RESULT]]

// -----

#map0 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
module {
  func.func @bufferize_dynamic_inplace() {
    %c1 = arith.constant 1 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.constant.load[4] : index
    %5 = hal.interface.constant.load[5] : index
    %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
    %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
    %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%4, %5}
    %workgroup_size_x = hal.interface.workgroup.size[0] : index
    %workgroup_size_y = hal.interface.workgroup.size[1] : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %9 = arith.muli %workgroup_size_y, %workgroup_id_y : index
    %10 = arith.muli %workgroup_size_y, %workgroup_count_y : index
    scf.for %arg0 = %9 to %0 step %10 {
      %11 = arith.muli %workgroup_size_x, %workgroup_id_x : index
      %12 = arith.muli %workgroup_size_x, %workgroup_count_x : index
      scf.for %arg1 = %11 to %3 step %12 {
        %13 = affine.min #map0(%arg0)[%0, %workgroup_size_y]
        %14 = flow.dispatch.tensor.load %6, offsets = [%arg0, 0], sizes = [%13, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %15 = affine.min #map0(%arg1)[%3, %workgroup_size_x]
        %16 = flow.dispatch.tensor.load %7, offsets = [0, %arg1], sizes = [%2, %15], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
        %17 = affine.min #map1(%arg0)[%workgroup_size_y, %4]
        %18 = affine.min #map1(%arg1)[%workgroup_size_x, %5]
        %19 = flow.dispatch.tensor.load %8, offsets = [%arg0, %arg1], sizes = [%17, %18], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%4, %5} -> tensor<?x?xf32>
        %20 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%14, %16 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %20, %8, offsets = [%arg0, %arg1], sizes = [%17, %18], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%4, %5}
      }
    }
    return
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
//       CHECK: func.func @bufferize_dynamic_inplace()
//       CHECK:   %[[DIM0:.+]] = hal.interface.constant.load[0] : index
//       CHECK:   %[[DIM1:.+]] = hal.interface.constant.load[1] : index
//       CHECK:   %[[DIM2:.+]] = hal.interface.constant.load[2] : index
//       CHECK:   %[[DIM3:.+]] = hal.interface.constant.load[3] : index
//       CHECK:   %[[DIM4:.+]] = hal.interface.constant.load[4] : index
//       CHECK:   %[[DIM5:.+]] = hal.interface.constant.load[5] : index
//       CHECK:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%[[DIM0]], %[[DIM1]]}
//       CHECK:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%[[DIM2]], %[[DIM3]]}
//       CHECK:   %[[RESULT:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%[[DIM4]], %[[DIM5]]}
//   CHECK-DAG:   %[[WGSIZE_X:.+]] = hal.interface.workgroup.size[0]
//   CHECK-DAG:   %[[WGSIZE_Y:.+]] = hal.interface.workgroup.size[1]
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//       CHECK:       %[[TILE_M:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[DIM0]], %[[WGSIZE_Y]]]
//       CHECK:       %[[LHS_TILE:.+]] = memref.subview %[[LHS]][%[[IV0]], 0] [%[[TILE_M]], %[[DIM1]]]
//       CHECK:       %[[TILE_N:.+]] = affine.min #[[MAP0]](%[[IV1]])[%[[DIM3]], %[[WGSIZE_X]]]
//   CHECK-DAG:       %[[RHS_TILE:.+]] = memref.subview %[[RHS]][0, %[[IV1]]] [%[[DIM2]], %[[TILE_N]]]
//       CHECK:       %[[TILE_M_2:.+]] = affine.min #[[MAP2]](%[[IV0]])[%[[WGSIZE_Y]], %[[DIM4]]]
//       CHECK:       %[[TILE_N_2:.+]] = affine.min #[[MAP2]](%[[IV1]])[%[[WGSIZE_X]], %[[DIM5]]]
//   CHECK-DAG:       %[[RESULT_TILE:.+]] = memref.subview %[[RESULT]][%[[IV0]], %[[IV1]]] [%[[TILE_M_2]], %[[TILE_N_2]]]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS_TILE]], %[[RHS_TILE]]
//  CHECK-SAME:         outs(%[[RESULT_TILE]]

// -----

module {
  func.func @reshape_simple() {
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<12xi32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [12], strides = [1] : !flow.dispatch.tensor<readonly:tensor<12xi32>> -> tensor<12xi32>
    %3 = tensor.expand_shape %2 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
    flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
    return
  }
}
// CHECK-LABEL: func.func @reshape_simple()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//       CHECK:   %[[RESHAPE:.+]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1]]
//       CHECK:   linalg.generic {{.*}} ins(%[[RESHAPE]] {{.*}} outs(%[[RET0]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @reshape_fused_source() {
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<12xi32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
    %2 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<3x4xi32>> -> tensor<3x4xi32>
    %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [12], strides = [1] : !flow.dispatch.tensor<readonly:tensor<12xi32>> -> tensor<12xi32>
    %4 = tensor.expand_shape %3 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<3x4xi32>) outs(%2 : tensor<3x4xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %6 = arith.addi %arg0, %arg0 : i32
      linalg.yield %6 : i32
    } -> tensor<3x4xi32>
    flow.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
    return
  }
}
// CHECK-LABEL: func.func @reshape_fused_source()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<12xi32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<3x4xi32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[RESHAPE:.+]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[RESHAPE]] : memref<3x4xi32
//  CHECK-SAME:     outs(%[[RET0]] : memref<3x4xi32

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @reshape_fused_source_and_copyout() {
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<12xi32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
    %2 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<3x4xi32>> -> tensor<3x4xi32>
    %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
    %4 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [12], strides = [1] : !flow.dispatch.tensor<readonly:tensor<12xi32>> -> tensor<12xi32>
    %5 = tensor.expand_shape %4 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<3x4xi32>) outs(%2 : tensor<3x4xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %7 = arith.addi %arg0, %arg0 : i32
      linalg.yield %7 : i32
    } -> tensor<3x4xi32>
    flow.dispatch.tensor.store %6, %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
    flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
    return
  }
}
// CHECK-LABEL: func.func @reshape_fused_source_and_copyout()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<12xi32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<3x4xi32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[RET1:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<3x4xi32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[RESHAPE:.+]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[RESHAPE]] : memref<3x4xi32
//  CHECK-SAME:     outs(%[[RET0]] : memref<3x4xi32
//       CHECK:   linalg.generic {{.*}} ins(%[[RESHAPE]] {{.*}} outs(%[[RET1]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @reshape_fused_target() {
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x4xi32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<12xi32>>
    %2 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [12], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<12xi32>> -> tensor<12xi32>
    %3 = tensor.expand_shape %2 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
    %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<3x4xi32>> -> tensor<3x4xi32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<3x4xi32>) outs(%3 : tensor<3x4xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %7 = arith.addi %arg0, %arg0 : i32
      linalg.yield %7 : i32
    } -> tensor<3x4xi32>
    %6 = tensor.collapse_shape %5 [[0, 1]] : tensor<3x4xi32> into tensor<12xi32>
    flow.dispatch.tensor.store %6, %1, offsets = [0], sizes = [12], strides = [1] : tensor<12xi32> -> !flow.dispatch.tensor<writeonly:tensor<12xi32>>
    return
  }
}
// CHECK-LABEL: func.func @reshape_fused_target()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<3x4xi32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<12xi32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[RESHAPE:.+]] = memref.expand_shape %[[RET0]] {{\[}}[0, 1]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[ARG0]] : memref<3x4xi32
//  CHECK-SAME:     outs(%[[RESHAPE]] : memref<3x4xi32

// -----

#map0 = affine_map<(d0)[s0] -> (-d0 + 1, s0)>
#map1 = affine_map<(d0)[s0] -> (-d0 + 3, s0)>
module {
  func.func @dot_general_lowering() {
    %cst = arith.constant 0.000000e+00 : f32
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x1x2xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<2x3xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x3xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 1, 2], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x2xf32>> -> tensor<1x1x2xf32>
    %4 = tensor.collapse_shape %3 [[0, 1], [2]] : tensor<1x1x2xf32> into tensor<1x2xf32>
    %workgroup_size_x = hal.interface.workgroup.size[0] : index
    %workgroup_size_y = hal.interface.workgroup.size[1] : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %5 = arith.muli %workgroup_size_y, %workgroup_id_y : index
    %6 = arith.muli %workgroup_size_y, %workgroup_count_y : index
    scf.for %arg0 = %5 to %c1 step %6 {
      %7 = arith.muli %workgroup_size_x, %workgroup_id_x : index
      %8 = arith.muli %workgroup_size_x, %workgroup_count_x : index
      scf.for %arg1 = %7 to %c3 step %8 {
        %9 = affine.min #map0(%arg0)[%workgroup_size_y]
        %10 = affine.min #map1(%arg1)[%workgroup_size_x]
        %11 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%9, %10], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x3xf32>> -> tensor<?x?xf32>
        %12 = tensor.extract_slice %4[%arg0, 0] [%9, 2] [1, 1] : tensor<1x2xf32> to tensor<?x2xf32>
        %13 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [2, %10], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x3xf32>> -> tensor<2x?xf32>
        %14 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %15 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%12, %13 : tensor<?x2xf32>, tensor<2x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %15, %2, offsets = [%arg0, %arg1], sizes = [%9, %10], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x3xf32>>
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @dot_general_lowering()
//   CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG:   %[[RESHAPE_LHS:.+]] = memref.collapse_shape %[[LHS]]
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS_TILE:.+]] = memref.subview %[[RESHAPE_LHS]][%[[IV0]], 0]
//   CHECK-DAG:       %[[RESULT_TILE:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:       %[[RHS_TILE:.+]] = memref.subview %[[RHS]][0, %[[IV1]]]
//       CHECK:       linalg.fill
//  CHECK-SAME:           outs(%[[RESULT_TILE]] :
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS_TILE]], %[[RHS_TILE]]
//  CHECK-SAME:         outs(%[[RESULT_TILE]]

// -----

module {
  func.func @slice() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1}
    %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%2, %3}
    %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1} -> tensor<?x?xi32>
    %7 = tensor.extract_slice %6[%0, %1] [%2, %3] [1, 1] : tensor<?x?xi32> to tensor<?x?xi32>
    flow.dispatch.tensor.store %7, %5, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%2, %3}
    return
  }
}
// CHECK-LABEL: func.func @slice()
//   CHECK-DAG: %[[ARG:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG: %[[RETURN:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG: %[[SUBVIEW:.+]] = memref.subview %[[ARG]]
//       CHECK: linalg.generic {{.*}} ins(%[[SUBVIEW]] {{.*}} outs(%[[RETURN]]

// -----

module {
  func.func @slice_rank_reducing() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.constant.load[4] : index
    %5 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%4, %4, %4}
    %6 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%2, %3}
    %7 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [%4, %4, %4], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%4, %4, %4} -> tensor<?x?x?xi32>
    %8 = tensor.extract_slice %7[%0, %0, %1] [%2, 1, %3] [1, 1, 1] : tensor<?x?x?xi32> to tensor<?x?xi32>
    flow.dispatch.tensor.store %8, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%2, %3}
    return
  }
}
// CHECK-LABEL: func.func @slice_rank_reducing()
//   CHECK-DAG: %[[ARG:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG: %[[RETURN:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG: %[[SUBVIEW:.+]] = memref.subview %[[ARG]]
//       CHECK: linalg.generic {{.*}} ins(%[[SUBVIEW]] {{.*}} outs(%[[RETURN]]

// -----

module {
  func.func @slice_multiple_copy() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.constant.load[4] : index
    %5 = hal.interface.constant.load[5] : index
    %6 = hal.interface.constant.load[6] : index
    %7 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%6, %6, %6}
    %8 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?x?xi32>>{%3, %4, %5}
    %9 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%3, %5}
    %10 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0], sizes = [%6, %6, %6], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%6, %6, %6} -> tensor<?x?x?xi32>
    %11 = tensor.extract_slice %10[%0, %1, %2] [%3, %4, %5] [1, 1, 1] : tensor<?x?x?xi32> to tensor<?x?x?xi32>
    %12 = tensor.extract_slice %10[%0, %1, %2] [%3, 1, %5] [1, 1, 1] : tensor<?x?x?xi32> to tensor<?x?xi32>
    flow.dispatch.tensor.store %11, %8, offsets = [0, 0, 0], sizes = [%3, %4, %5], strides = [1, 1, 1] : tensor<?x?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?xi32>>{%3, %4, %5}
    flow.dispatch.tensor.store %12, %9, offsets = [%0, %2], sizes = [%3, %5], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%3, %5}
    return
  }
}
// CHECK-LABEL: func.func @slice_multiple_copy()
//   CHECK-DAG: %[[ARG:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG: %[[RETURN1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG: %[[RETURN2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//   CHECK-DAG: %[[SIZE1:.+]] = hal.interface.constant.load[3] : index
//   CHECK-DAG: %[[SIZE2:.+]] = hal.interface.constant.load[4] : index
//   CHECK-DAG: %[[SIZE3:.+]] = hal.interface.constant.load[5] : index
//   CHECK-DAG: %[[SUBVIEW1:.+]] = memref.subview %[[ARG]][%{{.+}}, %{{.+}}, %{{.+}}] [%[[SIZE1]], %[[SIZE2]], %[[SIZE3]]]
//   CHECK-DAG: %[[SUBVIEW2:.+]] = memref.subview %[[ARG]][%{{.+}}, %{{.+}}, %{{.+}}] [%[[SIZE1]], 1, %[[SIZE3]]]
//       CHECK: linalg.generic {{.*}} ins(%[[SUBVIEW1]] {{.*}} outs(%[[RETURN1]]
//       CHECK: linalg.generic {{.*}} ins(%[[SUBVIEW2]] {{.*}} outs(%[[RETURN2]]

// -----

module {
  func.func @slice_in_place() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>{%0, %1}
    %3 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>{%0, %1} -> tensor<?x?xi32>
    flow.dispatch.tensor.store %3, %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>{%0, %1}
    return
  }
}
// CHECK-LABEL: func.func @slice_in_place()
//   CHECK-NOT:   linalg.generic

// -----

module {
  func.func @slice_whole_stride_dispatch_0() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1}
    %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%2, %3}
    %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1} -> tensor<?x?xi32>
    %7 = tensor.extract_slice %6[1, 0] [1, 4] [1, 1] : tensor<?x?xi32> to tensor<1x4xi32>
    flow.dispatch.tensor.store %7, %5, offsets = [0, 0], sizes = [1, 4], strides = [1, 1] : tensor<1x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%2, %3}
    return
  }
}
// CHECK-LABEL: func.func @slice_whole_stride_dispatch_0()
//   CHECK-DAG:   %[[INPUT:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG:   %[[SUB_IN_FIXED:.+]] = memref.subview %[[INPUT]][1, 0] [1, 4] [1, 1]
//   CHECK-DAG:   %[[SUB_OUT_FIXED:.+]] = memref.subview %[[OUTPUT]][0, 0] [1, 4] [1, 1]
//       CHECK:   linalg.generic {{.*}} ins(%[[SUB_IN_FIXED]] {{.*}} outs(%[[SUB_OUT_FIXED]]

// -----

module {
  func.func @subtensor_insert() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.constant.load[4] : index
    %5 = hal.interface.constant.load[5] : index
    %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1}
    %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%2, %3}
    %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%4, %5}
    %9 = flow.dispatch.tensor.load %6, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1} -> tensor<?x?xi32>
    %10 = flow.dispatch.tensor.load %7, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%2, %3} -> tensor<?x?xi32>
    %11 = tensor.insert_slice %9 into %10[3, 4] [%0, %1] [1, 1] : tensor<?x?xi32> into tensor<?x?xi32>
    flow.dispatch.tensor.store %11, %8, offsets = [0, 0], sizes = [%4, %5], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%4, %5}
    return
  }
}
// CHECK-LABEL: func.func @subtensor_insert()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//   CHECK-DAG:   %[[D0:.+]] = hal.interface.constant.load[0] : index
//   CHECK-DAG:   %[[D1:.+]] = hal.interface.constant.load[1] : index
//   CHECK-DAG:   %[[D2:.+]] = hal.interface.constant.load[2] : index
//   CHECK-DAG:   %[[D3:.+]] = hal.interface.constant.load[3] : index
//   CHECK-DAG:   %[[D4:.+]] = hal.interface.constant.load[4] : index
//   CHECK-DAG:   %[[D5:.+]] = hal.interface.constant.load[5] : index
//   CHECK-DAG:   %[[ALLOC:.+]] = memref.alloc(%[[D2]], %[[D3]]) : memref<?x?xi32>
//       CHECK:   linalg.generic {{.*}} ins(%[[ARG1]] {{.*}} outs(%[[ALLOC]]
//       CHECK:   %[[SUB_ALLOC:.+]] = memref.subview %[[ALLOC]]
//       CHECK:   linalg.generic {{.*}} ins(%[[ARG0]] {{.*}} outs(%[[SUB_ALLOC]]
//       CHECK:   linalg.generic {{.*}} ins(%[[ALLOC]] {{.*}} outs(%[[RET0]]
//       CHECK:   memref.dealloc %[[ALLOC]]

// -----

module {
  func.func @tensor_extract() {
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<i32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<3x9xi32>>
    %2 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [3, 9], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<3x9xi32>> -> tensor<3x9xi32>
    %3 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<i32>> -> tensor<i32>
    %4 = tensor.extract %3[] : tensor<i32>
    %5 = linalg.fill ins(%4 : i32) outs(%2 : tensor<3x9xi32>) -> tensor<3x9xi32>
    flow.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [3, 9], strides = [1, 1] : tensor<3x9xi32> -> !flow.dispatch.tensor<writeonly:tensor<3x9xi32>>
    return
  }
}
// CHECK-LABEL: func.func @tensor_extract()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//       CHECK:   %[[LOAD:.+]] = memref.load %[[ARG0]]
//       CHECK:   linalg.fill
//  CHECK-SAME:       ins(%[[LOAD]] :
//  CHECK-SAME:       outs(%[[RET0]] :

// -----

module {
  func.func @load_to_store() {
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x4xi32>>
    %2 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<3x4xi32>> -> tensor<3x4xi32>
    flow.dispatch.tensor.store %2, %0, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
    return
  }
}
// CHECK-LABEL: func.func @load_to_store()
//       CHECK:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<3x4xi32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<3x4xi32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   linalg.generic {{.*}} ins(%[[IN]] {{.*}} outs(%[[OUT]]

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (-d0 + 5, s0)>
module {
  func.func @rhs_non_splat_constant() {
    %cst = arith.constant dense<[[0.706495285, -0.567672312, 0.483717591, 0.522725761, 0.7563259], [-0.0899272263, -0.283501834, -0.350822538, -0.351515919, -0.337136656], [-0.451804549, 0.372324884, -0.620518147, 0.235451385, 0.851095855]]> : tensor<3x5xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x5x3x1xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<5x5xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 5, 3, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x5x3x1xf32>> -> tensor<1x5x3x1xf32>
    %3 = tensor.collapse_shape %2 [[0, 1], [2, 3]] : tensor<1x5x3x1xf32> into tensor<5x3xf32>
    %workgroup_size_x = hal.interface.workgroup.size[0] : index
    %workgroup_size_y = hal.interface.workgroup.size[1] : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %4 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
    %5 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
    scf.for %arg0 = %4 to %c5 step %5 {
      %6 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
      %7 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
      scf.for %arg1 = %6 to %c5 step %7 {
        %8 = affine.min #map1(%arg0)[%workgroup_size_y]
        %9 = affine.min #map1(%arg1)[%workgroup_size_x]
        %10 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1], sizes = [%8, %9], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<5x5xf32>> -> tensor<?x?xf32>
        %11 = tensor.extract_slice %3[%arg0, 0] [%8, 3] [1, 1] : tensor<5x3xf32> to tensor<?x3xf32>
        %12 = tensor.extract_slice %cst[0, %arg1] [3, %9] [1, 1] : tensor<3x5xf32> to tensor<3x?xf32>
        %13 = linalg.fill ins(%cst_0 : f32) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %14 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%11, %12 : tensor<?x3xf32>, tensor<3x?xf32>) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %14, %1, offsets = [%arg0, %arg1], sizes = [%8, %9], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<5x5xf32>>
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @rhs_non_splat_constant
//   CHECK-DAG:   %[[CONSTANT:.+]] = arith.constant {{.+}} : tensor<3x5xf32>
//   CHECK-DAG:   %[[RHS:.+]] = bufferization.to_memref %[[CONSTANT]]
//   CHECK-DAG:   %[[LHS_INPUT:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1x5x3x1xf32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<5x5xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[LHS:.+]] = memref.collapse_shape %[[LHS_INPUT]]
//       CHECK:   scf.for %[[IV0:.+]] =
//       CHECK:     scf.for %[[IV1:.+]] =
//   CHECK-DAG:       %[[LHS_SUBVIEW:.+]] = memref.subview %[[LHS]][%[[IV0]], 0]
//   CHECK-DAG:       %[[RHS_SUBVIEW:.+]] = memref.subview %[[RHS]][0, %[[IV1]]]
//   CHECK-DAG:       %[[RESULT_SUBVIEW:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]]
//       CHECK:       linalg.fill
//  CHECK-SAME:           outs(%[[RESULT_SUBVIEW]] :
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS_SUBVIEW]], %[[RHS_SUBVIEW]]
//  CHECK-SAME:         outs(%[[RESULT_SUBVIEW]]

// -----

#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @gather() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.constant.load[4] : index
    %5 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
    %6 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xi32>>{%2}
    %7 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%3, %4}
    %8 = flow.dispatch.tensor.load %7, offsets = [0, 0], sizes = [%3, %4], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%3, %4} -> tensor<?x?xf32>
    %9 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
    %10 = flow.dispatch.tensor.load %6, offsets = [0], sizes = [%2], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xi32>>{%2} -> tensor<?xi32>
    %11 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<?xi32>) outs(%8 : tensor<?x?xf32>) {
    ^bb0(%arg0: i32, %arg1: f32):
      %12 = linalg.index 1 : index
      %13 = arith.index_cast %arg0 : i32 to index
      %14 = tensor.extract %9[%13, %12] : tensor<?x?xf32>
      linalg.yield %14 : f32
    } -> tensor<?x?xf32>
    flow.dispatch.tensor.store %11, %7, offsets = [0, 0], sizes = [%3, %4], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%3, %4}
    return
  }
}
// CHECK-LABEL: func.func @gather()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//       CHECK:   linalg.generic
//       CHECK:     %[[VAL:.+]] = memref.load %[[ARG0]]
//       CHECK:     linalg.yield %[[VAL]]

// -----

module {
  func.func @pooling_nhwc_sum() {
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<f32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x4x6x1xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x2x2x1xf32>>
    %3 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [1, 2, 2, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x2x2x1xf32>> -> tensor<1x2x2x1xf32>
    %4 = bufferization.alloc_tensor() : tensor<2x3xf32>
    %5 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
    %6 = tensor.extract %5[] : tensor<f32>
    %7 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1, 4, 6, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4x6x1xf32>> -> tensor<1x4x6x1xf32>
    %8 = linalg.fill ins(%6 : f32) outs(%3 : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
    %9 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<[2, 3]> : vector<2xi64>} ins(%7, %4 : tensor<1x4x6x1xf32>, tensor<2x3xf32>) outs(%8 : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
    flow.dispatch.tensor.store %9, %2, offsets = [0, 0, 0, 0], sizes = [1, 2, 2, 1], strides = [1, 1, 1, 1] : tensor<1x2x2x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x2x2x1xf32>>
    return
  }
}
// CHECK-LABEL: func.func @pooling_nhwc_sum
//   CHECK-DAG:   %[[WINDOW:.+]] = memref.alloc() : memref<2x3xf32>
//   CHECK-DAG:   %[[INIT:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<f32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[INPUT:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<1x4x6x1xf32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<1x2x2x1xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[INIT_VAL:.+]] = memref.load %[[INIT]][] : memref<f32{{.+}}>
//       CHECK:   linalg.fill
//  CHECK-SAME:       ins(%[[INIT_VAL]] :
//  CHECK-SAME:       outs(%[[RET0]] :
//       CHECK:   linalg.pooling_nhwc_sum
//  CHECK-SAME:     dilations = dense<1> : vector<2xi64>
//  CHECK-SAME:     strides = dense<[2, 3]> : vector<2xi64>
//  CHECK-SAME:     ins(%[[INPUT]], %[[WINDOW]] : memref<1x4x6x1xf32{{.+}}>, memref<2x3xf32>)
//  CHECK-SAME:    outs(%[[RET0]] : memref<1x2x2x1xf32{{.+}}>)

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @read_only_subtensor() {
    %c1 = arith.constant 1 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.constant.load[4] : index
    %5 = hal.interface.constant.load[5] : index
    %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    %7 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
    %8 = flow.dispatch.tensor.load %7, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
    %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%4, %5}
    %10 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [%4, %5], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%4, %5} -> tensor<?x?xf32>
    %workgroup_size_x = hal.interface.workgroup.size[0] : index
    %workgroup_size_y = hal.interface.workgroup.size[1] : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %11 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
    %12 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
    scf.for %arg0 = %11 to %2 step %12 {
      %13 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
      %14 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
      scf.for %arg1 = %13 to %3 step %14 {
        %15 = affine.min #map1(%arg0)[%workgroup_size_y, %2]
        %16 = affine.min #map1(%arg1)[%workgroup_size_x, %3]
        %17 = flow.dispatch.tensor.load %6, offsets = [%arg0, %arg1], sizes = [%15, %16], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %18 = tensor.extract_slice %8[%arg0, %arg1] [%15, %16] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %19 = tensor.extract_slice %10[%arg0, %arg1] [%15, %16] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %20 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%18, %18, %19, %19 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
        ^bb0(%arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
          %21 = arith.mulf %arg4, %arg5 : f32
          %22 = arith.mulf %arg2, %arg3 : f32
          %23 = arith.addf %22, %21 : f32
          %24 = math.sqrt %23 : f32
          linalg.yield %24 : f32
        } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %20, %6, offsets = [%arg0, %arg1], sizes = [%15, %16], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @read_only_subtensor
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   scf.for
//       CHECK:     scf.for
//   CHECK-DAG:       %[[SV1:.+]] = memref.subview %[[ARG0]]
//   CHECK-DAG:       %[[SV2:.+]] = memref.subview %[[ARG1]]
//   CHECK-DAG:       %[[SV3:.+]] = memref.subview %[[RET0]]
//       CHECK:       linalg.generic
//  CHECK-SAME:         ins(%[[SV1]], %[[SV2]] :
//  CHECK-SAME:         outs(%[[SV3]] :

// -----

#map = affine_map<(d0) -> (d0)>
module {
  func.func @reshape_read_only() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%2}
    %5 = flow.dispatch.tensor.load %4, offsets = [0], sizes = [%2], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%2} -> tensor<?xf32>
    %6 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
    %7 = tensor.collapse_shape %6 [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%7 : tensor<?xf32>)
      outs(%5 : tensor<?xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %9 = arith.addf %arg0, %arg0 : f32
      linalg.yield %9 : f32
    } -> tensor<?xf32>
    flow.dispatch.tensor.store %8, %4, offsets = [0], sizes = [%2], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%2}
    return
  }
}
// CHECK-LABEL: func.func @reshape_read_only
//   CHECK-DAG:   %[[INPUT:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//       CHECK:   %[[RESHAPE:.+]] = memref.collapse_shape %[[INPUT]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[RESHAPE]] : memref<?xf32
//  CHECK-SAME:     outs(%[[OUTPUT]] : memref<?xf32

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @use_buffer_for_operand_when_output_tensor_not_used() {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x225x225x16xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x16x32xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<32xf32>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
    %4 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>> -> tensor<1x112x112x32xf32>
    %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 255, 255, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x16xf32>> -> tensor<1x225x225x16xf32>
    %6 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 16, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x16x32xf32>> -> tensor<3x3x16x32xf32>
    %7 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readonly:tensor<32xf32>> -> tensor<32xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    %9 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%5, %6 : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) outs(%8 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    %10 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<32xf32>) outs(%9 : tensor<1x112x112x32xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %11 = arith.subf %arg1, %arg0 : f32
      linalg.yield %11 : f32
    } -> tensor<1x112x112x32xf32>
    flow.dispatch.tensor.store %10, %3, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 32], strides = [1, 1, 1, 1] : tensor<1x112x112x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
    return
  }
}
// CHECK: func.func @use_buffer_for_operand_when_output_tensor_not_used()

//  CHECK-NOT: memref.alloc
//      CHECK: %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
//      CHECK: linalg.fill
// CHECK-SAME:     outs(%[[OUTPUT]] :
// CHECK-NEXT: linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:   outs(%[[OUTPUT]] : memref<1x112x112x32xf32{{.+}}>)
// CHECK-NEXT: linalg.generic
// CHECK-SAME:   ins(%{{.+}} : memref<32xf32
// CHECK-SAME:   outs(%[[OUTPUT]] : memref<1x112x112x32xf32{{.+}}>)

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3)>
module {
  func.func @dont_use_buffer_for_operand_when_output_tensor_used() {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x225x225x16xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x16x32xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<32xf32>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
    %4 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>> -> tensor<1x112x112x32xf32>
    %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x16xf32>> -> tensor<1x225x225x16xf32>
    %6 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 16, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x16x32xf32>> -> tensor<3x3x16x32xf32>
    %7 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readonly:tensor<32xf32>> -> tensor<32xf32>
    %8 = bufferization.alloc_tensor() : tensor<1x112x112x32xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    %10 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%5, %6 : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>) outs(%9 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    %12 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10, %7 : tensor<1x112x112x32xf32>, tensor<32xf32>) outs(%11 : tensor<1x112x112x32xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %13 = arith.subf %arg0, %arg1 : f32
      %14 = arith.addf %13, %arg2 : f32
      linalg.yield %14 : f32
    } -> tensor<1x112x112x32xf32>
    flow.dispatch.tensor.store %12, %3, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 32], strides = [1, 1, 1, 1] : tensor<1x112x112x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
    return
  }
}
// CHECK-LABEL: func.func @dont_use_buffer_for_operand_when_output_tensor_used()
//   CHECK-DAG:   %[[ALLOC:.+]] = memref.alloc
//   CHECK-DAG:   %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
//       CHECK:   linalg.fill
//  CHECK-SAME:       outs(%[[ALLOC]] :
//  CHECK-NEXT:   linalg.conv_2d_nhwc_hwcf
//  CHECK-SAME:     outs(%[[ALLOC]] : memref<1x112x112x32xf32>)
//  CHECK-NEXT:   linalg.fill
//  CHECK-SAME:       outs(%[[OUTPUT]] :
//  CHECK-NEXT:   linalg.generic
//  CHECK-SAME:     ins(%[[ALLOC]], %{{.+}} : memref<1x112x112x32xf32>, memref<32xf32
//  CHECK-SAME:     outs(%[[OUTPUT]] : memref<1x112x112x32xf32{{.+}}>)

// -----

#map0 = affine_map<(d0) -> (-d0 + 4)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module {
  func.func @bufferize_cst_output_tensor() {
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<5xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<i32>>
    %2 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<writeonly:tensor<i32>> -> tensor<i32>
    %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [5], strides = [1] : !flow.dispatch.tensor<readonly:tensor<5xf32>> -> tensor<5xf32>
    %4 = linalg.fill ins(%c-2147483648_i32 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    %5 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["reduction"]} ins(%3, %cst_0 : tensor<5xf32>, tensor<5xi32>) outs(%4 : tensor<i32>) {
    ^bb0(%arg0: f32, %arg1: i32, %arg2: i32):
      %6 = arith.cmpf oeq, %arg0, %cst : f32
      %7 = arith.extui %6 : i1 to i32
      %8 = arith.muli %7, %arg1 : i32
      %9 = arith.cmpi sgt, %8, %arg2 : i32
      %10 = arith.select %9, %8, %arg2 : i32
      linalg.yield %10 : i32
    } -> tensor<i32>
    flow.dispatch.tensor.store %5, %1, offsets = [], sizes = [], strides = [] : tensor<i32> -> !flow.dispatch.tensor<writeonly:tensor<i32>>
    return
  }
}
// CHECK-LABEL: func.func @bufferize_cst_output_tensor()

//       CHECK-DAG: %[[CST1:.+]] = arith.constant -2147483648 : i32
//       CHECK-DAG: %[[CST5:.+]] = arith.constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
//       CHECK: %[[CAST5:.+]] = bufferization.to_memref %[[CST5]] : memref<5xi32>
//       CHECK: %[[INPUT:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<5xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK: %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<i32, #hal.descriptor_type<storage_buffer>>
//       CHECK: linalg.fill ins(%[[CST1]] : i32) outs(%[[OUTPUT]] : memref<i32{{.+}}>)
//       CHECK: linalg.generic
//  CHECK-SAME:   ins(%[[INPUT]], %[[CAST5]] : {{.*}}) outs(%[[OUTPUT]] : memref<i32{{.+}}>)

// -----

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  func.func @cast_follwed_by_store() {
    %cst = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4x32x1024xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4x1024x64xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<4x32x64xf32>>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %workgroup_count_z = hal.interface.workgroup.count[2] : index
    scf.for %arg0 = %workgroup_id_z to %c4 step %workgroup_count_z {
      %3 = affine.apply #map()[%workgroup_id_y]
      %4 = affine.apply #map()[%workgroup_count_y]
      scf.for %arg1 = %3 to %c32 step %4 {
        %5 = affine.apply #map()[%workgroup_id_x]
        %6 = affine.apply #map()[%workgroup_count_x]
        scf.for %arg2 = %5 to %c64 step %6 {
          %7 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1, %arg2], sizes = [1, 32, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<4x32x64xf32>> -> tensor<1x32x32xf32>
          %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1, 0], sizes = [1, 32, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x32x1024xf32>> -> tensor<1x32x1024xf32>
          %9 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0, %arg2], sizes = [1, 1024, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x1024x64xf32>> -> tensor<1x1024x32xf32>
          %10 = linalg.fill {__internal_linalg_transform__ = "workgroup"} ins(%cst : f32) outs(%7 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>
          %11 = linalg.batch_matmul {__internal_linalg_transform__ = "workgroup", is_root_op} ins(%8, %9 : tensor<1x32x1024xf32>, tensor<1x1024x32xf32>) outs(%10 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>
          flow.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1, %arg2], sizes = [%c1, %c32, %c32], strides = [1, 1, 1] : tensor<1x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x32x64xf32>>
        }
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @cast_follwed_by_store()
//   CHECK-DAG: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG: %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x32x1024xf32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG: %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<4x1024x64xf32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG: %[[RESULT:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x32x64xf32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG: %[[LHSV:.+]] = memref.subview %[[LHS]]
//   CHECK-DAG: %[[RHSV:.+]] = memref.subview %[[RHS]]
//   CHECK-DAG: %[[RESULTV:.+]] = memref.subview %[[RESULT]]
//        CHECK: linalg.fill
//   CHECK-SAME:     ins(%[[ZERO]] :
//   CHECK-SAME:     outs(%[[RESULTV]] :
//        CHECK: linalg.batch_matmul {{.*}} ins(%[[LHSV]], %[[RHSV]] : {{.*}}) outs(%[[RESULTV]]

// -----

module {
  func.func @rank_reduced_subtensor_insert() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.constant.load[4] : index
    %5 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
    %6 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<?x?x?xf32>>{%2, %3, %4}
    %7 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
    %8 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0], sizes = [%2, %3, %4], strides = [1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?x?xf32>>{%2, %3, %4} -> tensor<?x?x?xf32>
    %9 = tensor.insert_slice %7 into %8[0, 0, 0] [1, %3, %4] [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
    flow.dispatch.tensor.store %9, %6, offsets = [0, 0, 0], sizes = [%2, %3, %4], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?x?xf32>>{%2, %3, %4}
    return
  }
}
// CHECK-LABEL: func.func @rank_reduced_subtensor_insert()
//   CHECK-DAG:   %[[ARG:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[RET:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[RET]]
//       CHECK:   linalg.generic {{.*}} ins(%[[ARG]] {{.*}} outs(%[[SUBVIEW]]

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @bufferize_transfer_op_inplace() {
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<2x3xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x4xf32>>
  %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<2x4xf32>>
  %4 = flow.dispatch.tensor.load %0, offsets = [%c0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:tensor<2x3xf32>> -> tensor<2x3xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [%c0, %c0], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:tensor<3x4xf32>> -> tensor<3x1xf32>
  %6 = flow.dispatch.tensor.load %3, offsets = [%c0, %c0], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readwrite:tensor<2x4xf32>> -> tensor<2x1xf32>
  %7 = vector.transfer_read %4[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<2x3xf32>, vector<1x1xf32>
  %8 = vector.transfer_read %4[%c0, %c1], %cst {in_bounds = [true, true]} : tensor<2x3xf32>, vector<1x1xf32>
  %9 = vector.transfer_read %4[%c0, %c2], %cst {in_bounds = [true, true]} : tensor<2x3xf32>, vector<1x1xf32>
  %10 = vector.transfer_read %4[%c1, %c0], %cst {in_bounds = [true, true]} : tensor<2x3xf32>, vector<1x1xf32>
  %11 = vector.transfer_read %4[%c1, %c1], %cst {in_bounds = [true, true]} : tensor<2x3xf32>, vector<1x1xf32>
  %12 = vector.transfer_read %4[%c1, %c2], %cst {in_bounds = [true, true]} : tensor<2x3xf32>, vector<1x1xf32>
  %13 = vector.transfer_read %5[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<3x1xf32>, vector<1x1xf32>
  %14 = vector.transfer_read %5[%c1, %c0], %cst {in_bounds = [true, true]} : tensor<3x1xf32>, vector<1x1xf32>
  %15 = vector.transfer_read %5[%c2, %c0], %cst {in_bounds = [true, true]} : tensor<3x1xf32>, vector<1x1xf32>
  %16 = vector.transfer_read %6[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<2x1xf32>, vector<1x1xf32>
  %17 = vector.transfer_read %6[%c1, %c0], %cst {in_bounds = [true, true]} : tensor<2x1xf32>, vector<1x1xf32>
  %18 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %7, %13, %16 : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
  %19 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %8, %14, %18 : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
  %20 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %9, %15, %19 : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
  %21 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %10, %13, %17 : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
  %22 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %11, %14, %21 : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
  %23 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %12, %15, %22 : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
  %24 = vector.transfer_write %20, %6[%c0, %c0] {in_bounds = [true, true]} : vector<1x1xf32>, tensor<2x1xf32>
  %25 = vector.transfer_write %23, %24[%c1, %c0] {in_bounds = [true, true]} : vector<1x1xf32>, tensor<2x1xf32>
  flow.dispatch.tensor.store %25, %3, offsets = [%c0, %c0], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<2x1xf32> -> !flow.dispatch.tensor<readwrite:tensor<2x4xf32>>
  return
}

//   CHECK-LABEL: func.func @bufferize_transfer_op_inplace()
//     CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//     CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//     CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//     CHECK-DAG:   %[[ARG0V:.+]] = memref.subview %[[ARG0]]
//     CHECK-DAG:   %[[ARG1V:.+]] = memref.subview %[[ARG1]]
//     CHECK-DAG:   %[[RET0V:.+]] = memref.subview %[[RET0]]
// CHECK-COUNT-6:   vector.transfer_read %[[ARG0V]]
// CHECK-COUNT-3:   vector.transfer_read %[[ARG1V]]
// CHECK-COUNT-2:   vector.transfer_read %[[RET0V]]
//     CHECK-NOT:   linalg.generic
//         CHECK:   vector.transfer_write %{{.+}}, %[[RET0V]]
//         CHECK:   vector.transfer_write %{{.+}}, %[[RET0V]]

// -----

#map0 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @multi_result() {
    %c1 = arith.constant 1 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.constant.load[4] : index
    %5 = hal.interface.constant.load[5] : index
    %6 = hal.interface.constant.load[6] : index
    %7 = hal.interface.constant.load[7] : index
    %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
    %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
    %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%4, %5}
    %11 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
    %12 = hal.interface.constant.load[8] : index
    %13 = hal.interface.constant.load[9] : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %workgroup_size_x = hal.interface.workgroup.size[0] : index
    %workgroup_size_y = hal.interface.workgroup.size[1] : index
    %14 = arith.muli %workgroup_id_y, %workgroup_size_y : index
    %15 = arith.muli %workgroup_count_y, %workgroup_size_y : index
    %16 = arith.muli %workgroup_id_x, %workgroup_size_x : index
    %17 = arith.muli %workgroup_count_x, %workgroup_size_x : index
    scf.for %arg0 = %14 to %12 step %15 {
      scf.for %arg1 = %16 to %13 step %17 {
        %18 = affine.min #map0(%arg0)[%12, %workgroup_size_y]
        %19 = affine.min #map0(%arg1)[%13, %workgroup_size_x]
        %20 = flow.dispatch.tensor.load %11, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %21 = flow.dispatch.tensor.load %10, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%4, %5} -> tensor<?x?xf32>
        %22 = flow.dispatch.tensor.load %8, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %23 = flow.dispatch.tensor.load %9, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
        %24:2 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%22, %23 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%21, %20 : tensor<?x?xf32>, tensor<?x?xf32>) {
        ^bb0(%arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32):
          %25 = arith.mulf %arg2, %arg3 : f32
          %26 = arith.addf %arg2, %arg3 : f32
          linalg.yield %25, %26 : f32, f32
        } -> (tensor<?x?xf32>, tensor<?x?xf32>)
        flow.dispatch.tensor.store %24#0, %10, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%4, %5}
        flow.dispatch.tensor.store %24#1, %11, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @multi_result()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//   CHECK-DAG:   %[[RET1:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
//   CHECK-DAG:   %[[ARG0V:.+]] = memref.subview %[[ARG0]]
//   CHECK-DAG:   %[[ARG1V:.+]] = memref.subview %[[ARG1]]
//   CHECK-DAG:   %[[RET0V:.+]] = memref.subview %[[RET0]]
//   CHECK-DAG:   %[[RET1V:.+]] = memref.subview %[[RET1]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[ARG0V]], %[[ARG1V]]
//  CHECK-SAME:     outs(%[[RET0V]], %[[RET1V]]

// -----

#map0 = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
#map3 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @multi_result_reduce() {
    %c0_i32 = arith.constant 0 : i32
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1}
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1}
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?xi32>>{%2}
    %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?xi32>>{%2}
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %7 = affine.apply #map0()[%workgroup_id_x]
    %8 = affine.apply #map0()[%workgroup_count_x]
    scf.for %arg0 = %7 to %1 step %8 {
      %9 = affine.min #map1(%arg0)[%1]
      %10 = flow.dispatch.tensor.load %6, offsets = [%arg0], sizes = [%9], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<?xi32>>{%2} -> tensor<?xi32>
      %11 = flow.dispatch.tensor.load %5, offsets = [%arg0], sizes = [%9], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<?xi32>>{%2} -> tensor<?xi32>
      %12 = flow.dispatch.tensor.load %3, offsets = [0, %arg0], sizes = [%0, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1} -> tensor<?x?xi32>
      %13 = flow.dispatch.tensor.load %4, offsets = [0, %arg0], sizes = [%0, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1} -> tensor<?x?xi32>
      %14 = linalg.fill {__internal_linalg_transform__ = "workgroup", lowering_config = {tileSizes = [[128]]}} ins(%c-2147483648_i32 : i32) outs(%11 : tensor<?xi32>) -> tensor<?xi32>
      %15 = linalg.fill {__internal_linalg_transform__ = "workgroup", lowering_config = {tileSizes = [[128]]}} ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
      %16:2 = linalg.generic {indexing_maps = [#map2, #map2, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%12, %13 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%14, %15 : tensor<?xi32>, tensor<?xi32>) attrs =  {__internal_linalg_transform__ = "workgroup", lowering_config = {tileSizes = [[128]]}} {
      ^bb0(%arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32):
        %17 = arith.cmpi sge, %arg1, %arg3 : i32
        %18 = arith.select %17, %arg1, %arg3 : i32
        %19 = arith.cmpi eq, %arg1, %arg3 : i32
        %20 = arith.cmpi slt, %arg2, %arg4 : i32
        %21 = arith.select %20, %arg2, %arg4 : i32
        %22 = arith.select %17, %arg2, %arg4 : i32
        %23 = arith.select %19, %21, %22 : i32
        linalg.yield %18, %23 : i32, i32
      } -> (tensor<?xi32>, tensor<?xi32>)
      flow.dispatch.tensor.store %16#0, %5, offsets = [%arg0], sizes = [%9], strides = [1] : tensor<?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?xi32>>{%2}
      flow.dispatch.tensor.store %16#1, %6, offsets = [%arg0], sizes = [%9], strides = [1] : tensor<?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?xi32>>{%2}
    }
    return
  }
}
// CHECK-LABEL: func.func @multi_result_reduce
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//   CHECK-DAG:   %[[RET1:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
//       CHECK:   scf.for
//   CHECK-DAG:     %[[ARG0_SV:.+]] = memref.subview %[[ARG0]]
//   CHECK-DAG:     %[[ARG1_SV:.+]] = memref.subview %[[ARG1]]
//   CHECK-DAG:     %[[RET0_SV:.+]] = memref.subview %[[RET0]]
//   CHECK-DAG:     %[[RET1_SV:.+]] = memref.subview %[[RET1]]
//   CHECK-DAG:     linalg.fill
//  CHECK-SAME:         outs(%[[RET0_SV]] :
//   CHECK-DAG:     linalg.fill
//  CHECK-SAME:         outs(%[[RET1_SV]] :
//       CHECK:     linalg.generic
//  CHECK-SAME:       ins(%[[ARG0_SV]], %[[ARG1_SV]]
//  CHECK-SAME:       outs(%[[RET0_SV]], %[[RET1_SV]]

// -----

#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0) -> (-d0 + 250, 64)>
#map2 = affine_map<(d0) -> (-d0 + 370, 64)>
#map3 = affine_map<(d0) -> (-d0 + 250, 32)>
#map4 = affine_map<(d0) -> (-d0 + 144, 24)>
#map5 = affine_map<(d0) -> (-d0 + 370, 32)>
#map6 = affine_map<(d0, d1) -> (32, d0 - d1)>
module {
  func.func @l1_tiled_matmul_no_fill_readwrite() {
    %c32 = arith.constant 32 : index
    %c24 = arith.constant 24 : index
    %c144 = arith.constant 144 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c250 = arith.constant 250 : index
    %c370 = arith.constant 370 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<250x144xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<144x370xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<250x370xf32>>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %3 = affine.apply #map0()[%workgroup_id_y]
    %4 = affine.apply #map0()[%workgroup_count_y]
    scf.for %arg0 = %3 to %c250 step %4 {
      %5 = affine.apply #map0()[%workgroup_id_x]
      %6 = affine.apply #map0()[%workgroup_count_x]
      scf.for %arg1 = %5 to %c370 step %6 {
        %7 = affine.min #map1(%arg0)
        %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 144], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<250x144xf32>> -> tensor<?x144xf32>
        %9 = affine.min #map2(%arg1)
        %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [144, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<144x370xf32>> -> tensor<144x?xf32>
        %11 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%7, %9], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<250x370xf32>> -> tensor<?x?xf32>
        %12 = scf.for %arg2 = %c0 to %c250 step %c32 iter_args(%arg3 = %11) -> (tensor<?x?xf32>) {
          %13 = scf.for %arg4 = %c0 to %c370 step %c32 iter_args(%arg5 = %arg3) -> (tensor<?x?xf32>) {
            %14 = scf.for %arg6 = %c0 to %c144 step %c24 iter_args(%arg7 = %arg5) -> (tensor<?x?xf32>) {
              %15 = affine.min #map3(%arg2)
              %16 = affine.min #map4(%arg6)
              %17 = tensor.extract_slice %8[%arg2, %arg6] [%15, %16] [1, 1] : tensor<?x144xf32> to tensor<?x?xf32>
              %18 = affine.min #map5(%arg4)
              %19 = tensor.extract_slice %10[%arg6, %arg4] [%16, %18] [1, 1] : tensor<144x?xf32> to tensor<?x?xf32>
              %20 = tensor.dim %arg7, %c0 : tensor<?x?xf32>
              %21 = affine.min #map6(%20, %arg2)
              %22 = tensor.dim %arg7, %c1 : tensor<?x?xf32>
              %23 = affine.min #map6(%22, %arg4)
              %24 = tensor.extract_slice %arg7[%arg2, %arg4] [%21, %23] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %25 = linalg.matmul {__internal_linalg_transform__ = "workgroup_l1_tile", lowering_config = {nativeVectorSize = [4, 4, 4], tileSizes = [[64, 64], [32, 32, 24], [4, 4, 4]]}} ins(%17, %19 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%24 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %26 = tensor.insert_slice %25 into %arg7[%arg2, %arg4] [%21, %23] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
              scf.yield %26 : tensor<?x?xf32>
            }
            scf.yield %14 : tensor<?x?xf32>
          }
          scf.yield %13 : tensor<?x?xf32>
        }
        flow.dispatch.tensor.store %12, %2, offsets = [%arg0, %arg1], sizes = [%7, %9], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<250x370xf32>>
      }
    }
    return
  }
}
// CHECK-LABEL: l1_tiled_matmul_no_fill_readwrite
//    CHECK-DAG: %[[M:.+]] = arith.constant 250 : index
//    CHECK-DAG: %[[N:.+]] = arith.constant 370 : index
//    CHECK-DAG: %[[K:.+]] = arith.constant 144 : index
//    CHECK-DAG: %[[L1_MN_SIZE:.+]] = arith.constant 32 : index
//    CHECK-DAG: %[[L1_K_SIZE:.+]] = arith.constant 24 : index
//    CHECK-DAG: %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<250x144xf32, #hal.descriptor_type<storage_buffer>>
//    CHECK-DAG: %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<144x370xf32, #hal.descriptor_type<storage_buffer>>
//    CHECK-DAG: %[[DST:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<250x370xf32, #hal.descriptor_type<storage_buffer>>
//        CHECK: scf.for %[[WORKGROUP_I:.+]] = %{{.*}} to %[[M]] step %{{.*}} {
//        CHECK:    scf.for %[[WORKGROUP_J:.+]] = %{{.*}} to %[[N]] step %{{.*}} {
//    CHECK-DAG:        %[[WORKGROUP_I_SIZE:.+]] = affine.min #{{.*}}(%[[WORKGROUP_I]])
//    CHECK-DAG:        %[[LHS_WORKGROUP_TILE:.+]] = memref.subview %[[LHS]][%[[WORKGROUP_I]], 0] [%[[WORKGROUP_I_SIZE]], 144] [1, 1] : memref<250x144xf32{{.+}}> to memref<?x144xf32
//    CHECK-DAG:        %[[WORKGROUP_J_SIZE:.+]] = affine.min #{{.*}}(%[[WORKGROUP_J]])
//    CHECK-DAG:        %[[RHS_WORKGROUP_TILE:.+]] = memref.subview %[[RHS]][0, %[[WORKGROUP_J]]] [144, %[[WORKGROUP_J_SIZE]]] [1, 1] : memref<144x370xf32{{.+}}> to memref<144x?xf32
//    CHECK-DAG:            %[[DST_WORKGROUP_TILE:.+]] = memref.subview %[[DST]][%[[WORKGROUP_I]], %[[WORKGROUP_J]]] [%[[WORKGROUP_I_SIZE]], %[[WORKGROUP_J_SIZE]]]
//        CHECK:            scf.for %[[L1_I:.+]] = %{{.*}} to %[[M]] step %[[L1_MN_SIZE]] {
//        CHECK:              scf.for %[[L1_J:.+]] = %{{.*}} to %[[N]] step %[[L1_MN_SIZE]] {
//        CHECK:                scf.for %[[L1_K:.+]] = %{{.*}} to %[[K]] step %[[L1_K_SIZE]] {
//    CHECK-DAG:                    %[[LHS_L1_TILE:.+]] = memref.subview %[[LHS_WORKGROUP_TILE]][%[[L1_I]], %[[L1_K]]]
//    CHECK-DAG:                    %[[RHS_L1_TILE:.+]] = memref.subview %[[RHS_WORKGROUP_TILE]][%[[L1_K]], %[[L1_J]]]
//    CHECK-DAG:                    %[[L1_I_SIZE:.+]] = affine.min #{{.*}}(%[[WORKGROUP_I_SIZE]], %[[L1_I]])
//    CHECK-DAG:                    %[[L1_J_SIZE:.+]] = affine.min #{{.*}}(%[[WORKGROUP_J_SIZE]], %[[L1_J]])
//    CHECK-DAG:                    %[[DST_L1_TILE:.+]] = memref.subview %[[DST_WORKGROUP_TILE]][%[[L1_I]], %[[L1_J]]]
//        CHECK:                    linalg.matmul
//   CHECK-SAME:                    ins(%[[LHS_L1_TILE]], %[[RHS_L1_TILE]]
//   CHECK-SAME:                    outs(%[[DST_L1_TILE]]

// -----

#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0) -> (-d0 + 250, 64)>
#map2 = affine_map<(d0) -> (-d0 + 370, 64)>
#map3 = affine_map<(d0) -> (-d0 + 250, 32)>
#map4 = affine_map<(d0) -> (-d0 + 144, 24)>
#map5 = affine_map<(d0) -> (-d0 + 370, 32)>
#map6 = affine_map<(d0, d1) -> (32, d0 - d1)>
module {
  func.func @l1_tiled_matmul() {
    %cst = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %c24 = arith.constant 24 : index
    %c144 = arith.constant 144 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c250 = arith.constant 250 : index
    %c370 = arith.constant 370 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<250x144xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<144x370xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<250x370xf32>>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %3 = affine.apply #map0()[%workgroup_id_y]
    %4 = affine.apply #map0()[%workgroup_count_y]
    scf.for %arg0 = %3 to %c250 step %4 {
      %5 = affine.apply #map0()[%workgroup_id_x]
      %6 = affine.apply #map0()[%workgroup_count_x]
      scf.for %arg1 = %5 to %c370 step %6 {
        %7 = affine.min #map1(%arg0)
        %8 = affine.min #map2(%arg1)
        %9 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%7, %8], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<250x370xf32>> -> tensor<?x?xf32>
        %10 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 144], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<250x144xf32>> -> tensor<?x144xf32>
        %11 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [144, %8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<144x370xf32>> -> tensor<144x?xf32>
        %12 = linalg.fill {__internal_linalg_transform__ = "workgroup", lowering_config = {tileSizes = [[64, 64]]}} ins(%cst : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %13 = scf.for %arg2 = %c0 to %c250 step %c32 iter_args(%arg3 = %12) -> (tensor<?x?xf32>) {
          %14 = scf.for %arg4 = %c0 to %c370 step %c32 iter_args(%arg5 = %arg3) -> (tensor<?x?xf32>) {
            %15 = scf.for %arg6 = %c0 to %c144 step %c24 iter_args(%arg7 = %arg5) -> (tensor<?x?xf32>) {
              %16 = affine.min #map3(%arg2)
              %17 = affine.min #map4(%arg6)
              %18 = tensor.extract_slice %10[%arg2, %arg6] [%16, %17] [1, 1] : tensor<?x144xf32> to tensor<?x?xf32>
              %19 = affine.min #map5(%arg4)
              %20 = tensor.extract_slice %11[%arg6, %arg4] [%17, %19] [1, 1] : tensor<144x?xf32> to tensor<?x?xf32>
              %21 = tensor.dim %arg7, %c0 : tensor<?x?xf32>
              %22 = affine.min #map6(%21, %arg2)
              %23 = tensor.dim %arg7, %c1 : tensor<?x?xf32>
              %24 = affine.min #map6(%23, %arg4)
              %25 = tensor.extract_slice %arg7[%arg2, %arg4] [%22, %24] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %26 = linalg.matmul {__internal_linalg_transform__ = "workgroup_l1_tile", lowering_config = {nativeVectorSize = [4, 4, 4], tileSizes = [[64, 64], [32, 32, 24], [4, 4, 4]]}} ins(%18, %20 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%25 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %27 = tensor.insert_slice %26 into %arg7[%arg2, %arg4] [%22, %24] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
              scf.yield %27 : tensor<?x?xf32>
            }
            scf.yield %15 : tensor<?x?xf32>
          }
          scf.yield %14 : tensor<?x?xf32>
        }
        flow.dispatch.tensor.store %13, %2, offsets = [%arg0, %arg1], sizes = [%7, %8], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<250x370xf32>>
      }
    }
    return
  }
}
// CHECK-LABEL: l1_tiled_matmul
//    CHECK-DAG: %[[M:.+]] = arith.constant 250 : index
//    CHECK-DAG: %[[N:.+]] = arith.constant 370 : index
//    CHECK-DAG: %[[K:.+]] = arith.constant 144 : index
//    CHECK-DAG: %[[L1_MN_SIZE:.+]] = arith.constant 32 : index
//    CHECK-DAG: %[[L1_K_SIZE:.+]] = arith.constant 24 : index
//    CHECK-DAG: %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<250x144xf32, #hal.descriptor_type<storage_buffer>>
//    CHECK-DAG: %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<144x370xf32, #hal.descriptor_type<storage_buffer>>
//    CHECK-DAG: %[[DST:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<250x370xf32, #hal.descriptor_type<storage_buffer>>
//        CHECK: scf.for %[[WORKGROUP_I:.+]] = %{{.*}} to %[[M]] step %{{.*}} {
//        CHECK:    scf.for %[[WORKGROUP_J:.+]] = %{{.*}} to %[[N]] step %{{.*}} {
//    CHECK-DAG:        %[[WORKGROUP_I_SIZE:.+]] = affine.min #{{.*}}(%[[WORKGROUP_I]])
//    CHECK-DAG:        %[[LHS_WORKGROUP_TILE:.+]] = memref.subview %[[LHS]][%[[WORKGROUP_I]], 0] [%[[WORKGROUP_I_SIZE]], 144] [1, 1] : memref<250x144xf32{{.+}}> to memref<?x144xf32
//    CHECK-DAG:        %[[WORKGROUP_J_SIZE:.+]] = affine.min #{{.*}}(%[[WORKGROUP_J]])
//    CHECK-DAG:        %[[RHS_WORKGROUP_TILE:.+]] = memref.subview %[[RHS]][0, %[[WORKGROUP_J]]] [144, %[[WORKGROUP_J_SIZE]]] [1, 1] : memref<144x370xf32{{.+}}> to memref<144x?xf32
//    CHECK-DAG:            %[[DST_WORKGROUP_TILE:.+]] = memref.subview %[[DST]][%[[WORKGROUP_I]], %[[WORKGROUP_J]]] [%[[WORKGROUP_I_SIZE]], %[[WORKGROUP_J_SIZE]]]
//        CHECK:            scf.for %[[L1_I:.+]] = %{{.*}} to %[[M]] step %[[L1_MN_SIZE]] {
//        CHECK:              scf.for %[[L1_J:.+]] = %{{.*}} to %[[N]] step %[[L1_MN_SIZE]] {
//        CHECK:                scf.for %[[L1_K:.+]] = %{{.*}} to %[[K]] step %[[L1_K_SIZE]] {
//    CHECK-DAG:                    %[[LHS_L1_TILE:.+]] = memref.subview %[[LHS_WORKGROUP_TILE]][%[[L1_I]], %[[L1_K]]]
//    CHECK-DAG:                    %[[RHS_L1_TILE:.+]] = memref.subview %[[RHS_WORKGROUP_TILE]][%[[L1_K]], %[[L1_J]]]
//    CHECK-DAG:                    %[[L1_I_SIZE:.+]] = affine.min #{{.*}}(%[[WORKGROUP_I_SIZE]], %[[L1_I]])
//    CHECK-DAG:                    %[[L1_J_SIZE:.+]] = affine.min #{{.*}}(%[[WORKGROUP_J_SIZE]], %[[L1_J]])
//    CHECK-DAG:                    %[[DST_L1_TILE:.+]] = memref.subview %[[DST_WORKGROUP_TILE]][%[[L1_I]], %[[L1_J]]]
//        CHECK:                    linalg.matmul
//   CHECK-SAME:                    ins(%[[LHS_L1_TILE]], %[[RHS_L1_TILE]]
//   CHECK-SAME:                    outs(%[[DST_L1_TILE]]

// -----

#map0 = affine_map<()[s0, s1] -> (s1 * s0)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
#map2 = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  func.func @tensor_insert_slice() {
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.constant.load[3] : index
    %4 = hal.interface.constant.load[4] : index
    %5 = hal.interface.constant.load[5] : index
    %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%2, %3}
    %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%4, %5}
    %workgroup_size_x = hal.interface.workgroup.size[0] : index
    %workgroup_size_y = hal.interface.workgroup.size[1] : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %8 = affine.apply #map0()[%workgroup_size_y, %workgroup_id_y]
    %9 = affine.apply #map0()[%workgroup_size_y, %workgroup_count_y]
    scf.for %arg0 = %8 to %2 step %9 {
      %10 = affine.min #map1(%arg0)[%workgroup_size_y, %2]
      %11 = affine.apply #map0()[%workgroup_size_x, %workgroup_id_x]
      %12 = affine.apply #map0()[%workgroup_size_x, %workgroup_count_x]
      scf.for %arg1 = %11 to %3 step %12 {
        %13 = affine.min #map1(%arg1)[%workgroup_size_x, %3]
        %14 = flow.dispatch.tensor.load %6, offsets = [%arg0, %arg1], sizes = [%10, %13], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%2, %3} -> tensor<?x?xi32>
        %15 = affine.apply #map2(%arg0)[%0]
        %16 = affine.apply #map2(%arg1)[%1]
        flow.dispatch.tensor.store %14, %7, offsets = [%15, %16], sizes = [%10, %13], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%4, %5}
      }
    }
    return
  }
}
//       CHECK: #[[MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//       CHECK: func.func @tensor_insert_slice()
//   CHECK-DAG:   %[[SRC:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?xi32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[DST:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x?xi32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[OFFSET_Y:.+]] = hal.interface.constant.load[0]
//   CHECK-DAG:   %[[OFFSET_X:.+]] = hal.interface.constant.load[1]
//       CHECK:   scf.for %[[IV0:.+]] =
//       CHECK:     scf.for %[[IV1:.+]] =
//   CHECK-DAG:       %[[SRC_VIEW:.+]] = memref.subview %[[SRC]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:       %[[DST_IDX_Y:.+]] = affine.apply #[[MAP]](%[[IV0]])[%[[OFFSET_Y]]]
//   CHECK-DAG:       %[[DST_IDX_X:.+]] = affine.apply #[[MAP]](%[[IV1]])[%[[OFFSET_X]]]
//       CHECK:       %[[DST_VIEW:.+]] = memref.subview %[[DST]][%[[DST_IDX_Y]], %[[DST_IDX_X]]]
//       CHECK:       linalg.generic {{.*}} ins(%[[SRC_VIEW]] {{.*}} outs(%[[DST_VIEW]]

// -----

#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
#map2 = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  func.func @dynamic_update_slice() {
    %c0_i32 = arith.constant 0 : i32
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xi32>>{%0}
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<i32>>
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%1, %0}
    %5 = flow.dispatch.tensor.load %3, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<i32>> -> tensor<i32>
    %6 = tensor.extract %5[] : tensor<i32>
    %7 = arith.cmpi slt, %6, %c0_i32 : i32
    %8 = arith.select %7, %6, %c0_i32 : i32
    %9 = arith.cmpi sgt, %8, %c0_i32 : i32
    %10 = arith.select %9, %8, %c0_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %12 = affine.apply #map0()[%workgroup_id_x]
    %13 = affine.apply #map0()[%workgroup_count_x]
    scf.for %arg0 = %12 to %0 step %13 {
      %14 = affine.min #map1(%arg0)[%0]
      %15 = flow.dispatch.tensor.load %2, offsets = [%arg0], sizes = [%14], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xi32>>{%0} -> tensor<?xi32>
      %16 = affine.apply #map2(%arg0)[%11]
      flow.dispatch.tensor.store %15, %4, offsets = [0, %16], sizes = [1, %14], strides = [1, 1] : tensor<?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%1, %0}
    }
    return
  }
}
// CHECK-LABEL: func.func @dynamic_update_slice()
//   CHECK-DAG:   %[[SRC:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?xi32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[DST:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<?x?xi32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[OFFSET_Y:.+]] = hal.interface.constant.load[0]
//   CHECK-DAG:   %[[OFFSET_X:.+]] = hal.interface.constant.load[1]
//       CHECK:   scf.for %[[IV0:.+]] =
//       CHECK:     %[[SRC_VIEW:.+]] = memref.subview %[[SRC]][%[[IV0]]]
//  CHECK-SAME:         : memref<?xi32{{.+}}> to memref<?xi32, strided<[1], offset: ?>{{.+}}>
//       CHECK:     %[[DST_VIEW:.+]] = memref.subview %[[DST]][0, %{{[a-zA-Z0-9]+}}] [1, %{{[a-zA-Z0-9]+}}]
//  CHECK-SAME:         : memref<?x?xi32{{.+}}> to memref<?xi32, strided<[1], offset: ?>{{.+}}>
//       CHECK:     linalg.generic {{.*}} ins(%[[SRC_VIEW]] {{.*}} outs(%[[DST_VIEW]]

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
#map2 = affine_map<(d0, d1) -> (4, d0 - d1)>
#map3 = affine_map<(d0, d1) -> (d0 + d1)>
#map4 = affine_map<(d0, d1) -> ()>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @multi_level_tile_fuse() {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<f32>>
    %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    %7 = flow.dispatch.tensor.load %5, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_size_x = hal.interface.workgroup.size[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %workgroup_size_y = hal.interface.workgroup.size[1] : index
    %8 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
    %9 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
    scf.for %arg0 = %8 to %0 step %9 {
      %10 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
      %11 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
      scf.for %arg1 = %10 to %1 step %11 {
        %12 = affine.min #map1(%arg0)[%workgroup_size_y, %0]
        %13 = affine.min #map1(%arg1)[%workgroup_size_x, %1]
        %14 = flow.dispatch.tensor.load %6, offsets = [%arg0, %arg1], sizes = [%12, %13], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %15 = flow.dispatch.tensor.load %3, offsets = [%arg0, 0], sizes = [%12, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
        %16 = flow.dispatch.tensor.load %4, offsets = [0, %arg1], sizes = [%2, %13], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
        %17 = linalg.fill {__internal_linalg_transform__ = "workgroup"} ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %18 = scf.for %arg2 = %c0 to %12 step %c4 iter_args(%arg3 = %17) -> (tensor<?x?xf32>) {
          %20 = scf.for %arg4 = %c0 to %13 step %c4 iter_args(%arg5 = %arg3) -> (tensor<?x?xf32>) {
            %21 = affine.min #map2(%12, %arg2)
            %22 = affine.min #map2(%13, %arg4)
            %23 = tensor.extract_slice %arg5[%arg2, %arg4] [%21, %22] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %24 = scf.for %arg6 = %c0 to %21 step %c4 iter_args(%arg7 = %23) -> (tensor<?x?xf32>) {
              %26 = scf.for %arg8 = %c0 to %22 step %c4 iter_args(%arg9 = %arg7) -> (tensor<?x?xf32>) {
                %27 = affine.min #map2(%21, %arg6)
                %28 = affine.apply #map3(%arg6, %arg2)
                %29 = tensor.extract_slice %15[%28, 0] [%27, %2] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
                %30 = affine.min #map2(%22, %arg8)
                %31 = affine.apply #map3(%arg8, %arg4)
                %32 = tensor.extract_slice %16[0, %31] [%2, %30] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
                %33 = tensor.extract_slice %arg9[%arg6, %arg8] [%27, %30] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
                %34 = linalg.matmul {__internal_linalg_transform__ = "vectorize", lowering_config = {nativeVectorSize = [4, 4, 4], tileSizes = [[], [4, 4, 4], [4, 4, 4]]}} ins(%29, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%33 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %35 = tensor.insert_slice %34 into %arg9[%arg6, %arg8] [%27, %30] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
                scf.yield %35 : tensor<?x?xf32>
              }
              scf.yield %26 : tensor<?x?xf32>
            }
            %25 = tensor.insert_slice %24 into %arg5[%arg2, %arg4] [%21, %22] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
            scf.yield %25 : tensor<?x?xf32>
          }
          scf.yield %20 : tensor<?x?xf32>
        }
        %19 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<f32>) outs(%18 : tensor<?x?xf32>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
        ^bb0(%arg2: f32, %arg3: f32):
          %20 = arith.addf %arg2, %arg3 : f32
          linalg.yield %20 : f32
        } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %19, %6, offsets = [%arg0, %arg1], sizes = [%12, %13], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @multi_level_tile_fuse()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//   CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%[[M]], %[[K]]}
//   CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%[[K]], %[[N]]}
//   CHECK-DAG:   %[[SCALAR:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<f32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%[[M]], %[[N]]}
//       CHECK:   scf.for
//       CHECK:     scf.for
//   CHECK-DAG:       %[[LHS_SUBVIEW1:.+]] = memref.subview %[[LHS]]
//   CHECK-DAG:       %[[RHS_SUBVIEW1:.+]] = memref.subview %[[RHS]]
//   CHECK-DAG:       %[[OUT_SUBVIEW1:.+]] = memref.subview %[[OUT]]
//       CHECK:       linalg.fill
//  CHECK-SAME:           outs(%[[OUT_SUBVIEW1]] :
//       CHECK:       scf.for
//       CHECK:         scf.for
//       CHECK:           %[[OUT_SUBVIEW2:.+]] = memref.subview %[[OUT_SUBVIEW1]]
//       CHECK:           scf.for
//       CHECK:             scf.for
//   CHECK-DAG:               %[[LHS_SUBVIEW2:.+]] = memref.subview %[[LHS_SUBVIEW1]]
//   CHECK-DAG:               %[[RHS_SUBVIEW2:.+]] = memref.subview %[[RHS_SUBVIEW1]]
//   CHECK-DAG:               %[[OUT_SUBVIEW3:.+]] = memref.subview %[[OUT_SUBVIEW2]]
//       CHECK:               linalg.matmul
//  CHECK-SAME:                   ins(%[[LHS_SUBVIEW2]], %[[RHS_SUBVIEW2]] :
//  CHECK-SAME:                   outs(%[[OUT_SUBVIEW3]] :
//       CHECK:       linalg.generic
//  CHECK-SAME:           ins(%[[SCALAR]] :
//  CHECK-SAME:           outs(%[[OUT_SUBVIEW1]] :

// -----

#map0 = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<(d0) -> (-d0 + 2, 4)>
#map2 = affine_map<(d0) -> (-d0 + 1, 4)>
#map3 = affine_map<(d0, d1) -> ()>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @operand_fusion() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.constant.load[2] : index
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2}
    %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1}
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<f32>>
    %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    %7 = flow.dispatch.tensor.load %5, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %8 = affine.apply #map0()[%workgroup_id_y]
    %9 = affine.apply #map0()[%workgroup_count_y]
    scf.for %arg0 = %8 to %c2 step %9 {
      %10 = affine.apply #map0()[%workgroup_id_x]
      %11 = affine.apply #map0()[%workgroup_count_x]
      scf.for %arg1 = %10 to %c1 step %11 {
        %12 = affine.min #map1(%arg0)
        %13 = affine.min #map2(%arg1)
        %14 = flow.dispatch.tensor.load %6, offsets = [%arg0, %arg1], sizes = [%12, %13], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %15 = flow.dispatch.tensor.load %3, offsets = [%arg0, 0], sizes = [%12, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %2} -> tensor<?x?xf32>
        %16 = flow.dispatch.tensor.load %4, offsets = [0, %arg1], sizes = [3, %13], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %1} -> tensor<?x?xf32>
        %17 = linalg.fill {__internal_linalg_transform__ = "workgroup"} ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %18 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%15, %16 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %19 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<f32>) outs(%18 : tensor<?x?xf32>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
        ^bb0(%arg2: f32, %arg3: f32):
          %20 = arith.addf %arg2, %arg3 : f32
          linalg.yield %20 : f32
        } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %19, %6, offsets = [%arg0, %arg1], sizes = [%12, %13], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @operand_fusion()
//   CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load[0]
//   CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load[1]
//   CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load[2]
//   CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%[[M]], %[[K]]}
//   CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%[[K]], %[[N]]}
//   CHECK-DAG:   %[[SCALAR:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<f32, #hal.descriptor_type<storage_buffer>>
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%[[M]], %[[N]]}
//       CHECK:   scf.for
//       CHECK:     scf.for
//   CHECK-DAG:       %[[LHS_SUBVIEW1:.+]] = memref.subview %[[LHS]]
//   CHECK-DAG:       %[[RHS_SUBVIEW1:.+]] = memref.subview %[[RHS]]
//   CHECK-DAG:       %[[OUT_SUBVIEW1:.+]] = memref.subview %[[OUT]]
//       CHECK:       linalg.fill
//  CHECK-SAME:           outs(%[[OUT_SUBVIEW1]] :
//       CHECK:       linalg.matmul
//  CHECK-SAME:           ins(%[[LHS_SUBVIEW1]], %[[RHS_SUBVIEW1]] :
//  CHECK-SAME:           outs(%[[OUT_SUBVIEW1]] :
//       CHECK:       linalg.generic
//  CHECK-SAME:           ins(%[[SCALAR]] :
//  CHECK-SAME:           outs(%[[OUT_SUBVIEW1]] :

// -----

// This test is a repro from a failure. No checking needed.
#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func @forward_dispatch_3() {
    %c384 = arith.constant 384 : index
    %c512_i64 = arith.constant 512 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c592896 = arith.constant 592896 : index
    %c47481856 = arith.constant 47481856 : index
    %c64 = arith.constant 64 : index
    %0 = hal.interface.constant.load[0] : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c592896) : !flow.dispatch.tensor<readonly:tensor<1x512xi64>>
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c47481856) : !flow.dispatch.tensor<readonly:tensor<512x384xf32>>
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x384xf32>>{%1}
    %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x512xi64>> -> tensor<1x512xi64>
    %6 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [512, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x384xf32>> -> tensor<512x384xf32>
    %7 = tensor.extract_slice %5[0, 0] [1, %1] [1, 1] : tensor<1x512xi64> to tensor<?xi64>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %8 = affine.apply #map0()[%workgroup_id_y]
    %9 = affine.apply #map0()[%workgroup_count_y]
    scf.for %arg0 = %8 to %1 step %9 {
      %10 = affine.apply #map0()[%workgroup_id_x]
      %11 = affine.apply #map0()[%workgroup_count_x]
      scf.for %arg1 = %10 to %c384 step %11 {
        %12 = affine.min #map1(%arg0)[%1]
        %13 = flow.dispatch.tensor.load %4, offsets = [%arg0, %arg1], sizes = [%12, 64], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<?x384xf32>>{%1} -> tensor<?x64xf32>
        %14 = tensor.extract_slice %7[%arg0] [%12] [1] : tensor<?xi64> to tensor<?xi64>
        %15 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<?xi64>) outs(%13 : tensor<?x64xf32>) {
        ^bb0(%arg2: i64, %arg3: f32):
          %16 = arith.index_cast %arg2 : i64 to index
          %17 = linalg.index 1 : index
          %18 = affine.apply #map4(%17, %arg1)
          %19 = arith.cmpi slt, %arg2, %c512_i64 : i64
          cf.assert %19, "index must be smaller than dim size"
          %20 = arith.cmpi sge, %arg2, %c0_i64 : i64
          cf.assert %20, "index must be larger or equal to 0"
          %21 = tensor.extract %6[%16, %18] : tensor<512x384xf32>
          linalg.yield %21 : f32
        } -> tensor<?x64xf32>
        flow.dispatch.tensor.store %15, %4, offsets = [%arg0, %arg1], sizes = [%12, %c64], strides = [1, 1] : tensor<?x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x384xf32>>{%1}
      }
    }
    return
  }
}
// CHECK: func.func @forward_dispatch_3()

// -----

#map0 = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<()[s0] -> (s0 * 2)>
#map2 = affine_map<(d0) -> (-d0 + 6, 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @dot_general_nontrivial_batching_mutliple_parallel_dimension() {
    %cst = arith.constant dense<0.000000e+00> : vector<1x4x2xf32>
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c2 = arith.constant 2 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x6x1xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c64) : !flow.dispatch.tensor<readonly:tensor<2x1x2xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x6x2xf32>>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %workgroup_count_z = hal.interface.workgroup.count[2] : index
    %3 = affine.apply #map0()[%workgroup_id_y]
    %4 = affine.apply #map0()[%workgroup_count_y]
    %5 = affine.apply #map1()[%workgroup_id_x]
    %6 = affine.apply #map1()[%workgroup_count_x]
    scf.for %arg0 = %workgroup_id_z to %c2 step %workgroup_count_z {
      scf.for %arg1 = %3 to %c6 step %4 {
        %7 = affine.min #map2(%arg1)
        %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1, 0], sizes = [1, %7, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x6x1xf32>> -> tensor<1x?x1xf32>
        %9 = tensor.extract_slice %8[0, 0, 0] [1, %7, 1] [1, 1, 1] : tensor<1x?x1xf32> to tensor<1x?x1xf32>
        %10 = vector.transfer_read %9[%c0, %c0, %c0], %cst_0 {in_bounds = [true, false, true]} : tensor<1x?x1xf32>, vector<1x4x1xf32>
        scf.for %arg2 = %5 to %c2 step %6 {
          %11 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1, %arg2], sizes = [1, %7, 2], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<2x6x2xf32>> -> tensor<1x?x2xf32>
          %12 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0, %arg2], sizes = [1, 1, 2], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1x2xf32>> -> tensor<1x1x2xf32>
          %13 = tensor.extract_slice %11[0, 0, 0] [1, %7, 2] [1, 1, 1] : tensor<1x?x2xf32> to tensor<1x?x2xf32>
          %14 = vector.transfer_write %cst, %13[%c0, %c0, %c0] {in_bounds = [true, false, true]} : vector<1x4x2xf32>, tensor<1x?x2xf32>
          %15 = tensor.extract_slice %14[0, 0, 0] [1, %7, 2] [1, 1, 1] : tensor<1x?x2xf32> to tensor<1x?x2xf32>
          %16 = vector.transfer_read %12[%c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true]} : tensor<1x1x2xf32>, vector<1x1x2xf32>
          %17 = vector.transfer_read %15[%c0, %c0, %c0], %cst_0 {in_bounds = [true, false, true]} : tensor<1x?x2xf32>, vector<1x4x2xf32>
          %18 = vector.contract {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %10, %16, %17 : vector<1x4x1xf32>, vector<1x1x2xf32> into vector<1x4x2xf32>
          %19 = vector.transfer_write %18, %15[%c0, %c0, %c0] {in_bounds = [true, false, true]} : vector<1x4x2xf32>, tensor<1x?x2xf32>
          %20 = tensor.insert_slice %19 into %14[0, 0, 0] [1, %7, 2] [1, 1, 1] : tensor<1x?x2xf32> into tensor<1x?x2xf32>
          %21 = tensor.insert_slice %20 into %11[0, 0, 0] [1, %7, 2] [1, 1, 1] : tensor<1x?x2xf32> into tensor<1x?x2xf32>
          flow.dispatch.tensor.store %21, %2, offsets = [%arg0, %arg1, %arg2], sizes = [%c1, %7, %c2], strides = [1, 1, 1] : tensor<1x?x2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x6x2xf32>>
        }
      }
    }
    return
  }
}
// CHECK-LABEL: func.func @dot_general_nontrivial_batching_mutliple_parallel_dimension()
//   CHECK-NOT:   memref.alloc

// -----

module {
  func.func @no_op_subview() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.constant.load[1] : index
    %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
    %5 = tensor.extract_slice %4[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
    return
  }
}
// CHECK-LABEL: func.func @no_op_subview()
//   CHECK-DAG:   %[[SRC:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[DEST:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[SRC_DUP:.+]] = memref.subview %[[SRC]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%[[SRC_DUP]] :
//  CHECK-SAME:       outs(%[[DEST]] :

// -----

module {
  func.func @rank_reducing_no_op_subview() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load[0] : index
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x?xf32>>{%0}
    %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
    %3 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, %0], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x?xf32>>{%0} -> tensor<1x?xf32>
    %4 = tensor.extract_slice %3[0, 0] [1, %0] [1, 1] : tensor<1x?xf32> to tensor<?xf32>
    flow.dispatch.tensor.store %4, %2, offsets = [0], sizes = [%0], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
    return
  }
}
// CHECK-LABEL: func.func @rank_reducing_no_op_subview()
//   CHECK-DAG:   %[[SRC:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[DEST:.+]] = hal.interface.binding.subspan set(0) binding(1)
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[SRC]][0, 0] [1, %{{.+}}]
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%[[SUBVIEW]] :
//  CHECK-SAME:       outs(%[[DEST]] :

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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<6xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<f32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<6xf32>>
  %3 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [6], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<6xf32>> -> tensor<6xf32>
  %4 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [6], strides = [1] : !flow.dispatch.tensor<readonly:tensor<6xf32>> -> tensor<6xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:tensor<f32>> -> tensor<f32>
  %6:2 = iree_linalg_ext.scan dimension(0) inclusive(true) ins(%4 : tensor<6xf32>) outs(%3, %5 : tensor<6xf32>, tensor<f32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %7 = arith.addf %arg0, %arg1 : f32
    iree_linalg_ext.yield %7 : f32
  } -> tensor<6xf32>, tensor<f32>
  flow.dispatch.tensor.store %6#0, %2, offsets = [0], sizes = [6], strides = [1] : tensor<6xf32> -> !flow.dispatch.tensor<writeonly:tensor<6xf32>>
  flow.dispatch.tensor.store %6#1, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<readwrite:tensor<f32>>
  return
}
// CHECK-LABEL: func.func @scan_1d_dim0_inclusive_sum
// CHECK-NOT:    memref.alloca
// CHECK:        iree_linalg_ext.scan
// CHECK-SAME:     ins(%{{[a-z0-9]+}} : memref<6xf32
// CHECK-SAME:     outs(%{{.*}}, %{{.*}} : memref<6xf32{{.+}}>, memref<f32{{.+}}>)

// -----

func.func @sort1D() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<4xi32>>
  %1 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<4xi32>> -> tensor<4xi32>
  %2 = iree_linalg_ext.sort dimension(0) outs(%1 : tensor<4xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %3 = arith.cmpi slt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %3 : i1
  } -> tensor<4xi32>
  flow.dispatch.tensor.store %2, %0, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !flow.dispatch.tensor<readwrite:tensor<4xi32>>
  return
}
// CHECK-LABEL: func.func @sort1D
// CHECK:        %[[BUF:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<4xi32, #hal.descriptor_type<storage_buffer>>
// CHECK:        iree_linalg_ext.sort
// CHECK-SAME:     outs(%[[BUF]] : memref<4xi32{{.+}}>)

// -----

func.func @scatter_update_scalar_1D() {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x1xi32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<8xi32>>
  %3 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [8], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<8xi32>> -> tensor<8xi32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
  scf.for %arg0 = %4 to %c4 step %5 {
    %6 = flow.dispatch.tensor.load %0, offsets = [%arg0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<4xi32>
    %7 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0], sizes = [4, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x1xi32>> -> tensor<4x1xi32>
    %8 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%6, %7 : tensor<4xi32>, tensor<4x1xi32>) outs(%3 : tensor<8xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      iree_linalg_ext.yield %arg1 : i32
    } -> tensor<8xi32>
    flow.dispatch.tensor.store %8, %2, offsets = [0], sizes = [8], strides = [1] : tensor<8xi32> -> !flow.dispatch.tensor<readwrite:tensor<8xi32>>
  }
  return
}
// CHECK:      func.func @scatter_update_scalar_1D
// CHECK-DAG:    %[[UPDATE:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<4xi32, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:    %[[INDICES:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<4x1xi32, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:    %[[ORIGINAL:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<8xi32, #hal.descriptor_type<storage_buffer>>
// CHECK:        scf.for %[[I:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK:          iree_linalg_ext.scatter
// CHECK-SAME:     ins(%[[UPDATE]], %[[INDICES]]
// CHECK-SAME:     outs(%[[ORIGINAL:.+]]

// -----

func.func @topk() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<200x8xf32>>
  %input_values = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [200, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<200x8xf32>> -> tensor<200x8xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<200x8xi32>>
  %input_indices = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [200, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<200x8xi32>> -> tensor<200x8xi32>
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
// CHECK:      func.func @topk
// CHECK-DAG:    %[[INPUT_VALUES:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<200x8xf32, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:    %[[INPUT_INDICES:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<200x8xi32, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:    %[[OUTPUT_VALUES:.+]] = memref.alloc() : memref<200x3xf32>
// CHECK-DAG:    %[[OUTPUT_INDICES:.+]] = memref.alloc() : memref<200x3xi32>
// CHECK:        iree_linalg_ext.topk
// CHECK-SAME:     ins(%[[INPUT_VALUES]], %[[INPUT_INDICES]]
// CHECK-SAME:     outs(%[[OUTPUT_VALUES]], %[[OUTPUT_INDICES]]

// -----

func.func @iree_linalg_ext_pack() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x4xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x2x3x3xi32>>
  %2 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [2, 2, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<2x2x3x3xi32>> -> tensor<2x2x3x3xi32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x4xi32>> -> tensor<4x4xi32>
  %4 = iree_linalg_ext.pack %3 padding_value(%c0_i32 : i32) inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %2 : (tensor<4x4xi32> tensor<2x2x3x3xi32>) -> tensor<2x2x3x3xi32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [2, 2, 3, 3], strides = [1, 1, 1, 1] : tensor<2x2x3x3xi32> -> !flow.dispatch.tensor<writeonly:tensor<2x2x3x3xi32>>
  return
}
// CHECK: func.func @iree_linalg_ext_pack
// CHECK-DAG:  %[[PAD:.+]] = arith.constant 0 : i32
// CHECK-DAG:  %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<4x4xi32, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:  %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<2x2x3x3xi32, #hal.descriptor_type<storage_buffer>>
// CHECK:      iree_linalg_ext.pack %[[IN]]
// CHECK-SAME:   padding_value(%[[PAD]] : i32)
// CHECK-SAME:   inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %[[OUT]]

// -----

func.func @iree_linalg_ext_unpack() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x2x2x2xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x4xi32>>
  %2 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<4x4xi32>> -> tensor<4x4xi32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 2, 2, 2], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x2x2x2xi32>> -> tensor<2x2x2x2xi32>
  %4 = iree_linalg_ext.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %2 : (tensor<2x2x2x2xi32> tensor<4x4xi32>) -> tensor<4x4xi32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<4x4xi32>>
  return
}
// CHECK: func.func @iree_linalg_ext_unpack
// CHECK-DAG:  %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<2x2x2x2xi32, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:  %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<4x4xi32, #hal.descriptor_type<storage_buffer>>
// CHECK:      iree_linalg_ext.unpack %[[IN]]
// CHECK-SAME:   inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %[[OUT]]

// -----

func.func @iree_linalg_ext_unpack_fully_dynamic() {
  %c0 = arith.constant 0 : index
  %inner_d0 = util.unfoldable_constant 2 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x2x2x2xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x4xi32>>
  %2 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<4x4xi32>> -> tensor<4x4xi32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 2, %inner_d0, %inner_d0], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x2x2x2xi32>> -> tensor<2x2x?x?xi32>
  %4 = iree_linalg_ext.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [%inner_d0, %inner_d0] into %2 : (tensor<2x2x?x?xi32> tensor<4x4xi32>) -> tensor<4x4xi32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<4x4xi32>>
  return
}

// CHECK:      func.func @iree_linalg_ext_unpack_fully_dynamic
// CHECK-DAG:  %[[D:.+]] = util.optimization_barrier %c2 : index
// CHECK:      iree_linalg_ext.unpack
// CHECK-SAME:   inner_dims_pos = [0, 1] inner_tiles = [%[[D]], %[[D]]]

// -----

func.func @tensor_pack() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x4xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x2x3x3xi32>>
  %2 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [2, 2, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<2x2x3x3xi32>> -> tensor<2x2x3x3xi32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x4xi32>> -> tensor<4x4xi32>
  %4 = tensor.pack %3 padding_value(%c0_i32 : i32) inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %2 : tensor<4x4xi32> -> tensor<2x2x3x3xi32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [2, 2, 3, 3], strides = [1, 1, 1, 1] : tensor<2x2x3x3xi32> -> !flow.dispatch.tensor<writeonly:tensor<2x2x3x3xi32>>
  return
}
// CHECK: func.func @tensor_pack
// CHECK-DAG:  %[[PAD:.+]] = arith.constant 0 : i32
// CHECK-DAG:  %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<4x4xi32, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:  %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<2x2x3x3xi32, #hal.descriptor_type<storage_buffer>>
// CHECK:      iree_linalg_ext.pack %[[IN]]
// CHECK-SAME:   padding_value(%[[PAD]] : i32)
// CHECK-SAME:   inner_dims_pos = [0, 1] inner_tiles = [3, 3] into %[[OUT]]

// -----

func.func @tensor_unpack() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x2x2x2xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x4xi32>>
  %2 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<4x4xi32>> -> tensor<4x4xi32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 2, 2, 2], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x2x2x2xi32>> -> tensor<2x2x2x2xi32>
  %4 = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %2 : tensor<2x2x2x2xi32> -> tensor<4x4xi32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<4x4xi32>>
  return
}
// CHECK: func.func @tensor_unpack
// CHECK-DAG:  %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<2x2x2x2xi32, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:  %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<4x4xi32, #hal.descriptor_type<storage_buffer>>
// CHECK:      iree_linalg_ext.unpack %[[IN]]
// CHECK-SAME:   inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %[[OUT]]

// -----

func.func @tensor_unpack_fully_dynamic() {
  %c0 = arith.constant 0 : index
  %inner_d0 = util.unfoldable_constant 2 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x2x2x2xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x4xi32>>
  %2 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<4x4xi32>> -> tensor<4x4xi32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 2, %inner_d0, %inner_d0], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x2x2x2xi32>> -> tensor<2x2x?x?xi32>
  %4 = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [%inner_d0, %inner_d0] into %2 : tensor<2x2x?x?xi32> -> tensor<4x4xi32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<4x4xi32>>
  return
}

// CHECK:      func.func @tensor_unpack_fully_dynamic
// CHECK-DAG:  %[[D:.+]] = util.optimization_barrier %c2 : index
// CHECK:      iree_linalg_ext.unpack
// CHECK-SAME:   inner_dims_pos = [0, 1] inner_tiles = [%[[D]], %[[D]]]

// -----

module {
  func.func @reduction_ew() {
    %c5120 = arith.constant 5120 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c5120) : !flow.dispatch.tensor<readonly:tensor<1001xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c5120) : !flow.dispatch.tensor<readonly:tensor<1x1001xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x1001xf32>>
    %3 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1, 1001], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x1001xf32>> -> tensor<1x1001xf32>
    %4 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [1001], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1001xf32>> -> tensor<1001xf32>
    %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, 1001], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1001xf32>> -> tensor<1x1001xf32>
    %6 = bufferization.alloc_tensor() : tensor<f32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<f32>) -> tensor<f32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%4 : tensor<1001xf32>) outs(%7 : tensor<f32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0]]>} {
    ^bb0(%arg0: f32, %arg1: f32):
      %10 = arith.addf %arg0, %arg1 : f32
      linalg.yield %10 : f32
    } -> tensor<f32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %8 : tensor<1x1001xf32>, tensor<f32>) outs(%3 : tensor<1x1001xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %10 = arith.divf %cst_0, %arg1 : f32
      %11 = arith.mulf %arg0, %10 : f32
      linalg.yield %11 : f32
    } -> tensor<1x1001xf32>
    flow.dispatch.tensor.store %9, %2, offsets = [0, 0], sizes = [1, 1001], strides = [1, 1] : tensor<1x1001xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x1001xf32>>
    return
  }
}

// CHECK: func.func @reduction_ew
// CHECK: hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c5120) : memref<1001xf32, strided<[1], offset: 1280>, #hal.descriptor_type<storage_buffer>>
// CHECK: hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c5120) : memref<1x1001xf32, strided<[1001, 1], offset: 1280>, #hal.descriptor_type<storage_buffer>>
// CHECK: hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<1x1001xf32, #hal.descriptor_type<storage_buffer>>

// -----

func.func @uniform_storage_buffer() {
  %c0 = arith.constant 0 : index
  %m = hal.interface.constant.load[0] : index
  %n = hal.interface.constant.load[1] : index
  %k = hal.interface.constant.load[2] : index
  %lhs = hal.interface.binding.subspan set(0) binding(0) type(uniform_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %k}
  %rhs = hal.interface.binding.subspan set(0) binding(1) type(uniform_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%k, %n}
  %init = hal.interface.binding.subspan set(0) binding(2) type(uniform_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %n}
  %result = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%m, %n}
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
      %lhs_tile = flow.dispatch.tensor.load %lhs, offsets = [%iv0, 0], sizes = [%tilesize_y, %k], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %k} -> tensor<?x?xf32>
      %rhs_tile = flow.dispatch.tensor.load %rhs, offsets = [0, %iv1], sizes = [%k, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%k, %n} -> tensor<?x?xf32>
      %init_tile = flow.dispatch.tensor.load %init, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %n} -> tensor<?x?xf32>
      %matmul_tile = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%m, %n}
    }
  }
  return
}

// CHECK-LABEL: func.func @uniform_storage_buffer()
//       CHECK: hal.interface.binding.subspan set(0) binding(0) type(uniform_buffer) : memref<?x?xf32, #hal.descriptor_type<uniform_buffer>>
//       CHECK: hal.interface.binding.subspan set(0) binding(1) type(uniform_buffer) : memref<?x?xf32, #hal.descriptor_type<uniform_buffer>>
//       CHECK: hal.interface.binding.subspan set(0) binding(2) type(uniform_buffer) : memref<?x?xf32, #hal.descriptor_type<uniform_buffer>>
//       CHECK: hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>

// -----

func.func @micro_kernel_op() {
  %d0 = hal.interface.constant.load[0] : index
  %d1 = hal.interface.constant.load[1] : index
  %s0 = hal.interface.constant.load[2] : f32
  %s1 = hal.interface.constant.load[3] : i64
  %arg0_binding = hal.interface.binding.subspan set(0) binding(0) type(uniform_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d0, %d1}
  %arg1_binding = hal.interface.binding.subspan set(0) binding(1) type(uniform_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%d0, %d1}
  %arg2_binding = hal.interface.binding.subspan set(0) binding(2) type(uniform_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%d0, %d1}
  %arg3_binding = hal.interface.binding.subspan set(0) binding(3) type(uniform_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d0, %d1}
  %arg0 = flow.dispatch.tensor.load %arg0_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d0, %d1} -> tensor<?x?xf32>
  %arg1 = flow.dispatch.tensor.load %arg1_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%d0, %d1} -> tensor<?x?xf32>
  %arg2 = flow.dispatch.tensor.load %arg2_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%d0, %d1} -> tensor<?x?xf32>
  %arg3 = flow.dispatch.tensor.load %arg3_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d0, %d1} -> tensor<?x?xf32>
  %0:2 = iree_codegen.ukernel.generic "foo" ins(%arg0 : tensor<?x?xf32>)
      outs(%arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      (%s0, %arg3, %s1 : f32, tensor<?x?xf32>, i64) -> tensor<?x?xf32>, tensor<?x?xf32>
  flow.dispatch.tensor.store %0#0, %arg1_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%d0, %d1}
  flow.dispatch.tensor.store %0#1, %arg2_binding, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%d0, %d1}
  return
}
// CHECK-LABEL: func @micro_kernel_op()
//   CHECK-DAG:   %[[S0:.+]] = hal.interface.constant.load[2]
//   CHECK-DAG:   %[[S1:.+]] = hal.interface.constant.load[3]
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(uniform_buffer) : memref<?x?xf32, #hal.descriptor_type<uniform_buffer>>
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(uniform_buffer) : memref<?x?xf32, #hal.descriptor_type<uniform_buffer>>
//   CHECK-DAG:   %[[ARG2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(uniform_buffer) : memref<?x?xf32, #hal.descriptor_type<uniform_buffer>>
//   CHECK-DAG:   %[[ARG3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(uniform_buffer) : memref<?x?xf32, #hal.descriptor_type<uniform_buffer>>
//       CHECK:   iree_codegen.ukernel.generic "foo"
//  CHECK-SAME:       ins(%[[ARG0]] :
//  CHECK-SAME:       outs(%[[ARG1]], %[[ARG2]] :
//  CHECK-SAME:       (%[[S0]], %[[ARG3]], %[[S1]] :
//       CHECK:   return
