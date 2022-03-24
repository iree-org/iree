// RUN: iree-opt %s --iree-codegen-iree-comprehensive-bufferize -canonicalize -cse -canonicalize -split-input-file | FileCheck %s

func @matmul() {
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
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//      CHECK: func @matmul()
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

func @matmul_fill() {
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
      %init_tile = linalg.init_tensor [%tilesize_y, %tilesize_x] : tensor<?x?xf32>
      %fill_tile = linalg.fill ins(%cst : f32) outs(%init_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      %matmul_tile = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>{%m, %n}
    }
  }
  return
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
//      CHECK: func @matmul_fill()
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

func @elementwise() {
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
//      CHECK: func @elementwise()
//  CHECK-DAG:   %[[GLB_CST:.+]] = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
//  CHECK-DAG:   %[[IN_BUF:.+]] = hal.interface.binding.subspan set(0)  binding(0) {{.+}} : memref<1x10xf32>
//  CHECK-DAG:   %[[OUT_BUF:.+]] = hal.interface.binding.subspan set(0)  binding(1) {{.+}} : memref<1x10xf32>
//      CHECK:   scf.for
//  CHECK-DAG:     %[[SUB_IN1:.+]] = memref.subview %[[IN_BUF]]
//  CHECK-DAG:     %[[SUB_OUT1:.+]] = memref.subview %[[OUT_BUF]]
//      CHECK:     scf.for
//  CHECK-DAG:       %[[SUB_IN2:.+]] = memref.subview %[[SUB_IN1]]
//  CHECK-DAG:       %[[SUB_CST:.+]] = memref.subview %[[GLB_CST]]
//  CHECK-DAG:       %[[SUB_OUT2:.+]] = memref.subview %[[SUB_OUT1]]
//      CHECK:       linalg.generic
// CHECK-SAME:         ins(%[[SUB_IN2]], %[[SUB_CST]]
// CHECK-SAME:         outs(%[[SUB_OUT2]]
