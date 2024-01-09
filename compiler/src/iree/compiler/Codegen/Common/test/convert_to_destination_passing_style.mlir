// RUN: iree-opt %s --iree-codegen-convert-to-destination-passing-style --canonicalize -cse --split-input-file | FileCheck %s

func.func @matmul() {
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
//      CHECK: func.func @matmul()
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//  CHECK-DAG:   %[[INIT:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//  CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//  CHECK-DAG:       %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[LHS]]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[RHS]]
//  CHECK-DAG:       %[[INIT_TILE:.+]] = flow.dispatch.tensor.load %[[INIT]]
//      CHECK:       %[[MATMUL_TILE:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:           outs(%[[INIT_TILE]] : tensor<?x?xf32>)
//      CHECK:       flow.dispatch.tensor.store %[[MATMUL_TILE]], %[[RESULT]]

// -----

func.func @matmul_fill() {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %m = hal.interface.constant.load[0] : index
  %n = hal.interface.constant.load[1] : index
  %k = hal.interface.constant.load[2] : index
  %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %k}
  %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%k, %n}
  %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%m, %n}
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
      %init_tile = tensor.empty(%tilesize_y, %tilesize_x) : tensor<?x?xf32>
      %fill_tile = linalg.fill ins(%cst : f32) outs(%init_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      %matmul_tile = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%m, %n}
    }
  }
  return
}
//      CHECK: func.func @matmul_fill()
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//  CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//  CHECK-DAG:       %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[LHS]]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[RHS]]
//  CHECK-DAG:       %[[RESULT_TILE:.+]] = flow.dispatch.tensor.load %[[RESULT]]
//      CHECK:       %[[FILL_TILE:.+]] = linalg.fill
// CHECK-SAME:           outs(%[[RESULT_TILE]] :
//      CHECK:       %[[MATMUL_TILE:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:           outs(%[[FILL_TILE]] : tensor<?x?xf32>)
//      CHECK:       flow.dispatch.tensor.store %[[MATMUL_TILE]], %[[RESULT]]

// -----

func.func @matmul_inplace() {
  %c0 = arith.constant 0 : index
  %m = hal.interface.constant.load[0] : index
  %n = hal.interface.constant.load[1] : index
  %k = hal.interface.constant.load[2] : index
  %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%m, %k}
  %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%k, %n}
  %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%m, %n}
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
      %matmul_tile = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%m, %n}
    }
  }
  return
}
//      CHECK: func.func @matmul_inplace()
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//  CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//  CHECK-DAG:       %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[LHS]]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[RHS]]
//  CHECK-DAG:       %[[RESULT_TILE:.+]] = flow.dispatch.tensor.load %[[RESULT]]
//      CHECK:       %[[MATMUL_TILE:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:           outs(%[[RESULT_TILE]] : tensor<?x?xf32>)
//      CHECK:       flow.dispatch.tensor.store %[[MATMUL_TILE]], %[[RESULT]]

// -----

func.func @reshape_simple() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<12xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [12], strides = [1] : !flow.dispatch.tensor<readonly:tensor<12xi32>> -> tensor<12xi32>
  %3 = tensor.expand_shape %2 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
  return
}
//      CHECK: func.func @reshape_simple()
//  CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//  CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//      CHECK:   %[[SOURCE:.+]] = flow.dispatch.tensor.load %[[ARG0]]
//      CHECK:   %[[RESHAPE:.+]] = tensor.expand_shape %[[SOURCE]]
//      CHECK:   flow.dispatch.tensor.store %[[RESHAPE]], %[[RET0]]

// -----

func.func @reshape_fused_source() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<12xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [12], strides = [1] : !flow.dispatch.tensor<readonly:tensor<12xi32>> -> tensor<12xi32>
  %3 = tensor.expand_shape %2 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
  %4 = tensor.empty() : tensor<3x4xi32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%3 : tensor<3x4xi32>) outs(%4 : tensor<3x4xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %6 = arith.addi %arg0, %arg0 : i32
      linalg.yield %6 : i32
    } -> tensor<3x4xi32>
  flow.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
  return
}
//      CHECK: func.func @reshape_fused_source()
//  CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//  CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//      CHECK:   %[[TARGET:.+]] = flow.dispatch.tensor.load %[[RET0]]
//      CHECK:   %[[SOURCE:.+]] = flow.dispatch.tensor.load %[[ARG0]]
//      CHECK:   %[[RESHAPE:.+]] = tensor.expand_shape %[[SOURCE]]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[RESHAPE]]
// CHECK-SAME:       outs(%[[TARGET]]
//      CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[RET0]]

// -----

func.func @reshape_fused_source_and_copyout() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<12xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [12], strides = [1] : !flow.dispatch.tensor<readonly:tensor<12xi32>> -> tensor<12xi32>
  %4 = tensor.expand_shape %3 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
  %5 = tensor.empty() : tensor<3x4xi32>
  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%4 : tensor<3x4xi32>) outs(%5 : tensor<3x4xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %7 = arith.addi %arg0, %arg0 : i32
      linalg.yield %7 : i32
    } -> tensor<3x4xi32>
  flow.dispatch.tensor.store %6, %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
  flow.dispatch.tensor.store %4, %2, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<3x4xi32>>
  return
}
//      CHECK: func.func @reshape_fused_source_and_copyout()
//  CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//  CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//  CHECK-DAG:   %[[RET1:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//  CHECK-DAG:   %[[TARGET:.+]] = flow.dispatch.tensor.load %[[RET0]]
//      CHECK:   %[[SOURCE:.+]] = flow.dispatch.tensor.load %[[ARG0]]
//      CHECK:   %[[RESHAPE:.+]] = tensor.expand_shape %[[SOURCE]]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[RESHAPE]]
// CHECK-SAME:       outs(%[[TARGET]]
//      CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[RET0]]
//      CHECK:   flow.dispatch.tensor.store %[[RESHAPE]], %[[RET1]]

// -----

func.func @reshape_fused_target() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x4xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<12xi32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<3x4xi32>> -> tensor<3x4xi32>
  %3 = tensor.empty() : tensor<3x4xi32>
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%2 : tensor<3x4xi32>) outs(%3 : tensor<3x4xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %5 = arith.addi %arg0, %arg0 : i32
      linalg.yield %5 : i32
    } -> tensor<3x4xi32>
  %5 = tensor.collapse_shape %4 [[0, 1]] : tensor<3x4xi32> into tensor<12xi32>
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [12], strides = [1] : tensor<12xi32> -> !flow.dispatch.tensor<writeonly:tensor<12xi32>>
  return
}
//      CHECK: func.func @reshape_fused_target()
//  CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//  CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//  CHECK-DAG:   %[[SOURCE:.+]] = flow.dispatch.tensor.load %[[ARG0]]
//  CHECK-DAG:   %[[TARGET:.+]] = flow.dispatch.tensor.load %[[RET0]]
//  CHECK-DAG:   %[[RESHAPE_EXPAND:.+]] = tensor.expand_shape %[[TARGET]] {{\[}}[0, 1]{{\]}}
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[SOURCE]]
// CHECK-SAME:       outs(%[[RESHAPE_EXPAND]]
//      CHECK:   %[[RESHAPE_COLLAPSE:.+]] = tensor.collapse_shape %[[GENERIC]] {{\[}}[0, 1]{{\]}}
//      CHECK:   flow.dispatch.tensor.store %[[RESHAPE_COLLAPSE]], %[[RET0]]

// -----

func.func @cast_followed_by_store() {
  %c0 = arith.constant 0 : index
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
    %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
    %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_y]
    scf.for %arg1 = %3 to %c32 step %4 {
      %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
      %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
      scf.for %arg2 = %5 to %c64 step %6 {
        %7 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1, 0], sizes = [%c1, %c32, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x32x1024xf32>> -> tensor<?x?x1024xf32>
        %8 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0, %arg2], sizes = [%c1, 1024, %c32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x1024x64xf32>> -> tensor<?x1024x?xf32>
        %9 = tensor.empty() : tensor<1x32x32xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>
        %11 = linalg.batch_matmul ins(%7, %8 : tensor<?x?x1024xf32>, tensor<?x1024x?xf32>) outs(%10 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>
        %12 = tensor.cast %11 : tensor<1x32x32xf32> to tensor<?x?x?xf32>
        flow.dispatch.tensor.store %12, %2, offsets = [%arg0, %arg1, %arg2], sizes = [%c1, %c32, %c32], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x32x64xf32>>
      }
    }
  }
  return
}
//      CHECK: func.func @cast_followed_by_store()
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//  CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//  CHECK-DAG:       %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[LHS]]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[RHS]]
//  CHECK-DAG:       %[[RESULT_TILE:.+]] = flow.dispatch.tensor.load %[[RESULT]]
//      CHECK:       %[[FILL_TILE:.+]] = linalg.fill
//  CHECK-SAME:          outs(%[[RESULT_TILE]] :
//      CHECK:       %[[MATMUL_TILE:.+]] = linalg.batch_matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]]
// CHECK-SAME:           outs(%[[FILL_TILE]]
//      CHECK:       flow.dispatch.tensor.store %[[MATMUL_TILE]], %[[RESULT]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @multi_result() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %dim0 = hal.interface.constant.load[0] : index
  %dim1 = hal.interface.constant.load[1] : index
  %dim2 = hal.interface.constant.load[2] : index
  %dim3 = hal.interface.constant.load[3] : index
  %dim4 = hal.interface.constant.load[4] : index
  %dim5 = hal.interface.constant.load[5] : index
  %dim6 = hal.interface.constant.load[6] : index
  %dim7 = hal.interface.constant.load[7] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim0, %dim1}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim2, %dim3}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim4, %dim5}
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim6, %dim7}
  %4 = hal.interface.constant.load[8] : index
  %5 = hal.interface.constant.load[9] : index
  %6 = hal.interface.constant.load[10] : index
  %7 = hal.interface.constant.load[11] : index
  %8 = hal.interface.workgroup.id[0] : index
  %9 = hal.interface.workgroup.id[1] : index
  %10 = hal.interface.workgroup.count[0] : index
  %11 = hal.interface.workgroup.count[1] : index
  %12 = hal.interface.workgroup.size[0] : index
  %13 = hal.interface.workgroup.size[1] : index
  %14 = arith.muli %9, %13 : index
  %15 = arith.muli %11, %13 : index
  %16 = arith.muli %8, %12 : index
  %17 = arith.muli %10, %12 : index
  scf.for %arg0 = %14 to %4 step %15 {
    scf.for %arg1 = %16 to %5 step %17 {
      %18 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg0)[%4, %13]
      %19 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg1)[%5, %12]
      %20 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim0, %dim1} -> tensor<?x?xf32>
      %21 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim2, %dim3} -> tensor<?x?xf32>
      %shape = tensor.empty(%18, %19) : tensor<?x?xf32>
      %22:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%20, %21 : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%shape, %shape : tensor<?x?xf32>, tensor<?x?xf32>) {
        ^bb0(%arg2: f32, %arg3 : f32, %arg4 : f32, %arg5 : f32):  // no predecessors
          %23 = arith.mulf %arg2, %arg3 : f32
          %24 = arith.addf %arg2, %arg3 : f32
          linalg.yield %23, %24 : f32, f32
        } -> (tensor<?x?xf32>, tensor<?x?xf32>)
      flow.dispatch.tensor.store %22#0, %2, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim4, %dim5}
      flow.dispatch.tensor.store %22#1, %3, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim6, %dim7}
    }
  }
  return
}
//      CHECK: func.func @multi_result()
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//  CHECK-DAG:   %[[RESULT0:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//  CHECK-DAG:   %[[RESULT1:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//  CHECK-DAG:       %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[LHS]]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[RHS]]
//  CHECK-DAG:       %[[RESULT0_TILE:.+]] = flow.dispatch.tensor.load %[[RESULT0]]
//  CHECK-DAG:       %[[RESULT1_TILE:.+]] = flow.dispatch.tensor.load %[[RESULT1]]
//      CHECK:       %[[GENERIC_TILE:.+]]:2 = linalg.generic
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]]
// CHECK-SAME:           outs(%[[RESULT0_TILE]], %[[RESULT1_TILE]]
//      CHECK:       flow.dispatch.tensor.store %[[GENERIC_TILE]]#0, %[[RESULT0]]
//      CHECK:       flow.dispatch.tensor.store %[[GENERIC_TILE]]#1, %[[RESULT1]]

// -----

func.func @unused_ins_operand() {
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = hal.interface.constant.load[4] : i32
  %5 = hal.interface.constant.load[5] : i32
  %6 = arith.index_cast %0 : i32 to index
  %7 = arith.index_cast %1 : i32 to index
  %8 = arith.index_cast %2 : i32 to index
  %9 = arith.index_cast %3 : i32 to index
  %10 = arith.index_cast %4 : i32 to index
  %11 = arith.index_cast %5 : i32 to index
  %12 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c32) : !flow.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%6, %7, %8}
  %13 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c64) : !flow.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%9, %10, %11}
  %14 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?x?xi32>>{%9, %10, %8}
  %15 = flow.dispatch.tensor.load %13, offsets = [0, 0, 0], sizes = [%9, %10, %11], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%9, %10, %11} -> tensor<?x?x?xi32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_count_z = hal.interface.workgroup.count[2] : index
  %16 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_z]
  %17 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_z]
  scf.for %arg0 = %16 to %6 step %17 {
    %18 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
    %19 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
    scf.for %arg1 = %18 to %7 step %19 {
      %20 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
      %21 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
      scf.for %arg2 = %20 to %8 step %21 {
        %22 = affine.min affine_map<(d0)[s0] -> (64, -d0 + s0)>(%arg0)[%6]
        %23 = affine.min affine_map<(d0)[s0] -> (64, -d0 + s0)>(%arg1)[%7]
        %24 = affine.min affine_map<(d0)[s0] -> (64, -d0 + s0)>(%arg2)[%8]
        %25 = flow.dispatch.tensor.load %12, offsets = [%arg0, %arg1, %arg2], sizes = [%22, %23, %24], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?xi32>>{%6, %7, %8} -> tensor<?x?x?xi32>
        %26 = tensor.empty(%22, %23) : tensor<?x?xi32>
        %27 = tensor.empty(%22, %23, %24) : tensor<?x?x?xi32>
        %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %26 : tensor<?x?x?xi32>, tensor<?x?xi32>) outs(%27 : tensor<?x?x?xi32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[], [1, 4, 4]]>} {
        ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):  // no predecessors
          %29 = arith.index_cast %arg3 : i32 to index
          %30 = linalg.index 0 : index
          %31 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%30, %arg0)
          %32 = linalg.index 1 : index
          %33 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%32, %arg1)
          %34 = tensor.extract %15[%31, %33, %29] : tensor<?x?x?xi32>
          linalg.yield %34 : i32
        } -> tensor<?x?x?xi32>
        flow.dispatch.tensor.store %28, %14, offsets = [%arg0, %arg1, %arg2], sizes = [%22, %23, %24], strides = [1, 1, 1] : tensor<?x?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?xi32>>{%9, %10, %8}
      }
    }
  }
  return
}
// CHECK-LABEL: func.func @unused_ins_operand()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(2)
//   CHECK-DAG:   %[[IN_VIEW:.+]] = flow.dispatch.tensor.load %[[IN]]
//  CHECK-DAG:    %[[OUT_VIEW:.+]] = flow.dispatch.tensor.load %[[OUT]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[IN_VIEW]] :
//  CHECK-SAME:     outs(%[[OUT_VIEW]] :

// -----

func.func @cumsum__2x2x2x2x2x2x2() {
  %cst = arith.constant dense<0.000000e+00> : tensor<2x2x2x2x2x2x2xf32>
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<3x2x2x2x2x2x2xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x2x2x2x2x2x2xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0, 0, 0], sizes = [3, 2, 2, 2, 2, 2, 2], strides = [1, 1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x2x2x2x2x2x2xf32>> -> tensor<3x2x2x2x2x2x2xf32>
  %3 = tensor.empty() : tensor<2xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0 + d7, d1, d2, d3, d4, d5, d6)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d7)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]}
    ins(%2, %3 : tensor<3x2x2x2x2x2x2xf32>, tensor<2xf32>)
    outs(%cst : tensor<2x2x2x2x2x2x2xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %5 = arith.addf %arg0, %arg2 : f32
    linalg.yield %5 : f32
  } -> tensor<2x2x2x2x2x2x2xf32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0, 0, 0, 0], sizes = [2, 2, 2, 2, 2, 2, 2], strides = [1, 1, 1, 1, 1, 1, 1] : tensor<2x2x2x2x2x2x2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x2x2x2x2x2x2xf32>>
  return
}

// CHECK-LABEL: func.func @cumsum__2x2x2x2x2x2x2()
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[DST:.+]] = flow.dispatch.tensor.load {{.+}} !flow.dispatch.tensor<writeonly:tensor<2x2x2x2x2x2x2xf32>> -> tensor<2x2x2x2x2x2x2xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[DST]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%[[FILL]]

// -----

func.func @reduce_window_max_4x6xf32() {
  %cst = arith.constant dense<0xFF800000> : tensor<2x2xf32>
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x4x6xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x2xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 4, 6], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x6xf32>> -> tensor<2x4x6xf32>
  %3 = tensor.empty() : tensor<2x2x3xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d2, d0 * 2 + d3, d1 * 3 + d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%2, %3 : tensor<2x4x6xf32>, tensor<2x2x3xf32>) outs(%cst : tensor<2x2xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %5 = arith.maximumf %arg0, %arg2 : f32
    linalg.yield %5 : f32
  } -> tensor<2x2xf32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x2xf32>>
  return
}
// CHECK-LABEL: func.func @reduce_window_max_4x6xf32()
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0xFF800000 : f32
//       CHECK:   %[[DST:.+]] = flow.dispatch.tensor.load {{.+}} !flow.dispatch.tensor<writeonly:tensor<2x2xf32>> -> tensor<2x2xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[DST]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%[[FILL]]

// -----

func.func @linalg_ext_reverse_dim0() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x3xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x3xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %2 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
  %3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
  scf.for %arg0 = %2 to %c2 step %3 {
    %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
    %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
    scf.for %arg1 = %4 to %c3 step %5 {
      %6 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1], sizes = [2, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x3xf32>> -> tensor<2x3xf32>
      %7 = tensor.empty() : tensor<2x3xf32>
      %8 = iree_linalg_ext.reverse dimensions(dense<0> : tensor<1xi64>) ins(%6 : tensor<2x3xf32>) outs(%7 : tensor<2x3xf32>) : tensor<2x3xf32>
      %9 = affine.apply affine_map<()[s0] -> (-s0)>()[%arg0]
      flow.dispatch.tensor.store %8, %1, offsets = [%9, %arg1], sizes = [2, 3], strides = [%c1, %c1] : tensor<2x3xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x3xf32>>
    }
  }
  return
}
//      CHECK: func.func @linalg_ext_reverse_dim0()
//  CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//  CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//  CHECK-DAG:       %[[IN_TILE:.+]] = flow.dispatch.tensor.load %[[IN]]
//  CHECK-DAG:       %[[OUT_TILE:.+]] = flow.dispatch.tensor.load %[[OUT]]
//      CHECK:       %[[REV_TILE:.+]] = iree_linalg_ext.reverse
// CHECK-SAME:           ins(%[[IN_TILE]] : tensor<2x3xf32>)
// CHECK-SAME:           outs(%[[OUT_TILE]] : tensor<2x3xf32>)
//      CHECK:       flow.dispatch.tensor.store %[[REV_TILE]], %[[OUT]]

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
//      CHECK: func.func @sort1D()
//  CHECK-DAG:   %[[BUF:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
//  CHECK-DAG:   %[[IN:.+]] = flow.dispatch.tensor.load %[[BUF]]
//      CHECK:   %[[SORT:.+]] = iree_linalg_ext.sort
// CHECK-SAME:       outs(%[[IN]] : tensor<4xi32>)
//      CHECK:   flow.dispatch.tensor.store %[[SORT]], %[[BUF]]

// -----

func.func @clone_index_computations() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = arith.index_castui %0 : i32 to index
  %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%1}
  %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%1}
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
  scf.for %arg0 = %4 to %1 step %5 {
    %6 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg0)[%1]
    %7 = flow.dispatch.tensor.load %2, offsets = [%arg0], sizes = [%6], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%1} -> tensor<?xf32>
    %8 = tensor.empty(%6) : tensor<?xf32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7 : tensor<?xf32>) outs(%8 : tensor<?xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64]]>} {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.addf %in, %in : f32
      linalg.yield %11 : f32
    } -> tensor<?xf32>
    %10 = arith.index_castui %0 : i32 to index
    flow.dispatch.tensor.store %9, %3, offsets = [%arg0], sizes = [%6], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%10}
  }
  return
}
// CHECK-LABEL: func @clone_index_computations()
//   CHECK-DAG:   %[[INPUT:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//       CHECK:   scf.for
//       CHECK:     %[[TILESIZE:.+]] = affine.min
//       CHECK:     %[[LOAD:.+]] = flow.dispatch.tensor.load %[[OUTPUT]], offsets = [{{.+}}], sizes = [%[[TILESIZE]]]

// -----

func.func @gemm_gather() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x512xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<2x512xf32>>
  %3 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128xi32>>
  %result = hal.interface.binding.subspan set(0) binding(5) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 512], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<256x512xf32>> -> tensor<256x512xf32>
  %6 = flow.dispatch.tensor.load %2, offsets= [0, 0], sizes= [2, 512], strides= [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<2x512xf32>> -> tensor<2x512xf32>
  %7 = flow.dispatch.tensor.load %3, offsets = [0], sizes= [128], strides = [1]
      : !flow.dispatch.tensor<readonly:tensor<128xi32>> -> tensor<128xi32>
  %8 = tensor.empty() : tensor<128x512xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %10 = linalg.matmul ins(%4, %5 : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%9 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %11 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%10, %7 : tensor<128x512xf32>, tensor<128xi32>) outs(%8 : tensor<128x512xf32>) {
    ^bb0(%b0 : f32, %b1 : i32, %b2 : f32):
        %12 = linalg.index 1 : index
        %13 = arith.index_cast %b1 : i32 to index
        %14 = tensor.extract %6[%13, %12] : tensor<2x512xf32>
        %15 = arith.addf %b0, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<128x512xf32>
  flow.dispatch.tensor.store %11, %result, offsets= [0, 0], sizes = [128, 512], strides= [1, 1]
      : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
  return
}
// CHECK-LABEL: func @gemm_gather
//       CHECK:   %[[GEMM:.+]] = linalg.matmul
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%{{[a-zA-Z0-9]+}} :
//  CHECK-SAME:       outs(%[[GEMM]] :

// -----

func.func @reduce_broadcast_generic() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<10x1024xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<10xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<10x1024xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets= [0, 0], sizes = [10, 1024], strides= [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<10x1024xf32>> -> tensor<10x1024xf32>
  %4 = flow.dispatch.tensor.load %1, offsets= [0], sizes = [10], strides= [1]
      : !flow.dispatch.tensor<readonly:tensor<10xf32>> -> tensor<10xf32>
  %5 = tensor.empty() : tensor<10x1024xf32>
  %6 = tensor.empty() : tensor<10xf32>
  %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<10xf32>) -> tensor<10xf32>
  %8:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%3, %4 : tensor<10x1024xf32>, tensor<10xf32>) outs(%5, %7 : tensor<10x1024xf32>, tensor<10xf32>) {
   ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
      %18 = arith.subf %arg1, %arg2 : f32
      %19 = math.exp %18 : f32
      %20 = arith.addf %19, %arg4 : f32
      linalg.yield %19, %20 : f32, f32
  } -> (tensor<10x1024xf32>, tensor<10xf32>)
  %9 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%8#0, %8#1 : tensor<10x1024xf32>, tensor<10xf32>) outs(%5 : tensor<10x1024xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %18 = arith.divf %arg1, %arg2 : f32
      linalg.yield %18 : f32
    } -> tensor<10x1024xf32>
  flow.dispatch.tensor.store %9, %2, offsets = [0, 0], sizes= [10, 1024], strides= [1, 1]
      : tensor<10x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<10x1024xf32>>
  return
}
// CHECK-LABEL: func @reduce_broadcast_generic
//       CHECK:   %[[OUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(4)
//       CHECK:   %[[OUT:.+]] = flow.dispatch.tensor.load %[[OUT_BINDING]]
//       CHECK:   %[[RESULT:.+]]:2 = linalg.generic
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%[[RESULT]]#0, %[[RESULT]]#1 :
//  CHECK-SAME:       outs(%[[OUT]] :

// -----

func.func @pack() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x4xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x2x2x2xi32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x4xi32>> -> tensor<4x4xi32>
  %3 = tensor.empty() : tensor<2x2x2x2xi32>
  %pack = tensor.pack %2 inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %3 : tensor<4x4xi32> -> tensor<2x2x2x2xi32>
  flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [2, 2, 2, 2], strides = [1, 1, 1, 1] : tensor<2x2x2x2xi32> -> !flow.dispatch.tensor<writeonly:tensor<2x2x2x2xi32>>
  return
}
// CHECK-LABEL: func.func @pack
// CHECK-DAG:     %[[IN_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-DAG:     %[[OUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-DAG:     %[[IN:.+]] = flow.dispatch.tensor.load %[[IN_BINDING]]
// CHECK-DAG:     %[[OUT:.+]] = flow.dispatch.tensor.load %[[OUT_BINDING]]
// CHECK:         tensor.pack %[[IN]] inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %[[OUT]]

// -----

func.func @unpack() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x2x2x2xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x4xi32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 2, 2, 2], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x2x2x2xi32>> -> tensor<2x2x2x2xi32>
  %3 = tensor.empty() : tensor<4x4xi32>
  %4 = tensor.unpack %2 inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %3 : tensor<2x2x2x2xi32> -> tensor<4x4xi32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xi32> -> !flow.dispatch.tensor<writeonly:tensor<4x4xi32>>
  return
}
// CHECK-LABEL: func.func @unpack
// CHECK-DAG:     %[[IN_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-DAG:     %[[OUT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-DAG:     %[[IN:.+]] = flow.dispatch.tensor.load %[[IN_BINDING]]
// CHECK-DAG:     %[[OUT:.+]] = flow.dispatch.tensor.load %[[OUT_BINDING]]
// CHECK:         tensor.unpack %[[IN]] inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %[[OUT]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @non_perfect_tiling_unpack() {
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0:2 = iree_codegen.query_tile_sizes tensor<16x16xi32, #iree_linalg_ext.encoding<role = RESULT, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
  %1 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%0#0]
  %2 = affine.apply affine_map<()[s0] -> (16 ceildiv s0)>()[%0#1]
  %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c512) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi32>>{%1, %2, %0#0, %0#1}
  %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x1xi32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %9 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_y]
  %10 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_y]
  scf.for %arg0 = %9 to %c1 step %10 {
    %11 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_x]
    %12 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_x]
    scf.for %arg1 = %11 to %c1 step %12 {
      %13 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg0)[%0#0]
      %14 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%0#1]
      %15 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg0)[%0#0]
      %16 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%0#1]
      %17 = flow.dispatch.tensor.load %3, offsets = [%15, %16, 0, 0], sizes = [%c1, %c1, %0#0, %0#1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi32>>{%1, %2, %0#0, %0#1} -> tensor<?x?x?x?xi32>
      %18 = tensor.empty(%0#0, %0#1) : tensor<?x?xi32>
      %19 = tensor.unpack %17 inner_dims_pos = [0, 1] inner_tiles = [%0#0, %0#1] into %18 : tensor<?x?x?x?xi32> -> tensor<?x?xi32>
      %extracted_slice = tensor.extract_slice %19[%13, %14] [1, 1] [1, 1] : tensor<?x?xi32> to tensor<1x1xi32>
      %cast = tensor.cast %extracted_slice : tensor<1x1xi32> to tensor<?x?xi32>
      flow.dispatch.tensor.store %cast, %4, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<1x1xi32>>
    }
  }
  return
}
// CHECK-LABEL: func.func @non_perfect_tiling_unpack
// CHECK:         %[[ALLOC:.+]] = bufferization.alloc_tensor
// CHECK:         %[[UNPACK:.+]] = tensor.unpack
// CHECK-SAME:      into %[[ALLOC]]
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[UNPACK]]

// -----

func.func @multi_result_dispatches() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<120x240xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<240x360xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<120xf32>>
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<writeonly:tensor<120x360xf32>>
  %30 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<writeonly:tensor<120x360xf32>>
  %4 = tensor.empty() : tensor<120x360xf32>
  %cst = arith.constant 0.0 : f32
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<120x360xf32>) -> tensor<120x360xf32>
  %6 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [120, 240], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<120x240xf32>> -> tensor<120x240xf32>
  %7 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [240, 360], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<240x360xf32>> -> tensor<240x360xf32>
  %8 = linalg.matmul ins(%6, %7 : tensor<120x240xf32>, tensor<240x360xf32>)
      outs(%5 : tensor<120x360xf32>) -> tensor<120x360xf32>
  %9 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [120], strides = [1]
      : !flow.dispatch.tensor<readonly:tensor<120xf32>> -> tensor<120xf32>
  %10 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%8, %9 : tensor<120x360xf32>, tensor<120xf32>)
      outs(%4 : tensor<120x360xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %10 = arith.addf %b0, %b1 : f32
      linalg.yield %10 : f32
    } -> tensor<120x360xf32>
  flow.dispatch.tensor.store %8, %30, offsets = [0, 0], sizes = [120, 360], strides = [1, 1]
      : tensor<120x360xf32> -> !flow.dispatch.tensor<writeonly:tensor<120x360xf32>>
  flow.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [120, 360], strides = [1, 1]
      : tensor<120x360xf32> -> !flow.dispatch.tensor<writeonly:tensor<120x360xf32>>
  return
}
// CHECK-LABEL: func @multi_result_dispatches()
//   CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[BIAS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
//   CHECK-DAG:   %[[RESULT_BINDING0:.+]] = hal.interface.binding.subspan set(0) binding(3)
//   CHECK-DAG:   %[[RESULT0:.+]] = flow.dispatch.tensor.load %[[RESULT_BINDING0]]
//   CHECK-DAG:   %[[RESULT_BINDING1:.+]] = hal.interface.binding.subspan set(0) binding(4)
//   CHECK-DAG:   %[[RESULT1:.+]] = flow.dispatch.tensor.load %[[RESULT_BINDING1]]
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       outs(%[[RESULT1]] :
//       CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//       CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   %[[BIAS:.+]] = flow.dispatch.tensor.load %[[BIAS_BINDING]]
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[MATMUL]], %[[BIAS]] :
//  CHECK-SAME:       outs(%[[RESULT0]] :
//       CHECK:   flow.dispatch.tensor.store %[[MATMUL]], %[[RESULT_BINDING1]]
//       CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[RESULT_BINDING0]]

// -----

func.func @if_conversion() {
  %0 = hal.interface.constant.load[0] : index
  %offset = hal.interface.constant.load[1] : index
  %size = hal.interface.constant.load[2] : index
  %cond = hal.interface.constant.load[3] : i1
  %result_offset = hal.interface.constant.load[4] : index
  %then = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
    : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%0}
  %else = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
    : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%0}
  %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
    : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
  %then_value = flow.dispatch.tensor.load %then, offsets = [%offset], sizes = [%size], strides = [1]
    : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%0} -> tensor<?xf32>
  %else_value = flow.dispatch.tensor.load %else, offsets = [%offset], sizes = [%size], strides = [1]
    : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%0} -> tensor<?xf32>
  %if = scf.if %cond -> (tensor<?xf32>) {
    scf.yield %then_value : tensor<?xf32>
  } else {
    scf.yield %else_value : tensor<?xf32>
  }
  flow.dispatch.tensor.store %if, %result, offsets = [%result_offset], sizes = [%size], strides = [1]
    : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%0}
  return
}
// CHECK-LABEL: func @if_conversion()
//   CHECK-DAG:   %[[S0:.+]] = hal.interface.constant.load[0]
//   CHECK-DAG:   %[[S1:.+]] = hal.interface.constant.load[2]
//   CHECK-DAG:   %[[COND:.+]] = hal.interface.constant.load[3]
//   CHECK-DAG:   %[[OFFSET:.+]] = hal.interface.constant.load[4]
//   CHECK-DAG:   %[[THEN_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[ELSE_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
//   CHECK-DAG:   %[[THEN:.+]] = flow.dispatch.tensor.load %[[THEN_BINDING]]
//   CHECK-DAG:   %[[ELSE:.+]] = flow.dispatch.tensor.load %[[ELSE_BINDING]]
//       CHECK:   scf.if %[[COND]] {
//  CHECK-NEXT:     flow.dispatch.tensor.store %[[THEN]], %[[RESULT_BINDING]]
//  CHECK-SAME:         offsets = [%[[OFFSET]]], sizes = [%[[S1]]]
//  CHECK-SAME:         flow.dispatch.tensor<writeonly:tensor<?xf32>>{%[[S0]]}
//  CHECK-NEXT:   } else {
//  CHECK-NEXT:     flow.dispatch.tensor.store %[[ELSE]], %[[RESULT_BINDING]]
//  CHECK-SAME:         offsets = [%[[OFFSET]]], sizes = [%[[S1]]]
//  CHECK-SAME:         flow.dispatch.tensor<writeonly:tensor<?xf32>>{%[[S0]]}
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @if_conversion_clone_offsets() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = hal.interface.constant.load[4] : i32
  %5 = arith.index_castui %0 : i32 to index
  %6 = arith.index_castui %1 : i32 to index
  %7 = arith.index_castui %2 : i32 to index
  %8 = arith.index_castui %3 : i32 to index
  %9 = arith.index_castui %4 : i32 to index
  %10 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
  %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%8, %9}
  %12 = affine.apply affine_map<()[s0, s1] -> (-s0 + s1 + (s0 ceildiv 16) * 16)>()[%6, %6]
  %13 = affine.apply affine_map<()[s0, s1] -> (-s0 + s1 + (s0 ceildiv 16) * 16)>()[%7, %7]
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %14 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
  %15 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
  scf.for %arg0 = %14 to %12 step %15 {
    %16 = affine.min affine_map<(d0)[s0, s1] -> (64, -d0 - s0 + s1 + (s0 ceildiv 16) * 16)>(%arg0)[%6, %6]
    %17 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
    %18 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
    scf.for %arg1 = %17 to %13 step %18 {
      %19 = affine.min affine_map<(d0)[s0, s1] -> (64, -d0 - s0 + s1 + (s0 ceildiv 16) * 16)>(%arg1)[%7, %7]
      %20 = affine.min affine_map<(d0)[s0] -> (s0, d0)>(%arg0)[%6]
      %21 = affine.min affine_map<(d0, d1)[s0] -> (s0, d0 + d1)>(%arg0, %16)[%6]
      %22 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%21, %20)
      %23 = arith.cmpi eq, %22, %c0 : index
      %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 - d1 + d2)>(%16, %21, %20)
      %25 = affine.min affine_map<(d0)[s0] -> (s0, d0)>(%arg1)[%7]
      %26 = affine.min affine_map<(d0, d1)[s0] -> (s0, d0 + d1)>(%arg1, %19)[%7]
      %27 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%26, %25)
      %28 = arith.cmpi eq, %27, %c0 : index
      %29 = arith.ori %28, %23 : i1
      %30 = affine.apply affine_map<(d0, d1, d2) -> (d0 - d1 + d2)>(%19, %26, %25)
      %31 = scf.if %29 -> (tensor<?x?xf32>) {
        %generated = tensor.generate %16, %19 {
        ^bb0(%arg2: index, %arg3: index):
          tensor.yield %cst : f32
        } : tensor<?x?xf32>
        scf.yield %generated : tensor<?x?xf32>
      } else {
        %34 = flow.dispatch.tensor.load %10, offsets = [%20, %25], sizes = [%22, %27], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %padded = tensor.pad %34 low[0, 0] high[%24, %30] {
        ^bb0(%arg2: index, %arg3: index):
          tensor.yield %cst : f32
        } : tensor<?x?xf32> to tensor<?x?xf32>
        scf.yield %padded : tensor<?x?xf32>
      }
      %32 = arith.index_castui %3 : i32 to index
      %33 = arith.index_castui %4 : i32 to index
      flow.dispatch.tensor.store %31, %11, offsets = [%arg0, %arg1], sizes = [%16, %19], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%32, %33}
    }
  }
  return
}
// CHECK-LABEL: func @if_conversion_clone_offsets()
//       CHECK:   scf.if
//  CHECK-NEXT:     %[[GENERATED:.+]] = tensor.generate
//       CHECK:     flow.dispatch.tensor.store %[[GENERATED]]
//       CHECK:   else
//       CHECK:     %[[VAL:.+]] = flow.dispatch.tensor.load
//       CHECK:     %[[PADDED:.+]] = tensor.pad %[[VAL]]
//       CHECK:     flow.dispatch.tensor.store %[[PADDED]]
