// RUN: iree-opt %s -iree-codegen-convert-to-destination-passing-style -canonicalize -cse -split-input-file | FileCheck %s

func.func @matmul() {
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
//      CHECK: func @matmul()
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
  %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %k}
  %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:?x?xf32>{%k, %n}
  %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?x?xf32>{%m, %n}
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
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%m, %n}
    }
  }
  return
}
//      CHECK: func @matmul_fill()
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
  %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %k}
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
      %matmul_tile = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>{%m, %n}
    }
  }
  return
}
//      CHECK: func @matmul_inplace()
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:12xi32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:3x4xi32>
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [12], strides = [1] : !flow.dispatch.tensor<readonly:12xi32> -> tensor<12xi32>
  %3 = tensor.expand_shape %2 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  return
}
//      CHECK: func @reshape_simple()
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:12xi32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:3x4xi32>
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [12], strides = [1] : !flow.dispatch.tensor<readonly:12xi32> -> tensor<12xi32>
  %3 = tensor.expand_shape %2 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
  %4 = linalg.init_tensor [3, 4] : tensor<3x4xi32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%3 : tensor<3x4xi32>) outs(%4 : tensor<3x4xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %6 = arith.addi %arg0, %arg0 : i32
      linalg.yield %6 : i32
    } -> tensor<3x4xi32>
  flow.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  return
}
//      CHECK: func @reshape_fused_source()
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:12xi32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:3x4xi32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:3x4xi32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [12], strides = [1] : !flow.dispatch.tensor<readonly:12xi32> -> tensor<12xi32>
  %4 = tensor.expand_shape %3 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
  %5 = linalg.init_tensor [3, 4] : tensor<3x4xi32>
  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%4 : tensor<3x4xi32>) outs(%5 : tensor<3x4xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %7 = arith.addi %arg0, %arg0 : i32
      linalg.yield %7 : i32
    } -> tensor<3x4xi32>
  flow.dispatch.tensor.store %6, %1, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  flow.dispatch.tensor.store %4, %2, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  return
}
//      CHECK: func @reshape_fused_source_and_copyout()
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:3x4xi32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:12xi32>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [3, 4], strides = [1, 1] : !flow.dispatch.tensor<readonly:3x4xi32> -> tensor<3x4xi32>
  %3 = linalg.init_tensor [3, 4] : tensor<3x4xi32>
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%2 : tensor<3x4xi32>) outs(%3 : tensor<3x4xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %5 = arith.addi %arg0, %arg0 : i32
      linalg.yield %5 : i32
    } -> tensor<3x4xi32>
  %5 = tensor.collapse_shape %4 [[0, 1]] : tensor<3x4xi32> into tensor<12xi32>
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [12], strides = [1] : tensor<12xi32> -> !flow.dispatch.tensor<writeonly:12xi32>
  return
}
//      CHECK: func @reshape_fused_target()
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:4x32x1024xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:4x1024x64xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:4x32x64xf32>
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
        %7 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1, 0], sizes = [%c1, %c32, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:4x32x1024xf32> -> tensor<?x?x1024xf32>
        %8 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0, %arg2], sizes = [%c1, 1024, %c32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:4x1024x64xf32> -> tensor<?x1024x?xf32>
        %9 = linalg.init_tensor [1, 32, 32] : tensor<1x32x32xf32>
        %10 = linalg.fill  {__internal_linalg_transform__ = "workgroup"} ins(%cst : f32) outs(%9 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>
        %11 = linalg.batch_matmul {__internal_linalg_transform__ = "workgroup", is_root_op} ins(%7, %8 : tensor<?x?x1024xf32>, tensor<?x1024x?xf32>) outs(%10 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>
        %12 = tensor.cast %11 : tensor<1x32x32xf32> to tensor<?x?x?xf32>
        flow.dispatch.tensor.store %12, %2, offsets = [%arg0, %arg1, %arg2], sizes = [%c1, %c32, %c32], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:4x32x64xf32>
      }
    }
  }
  return
}
//      CHECK: func @cast_followed_by_store()
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?x?xf32>{%dim0, %dim1}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:?x?xf32>{%dim2, %dim3}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?x?xf32>{%dim4, %dim5}
  %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?x?xf32>{%dim6, %dim7}
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
      %20 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32>{%dim0, %dim1} -> tensor<?x?xf32>
      %21 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32>{%dim2, %dim3} -> tensor<?x?xf32>
      %shape = linalg.init_tensor [%18, %19] : tensor<?x?xf32>
      %22:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%20, %21 : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%shape, %shape : tensor<?x?xf32>, tensor<?x?xf32>) {
        ^bb0(%arg2: f32, %arg3 : f32, %arg4 : f32, %arg5 : f32):  // no predecessors
          %23 = arith.mulf %arg2, %arg3 : f32
          %24 = arith.addf %arg2, %arg3 : f32
          linalg.yield %23, %24 : f32, f32
        } -> (tensor<?x?xf32>, tensor<?x?xf32>)
      flow.dispatch.tensor.store %22#0, %2, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%dim4, %dim5}
      flow.dispatch.tensor.store %22#1, %3, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%dim6, %dim7}
    }
  }
  return
}
//      CHECK: func @multi_result()
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
  %12 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c32) alignment(32) : !flow.dispatch.tensor<readonly:?x?x?xi32>{%6, %7, %8}
  %13 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c64) alignment(32) : !flow.dispatch.tensor<readonly:?x?x?xi32>{%9, %10, %11}
  %14 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:?x?x?xi32>{%9, %10, %8}
  %15 = flow.dispatch.tensor.load %13, offsets = [0, 0, 0], sizes = [%9, %10, %11], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:?x?x?xi32>{%9, %10, %11} -> tensor<?x?x?xi32>
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
        %25 = flow.dispatch.tensor.load %12, offsets = [%arg0, %arg1, %arg2], sizes = [%22, %23, %24], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:?x?x?xi32>{%6, %7, %8} -> tensor<?x?x?xi32>
        %26 = linalg.init_tensor [%22, %23] : tensor<?x?xi32>
        %27 = linalg.init_tensor [%22, %23, %24] : tensor<?x?x?xi32>
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
        flow.dispatch.tensor.store %28, %14, offsets = [%arg0, %arg1, %arg2], sizes = [%22, %23, %24], strides = [1, 1, 1] : tensor<?x?x?xi32> -> !flow.dispatch.tensor<writeonly:?x?x?xi32>{%9, %10, %8}
      }
    }
  }
  return
}
// CHECK-LABEL: func @unused_ins_operand()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(2)
//   CHECK-DAG:   %[[IN_VIEW:.+]] = flow.dispatch.tensor.load %[[IN]]
//  CHECK-DAG:    %[[OUT_VIEW:.+]] = flow.dispatch.tensor.load %[[OUT]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[IN_VIEW]] :
//  CHECK-SAME:     outs(%[[OUT_VIEW]] :

// -----

func.func @three_init_tensor_uses() {
  %c6400 = arith.constant 6400 : index
  %c64 = arith.constant 64 : index
  %c1654784 = arith.constant 1654784 : index
  %c1638400 = arith.constant 1638400 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 3.40282347E+38 : f32
  %cst_0 = arith.constant opaque<"elided_large_const", "0xDEADBEEF"> : tensor<64xf32>
  %cst_1 = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:6400x64xf32>
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c1638400) alignment(32) : !flow.dispatch.tensor<readonly:64x64xf32>
  %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c1654784) alignment(32) : !flow.dispatch.tensor<writeonly:6400x64xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
  scf.for %arg0 = %3 to %c6400 step %4 {
    %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
    %6 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
    scf.for %arg1 = %5 to %c64 step %6 {
      %7 = linalg.init_tensor [64, 64] : tensor<64x64xf32>
      %8 = tensor.extract_slice %cst_0[%arg1] [64] [1] : tensor<64xf32> to tensor<64xf32>
      %9 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:6400x64xf32> -> tensor<64x64xf32>
      %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:64x64xf32> -> tensor<64x64xf32>
      %11 = linalg.fill ins(%cst_1 : f32) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %12 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [8, 32, 0], [0, 0, 16]]>} ins(%9, %10 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%11 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %12 : tensor<64xf32>, tensor<64x64xf32>) outs(%7 : tensor<64x64xf32>) {
      ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
        %15 = arith.addf %arg2, %arg3 : f32
        linalg.yield %15 : f32
      } -> tensor<64x64xf32>
      %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<64x64xf32>) outs(%7 : tensor<64x64xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %15 = arith.cmpf olt, %arg2, %cst_1 : f32
        %16 = arith.select %15, %cst_1, %arg2 : f32
        %17 = arith.cmpf olt, %cst, %arg2 : f32
        %18 = arith.select %17, %cst, %16 : f32
        linalg.yield %18 : f32
      } -> tensor<64x64xf32>
      flow.dispatch.tensor.store %14, %2, offsets = [%arg0, %arg1], sizes = [64, 64], strides = [1, 1] : tensor<64x64xf32> -> !flow.dispatch.tensor<writeonly:6400x64xf32>
    }
  }
  return
}
// CHECK-LABEL: func @three_init_tensor_uses()
//       CHECK: %[[OUTPUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-NOT:   linalg.init_tensor
//       CHECK:   %[[LOAD:.+]] = flow.dispatch.tensor.load %[[OUTPUT]]
//   CHECK-NOT:   linalg.init_tensor
//       CHECK:   linalg.fill
//  CHECK-SAME:       outs(%[[LOAD]] :
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       outs(%[[MATMUL]] :
//       CHECK:   linalg.generic
//  CHECK-SAME:       outs(%[[GENERIC]] :

// -----

func.func @fill_matmul_exp() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c33 = arith.constant 33 : index
  %c49 = arith.constant 49 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:33x16xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<readonly:16x49xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:33x49xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_y]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_y]
  scf.for %arg0 = %3 to %c33 step %4 {
    %5 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_x]
    %6 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_x]
    scf.for %arg1 = %5 to %c49 step %6 {
      %7 = affine.min affine_map<(d0) -> (16, -d0 + 33)>(%arg0)
      %8 = affine.min affine_map<(d0) -> (16, -d0 + 49)>(%arg1)
      %9 = linalg.init_tensor [%7, %8] : tensor<?x?xf32>
      %10 = affine.min affine_map<(d0) -> (-d0 + 33, 16)>(%arg0)
      %11 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%10, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:33x16xf32> -> tensor<?x16xf32>
      %12 = affine.min affine_map<(d0) -> (-d0 + 49, 16)>(%arg1)
      %13 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [16, %12], strides = [1, 1] : !flow.dispatch.tensor<readonly:16x49xf32> -> tensor<16x?xf32>
      %14 = linalg.init_tensor [%10, %12] : tensor<?x?xf32>
      %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %16 = linalg.matmul ins(%11, %13 : tensor<?x16xf32>, tensor<16x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %18 = math.exp %arg2 : f32
        linalg.yield %18 : f32
      } -> tensor<?x?xf32>
      flow.dispatch.tensor.store %17, %2, offsets = [%arg0, %arg1], sizes = [%7, %8], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:33x49xf32>
    }
  }
  return
}
// CHECK-LABEL: func @fill_matmul_exp()
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul
//       CHECK:   linalg.generic
//  CHECK-SAME:       outs(%[[MATMUL]]

// -----

func @cumsum__2x2x2x2x2x2x2() {
  %cst = arith.constant dense<0.000000e+00> : tensor<2x2x2x2x2x2x2xf32>
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:3x2x2x2x2x2x2xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:2x2x2x2x2x2x2xf32>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0, 0, 0], sizes = [3, 2, 2, 2, 2, 2, 2], strides = [1, 1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:3x2x2x2x2x2x2xf32> -> tensor<3x2x2x2x2x2x2xf32>
  %3 = linalg.init_tensor [2] : tensor<2xf32>
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
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0, 0, 0, 0], sizes = [2, 2, 2, 2, 2, 2, 2], strides = [1, 1, 1, 1, 1, 1, 1] : tensor<2x2x2x2x2x2x2xf32> -> !flow.dispatch.tensor<writeonly:2x2x2x2x2x2x2xf32>
  return
}

// CHECK-LABEL: func @cumsum__2x2x2x2x2x2x2()
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[DST:.+]] = flow.dispatch.tensor.load {{.+}} !flow.dispatch.tensor<writeonly:2x2x2x2x2x2x2xf32> -> tensor<2x2x2x2x2x2x2xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[DST]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%[[FILL]]

// -----

func @reduce_window_max_4x6xf32() {
  %cst = arith.constant dense<0xFF800000> : tensor<2x2xf32>
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:2x4x6xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:2x2xf32>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 4, 6], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:2x4x6xf32> -> tensor<2x4x6xf32>
  %3 = linalg.init_tensor [2, 2, 3] : tensor<2x2x3xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d2, d0 * 2 + d3, d1 * 3 + d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%2, %3 : tensor<2x4x6xf32>, tensor<2x2x3xf32>) outs(%cst : tensor<2x2xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %5 = arith.maxf %arg0, %arg2 : f32
    linalg.yield %5 : f32
  } -> tensor<2x2xf32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !flow.dispatch.tensor<writeonly:2x2xf32>
  return
}
// CHECK-LABEL: func @reduce_window_max_4x6xf32()
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0xFF800000 : f32
//       CHECK:   %[[DST:.+]] = flow.dispatch.tensor.load {{.+}} !flow.dispatch.tensor<writeonly:2x2xf32> -> tensor<2x2xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[DST]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%[[FILL]]
