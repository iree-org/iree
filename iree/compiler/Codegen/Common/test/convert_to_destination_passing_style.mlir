// RUN: iree-opt %s -iree-codegen-convert-to-destination-passing-style -canonicalize -cse -split-input-file | IreeFileCheck %s

func @matmul() {
  %c0 = arith.constant 0 : index
  %m = hal.interface.load.constant offset = 0 : index
  %n = hal.interface.load.constant offset = 1 : index
  %k = hal.interface.load.constant offset = 2 : index
  %lhs = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %k}
  %rhs = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>{%k, %n}
  %init = hal.interface.binding.subspan @io::@arg2[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %n}
  %result = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>{%m, %n}
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
      %lhs_tile = flow.dispatch.tensor.load %lhs, offsets = [%iv0, 0], sizes = [%tilesize_y, %k], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %rhs_tile = flow.dispatch.tensor.load %rhs, offsets = [0, %iv1], sizes = [%k, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %init_tile = flow.dispatch.tensor.load %init, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %matmul_tile = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
    }
  }
  return
}
hal.interface private @io  {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @arg2, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
//      CHECK: func @matmul()
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan @io::@arg0
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan @io::@arg1
//  CHECK-DAG:   %[[INIT:.+]] = hal.interface.binding.subspan @io::@arg2
//  CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan @io::@ret0
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

func @matmul_fill() {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %m = hal.interface.load.constant offset = 0 : index
  %n = hal.interface.load.constant offset = 1 : index
  %k = hal.interface.load.constant offset = 2 : index
  %lhs = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %k}
  %rhs = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>{%k, %n}
  %result = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>{%m, %n}
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
      %lhs_tile = flow.dispatch.tensor.load %lhs, offsets = [%iv0, 0], sizes = [%tilesize_y, %k], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %rhs_tile = flow.dispatch.tensor.load %rhs, offsets = [0, %iv1], sizes = [%k, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %init_tile = linalg.init_tensor [%tilesize_y, %tilesize_x] : tensor<?x?xf32>
      %fill_tile = linalg.fill(%cst, %init_tile) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
      %matmul_tile = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
    }
  }
  return
}
hal.interface private @io  {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}
//      CHECK: func @matmul_fill()
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan @io::@arg0
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan @io::@arg1
//  CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan @io::@ret0
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//  CHECK-DAG:       %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[LHS]]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[RHS]]
//  CHECK-DAG:       %[[RESULT_TILE:.+]] = flow.dispatch.tensor.load %[[RESULT]]
//      CHECK:       %[[FILL_TILE:.+]] = linalg.fill(%{{.+}}, %[[RESULT_TILE]])
//      CHECK:       %[[MATMUL_TILE:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:           outs(%[[FILL_TILE]] : tensor<?x?xf32>)
//      CHECK:       flow.dispatch.tensor.store %[[MATMUL_TILE]], %[[RESULT]]

// -----

func @matmul_inplace() {
  %c0 = arith.constant 0 : index
  %m = hal.interface.load.constant offset = 0 : index
  %n = hal.interface.load.constant offset = 1 : index
  %k = hal.interface.load.constant offset = 2 : index
  %lhs = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>{%m, %k}
  %rhs = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>{%k, %n}
  %result = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<readwrite:?x?xf32>{%m, %n}
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
      %lhs_tile = flow.dispatch.tensor.load %lhs, offsets = [%iv0, 0], sizes = [%tilesize_y, %k], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %rhs_tile = flow.dispatch.tensor.load %rhs, offsets = [0, %iv1], sizes = [%k, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %init_tile = flow.dispatch.tensor.load %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : !flow.dispatch.tensor<readwrite:?x?xf32> -> tensor<?x?xf32>
      %matmul_tile = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %matmul_tile, %result, offsets = [%iv0, %iv1], sizes = [%tilesize_y, %tilesize_x], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>
    }
  }
  return
}
hal.interface private @io  {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Read|Write"
}
//      CHECK: func @matmul_inplace()
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan @io::@arg0
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan @io::@arg1
//  CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan @io::@ret0
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

func @reshape_simple() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:12xi32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:3x4xi32>
  %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:12xi32> -> tensor<12xi32>
  %3 = linalg.tensor_expand_shape %2 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
  flow.dispatch.tensor.store %3, %1, offsets = [], sizes = [], strides = [] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  return
}
hal.interface private @io  {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
//      CHECK: func @reshape_simple()
//  CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//  CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//      CHECK:   %[[SOURCE:.+]] = flow.dispatch.tensor.load %[[ARG0]]
//      CHECK:   %[[RESHAPE:.+]] = linalg.tensor_expand_shape %[[SOURCE]]
//      CHECK:   flow.dispatch.tensor.store %[[RESHAPE]], %[[RET0]]

// -----

func @reshape_fused_source() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:12xi32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:3x4xi32>
  %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:12xi32> -> tensor<12xi32>
  %3 = linalg.tensor_expand_shape %2 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
  %4 = linalg.init_tensor [3, 4] : tensor<3x4xi32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%3 : tensor<3x4xi32>) outs(%4 : tensor<3x4xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %6 = arith.addi %arg0, %arg0 : i32
      linalg.yield %6 : i32
    } -> tensor<3x4xi32>
  flow.dispatch.tensor.store %5, %1, offsets = [], sizes = [], strides = [] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  return
}
hal.interface private @io  {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
//      CHECK: func @reshape_fused_source()
//  CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//  CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//      CHECK:   %[[SOURCE:.+]] = flow.dispatch.tensor.load %[[ARG0]]
//      CHECK:   %[[RESHAPE:.+]] = linalg.tensor_expand_shape %[[SOURCE]]
//      CHECK:   %[[TARGET:.+]] = flow.dispatch.tensor.load %[[RET0]]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[RESHAPE]]
// CHECK-SAME:       outs(%[[TARGET]]
//      CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[RET0]]

// -----

func @reshape_fused_source_and_copyout() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:12xi32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:3x4xi32>
  %2 = hal.interface.binding.subspan @io::@ret1[%c0] : !flow.dispatch.tensor<writeonly:3x4xi32>
  %3 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:12xi32> -> tensor<12xi32>
  %4 = linalg.tensor_expand_shape %3 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
  %5 = linalg.init_tensor [3, 4] : tensor<3x4xi32>
  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%4 : tensor<3x4xi32>) outs(%5 : tensor<3x4xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %7 = arith.addi %arg0, %arg0 : i32
      linalg.yield %7 : i32
    } -> tensor<3x4xi32>
  flow.dispatch.tensor.store %6, %1, offsets = [], sizes = [], strides = [] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  flow.dispatch.tensor.store %4, %2, offsets = [], sizes = [], strides = [] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  return
}
hal.interface private @io  {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  hal.interface.binding @ret1, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}
//      CHECK: func @reshape_fused_source_and_copyout()
//  CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//  CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//  CHECK-DAG:   %[[RET1:.+]] = hal.interface.binding.subspan @io::@ret1
//      CHECK:   %[[SOURCE:.+]] = flow.dispatch.tensor.load %[[ARG0]]
//      CHECK:   %[[RESHAPE:.+]] = linalg.tensor_expand_shape %[[SOURCE]]
//      CHECK:   %[[TARGET:.+]] = flow.dispatch.tensor.load %[[RET0]]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[RESHAPE]]
// CHECK-SAME:       outs(%[[TARGET]]
//      CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[RET0]]
//      CHECK:   flow.dispatch.tensor.store %[[RESHAPE]], %[[RET1]]

// -----

func @reshape_fused_target() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:3x4xi32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:12xi32>
  %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:3x4xi32> -> tensor<3x4xi32>
  %3 = linalg.init_tensor [3, 4] : tensor<3x4xi32>
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%2 : tensor<3x4xi32>) outs(%3 : tensor<3x4xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %5 = arith.addi %arg0, %arg0 : i32
      linalg.yield %5 : i32
    } -> tensor<3x4xi32>
  %5 = linalg.tensor_collapse_shape %4 [[0, 1]] : tensor<3x4xi32> into tensor<12xi32>
  flow.dispatch.tensor.store %5, %1, offsets = [], sizes = [], strides = [] : tensor<12xi32> -> !flow.dispatch.tensor<writeonly:12xi32>
  return
}
hal.interface private @io  {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
//      CHECK: func @reshape_fused_target()
//  CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//  CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//  CHECK-DAG:   %[[SOURCE:.+]] = flow.dispatch.tensor.load %[[ARG0]]
//  CHECK-DAG:   %[[TARGET:.+]] = flow.dispatch.tensor.load %[[RET0]]
//      CHECK:   %[[RESHAPE_EXPAND:.+]] = linalg.tensor_expand_shape %[[TARGET]] {{\[}}[0, 1]{{\]}}
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[SOURCE]]
// CHECK-SAME:       outs(%[[RESHAPE_EXPAND]]
//      CHECK:   %[[RESHAPE_COLLAPSE:.+]] = linalg.tensor_collapse_shape %[[GENERIC]] {{\[}}[0, 1]{{\]}}
//      CHECK:   flow.dispatch.tensor.store %[[RESHAPE_COLLAPSE]], %[[RET0]]

// -----

func @cast_followed_by_store() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:4x32x1024xf32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:4x1024x64xf32>
  %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:4x32x64xf32>
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
        %10 = linalg.fill(%cst, %9) {__internal_linalg_transform__ = "workgroup"} : f32, tensor<1x32x32xf32> -> tensor<1x32x32xf32>
        %11 = linalg.batch_matmul {__internal_linalg_transform__ = "workgroup", is_root_op} ins(%7, %8 : tensor<?x?x1024xf32>, tensor<?x1024x?xf32>) outs(%10 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>
        %12 = tensor.cast %11 : tensor<1x32x32xf32> to tensor<?x?x?xf32>
        flow.dispatch.tensor.store %12, %2, offsets = [%arg0, %arg1, %arg2], sizes = [%c1, %c32, %c32], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:4x32x64xf32>
      }
    }
  }
  return
}
hal.interface private @io  {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  hal.interface.binding @ret1, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}
//      CHECK: func @cast_followed_by_store()
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan @io::@arg0
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan @io::@arg1
//  CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan @io::@ret0
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//  CHECK-DAG:       %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[LHS]]
//  CHECK-DAG:       %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[RHS]]
//  CHECK-DAG:       %[[RESULT_TILE:.+]] = flow.dispatch.tensor.load %[[RESULT]]
//      CHECK:       %[[FILL_TILE:.+]] = linalg.fill(%{{.+}}, %[[RESULT_TILE]])
//      CHECK:       %[[MATMUL_TILE:.+]] = linalg.batch_matmul
// CHECK-SAME:           ins(%[[LHS_TILE]], %[[RHS_TILE]]
// CHECK-SAME:           outs(%[[FILL_TILE]]
//      CHECK:       flow.dispatch.tensor.store %[[MATMUL_TILE]], %[[RESULT]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func @multi_result() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %dim0 = hal.interface.load.constant offset = 0 : index
  %dim1 = hal.interface.load.constant offset = 1 : index
  %dim2 = hal.interface.load.constant offset = 2 : index
  %dim3 = hal.interface.load.constant offset = 3 : index
  %dim4 = hal.interface.load.constant offset = 4 : index
  %dim5 = hal.interface.load.constant offset = 5 : index
  %dim6 = hal.interface.load.constant offset = 6 : index
  %dim7 = hal.interface.load.constant offset = 7 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>{%dim0, %dim1}
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>{%dim2, %dim3}
  %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>{%dim4, %dim5}
  %3 = hal.interface.binding.subspan @io::@ret1[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>{%dim6, %dim7}
  %4 = hal.interface.load.constant offset = 8 : index
  %5 = hal.interface.load.constant offset = 9 : index
  %6 = hal.interface.load.constant offset = 10 : index
  %7 = hal.interface.load.constant offset = 11 : index
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
      %20 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %21 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %shape = linalg.init_tensor [%18, %19] : tensor<?x?xf32>
      %22:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%20, %21 : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%shape, %shape : tensor<?x?xf32>, tensor<?x?xf32>) {
        ^bb0(%arg2: f32, %arg3 : f32, %arg4 : f32, %arg5 : f32):  // no predecessors
          %23 = arith.mulf %arg2, %arg3 : f32
          %24 = arith.addf %arg2, %arg3 : f32
          linalg.yield %23, %24 : f32, f32
        } -> (tensor<?x?xf32>, tensor<?x?xf32>)
      flow.dispatch.tensor.store %22#0, %2, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
      flow.dispatch.tensor.store %22#1, %3, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
    }
  }
  return
}
hal.interface private @io  {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  hal.interface.binding @ret1, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
//      CHECK: func @multi_result()
//  CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan @io::@arg0
//  CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan @io::@arg1
//  CHECK-DAG:   %[[RESULT0:.+]] = hal.interface.binding.subspan @io::@ret0
//  CHECK-DAG:   %[[RESULT1:.+]] = hal.interface.binding.subspan @io::@ret1
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
