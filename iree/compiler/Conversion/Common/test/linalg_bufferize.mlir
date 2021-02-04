// RUN: iree-opt %s --iree-codegen-linalg-bufferize-llvm -canonicalize -cse -split-input-file | IreeFileCheck %s

func @tile_from_tensor_load() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %0 = hal.interface.binding.subspan @legacy_io::@TENSOR_LHS[%c0] : !flow.dispatch.input<?x?xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@TENSOR_RHS[%c0] : !flow.dispatch.input<?x?xf32>
  %2 = hal.interface.binding.subspan @legacy_io::@TENSOR_INIT[%c0] : !flow.dispatch.input<?x?xf32>
  %3 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : !flow.dispatch.output<?x?xf32>

  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %5 to %c2 step %c2 {
    scf.for %arg1 = %4 to %c4 step %c4 {
      %6 = flow.dispatch.input.load %0, offsets = [%arg0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.input<?x?xf32> -> tensor<1x3xf32>
      %7 = flow.dispatch.input.load %1, offsets = [%c0, %arg1], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.input<?x?xf32> -> tensor<3x1xf32>
      %8 = flow.dispatch.input.load %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.input<?x?xf32> -> tensor<1x1xf32>
      %9 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%8 : tensor<1x1xf32>) -> tensor<1x1xf32>
      flow.dispatch.output.store %9, %3, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.output<?x?xf32>
    }
  }
  return
}

hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @tile_from_tensor_load()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan @legacy_io::@TENSOR_LHS
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan @legacy_io::@TENSOR_RHS
//   CHECK-DAG:   %[[TENSOR_INIT:.+]] = hal.interface.binding.subspan @legacy_io::@TENSOR_INIT
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @legacy_io::@ret0
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS:.+]] = subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//   CHECK-DAG:       %[[INIT:.+]] = subview %[[TENSOR_INIT]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//   CHECK-DAG:       %[[RESULT:.+]] = subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.copy(%[[INIT]], %[[RESULT]])
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS]], %[[RHS]]
//  CHECK-SAME:         outs(%[[RESULT]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func @tile_from_pointwise_lhs() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %0 = hal.interface.binding.subspan @legacy_io::@TENSOR_LHS[%c0] : !flow.dispatch.input<?x?xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@TENSOR_RHS[%c0] : !flow.dispatch.input<?x?xf32>
  %2 = hal.interface.binding.subspan @legacy_io::@TENSOR_INIT[%c0] : !flow.dispatch.input<?x?xf32>
  %3 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : !flow.dispatch.output<?x?xf32>
  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %5 to %c2 step %c2 {
    scf.for %arg1 = %4 to %c4 step %c4 {
      %6 = flow.dispatch.input.load %0, offsets = [%arg0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.input<?x?xf32> -> tensor<1x3xf32>
      %7 = flow.dispatch.input.load %1, offsets = [%c0, %arg1], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.input<?x?xf32> -> tensor<3x1xf32>
      %shape = linalg.init_tensor [1, 3] : tensor<1x3xf32>
      %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%6 : tensor<1x3xf32>) outs(%shape : tensor<1x3xf32>) {
        ^bb0(%arg2: f32, %s: f32):  // no predecessors
          linalg.yield %arg2 : f32
        } -> tensor<1x3xf32>
      %9 = flow.dispatch.input.load %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.input<?x?xf32> -> tensor<1x1xf32>
      %10 = linalg.matmul ins(%8, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%9 : tensor<1x1xf32>) -> tensor<1x1xf32>
      flow.dispatch.output.store %10, %3, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.output<?x?xf32>
    }
  }
  return
}

hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @tile_from_pointwise_lhs()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan @legacy_io::@TENSOR_LHS
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan @legacy_io::@TENSOR_RHS
//   CHECK-DAG:   %[[TENSOR_INIT:.+]] = hal.interface.binding.subspan @legacy_io::@TENSOR_INIT
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @legacy_io::@ret0
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS:.+]] = subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//       CHECK:       %[[ALLOC:.+]] = alloc() : memref<1x3xf32>
//       CHECK:       linalg.generic
//  CHECK-SAME:         ins(%[[LHS]] :
//  CHECK-SAME:         outs(%[[ALLOC]]
//   CHECK-DAG:       %[[INIT:.+]] = subview %[[TENSOR_INIT]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//   CHECK-DAG:       %[[RESULT:.+]] = subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.copy(%[[INIT]], %[[RESULT]])
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[ALLOC]], %[[RHS]]
//  CHECK-SAME:         outs(%[[RESULT]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func @tile_from_pointwise_outs() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %0 = hal.interface.binding.subspan @legacy_io::@TENSOR_LHS[%c0] : !flow.dispatch.input<?x?xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@TENSOR_RHS[%c0] : !flow.dispatch.input<?x?xf32>
  %2 = hal.interface.binding.subspan @legacy_io::@TENSOR_INIT[%c0] : !flow.dispatch.input<?x?xf32>
  %3 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : !flow.dispatch.output<?x?xf32>
  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %5 to %c2 step %c2 {
    scf.for %arg1 = %4 to %c4 step %c4 {
      %6 = flow.dispatch.input.load %0, offsets = [%arg0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.input<?x?xf32> -> tensor<1x3xf32>
      %7 = flow.dispatch.input.load %1, offsets = [%c0, %arg1], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.input<?x?xf32> -> tensor<3x1xf32>
      %8 = flow.dispatch.input.load %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.input<?x?xf32> -> tensor<1x1xf32>
      %shape = linalg.init_tensor [1, 1] : tensor<1x1xf32>
      %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%8 : tensor<1x1xf32>) outs(%shape : tensor<1x1xf32>) {
        ^bb0(%arg2: f32, %s: f32):  // no predecessors
          linalg.yield %arg2 : f32
        } -> tensor<1x1xf32>
      %10 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%9 : tensor<1x1xf32>)  -> tensor<1x1xf32>
      flow.dispatch.output.store %10, %3, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.output<?x?xf32>
    }
  }
  return
}
hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @tile_from_pointwise_outs()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan @legacy_io::@TENSOR_LHS
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan @legacy_io::@TENSOR_RHS
//   CHECK-DAG:   %[[TENSOR_INIT:.+]] = hal.interface.binding.subspan @legacy_io::@TENSOR_INIT
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @legacy_io::@ret0
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS:.+]] = subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//   CHECK-DAG:       %[[INIT:.+]] = subview %[[TENSOR_INIT]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       %[[ALLOC:.+]] = alloc() : memref<1x1xf32>
//       CHECK:       linalg.generic
//  CHECK-SAME:         ins(%[[INIT]] :
//  CHECK-SAME:         outs(%[[ALLOC]]
//   CHECK-DAG:       %[[RESULT:.+]] = subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS]], %[[RHS]]
//  CHECK-SAME:         outs(%[[ALLOC]]
//       CHECK:       linalg.copy(%[[ALLOC]], %[[RESULT]])

// -----

func @bufferize_dynamic() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : !flow.dispatch.input<?x?xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : !flow.dispatch.input<?x?xf32>
  %2 = hal.interface.binding.subspan @legacy_io::@arg2[%c0] : !flow.dispatch.input<?x?xf32>
  %3 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : !flow.dispatch.output<?x?xf32>
  %4 = hal.interface.load.constant offset = 0 : index
  %5 = hal.interface.load.constant offset = 1 : index
  %6 = hal.interface.load.constant offset = 2 : index
  %7 = hal.interface.load.constant offset = 3 : index
  %8 = hal.interface.load.constant offset = 4 : index
  %9 = hal.interface.load.constant offset = 5 : index
  %10 = hal.interface.load.constant offset = 6 : index
  %11 = hal.interface.load.constant offset = 7 : index
  %12 = shapex.make_ranked_shape %4, %5 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %13 = flow.dispatch.tie_shape %0, %12 : (!flow.dispatch.input<?x?xf32>, !shapex.ranked_shape<[?,?]>) -> !flow.dispatch.input<?x?xf32>
  %14 = shapex.make_ranked_shape %6, %7 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %15 = flow.dispatch.tie_shape %1, %14 : (!flow.dispatch.input<?x?xf32>, !shapex.ranked_shape<[?,?]>) -> !flow.dispatch.input<?x?xf32>
  %16 = shapex.make_ranked_shape %8, %9 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %17 = flow.dispatch.tie_shape %2, %16 : (!flow.dispatch.input<?x?xf32>, !shapex.ranked_shape<[?,?]>) -> !flow.dispatch.input<?x?xf32>
  %18 = shapex.make_ranked_shape %10, %11 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %19 = flow.dispatch.tie_shape %3, %18 : (!flow.dispatch.output<?x?xf32>, !shapex.ranked_shape<[?,?]>) -> !flow.dispatch.output<?x?xf32>
  %workgroup_size_x = hal.interface.workgroup.size[0] : index
  %workgroup_size_y = hal.interface.workgroup.size[1] : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %20 = muli %workgroup_size_y, %workgroup_id_y : index
  %21 = muli %workgroup_size_y, %workgroup_count_y : index
  scf.for %arg0 = %20 to %4 step %21 {
    %22 = muli %workgroup_size_x, %workgroup_id_x : index
    %23 = muli %workgroup_size_x, %workgroup_count_x : index
    scf.for %arg1 = %22 to %7 step %23 {
      %24 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg0)[%4, %workgroup_size_y]
      %25 = flow.dispatch.input.load %13, offsets = [%arg0, %c0], sizes = [%24, %5], strides = [%c1, %c1] : !flow.dispatch.input<?x?xf32> -> tensor<?x?xf32>
      %26 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg1)[%7, %workgroup_size_x]
      %27 = flow.dispatch.input.load %15, offsets = [%c0, %arg1], sizes = [%6, %26], strides = [%c1, %c1] : !flow.dispatch.input<?x?xf32> -> tensor<?x?xf32>
      %28 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %8]
      %29 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %9]
      %30 = flow.dispatch.input.load %17, offsets = [%arg0, %arg1], sizes = [%28, %29], strides = [%c1, %c1] : !flow.dispatch.input<?x?xf32> -> tensor<?x?xf32>
      %31 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%25, %27 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%30 : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.output.store %31, %19, offsets = [%arg0, %arg1], sizes = [%28, %29], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.output<?x?xf32>
    }
  }
  return
}
hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @arg2, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
//       CHECK: func @bufferize_dynamic()
//   CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan @legacy_io::@arg0
//   CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan @legacy_io::@arg1
//   CHECK-DAG:   %[[INIT:.+]] = hal.interface.binding.subspan @legacy_io::@arg2
//   CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan @legacy_io::@ret0
//   CHECK-DAG:   %[[DIM0:.+]] = hal.interface.load.constant offset = 0 : index
//   CHECK-DAG:   %[[DIM1:.+]] = hal.interface.load.constant offset = 1 : index
//   CHECK-DAG:   %[[DIM2:.+]] = hal.interface.load.constant offset = 2 : index
//   CHECK-DAG:   %[[DIM3:.+]] = hal.interface.load.constant offset = 3 : index
//   CHECK-DAG:   %[[DIM4:.+]] = hal.interface.load.constant offset = 4 : index
//   CHECK-DAG:   %[[DIM5:.+]] = hal.interface.load.constant offset = 5 : index
//   CHECK-DAG:   %[[DIM6:.+]] = hal.interface.load.constant offset = 6 : index
//   CHECK-DAG:   %[[DIM7:.+]] = hal.interface.load.constant offset = 7 : index
//       CHECK:   %[[SHAPE_LHS:.+]] = shapex.make_ranked_shape %[[DIM0]], %[[DIM1]]
//       CHECK:   %[[LHS_SHAPED:.+]] = shapex.tie_shape %[[LHS]], %[[SHAPE_LHS]]
//       CHECK:   %[[SHAPE_RHS:.+]] = shapex.make_ranked_shape %[[DIM2]], %[[DIM3]]
//       CHECK:   %[[RHS_SHAPED:.+]] = shapex.tie_shape %[[RHS]], %[[SHAPE_RHS]]
//       CHECK:   %[[SHAPE_INIT:.+]] = shapex.make_ranked_shape %[[DIM4]], %[[DIM5]]
//       CHECK:   %[[INIT_SHAPED:.+]] = shapex.tie_shape %[[INIT]], %[[SHAPE_INIT]]
//       CHECK:   %[[SHAPE_RESULT:.+]] = shapex.make_ranked_shape %[[DIM6]], %[[DIM7]]
//       CHECK:   %[[RESULT_SHAPED:.+]] = shapex.tie_shape %[[RESULT]], %[[SHAPE_RESULT]]
//   CHECK-DAG:   %[[WGSIZE_X:.+]] = hal.interface.workgroup.size[0]
//   CHECK-DAG:   %[[WGSIZE_Y:.+]] = hal.interface.workgroup.size[1]
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//       CHECK:       %[[TILE_M:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[DIM0]], %[[WGSIZE_Y]]]
//       CHECK:       %[[LHS_TILE:.+]] = subview %[[LHS_SHAPED]][%[[IV0]], 0] [%[[TILE_M]], %[[DIM1]]]
//       CHECK:       %[[TILE_N:.+]] = affine.min #[[MAP0]](%[[IV1]])[%[[DIM3]], %[[WGSIZE_X]]]
//   CHECK-DAG:       %[[RHS_TILE:.+]] = subview %[[RHS_SHAPED]][0, %[[IV1]]] [%[[DIM2]], %[[TILE_N]]]
//       CHECK:       %[[TILE_M_2:.+]] = affine.min #[[MAP2]](%[[IV0]])[%[[WGSIZE_Y]], %[[DIM4]]]
//       CHECK:       %[[TILE_N_2:.+]] = affine.min #[[MAP2]](%[[IV1]])[%[[WGSIZE_X]], %[[DIM5]]]
//   CHECK-DAG:       %[[INIT_TILE:.+]] = subview %[[INIT_SHAPED]][%[[IV0]], %[[IV1]]] [%[[TILE_M_2]], %[[TILE_N_2]]]
//   CHECK-DAG:       %[[RESULT_TILE:.+]] = subview %[[RESULT_SHAPED]][%[[IV0]], %[[IV1]]] [%[[TILE_M_2]], %[[TILE_N_2]]]
//       CHECK:       linalg.copy(%[[INIT_TILE]], %[[RESULT_TILE]])
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS_TILE]], %[[RHS_TILE]]
//  CHECK-SAME:         outs(%[[RESULT_TILE]]

// -----

// TODO(GH-4734): Enable after fixing the allocation for vector.transfer_writes.
// #map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
// #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// module  {
//   func @bufferize_transfer_op() {
//     %c3 = constant 3 : index
//     %cst = constant 0.000000e+00 : f32
//     %c0 = constant 0 : index
//     %c2 = constant 2 : index
//     %c1 = constant 1 : index
//     %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : !flow.dispatch.input<2x3xf32>
//     %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : !flow.dispatch.input<3x4xf32>
//     %2 = hal.interface.binding.subspan @legacy_io::@arg2[%c0] : !flow.dispatch.input<2x4xf32>
//     %3 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : !flow.dispatch.output<2x4xf32>
//     %4 = flow.dispatch.input.load %0, offsets = [%c0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.input<2x3xf32> -> tensor<2x3xf32>
//     %5 = flow.dispatch.input.load %1, offsets = [%c0, %c0], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.input<3x4xf32> -> tensor<3x1xf32>
//     %6 = flow.dispatch.input.load %2, offsets = [%c0, %c0], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.input<2x4xf32> -> tensor<2x1xf32>
//     %7 = vector.transfer_read %4[%c0, %c0], %cst {masked = [false, false]} : tensor<2x3xf32>, vector<1x1xf32>
//     %8 = vector.transfer_read %4[%c0, %c1], %cst {masked = [false, false]} : tensor<2x3xf32>, vector<1x1xf32>
//     %9 = vector.transfer_read %4[%c0, %c2], %cst {masked = [false, false]} : tensor<2x3xf32>, vector<1x1xf32>
//     %10 = vector.transfer_read %4[%c1, %c0], %cst {masked = [false, false]} : tensor<2x3xf32>, vector<1x1xf32>
//     %11 = vector.transfer_read %4[%c1, %c1], %cst {masked = [false, false]} : tensor<2x3xf32>, vector<1x1xf32>
//     %12 = vector.transfer_read %4[%c1, %c2], %cst {masked = [false, false]} : tensor<2x3xf32>, vector<1x1xf32>
//     %13 = vector.transfer_read %5[%c0, %c0], %cst {masked = [false, false]} : tensor<3x1xf32>, vector<1x1xf32>
//     %14 = vector.transfer_read %5[%c1, %c0], %cst {masked = [false, false]} : tensor<3x1xf32>, vector<1x1xf32>
//     %15 = vector.transfer_read %5[%c2, %c0], %cst {masked = [false, false]} : tensor<3x1xf32>, vector<1x1xf32>
//     %16 = vector.transfer_read %6[%c0, %c0], %cst {masked = [false, false]} : tensor<2x1xf32>, vector<1x1xf32>
//     %17 = vector.transfer_read %6[%c1, %c0], %cst {masked = [false, false]} : tensor<2x1xf32>, vector<1x1xf32>
//     %18 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %7, %13, %16 : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
//     %19 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %8, %14, %18 : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
//     %20 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %9, %15, %19 : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
//     %21 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %10, %13, %17 : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
//     %22 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %11, %14, %21 : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
//     %23 = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} %12, %15, %22 : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
//     %24 = vector.transfer_write %20, %6[%c0, %c0] {masked = [false, false]} : vector<1x1xf32>, tensor<2x1xf32>
//     %25 = vector.transfer_write %23, %24[%c1, %c0] {masked = [false, false]} : vector<1x1xf32>, tensor<2x1xf32>
//     flow.dispatch.output.store %25, %3, offsets = [%c0, %c0], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<2x1xf32> -> !flow.dispatch.output<2x4xf32>
//     return
//   }
// }
