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

// TODO(GH-????): Enable after fixing the allocation for vector.transfer_writes.
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
