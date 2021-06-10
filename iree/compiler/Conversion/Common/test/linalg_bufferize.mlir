// RUN: iree-opt %s --iree-codegen-linalg-bufferize -canonicalize -cse -split-input-file | IreeFileCheck %s

func @tile_from_tensor_load() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %0 = hal.interface.binding.subspan @io::@TENSOR_LHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@TENSOR_RHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@TENSOR_INIT[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %3 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>
  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %5 to %c2 step %c2 {
    scf.for %arg1 = %4 to %c4 step %c4 {
      %6 = flow.dispatch.tensor.load %0, offsets = [%arg0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x3xf32>
      %7 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<3x1xf32>
      %8 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x1xf32>
      %9 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%8 : tensor<1x1xf32>) -> tensor<1x1xf32>
      flow.dispatch.tensor.store %9, %3, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
    }
  }
  return
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @tile_from_tensor_load()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_LHS
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_RHS
//   CHECK-DAG:   %[[TENSOR_INIT:.+]] = hal.interface.binding.subspan @io::@TENSOR_INIT
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS:.+]] = memref.subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = memref.subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//   CHECK-DAG:       %[[INIT:.+]] = memref.subview %[[TENSOR_INIT]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//   CHECK-DAG:       %[[RESULT:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.copy(%[[INIT]], %[[RESULT]])
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS]], %[[RHS]]
//  CHECK-SAME:         outs(%[[RESULT]]

// -----

func @tile_from_tensor_load_inplace() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %0 = hal.interface.binding.subspan @io::@TENSOR_LHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@TENSOR_RHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@TENSOR_INIT[%c0] : !flow.dispatch.tensor<readwrite:?x?xf32>
  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %5 to %c2 step %c2 {
    scf.for %arg1 = %4 to %c4 step %c4 {
      %6 = flow.dispatch.tensor.load %0, offsets = [%arg0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x3xf32>
      %7 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<3x1xf32>
      %8 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readwrite:?x?xf32> -> tensor<1x1xf32>
      %9 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%8 : tensor<1x1xf32>) -> tensor<1x1xf32>
      flow.dispatch.tensor.store %9, %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>
    }
  }
  return
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read|Write"
}
// CHECK-LABEL: func @tile_from_tensor_load_inplace()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_LHS
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_RHS
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @io::@TENSOR_INIT
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS:.+]] = memref.subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = memref.subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//   CHECK-DAG:       %[[RESULT:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS]], %[[RHS]]
//  CHECK-SAME:         outs(%[[RESULT]]

// -----

func @tile_from_tensor_load_inplace_and_copy() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %0 = hal.interface.binding.subspan @io::@TENSOR_LHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@TENSOR_RHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@TENSOR_INIT[%c0] : !flow.dispatch.tensor<readwrite:?x?xf32>
  %3 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>
  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %5 to %c2 step %c2 {
    scf.for %arg1 = %4 to %c4 step %c4 {
      %6 = flow.dispatch.tensor.load %0, offsets = [%arg0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x3xf32>
      %7 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<3x1xf32>
      %8 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readwrite:?x?xf32> -> tensor<1x1xf32>
      %9 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%8 : tensor<1x1xf32>) -> tensor<1x1xf32>
      flow.dispatch.tensor.store %9, %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>
      flow.dispatch.tensor.store %9, %3, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
    }
  }
  return
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read|Write"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @tile_from_tensor_load_inplace_and_copy()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_LHS
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_RHS
//   CHECK-DAG:   %[[RETURN1:.+]] = hal.interface.binding.subspan @io::@TENSOR_INIT
//   CHECK-DAG:   %[[RETURN2:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS:.+]] = memref.subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = memref.subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//   CHECK-DAG:       %[[RESULT1:.+]] = memref.subview %[[RETURN1]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS]], %[[RHS]]
//  CHECK-SAME:         outs(%[[RESULT1]]
//       CHECK:       %[[RESULT2:.+]] = memref.subview %[[RETURN2]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.copy(%[[RESULT1]], %[[RESULT2]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func @tile_from_pointwise_lhs() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %0 = hal.interface.binding.subspan @io::@TENSOR_LHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@TENSOR_RHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@TENSOR_INIT[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %3 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>
  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %5 to %c2 step %c2 {
    scf.for %arg1 = %4 to %c4 step %c4 {
      %6 = flow.dispatch.tensor.load %0, offsets = [%arg0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x3xf32>
      %7 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<3x1xf32>
      %shape = linalg.init_tensor [1, 3] : tensor<1x3xf32>
      %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%6 : tensor<1x3xf32>) outs(%shape : tensor<1x3xf32>) {
        ^bb0(%arg2: f32, %s: f32):  // no predecessors
          linalg.yield %arg2 : f32
        } -> tensor<1x3xf32>
      %9 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x1xf32>
      %10 = linalg.matmul ins(%8, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%9 : tensor<1x1xf32>) -> tensor<1x1xf32>
      flow.dispatch.tensor.store %10, %3, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
    }
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @tile_from_pointwise_lhs()
//       CHECK:       %[[ALLOC:.+]] = memref.alloc() : memref<1x3xf32>
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_LHS
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_RHS
//   CHECK-DAG:   %[[TENSOR_INIT:.+]] = hal.interface.binding.subspan @io::@TENSOR_INIT
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS:.+]] = memref.subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = memref.subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//       CHECK:       linalg.generic
//  CHECK-SAME:         ins(%[[LHS]] :
//  CHECK-SAME:         outs(%[[ALLOC]]
//   CHECK-DAG:       %[[INIT:.+]] = memref.subview %[[TENSOR_INIT]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//   CHECK-DAG:       %[[RESULT:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.copy(%[[INIT]], %[[RESULT]])
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[ALLOC]], %[[RHS]]
//  CHECK-SAME:         outs(%[[RESULT]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func @tile_from_pointwise_lhs_inplace() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %0 = hal.interface.binding.subspan @io::@TENSOR_LHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@TENSOR_RHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@TENSOR_INIT[%c0] : !flow.dispatch.tensor<readwrite:?x?xf32>
  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %5 to %c2 step %c2 {
    scf.for %arg1 = %4 to %c4 step %c4 {
      %6 = flow.dispatch.tensor.load %0, offsets = [%arg0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x3xf32>
      %7 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<3x1xf32>
      %shape = linalg.init_tensor [1, 3] : tensor<1x3xf32>
      %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%6 : tensor<1x3xf32>) outs(%shape : tensor<1x3xf32>) {
        ^bb0(%arg2: f32, %s: f32):  // no predecessors
          linalg.yield %arg2 : f32
        } -> tensor<1x3xf32>
      %9 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readwrite:?x?xf32> -> tensor<1x1xf32>
      %10 = linalg.matmul ins(%8, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%9 : tensor<1x1xf32>) -> tensor<1x1xf32>
      flow.dispatch.tensor.store %10, %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>
    }
  }
  return
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read|Write"
}
// CHECK-LABEL: func @tile_from_pointwise_lhs_inplace()
//       CHECK:       %[[ALLOC:.+]] = memref.alloc() : memref<1x3xf32>
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_LHS
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_RHS
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @io::@TENSOR_INIT
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS:.+]] = memref.subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = memref.subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//       CHECK:       linalg.generic
//  CHECK-SAME:         ins(%[[LHS]] :
//  CHECK-SAME:         outs(%[[ALLOC]]
//   CHECK-DAG:       %[[RESULT:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
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
  %0 = hal.interface.binding.subspan @io::@TENSOR_LHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@TENSOR_RHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@TENSOR_INIT[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %3 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>
  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %5 to %c2 step %c2 {
    scf.for %arg1 = %4 to %c4 step %c4 {
      %6 = flow.dispatch.tensor.load %0, offsets = [%arg0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x3xf32>
      %7 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<3x1xf32>
      %8 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x1xf32>
      %shape = linalg.init_tensor [1, 1] : tensor<1x1xf32>
      %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%8 : tensor<1x1xf32>) outs(%shape : tensor<1x1xf32>) {
        ^bb0(%arg2: f32, %s: f32):  // no predecessors
          linalg.yield %arg2 : f32
        } -> tensor<1x1xf32>
      %10 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%9 : tensor<1x1xf32>)  -> tensor<1x1xf32>
      flow.dispatch.tensor.store %10, %3, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
    }
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @tile_from_pointwise_outs()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_LHS
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_RHS
//   CHECK-DAG:   %[[TENSOR_INIT:.+]] = hal.interface.binding.subspan @io::@TENSOR_INIT
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @io::@ret0
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
func @tile_from_pointwise_outs_inplace() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %0 = hal.interface.binding.subspan @io::@TENSOR_LHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@TENSOR_RHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@TENSOR_INIT[%c0] : !flow.dispatch.tensor<readwrite:?x?xf32>
  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %5 to %c2 step %c2 {
    scf.for %arg1 = %4 to %c4 step %c4 {
      %6 = flow.dispatch.tensor.load %0, offsets = [%arg0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x3xf32>
      %7 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<3x1xf32>
      %8 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readwrite:?x?xf32> -> tensor<1x1xf32>
      %shape = linalg.init_tensor [1, 1] : tensor<1x1xf32>
      %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%8 : tensor<1x1xf32>) outs(%shape : tensor<1x1xf32>) {
        ^bb0(%arg2: f32, %s: f32):  // no predecessors
          linalg.yield %arg2 : f32
        } -> tensor<1x1xf32>
      %10 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%9 : tensor<1x1xf32>)  -> tensor<1x1xf32>
      flow.dispatch.tensor.store %10, %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>
    }
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read|Write"
}
// CHECK-LABEL: func @tile_from_pointwise_outs_inplace()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_LHS
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_RHS
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @io::@TENSOR_INIT
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

#map = affine_map<(d0, d1) -> (d0, d1)>
func @tile_from_matmul_outs() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %0 = hal.interface.binding.subspan @io::@TENSOR_LHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@TENSOR_RHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@TENSOR_INIT[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %3 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>
  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %5 to %c2 step %c2 {
    scf.for %arg1 = %4 to %c4 step %c4 {
      %6 = flow.dispatch.tensor.load %0, offsets = [%arg0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x3xf32>
      %7 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<3x1xf32>
      %8 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x1xf32>
      %shape = linalg.init_tensor [1, 1] : tensor<1x1xf32>
      %9 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%8 : tensor<1x1xf32>)  -> tensor<1x1xf32>
      %10 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%9 : tensor<1x1xf32>)  -> tensor<1x1xf32>
      flow.dispatch.tensor.store %10, %3, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
    }
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @tile_from_matmul_outs()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_LHS
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_RHS
//   CHECK-DAG:   %[[TENSOR_INIT:.+]] = hal.interface.binding.subspan @io::@TENSOR_INIT
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS:.+]] = memref.subview %[[TENSOR_LHS]][%[[IV0]], 0] [1, 3] [1, 1]
//   CHECK-DAG:       %[[RHS:.+]] = memref.subview %[[TENSOR_RHS]][0, %[[IV1]]] [3, 1] [1, 1]
//   CHECK-DAG:       %[[INIT:.+]] = memref.subview %[[TENSOR_INIT]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//   CHECK-DAG:       %[[RESULT:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1]
//       CHECK:       linalg.copy(%[[INIT]], %[[RESULT]])
//       CHECK:       linalg.matmul
//  CHECK-SAME:         outs(%[[RESULT]]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         outs(%[[RESULT]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func @tile_from_matmul_outs_inplace() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %0 = hal.interface.binding.subspan @io::@TENSOR_LHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@TENSOR_RHS[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@TENSOR_INIT[%c0] : !flow.dispatch.tensor<readwrite:?x?xf32>
  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %5 to %c2 step %c2 {
    scf.for %arg1 = %4 to %c4 step %c4 {
      %6 = flow.dispatch.tensor.load %0, offsets = [%arg0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<1x3xf32>
      %7 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<3x1xf32>
      %8 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readwrite:?x?xf32> -> tensor<1x1xf32>
      %9 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%8 : tensor<1x1xf32>)  -> tensor<1x1xf32>
      %10 = linalg.matmul ins(%6, %7 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%9 : tensor<1x1xf32>)  -> tensor<1x1xf32>
      flow.dispatch.tensor.store %10, %2, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>
    }
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read|Write"
}
// CHECK-LABEL: func @tile_from_matmul_outs_inplace()
//   CHECK-DAG:   %[[TENSOR_LHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_LHS
//   CHECK-DAG:   %[[TENSOR_RHS:.+]] = hal.interface.binding.subspan @io::@TENSOR_RHS
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @io::@TENSOR_INIT
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

func @bufferize_dynamic() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@arg2[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %3 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>
  %4 = hal.interface.load.constant offset = 0 : index
  %5 = hal.interface.load.constant offset = 1 : index
  %6 = hal.interface.load.constant offset = 2 : index
  %7 = hal.interface.load.constant offset = 3 : index
  %8 = hal.interface.load.constant offset = 4 : index
  %9 = hal.interface.load.constant offset = 5 : index
  %10 = hal.interface.load.constant offset = 6 : index
  %11 = hal.interface.load.constant offset = 7 : index
  %12 = shapex.make_ranked_shape %4, %5 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %13 = flow.dispatch.tie_shape %0, %12 : (!flow.dispatch.tensor<readonly:?x?xf32>, !shapex.ranked_shape<[?,?]>) -> !flow.dispatch.tensor<readonly:?x?xf32>
  %14 = shapex.make_ranked_shape %6, %7 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %15 = flow.dispatch.tie_shape %1, %14 : (!flow.dispatch.tensor<readonly:?x?xf32>, !shapex.ranked_shape<[?,?]>) -> !flow.dispatch.tensor<readonly:?x?xf32>
  %16 = shapex.make_ranked_shape %8, %9 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %17 = flow.dispatch.tie_shape %2, %16 : (!flow.dispatch.tensor<readonly:?x?xf32>, !shapex.ranked_shape<[?,?]>) -> !flow.dispatch.tensor<readonly:?x?xf32>
  %18 = shapex.make_ranked_shape %10, %11 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %19 = flow.dispatch.tie_shape %3, %18 : (!flow.dispatch.tensor<writeonly:?x?xf32>, !shapex.ranked_shape<[?,?]>) -> !flow.dispatch.tensor<writeonly:?x?xf32>
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
      %25 = flow.dispatch.tensor.load %13, offsets = [%arg0, %c0], sizes = [%24, %5], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %26 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg1)[%7, %workgroup_size_x]
      %27 = flow.dispatch.tensor.load %15, offsets = [%c0, %arg1], sizes = [%6, %26], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %28 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %8]
      %29 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %9]
      %30 = flow.dispatch.tensor.load %17, offsets = [%arg0, %arg1], sizes = [%28, %29], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %31 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%25, %27 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%30 : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %31, %19, offsets = [%arg0, %arg1], sizes = [%28, %29], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
    }
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @arg2, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
//       CHECK: func @bufferize_dynamic()
//   CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan @io::@arg1
//   CHECK-DAG:   %[[INIT:.+]] = hal.interface.binding.subspan @io::@arg2
//   CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan @io::@ret0
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
//       CHECK:       %[[LHS_TILE:.+]] = memref.subview %[[LHS_SHAPED]][%[[IV0]], 0] [%[[TILE_M]], %[[DIM1]]]
//       CHECK:       %[[TILE_N:.+]] = affine.min #[[MAP0]](%[[IV1]])[%[[DIM3]], %[[WGSIZE_X]]]
//   CHECK-DAG:       %[[RHS_TILE:.+]] = memref.subview %[[RHS_SHAPED]][0, %[[IV1]]] [%[[DIM2]], %[[TILE_N]]]
//       CHECK:       %[[TILE_M_2:.+]] = affine.min #[[MAP2]](%[[IV0]])[%[[WGSIZE_Y]], %[[DIM4]]]
//       CHECK:       %[[TILE_N_2:.+]] = affine.min #[[MAP2]](%[[IV1]])[%[[WGSIZE_X]], %[[DIM5]]]
//   CHECK-DAG:       %[[INIT_TILE:.+]] = memref.subview %[[INIT_SHAPED]][%[[IV0]], %[[IV1]]] [%[[TILE_M_2]], %[[TILE_N_2]]]
//   CHECK-DAG:       %[[RESULT_TILE:.+]] = memref.subview %[[RESULT_SHAPED]][%[[IV0]], %[[IV1]]] [%[[TILE_M_2]], %[[TILE_N_2]]]
//       CHECK:       linalg.copy(%[[INIT_TILE]], %[[RESULT_TILE]])
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS_TILE]], %[[RHS_TILE]]
//  CHECK-SAME:         outs(%[[RESULT_TILE]]

// -----

func @bufferize_dynamic_inplace() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@arg2[%c0] : !flow.dispatch.tensor<readwrite:?x?xf32>
  %4 = hal.interface.load.constant offset = 0 : index
  %5 = hal.interface.load.constant offset = 1 : index
  %6 = hal.interface.load.constant offset = 2 : index
  %7 = hal.interface.load.constant offset = 3 : index
  %8 = hal.interface.load.constant offset = 4 : index
  %9 = hal.interface.load.constant offset = 5 : index
  %12 = shapex.make_ranked_shape %4, %5 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %13 = flow.dispatch.tie_shape %0, %12 : (!flow.dispatch.tensor<readonly:?x?xf32>, !shapex.ranked_shape<[?,?]>) -> !flow.dispatch.tensor<readonly:?x?xf32>
  %14 = shapex.make_ranked_shape %6, %7 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %15 = flow.dispatch.tie_shape %1, %14 : (!flow.dispatch.tensor<readonly:?x?xf32>, !shapex.ranked_shape<[?,?]>) -> !flow.dispatch.tensor<readonly:?x?xf32>
  %16 = shapex.make_ranked_shape %8, %9 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %17 = flow.dispatch.tie_shape %2, %16 : (!flow.dispatch.tensor<readwrite:?x?xf32>, !shapex.ranked_shape<[?,?]>) -> !flow.dispatch.tensor<readwrite:?x?xf32>
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
      %25 = flow.dispatch.tensor.load %13, offsets = [%arg0, %c0], sizes = [%24, %5], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %26 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg1)[%7, %workgroup_size_x]
      %27 = flow.dispatch.tensor.load %15, offsets = [%c0, %arg1], sizes = [%6, %26], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
      %28 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %8]
      %29 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %9]
      %30 = flow.dispatch.tensor.load %17, offsets = [%arg0, %arg1], sizes = [%28, %29], strides = [%c1, %c1] : !flow.dispatch.tensor<readwrite:?x?xf32> -> tensor<?x?xf32>
      %31 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%25, %27 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%30 : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %31, %17, offsets = [%arg0, %arg1], sizes = [%28, %29], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>
    }
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @arg2, set=0, binding=2, type="StorageBuffer", access="Read|Write"
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
//       CHECK: func @bufferize_dynamic_inplace()
//   CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan @io::@arg1
//   CHECK-DAG:   %[[RESULT:.+]] = hal.interface.binding.subspan @io::@arg2
//   CHECK-DAG:   %[[DIM0:.+]] = hal.interface.load.constant offset = 0 : index
//   CHECK-DAG:   %[[DIM1:.+]] = hal.interface.load.constant offset = 1 : index
//   CHECK-DAG:   %[[DIM2:.+]] = hal.interface.load.constant offset = 2 : index
//   CHECK-DAG:   %[[DIM3:.+]] = hal.interface.load.constant offset = 3 : index
//   CHECK-DAG:   %[[DIM4:.+]] = hal.interface.load.constant offset = 4 : index
//   CHECK-DAG:   %[[DIM5:.+]] = hal.interface.load.constant offset = 5 : index
//       CHECK:   %[[SHAPE_LHS:.+]] = shapex.make_ranked_shape %[[DIM0]], %[[DIM1]]
//       CHECK:   %[[LHS_SHAPED:.+]] = shapex.tie_shape %[[LHS]], %[[SHAPE_LHS]]
//       CHECK:   %[[SHAPE_RHS:.+]] = shapex.make_ranked_shape %[[DIM2]], %[[DIM3]]
//       CHECK:   %[[RHS_SHAPED:.+]] = shapex.tie_shape %[[RHS]], %[[SHAPE_RHS]]
//       CHECK:   %[[SHAPE_RESULT:.+]] = shapex.make_ranked_shape %[[DIM4]], %[[DIM5]]
//       CHECK:   %[[RESULT_SHAPED:.+]] = shapex.tie_shape %[[RESULT]], %[[SHAPE_RESULT]]
//   CHECK-DAG:   %[[WGSIZE_X:.+]] = hal.interface.workgroup.size[0]
//   CHECK-DAG:   %[[WGSIZE_Y:.+]] = hal.interface.workgroup.size[1]
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//       CHECK:       %[[TILE_M:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[DIM0]], %[[WGSIZE_Y]]]
//       CHECK:       %[[LHS_TILE:.+]] = memref.subview %[[LHS_SHAPED]][%[[IV0]], 0] [%[[TILE_M]], %[[DIM1]]]
//       CHECK:       %[[TILE_N:.+]] = affine.min #[[MAP0]](%[[IV1]])[%[[DIM3]], %[[WGSIZE_X]]]
//   CHECK-DAG:       %[[RHS_TILE:.+]] = memref.subview %[[RHS_SHAPED]][0, %[[IV1]]] [%[[DIM2]], %[[TILE_N]]]
//       CHECK:       %[[TILE_M_2:.+]] = affine.min #[[MAP2]](%[[IV0]])[%[[WGSIZE_Y]], %[[DIM4]]]
//       CHECK:       %[[TILE_N_2:.+]] = affine.min #[[MAP2]](%[[IV1]])[%[[WGSIZE_X]], %[[DIM5]]]
//   CHECK-DAG:       %[[RESULT_TILE:.+]] = memref.subview %[[RESULT_SHAPED]][%[[IV0]], %[[IV1]]] [%[[TILE_M_2]], %[[TILE_N_2]]]
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS_TILE]], %[[RHS_TILE]]
//  CHECK-SAME:         outs(%[[RESULT_TILE]]

// -----

func @reshape_simple() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c12 = constant 12 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:12xi32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:3x4xi32>
  %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:12xi32> -> tensor<12xi32>
  %3 = linalg.tensor_expand_shape %2 [[0, 1]] : tensor<12xi32> into tensor<3x4xi32>
  flow.dispatch.tensor.store %3, %1, offsets = [], sizes = [], strides = [] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
//       CHECK: func @reshape_simple()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK:   %[[RESHAPE:.+]] = linalg.expand_shape %[[ARG0]] {{\[}}[0, 1]]
//       CHECK:   linalg.copy(%[[RESHAPE]], %[[RET0]])

// -----

func @reshape_fused_source() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c12 = constant 12 : index
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
      %6 = addi %arg0, %arg0 : i32
      linalg.yield %6 : i32
    } -> tensor<3x4xi32>
  flow.dispatch.tensor.store %5, %1, offsets = [], sizes = [], strides = [] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
//       CHECK: func @reshape_fused_source()
//       CHECK:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0[%[[C0]]] : memref<12xi32>
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0[%[[C0]]] : memref<3x4xi32>
//       CHECK:   %[[RESHAPE:.+]] = linalg.expand_shape %[[ARG0]] {{\[}}[0, 1]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[RESHAPE]] : memref<3x4xi32>)
//  CHECK-SAME:     outs(%[[RET0]] : memref<3x4xi32>)

// -----

func @reshape_fused_source_and_copyout() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c12 = constant 12 : index
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
      %7 = addi %arg0, %arg0 : i32
      linalg.yield %7 : i32
    } -> tensor<3x4xi32>
  flow.dispatch.tensor.store %6, %1, offsets = [], sizes = [], strides = [] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  flow.dispatch.tensor.store %4, %2, offsets = [], sizes = [], strides = [] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  hal.interface.binding @ret1, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}
//       CHECK: func @reshape_fused_source_and_copyout()
//       CHECK:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0[%[[C0]]] : memref<12xi32>
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0[%[[C0]]] : memref<3x4xi32>
//   CHECK-DAG:   %[[RET1:.+]] = hal.interface.binding.subspan @io::@ret1[%[[C0]]] : memref<3x4xi32>
//       CHECK:   %[[RESHAPE:.+]] = linalg.expand_shape %[[ARG0]] {{\[}}[0, 1]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[RESHAPE]] : memref<3x4xi32>)
//  CHECK-SAME:     outs(%[[RET0]] : memref<3x4xi32>)
//       CHECK:   linalg.copy(%[[RESHAPE]], %[[RET1]])

// -----

func @reshape_fused_target() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c12 = constant 12 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:3x4xi32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:12xi32>
  %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:3x4xi32> -> tensor<3x4xi32>
  %3 = linalg.init_tensor [3, 4] : tensor<3x4xi32>
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%2 : tensor<3x4xi32>) outs(%3 : tensor<3x4xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %5 = addi %arg0, %arg0 : i32
      linalg.yield %5 : i32
    } -> tensor<3x4xi32>
  %5 = linalg.tensor_collapse_shape %4 [[0, 1]] : tensor<3x4xi32> into tensor<12xi32>
  flow.dispatch.tensor.store %5, %1, offsets = [], sizes = [], strides = [] : tensor<12xi32> -> !flow.dispatch.tensor<writeonly:12xi32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
//       CHECK: func @reshape_fused_target()
//       CHECK:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0[%[[C0]]] : memref<3x4xi32>
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0[%[[C0]]] : memref<12xi32>
//       CHECK:   %[[RESHAPE:.+]] = linalg.expand_shape %[[RET0]] {{\[}}[0, 1]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[ARG0]] : memref<3x4xi32>)
//  CHECK-SAME:     outs(%[[RESHAPE]] : memref<3x4xi32>)

// -----

func @dot_general_lowering() {
  %cst = constant 0.000000e+00 : f32
  %c3 = constant 3 : index
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c1 = constant 1 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:1x1x2xf32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:2x3xf32>
  %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:1x3xf32>
  %3 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:1x1x2xf32> -> tensor<1x1x2xf32>
  %4 = linalg.tensor_collapse_shape %3 [[0, 1], [2]] : tensor<1x1x2xf32> into tensor<1x2xf32>
  %workgroup_size_x = hal.interface.workgroup.size[0] : index
  %workgroup_size_y = hal.interface.workgroup.size[1] : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %5 = muli %workgroup_size_y, %workgroup_id_y : index
  %6 = muli %workgroup_size_y, %workgroup_count_y : index
  scf.for %arg0 = %5 to %c1 step %6 {
    %7 = muli %workgroup_size_x, %workgroup_id_x : index
    %8 = muli %workgroup_size_x, %workgroup_count_x : index
    scf.for %arg1 = %7 to %c3 step %8 {
      %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1)>(%arg0)[%workgroup_size_y]
      %10 = subtensor %4[%arg0, 0] [%9, 2] [1, 1] : tensor<1x2xf32> to tensor<?x2xf32>
      %11 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 3)>(%arg1)[%workgroup_size_x]
      %12 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [%c2, %11], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:2x3xf32> -> tensor<2x?xf32>
      %13 = linalg.init_tensor [%9, %11] : tensor<?x?xf32>
      %14 = linalg.fill(%13, %cst) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
      %15 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%10, %12 : tensor<?x2xf32>, tensor<2x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %15, %2, offsets = [%arg0, %arg1], sizes = [%9, %11], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:1x3xf32>
    }
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @dot_general_lowering()
//   CHECK-DAG:   %[[LHS:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[RHS:.+]] = hal.interface.binding.subspan @io::@arg1
//   CHECK-DAG:   %[[RESHAPE_LHS:.+]] = linalg.collapse_shape %[[LHS]]
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK:   scf.for %[[IV0:.+]] = {{.+}} {
//       CHECK:     scf.for %[[IV1:.+]] = {{.+}} {
//   CHECK-DAG:       %[[LHS_TILE:.+]] = memref.subview %[[RESHAPE_LHS]][%[[IV0]], 0]
//   CHECK-DAG:       %[[RESULT_TILE:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]]
//   CHECK-DAG:       %[[RHS_TILE:.+]] = memref.subview %[[RHS]][0, %[[IV1]]]
//       CHECK:       linalg.fill(%[[RESULT_TILE]], %{{.+}})
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS_TILE]], %[[RHS_TILE]]
//  CHECK-SAME:         outs(%[[RESULT_TILE]]

// -----

func @slice() {
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xi32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xi32>
  %2 = hal.interface.load.constant offset = 0 : index
  %3 = hal.interface.load.constant offset = 1 : index
  %4 = hal.interface.load.constant offset = 2 : index
  %5 = hal.interface.load.constant offset = 3 : index
  %6 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:?x?xi32> -> tensor<?x?xi32>
  %7 = subtensor %6[%2, %3] [%4, %5] [1, 1] : tensor<?x?xi32> to tensor<?x?xi32>
  flow.dispatch.tensor.store %7, %1, offsets = [], sizes = [], strides = [] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:?x?xi32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @slice()
//   CHECK-DAG: %[[ARG:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG: %[[RETURN:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK: %[[SUBVIEW:.+]] = memref.subview %[[ARG]]
//       CHECK: linalg.copy(%[[SUBVIEW]], %[[RETURN]])

// -----

func @slice_rank_reducing() {
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?x?xi32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xi32>
  %2 = hal.interface.load.constant offset = 0 : index
  %3 = hal.interface.load.constant offset = 1 : index
  %4 = hal.interface.load.constant offset = 2 : index
  %5 = hal.interface.load.constant offset = 3 : index
  %6 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:?x?x?xi32> -> tensor<?x?x?xi32>
  %7 = subtensor %6[%2, %2, %3] [%4, 1, %5] [1, 1, 1] : tensor<?x?x?xi32> to tensor<?x?xi32>
  flow.dispatch.tensor.store %7, %1, offsets = [], sizes = [], strides = [] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:?x?xi32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @slice_rank_reducing()
//   CHECK-DAG: %[[ARG:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG: %[[RETURN:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK: %[[SUBVIEW:.+]] = memref.subview %[[ARG]]
//       CHECK: linalg.copy(%[[SUBVIEW]], %[[RETURN]])

// -----

func @slice_multiple_copy() {
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?x?xi32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?x?xi32>
  %2 = hal.interface.binding.subspan @io::@ret1[%c0] : !flow.dispatch.tensor<writeonly:?x?xi32>
  %3 = hal.interface.load.constant offset = 0 : index
  %4 = hal.interface.load.constant offset = 1 : index
  %5 = hal.interface.load.constant offset = 2 : index
  %6 = hal.interface.load.constant offset = 3 : index
  %7 = hal.interface.load.constant offset = 4 : index
  %8 = hal.interface.load.constant offset = 5 : index
  %9 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:?x?x?xi32> -> tensor<?x?x?xi32>
  %10 = subtensor %9[%3, %4, %5] [%6, %7, %8] [1, 1, 1] : tensor<?x?x?xi32> to tensor<?x?x?xi32>
  %11 = subtensor %9[%3, %4, %5] [%6, 1, %8] [1, 1, 1] : tensor<?x?x?xi32> to tensor<?x?xi32>
  flow.dispatch.tensor.store %10, %1, offsets = [], sizes = [], strides = [] : tensor<?x?x?xi32> -> !flow.dispatch.tensor<writeonly:?x?x?xi32>
  flow.dispatch.tensor.store %11, %2, offsets = [%3, %5], sizes = [%6, %8], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:?x?xi32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  hal.interface.binding @ret1, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @slice_multiple_copy()
//   CHECK-DAG: %[[ARG:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG: %[[RETURN1:.+]] = hal.interface.binding.subspan @io::@ret0
//   CHECK-DAG: %[[RETURN2:.+]] = hal.interface.binding.subspan @io::@ret1
//   CHECK-DAG: %[[SIZE1:.+]] = hal.interface.load.constant offset = 3 : index
//   CHECK-DAG: %[[SIZE2:.+]] = hal.interface.load.constant offset = 4 : index
//   CHECK-DAG: %[[SIZE3:.+]] = hal.interface.load.constant offset = 5 : index
//       CHECK: %[[SUBVIEW1:.+]] = memref.subview %[[ARG]][%{{.+}}, %{{.+}}, %{{.+}}] [%[[SIZE1]], %[[SIZE2]], %[[SIZE3]]]
//       CHECK: linalg.copy(%[[SUBVIEW1]], %[[RETURN1]])
//   CHECK-DAG: %[[SUBVIEW2:.+]] = memref.subview %[[ARG]][%{{.+}}, %{{.+}}, %{{.+}}] [%[[SIZE1]], 1, %[[SIZE3]]]
//   CHECK-DAG: %[[RETURNVIEW:.+]] = memref.subview %[[RETURN2]]
//       CHECK: linalg.copy(%[[SUBVIEW2]], %[[RETURNVIEW]])

// -----

func @slice_in_place() {
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readwrite:?x?xi32>
  %2 = hal.interface.load.constant offset = 0 : index
  %3 = hal.interface.load.constant offset = 1 : index
  %4 = hal.interface.load.constant offset = 2 : index
  %5 = hal.interface.load.constant offset = 3 : index
  %6 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:?x?xi32> -> tensor<?x?xi32>
  flow.dispatch.tensor.store %6, %0, offsets = [], sizes = [], strides = [] : tensor<?x?xi32> -> !flow.dispatch.tensor<readwrite:?x?xi32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read|Write"
}
// CHECK-LABEL: func @slice_in_place()
//   CHECK-NOT:   linalg.copy


// -----

func @slice_whole_stride_dispatch_0() {
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xi32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xi32>
  %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:?x?xi32> -> tensor<?x?xi32>
  %3 = subtensor %2[1, 0] [1, 4] [1, 1] : tensor<?x?xi32> to tensor<1x4xi32>
  flow.dispatch.tensor.store %3, %1, offsets = [], sizes = [], strides = [] : tensor<1x4xi32> -> !flow.dispatch.tensor<writeonly:?x?xi32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @slice_whole_stride_dispatch_0()
//   CHECK-DAG:   %[[INPUT:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[OUTPUT:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[INPUT]]
//       CHECK:   linalg.copy(%[[SUBVIEW]], %[[OUTPUT]])

// -----

func @subtensor_insert() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xi32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:?x?xi32>
  %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xi32>
  %3 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:?x?xi32> -> tensor<?x?xi32>
  %4 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:?x?xi32> -> tensor<?x?xi32>
  %5 = memref.dim %3, %c0 : tensor<?x?xi32>
  %6 = memref.dim %3, %c1 : tensor<?x?xi32>
  %7 = subtensor_insert %3 into %4[3, 4] [%5, %6] [1, 1] : tensor<?x?xi32> into tensor<?x?xi32>
  flow.dispatch.tensor.store %7, %2, offsets = [], sizes = [], strides = [] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:?x?xi32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @subtensor_insert()
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[C1:.+]] = constant 1
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = memref.dim %[[ARG0]], %[[C1]]
//       CHECK:   linalg.copy(%[[ARG1]], %[[RET0]])
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[RET0]][3, 4] [%[[D0]], %[[D1]]] [1, 1]
//       CHECK:   linalg.copy(%[[ARG0]], %[[SUBVIEW]])

// -----

func @tensor_extract() {
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:i32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:3x9xi32>
  %2 = linalg.init_tensor [3, 9] : tensor<3x9xi32>
  %3 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = []  : !flow.dispatch.tensor<readonly:i32> -> tensor<i32>
  %4 = tensor.extract %3[] : tensor<i32>
  %5 = linalg.fill(%2, %4) : tensor<3x9xi32>, i32 -> tensor<3x9xi32>
  flow.dispatch.tensor.store %5, %1, offsets = [], sizes = [], strides = [] : tensor<3x9xi32> -> !flow.dispatch.tensor<writeonly:3x9xi32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @tensor_extract()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK:   %[[LOAD:.+]] = memref.load %[[ARG0]]
//       CHECK:   linalg.fill(%[[RET0]], %[[LOAD]])

// -----

func @load_to_store() {
  %c0 = constant 0 : index
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:3x4xi32>
  %2 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:3x4xi32>
  %3 = flow.dispatch.tensor.load %2, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:3x4xi32> -> tensor<3x4xi32>
  flow.dispatch.tensor.store %3, %1, offsets = [], sizes = [], strides = [] : tensor<3x4xi32> -> !flow.dispatch.tensor<writeonly:3x4xi32>
  return
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}

// CHECK-LABEL: func @load_to_store()
//       CHECK:   %[[OUT:.+]] = hal.interface.binding.subspan @io::@ret0[%c0] : memref<3x4xi32>
//       CHECK:   %[[IN:.+]] = hal.interface.binding.subspan @io::@arg0[%c0] : memref<3x4xi32>
//       CHECK:   linalg.copy(%[[IN]], %[[OUT]]) : memref<3x4xi32>, memref<3x4xi32>

// -----

func @constant() {
  %c0 = constant 0 : index
  %cst = constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>
  %0 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:2x2x3xi32>
  flow.dispatch.tensor.store %cst, %0, offsets = [], sizes = [], strides = [] : tensor<2x2x3xi32> -> !flow.dispatch.tensor<writeonly:2x2x3xi32>
  return
}
// CHECK-LABEL: func @constant()
//       CHECK:   %[[CST:.+]] = constant {{.+}} : tensor<2x2x3xi32>
//       CHECK:   %[[MEMREF:.+]] = memref.buffer_cast %[[CST]] : memref<2x2x3xi32>
//       CHECK:   %[[RESULT:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK:   linalg.copy(%[[MEMREF]], %[[RESULT]])

// -----

func @rhs_non_splat_constant() {
  %c0 = constant 0 : index
  %cst = constant dense<[[0.706495285, -0.567672312, 0.483717591, 0.522725761, 0.7563259], [-0.0899272263, -0.283501834, -0.350822538, -0.351515919, -0.337136656], [-0.451804549, 0.372324884, -0.620518147, 0.235451385, 0.851095855]]> : tensor<3x5xf32>
  %cst_0 = constant 0.000000e+00 : f32
  %c5 = constant 5 : index
  %c1 = constant 1 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:1x5x3x1xf32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:5x5xf32>
  %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:1x5x3x1xf32> -> tensor<1x5x3x1xf32>
  %3 = linalg.tensor_collapse_shape %2 [[0, 1], [2, 3]] : tensor<1x5x3x1xf32> into tensor<5x3xf32>
  %workgroup_size_x = hal.interface.workgroup.size[0] : index
  %workgroup_size_y = hal.interface.workgroup.size[1] : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
  %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
  scf.for %arg0 = %4 to %c5 step %5 {
    %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
    %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
    scf.for %arg1 = %6 to %c5 step %7 {
      %8 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 5)>(%arg0)[%workgroup_size_y]
      %9 = subtensor %3[%arg0, 0] [%8, 3] [1, 1] : tensor<5x3xf32> to tensor<?x3xf32>
      %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 5)>(%arg1)[%workgroup_size_x]
      %11 = subtensor %cst[0, %arg1] [3, %10] [1, 1] : tensor<3x5xf32> to tensor<3x?xf32>
      %12 = linalg.init_tensor [%8, %10] : tensor<?x?xf32>
      %13 = linalg.fill(%12, %cst_0) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
      %14 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%9, %11 : tensor<?x3xf32>, tensor<3x?xf32>) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %14, %1, offsets = [%arg0, %arg1], sizes = [%8, %10], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:5x5xf32>
    }
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @rhs_non_splat_constant
//   CHECK-DAG:   %[[CONSTANT:.+]] = constant {{.+}} : tensor<3x5xf32>
//   CHECK-DAG:   %[[RHS:.+]] = memref.buffer_cast %[[CONSTANT]]
//   CHECK-DAG:   %[[LHS_INPUT:.+]] = hal.interface.binding.subspan @io::@arg0[%{{.+}}] : memref<1x5x3x1xf32>
//   CHECK-DAG:   %[[RETURN:.+]] = hal.interface.binding.subspan @io::@ret0[%{{.+}}] : memref<5x5xf32>
//       CHECK:   %[[LHS:.+]] = linalg.collapse_shape %[[LHS_INPUT]]
//       CHECK:   scf.for %[[IV0:.+]] =
//       CHECK:     scf.for %[[IV1:.+]] =
//   CHECK-DAG:       %[[LHS_SUBVIEW:.+]] = memref.subview %[[LHS]][%[[IV0]], 0]
//   CHECK-DAG:       %[[RHS_SUBVIEW:.+]] = memref.subview %[[RHS]][0, %[[IV1]]]
//   CHECK-DAG:       %[[RESULT_SUBVIEW:.+]] = memref.subview %[[RETURN]][%[[IV0]], %[[IV1]]]
//       CHECK:       linalg.fill(%[[RESULT_SUBVIEW]], %{{.+}})
//       CHECK:       linalg.matmul
//  CHECK-SAME:         ins(%[[LHS_SUBVIEW]], %[[RHS_SUBVIEW]]
//  CHECK-SAME:         outs(%[[RESULT_SUBVIEW]]

// -----

func @gather() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:?xi32>
  %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>
  %4 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = []: !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:?xi32> -> tensor<?xi32>
  %d0 = memref.dim %5, %c0 : tensor<?xi32>
  %d1 = memref.dim %4, %c1 : tensor<?x?xf32>
  %3 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %7 = linalg.indexed_generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<?xi32>) outs(%3 : tensor<?x?xf32>) {
  ^bb0(%arg0: index, %arg1: index, %arg2: i32, %arg3: f32):  // no predecessors
    %8 = index_cast %arg2 : i32 to index
    %9 = tensor.extract %4[%8, %arg1] : tensor<?x?xf32>
    linalg.yield %9 : f32
  } -> tensor<?x?xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [], sizes = [], strides = [] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @gather()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK:   linalg.generic
//       CHECK:     %[[VAL:.+]] = memref.load %[[ARG0]]
//       CHECK:     linalg.yield %[[VAL]]

// -----

func @pooling_nhwc_sum() {
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = hal.interface.binding.subspan @io::@ro0[%c0] : !flow.dispatch.tensor<readonly:f32>
  %1 = hal.interface.binding.subspan @io::@ro1[%c0] : !flow.dispatch.tensor<readonly:1x4x6x1xf32>
  %2 = hal.interface.binding.subspan @io::@wo2[%c0] : !flow.dispatch.tensor<writeonly:1x2x2x1xf32>
  %3 = linalg.init_tensor [2, 3] : tensor<2x3xf32>
  %4 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:f32> -> tensor<f32>
  %5 = tensor.extract %4[] : tensor<f32>
  %6 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:1x4x6x1xf32> -> tensor<1x4x6x1xf32>
  %7 = linalg.init_tensor [1, 2, 2, 1] : tensor<1x2x2x1xf32>
  %8 = linalg.fill(%7, %5) : tensor<1x2x2x1xf32>, f32 -> tensor<1x2x2x1xf32>
  %9 = linalg.pooling_nhwc_sum {
    dilations = dense<1> : vector<2xi64>,
    strides = dense<[2, 3]> : vector<2xi64>
  } ins(%6, %3 : tensor<1x4x6x1xf32>, tensor<2x3xf32>)
   outs(%8 : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
  flow.dispatch.tensor.store %9, %2, offsets = [], sizes = [], strides = [] : tensor<1x2x2x1xf32> -> !flow.dispatch.tensor<writeonly:1x2x2x1xf32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @pooling_nhwc_sum
//       CHECK:   %[[WINDOW:.+]] = memref.alloc() : memref<2x3xf32>
//   CHECK-DAG:   %[[INPUT:.+]] = hal.interface.binding.subspan @io::@ro1[%c0] : memref<1x4x6x1xf32>
//   CHECK-DAG:   %[[INIT:.+]] = hal.interface.binding.subspan @io::@ro0[%c0] : memref<f32>
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@wo2[%c0] : memref<1x2x2x1xf32>
//       CHECK:   %[[INIT_VAL:.+]] = memref.load %[[INIT]][] : memref<f32>
//       CHECK:   linalg.fill(%[[RET0]], %[[INIT_VAL]]) : memref<1x2x2x1xf32>, f32
//       CHECK:   linalg.pooling_nhwc_sum
//  CHECK-SAME:     dilations = dense<1> : vector<2xi64>
//  CHECK-SAME:     strides = dense<[2, 3]> : vector<2xi64>
//  CHECK-SAME:     ins(%[[INPUT]], %[[WINDOW]] : memref<1x4x6x1xf32>, memref<2x3xf32>)
//  CHECK-SAME:    outs(%[[RET0]] : memref<1x2x2x1xf32>)

// -----

func @read_only_subtensor() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = hal.interface.binding.subspan @io::@wo2[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@ro0[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
  %3 = hal.interface.binding.subspan @io::@ro1[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %4 = flow.dispatch.tensor.load %3, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
  %workgroup_size_x = hal.interface.workgroup.size[0] : index
  %workgroup_size_y = hal.interface.workgroup.size[1] : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
  %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
  %dim0 = memref.dim %2, %c0 : tensor<?x?xf32>
  %dim1 = memref.dim %2, %c1 : tensor<?x?xf32>
  scf.for %arg0 = %5 to %dim0 step %6 {
    %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
    %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
    scf.for %arg1 = %7 to %dim1 step %8 {
      %9 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %dim0]
      %10 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %dim1]
      %11 = subtensor %2[%arg0, %arg1] [%9, %10] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %12 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %dim0]
      %13 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %dim1]
      %14 = subtensor %2[%arg0, %arg1] [%12, %13] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %15 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %dim0]
      %16 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %dim1]
      %17 = subtensor %4[%arg0, %arg1] [%15, %16] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %18 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %dim0]
      %19 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %dim1]
      %20 = subtensor %4[%arg0, %arg1] [%18, %19] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %21 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %dim0]
      %22 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %dim1]
      %23 = linalg.init_tensor [%21, %22] : tensor<?x?xf32>
      %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11, %14, %17, %20 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%23 : tensor<?x?xf32>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
      ^bb0(%arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
        %25 = mulf %arg4, %arg5 : f32
        %26 = mulf %arg2, %arg3 : f32
        %27 = addf %26, %25 : f32
        %28 = math.sqrt %27 : f32
        linalg.yield %28 : f32
      } -> tensor<?x?xf32>
      flow.dispatch.tensor.store %24, %0, offsets = [%arg0, %arg1], sizes = [%21, %22], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
    }
  }
  return
}hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @wo2, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @read_only_subtensor
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@ro0[%c0] : memref<?x?xf32>
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@ro1[%c0] : memref<?x?xf32>
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@wo2[%c0] : memref<?x?xf32>
//       CHECK:   scf.for
//       CHECK:     scf.for
//   CHECK-DAG:       %[[SV1:.+]] = memref.subview %[[ARG0]]
//   CHECK-DAG:       %[[SV2:.+]] = memref.subview %[[ARG1]]
//   CHECK-DAG:       %[[SV3:.+]] = memref.subview %[[RET0]]
//       CHECK:       linalg.generic
//  CHECK-SAME:         ins(%[[SV1]], %[[SV1]], %[[SV2]], %[[SV2]] :
//  CHECK-SAME:         outs(%[[SV3]] :

// -----

func @reshape_read_only() {
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @io::@ro0[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@wo0[%c0] : !flow.dispatch.tensor<writeonly:?xf32>
  %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
  %3 = linalg.tensor_collapse_shape %2 [[0, 1]]
      : tensor<?x?xf32> into tensor<?xf32>
  %4 = memref.dim %3, %c0 : tensor<?xf32>
  %5 = linalg.init_tensor [%4] : tensor<?xf32>
  %6 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%3 : tensor<?xf32>) outs(%5 : tensor<?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32):
         %7 = addf %arg0, %arg0 : f32
         linalg.yield %7 : f32
      } -> tensor<?xf32>
  flow.dispatch.tensor.store %6, %1, offsets = [], sizes = [], strides = []: tensor<?xf32> -> !flow.dispatch.tensor<writeonly:?xf32>
  return
}
// CHECK-LABEL: func @reshape_read_only
//   CHECK-DAG:   %[[INPUT:.+]] = hal.interface.binding.subspan @io::@ro0
//   CHECK-DAG:   %[[OUTPUT:.+]] = hal.interface.binding.subspan @io::@wo0
//       CHECK:   %[[RESHAPE:.+]] = linalg.collapse_shape %[[INPUT]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[RESHAPE]] : memref<?xf32>)
//  CHECK-SAME:     outs(%[[OUTPUT]] : memref<?xf32>)

// -----

func @use_buffer_for_operand_when_output_tensor_not_used() {
  %c0 = constant 0 : index

  %input_subspan = hal.interface.binding.subspan @interface_io::@ro0[%c0] : !flow.dispatch.tensor<readonly:1x225x225x16xf32>
  %filter_subspan = hal.interface.binding.subspan @interface_io::@ro1[%c0] : !flow.dispatch.tensor<readonly:3x3x16x32xf32>
  %offset_subspan = hal.interface.binding.subspan @interface_io::@ro2[%c0] : !flow.dispatch.tensor<readonly:32xf32>
  %output_subspan = hal.interface.binding.subspan @interface_io::@wo3[%c0] : !flow.dispatch.tensor<writeonly:1x112x112x32xf32>

  %input = flow.dispatch.tensor.load %input_subspan, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:1x225x225x16xf32> -> tensor<1x225x225x16xf32>
  %filter = flow.dispatch.tensor.load %filter_subspan, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:3x3x16x32xf32> -> tensor<3x3x16x32xf32>
  %offset = flow.dispatch.tensor.load %offset_subspan, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:32xf32> -> tensor<32xf32>

  %cst = constant 0.0 : f32
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %1 = linalg.fill(%0, %cst) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32>
  %2 = linalg.conv_2d_input_nhwc_filter_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
         ins(%input, %filter : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>)
         outs(%1 : tensor<1x112x112x32xf32>)
         -> tensor<1x112x112x32xf32>
  %3 = linalg.generic {
         indexing_maps = [
           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
           affine_map<(d0, d1, d2, d3) -> (d3)>,
           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
         ins(%2, %offset: tensor<1x112x112x32xf32>, tensor<32xf32>)
         outs(%0 : tensor<1x112x112x32xf32>) {
         ^bb0(%a: f32, %b: f32, %c: f32):
            %sub = subf %a, %b : f32
            linalg.yield %sub : f32
         } -> tensor<1x112x112x32xf32>
  flow.dispatch.tensor.store %3, %output_subspan, offsets = [], sizes = [], strides = [] : tensor<1x112x112x32xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
  return
}

hal.interface @interface_io attributes {sym_visibility = "private"} {
  hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @ro2, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @wo3, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}

// CHECK: func @use_buffer_for_operand_when_output_tensor_not_used()

//  CHECK-NOT: memref.alloc
//      CHECK: %[[OUTPUT:.+]] = hal.interface.binding.subspan @interface_io::@wo3
//      CHECK: linalg.fill(%[[OUTPUT]], %{{.+}})
// CHECK-NEXT: linalg.conv_2d_input_nhwc_filter_hwcf
// CHECK-SAME:   outs(%[[OUTPUT]] : memref<1x112x112x32xf32>)
// CHECK-NEXT: linalg.generic
// CHECK-SAME:   ins(%[[OUTPUT]], %{{.+}} : memref<1x112x112x32xf32>, memref<32xf32>)
// CHECK-SAME:   outs(%[[OUTPUT]] : memref<1x112x112x32xf32>)

// -----

func @dont_use_buffer_for_operand_when_output_tensor_used() {
  %c0 = constant 0 : index

  %input_subspan = hal.interface.binding.subspan @interface_io::@ro0[%c0] : !flow.dispatch.tensor<readonly:1x225x225x16xf32>
  %filter_subspan = hal.interface.binding.subspan @interface_io::@ro1[%c0] : !flow.dispatch.tensor<readonly:3x3x16x32xf32>
  %offset_subspan = hal.interface.binding.subspan @interface_io::@ro2[%c0] : !flow.dispatch.tensor<readonly:32xf32>
  %output_subspan = hal.interface.binding.subspan @interface_io::@wo3[%c0] : !flow.dispatch.tensor<writeonly:1x112x112x32xf32>

  %input = flow.dispatch.tensor.load %input_subspan, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:1x225x225x16xf32> -> tensor<1x225x225x16xf32>
  %filter = flow.dispatch.tensor.load %filter_subspan, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:3x3x16x32xf32> -> tensor<3x3x16x32xf32>
  %offset = flow.dispatch.tensor.load %offset_subspan, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:32xf32> -> tensor<32xf32>

  %cst0 = constant 0.0 : f32
  %cst1 = constant 1.0 : f32
  %0 = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %1 = linalg.fill(%0, %cst0) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32>
  %2 = linalg.conv_2d_input_nhwc_filter_hwcf
         {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
         ins(%input, %filter : tensor<1x225x225x16xf32>, tensor<3x3x16x32xf32>)
         outs(%1 : tensor<1x112x112x32xf32>)
         -> tensor<1x112x112x32xf32>
  %3 = linalg.fill(%0, %cst1) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32>
  %4 = linalg.generic {
         indexing_maps = [
           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
           affine_map<(d0, d1, d2, d3) -> (d3)>,
           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
         ins(%2, %offset: tensor<1x112x112x32xf32>, tensor<32xf32>)
         outs(%3 : tensor<1x112x112x32xf32>) {
         ^bb0(%a: f32, %b: f32, %c: f32):
            %sub = subf %a, %b : f32
            %add = addf %sub, %c : f32
            linalg.yield %add : f32
         } -> tensor<1x112x112x32xf32>
  flow.dispatch.tensor.store %4, %output_subspan, offsets = [], sizes = [], strides = []: tensor<1x112x112x32xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
  return
}

// CHECK-LABEL: func @dont_use_buffer_for_operand_when_output_tensor_used()
//      CHECK: %[[ALLOC:.+]] = memref.alloc
//      CHECK: %[[OUTPUT:.+]] = hal.interface.binding.subspan @interface_io::@wo3
//      CHECK: linalg.fill(%[[ALLOC]], %{{.+}})
// CHECK-NEXT: linalg.conv_2d_input_nhwc_filter_hwcf
// CHECK-SAME:   outs(%[[ALLOC]] : memref<1x112x112x32xf32>)
// CHECK-NEXT: linalg.fill(%[[OUTPUT]], %{{.+}})
// CHECK-NEXT: linalg.generic
// CHECK-SAME:   ins(%[[ALLOC]], %{{.+}} : memref<1x112x112x32xf32>, memref<32xf32>)
// CHECK-SAME:   outs(%[[OUTPUT]] : memref<1x112x112x32xf32>)

// -----

func @bufferize_cst_output_tensor() {
  %c0 = constant 0 : index
  %cst1 = constant dense<-2147483648> : tensor<i32>
  %zero = constant 0.000000e+00 : f32
  %cst5 = constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
  %input = hal.interface.binding.subspan @io::@ro0[%c0] : !flow.dispatch.tensor<readonly:5xf32>
  %output = hal.interface.binding.subspan @io::@wo1[%c0] : !flow.dispatch.tensor<writeonly:i32>
  %1 = flow.dispatch.tensor.load %input, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:5xf32> -> tensor<5xf32>
  %2 = linalg.generic {
         indexing_maps = [affine_map<(d0) -> (-d0 + 4)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
         iterator_types = ["reduction"]}
         ins(%1, %cst5 : tensor<5xf32>, tensor<5xi32>)
         outs(%cst1 : tensor<i32>) {
  ^bb0(%arg0: f32, %arg1: i32, %arg2: i32):
    %8 = cmpf oeq, %arg0, %zero : f32
    %9 = zexti %8 : i1 to i32
    %10 = muli %9, %arg1 : i32
    %11 = cmpi sgt, %10, %arg2 : i32
    %12 = select %11, %10, %arg2 : i32
    linalg.yield %12 : i32
  } -> tensor<i32>
  flow.dispatch.tensor.store %2, %output, offsets=[], sizes=[], strides=[] : tensor<i32> -> !flow.dispatch.tensor<writeonly:i32>
  return
}

hal.interface @interface_io attributes {sym_visibility = "private"} {
  hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @wo1, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}

// CHECK-LABEL: func @bufferize_cst_output_tensor()

//       CHECK-DAG: %[[CST1:.+]] = constant dense<-2147483648> : tensor<i32>
//       CHECK-DAG: %[[CST5:.+]] = constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
//       CHECK: %[[CAST1:.+]] = memref.buffer_cast %[[CST1]] : memref<i32>
//       CHECK: %[[CAST5:.+]] = memref.buffer_cast %[[CST5]] : memref<5xi32>
//       CHECK: %[[INPUT:.+]] = hal.interface.binding.subspan @io::@ro0[%c0] : memref<5xf32>
//       CHECK: %[[OUTPUT:.+]] = hal.interface.binding.subspan @io::@wo1[%c0] : memref<i32>
//       CHECK: linalg.copy(%[[CAST1]], %[[OUTPUT]])
//       CHECK: linalg.generic
//  CHECK-SAME:   ins(%[[INPUT]], %[[CAST5]] : memref<5xf32>, memref<5xi32>)
//  CHECK-SAME:   outs(%[[OUTPUT]] : memref<i32>)

// -----

func @cast_follwed_by_store() {
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  %c4 = constant 4 : index
  %c64 = constant 64 : index
  %c1 = constant 1 : index
  %c32 = constant 32 : index
  %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:4x32x1024xf32>
  %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:4x1024x64xf32>
  %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:4x32x64xf32>
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
        %10 = linalg.fill(%9, %cst) {__internal_linalg_transform__ = "workgroup"} : tensor<1x32x32xf32>, f32 -> tensor<1x32x32xf32>
        %11 = linalg.batch_matmul {__internal_linalg_transform__ = "workgroup", is_root_op} ins(%7, %8 : tensor<?x?x1024xf32>, tensor<?x1024x?xf32>) outs(%10 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>
        %12 = tensor.cast %11 : tensor<1x32x32xf32> to tensor<?x?x?xf32>
        flow.dispatch.tensor.store %12, %2, offsets = [%arg0, %arg1, %arg2], sizes = [%c1, %c32, %c32], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:4x32x64xf32>
      }
    }
  }
  return
}

// CHECK-LABEL: func @cast_follwed_by_store()
//   CHECK-DAG: %[[ZERO:.+]] = constant 0.000000e+00 : f32
//   CHECK-DAG: %[[LHS:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<4x32x1024xf32>
//   CHECK-DAG: %[[RHS:.+]] = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : memref<4x1024x64xf32>
//   CHECK-DAG: %[[RESULT:.+]] = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : memref<4x32x64xf32>
//       CHECK: %[[LHSV:.+]] = memref.subview %[[LHS]]
//       CHECK: %[[RHSV:.+]] = memref.subview %[[RHS]]
//       CHECK: %[[RESULTV:.+]] = memref.subview %[[RESULT]]
//        CHECK: linalg.fill(%[[RESULTV]], %[[ZERO]])
//        CHECK: linalg.batch_matmul {{.*}} ins(%[[LHSV]], %[[RHSV]] : {{.*}}) outs(%[[RESULTV]]

// -----

func @rank_reduced_subtensor_insert() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<readwrite:?x?x?xf32>
  %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
  %3 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:?x?x?xf32> -> tensor<?x?x?xf32>
  %4 = memref.dim %3, %c1 : tensor<?x?x?xf32>
  %5 = memref.dim %3, %c2 : tensor<?x?x?xf32>
  %6 = subtensor_insert %2 into %3[0, 0, 0] [1, %4, %5] [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
  flow.dispatch.tensor.store %6, %1, offsets = [], sizes = [], strides = [] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?x?xf32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Read|Write"
}
// CHECK-LABEL: func @rank_reduced_subtensor_insert()
//   CHECK-DAG:   %[[ARG:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[RET:.+]] = hal.interface.binding.subspan @io::@ret0
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[RET]]
//       CHECK:   linalg.copy(%[[ARG]], %[[SUBVIEW]])

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func @bufferize_transfer_op() {
  %c3 = constant 3 : index
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c1 = constant 1 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:2x3xf32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:3x4xf32>
  %2 = hal.interface.binding.subspan @io::@arg2[%c0] : !flow.dispatch.tensor<readonly:2x4xf32>
  %3 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:2x4xf32>
  %4 = flow.dispatch.tensor.load %0, offsets = [%c0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:2x3xf32> -> tensor<2x3xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [%c0, %c0], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:3x4xf32> -> tensor<3x1xf32>
  %6 = flow.dispatch.tensor.load %2, offsets = [%c0, %c0], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:2x4xf32> -> tensor<2x1xf32>
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
  flow.dispatch.tensor.store %25, %3, offsets = [%c0, %c0], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<2x1xf32> -> !flow.dispatch.tensor<writeonly:2x4xf32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @arg2, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
//   CHECK-LABEL: func @bufferize_transfer_op()
//     CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//     CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1
//     CHECK-DAG:   %[[ARG2:.+]] = hal.interface.binding.subspan @io::@arg2
//     CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//     CHECK-DAG:   %[[ARG0V:.+]] = memref.subview %[[ARG0]]
//     CHECK-DAG:   %[[ARG1V:.+]] = memref.subview %[[ARG1]]
//     CHECK-DAG:   %[[ARG2V:.+]] = memref.subview %[[ARG2]]
// CHECK-COUNT-6:   vector.transfer_read %[[ARG0V]]
// CHECK-COUNT-3:   vector.transfer_read %[[ARG1V]]
// CHECK-COUNT-2:   vector.transfer_read %[[ARG2V]]
//         CHECK:   %[[RET0V:.+]] = memref.subview %[[RET0]]
//         CHECK:   linalg.copy(%[[ARG2V]], %[[RET0V]])
//         CHECK:   vector.transfer_write %{{.+}}, %[[RET0V]]
//         CHECK:   vector.transfer_write %{{.+}}, %[[RET0V]]

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func @bufferize_transfer_op_inplace() {
  %c3 = constant 3 : index
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c1 = constant 1 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:2x3xf32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:3x4xf32>
  %3 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<readwrite:2x4xf32>
  %4 = flow.dispatch.tensor.load %0, offsets = [%c0, %c0], sizes = [%c1, %c3], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:2x3xf32> -> tensor<2x3xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [%c0, %c0], sizes = [%c3, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:3x4xf32> -> tensor<3x1xf32>
  %6 = flow.dispatch.tensor.load %3, offsets = [%c0, %c0], sizes = [%c1, %c1], strides = [%c1, %c1] : !flow.dispatch.tensor<readwrite:2x4xf32> -> tensor<2x1xf32>
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
  flow.dispatch.tensor.store %25, %3, offsets = [%c0, %c0], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<2x1xf32> -> !flow.dispatch.tensor<readwrite:2x4xf32>
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
//   CHECK-LABEL: func @bufferize_transfer_op_inplace()
//     CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//     CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1
//     CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//     CHECK-DAG:   %[[ARG0V:.+]] = memref.subview %[[ARG0]]
//     CHECK-DAG:   %[[ARG1V:.+]] = memref.subview %[[ARG1]]
//     CHECK-DAG:   %[[RET0V:.+]] = memref.subview %[[RET0]]
// CHECK-COUNT-6:   vector.transfer_read %[[ARG0V]]
// CHECK-COUNT-3:   vector.transfer_read %[[ARG1V]]
// CHECK-COUNT-2:   vector.transfer_read %[[RET0V]]
//     CHECK-NOT:   linalg.copy
//         CHECK:   vector.transfer_write %{{.+}}, %[[RET0V]]
//         CHECK:   vector.transfer_write %{{.+}}, %[[RET0V]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func @multi_result() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>
  %3 = hal.interface.binding.subspan @io::@ret1[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>
  %4 = hal.interface.load.constant offset = 0 : index
  %5 = hal.interface.load.constant offset = 1 : index
  %6 = hal.interface.load.constant offset = 2 : index
  %7 = hal.interface.load.constant offset = 3 : index
  %8 = hal.interface.workgroup.id[0] : index
  %9 = hal.interface.workgroup.id[1] : index
  %10 = hal.interface.workgroup.count[0] : index
  %11 = hal.interface.workgroup.count[1] : index
  %12 = hal.interface.workgroup.size[0] : index
  %13 = hal.interface.workgroup.size[1] : index
  %14 = muli %9, %13 : index
  %15 = muli %11, %13 : index
  %16 = muli %8, %12 : index
  %17 = muli %10, %12 : index
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
          %23 = mulf %arg2, %arg3 : f32
          %24 = addf %arg2, %arg3 : f32
          linalg.yield %23, %24 : f32, f32
        } -> tensor<?x?xf32>, tensor<?x?xf32>
      flow.dispatch.tensor.store %22#0, %2, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
      flow.dispatch.tensor.store %22#1, %3, offsets = [%arg0, %arg1], sizes = [%18, %19], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
    }
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  hal.interface.binding @ret1, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @multi_result()
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//   CHECK-DAG:   %[[RET1:.+]] = hal.interface.binding.subspan @io::@ret1
//   CHECK-DAG:   %[[ARG0V:.+]] = memref.subview %[[ARG0]]
//   CHECK-DAG:   %[[ARG1V:.+]] = memref.subview %[[ARG1]]
//   CHECK-DAG:   %[[RET0V:.+]] = memref.subview %[[RET0]]
//   CHECK-DAG:   %[[RET1V:.+]] = memref.subview %[[RET1]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[ARG0V]], %[[ARG1V]]
//  CHECK-SAME:     outs(%[[RET0V]], %[[RET1V]]

// -----

#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
module  {
  func @padded_matmul() {
    %c0 = constant 0 : index
    %c12544 = constant 12544 : index
    %c64 = constant 64 : index
    %c16 = constant 16 : index
    %cst = constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:12544x27xf32>
    %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:27x16xf32>
    %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:12544x16xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %3 = affine.apply #map0()[%workgroup_id_y]
    %4 = affine.apply #map0()[%workgroup_count_y]
    scf.for %arg0 = %3 to %c12544 step %4 {
      %5 = affine.apply #map1()[%workgroup_id_x]
      %6 = affine.apply #map1()[%workgroup_count_x]
      scf.for %arg1 = %5 to %c16 step %6 {
        %7 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [64, 27], strides = [1, 1] : !flow.dispatch.tensor<readonly:12544x27xf32> -> tensor<64x27xf32>
        %8 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [27, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:27x16xf32> -> tensor<27x16xf32>
        %9 = linalg.init_tensor [64, 16] : tensor<64x16xf32>
        %10 = linalg.fill(%9, %cst) {__internal_linalg_transform__ = "workgroup"} : tensor<64x16xf32>, f32 -> tensor<64x16xf32>
        %11 = linalg.pad_tensor %7 low[0, 0] high[0, 5]  {
        ^bb0(%arg2: index, %arg3: index):  // no predecessors
          linalg.yield %cst : f32
        } : tensor<64x27xf32> to tensor<64x32xf32>
        %12 = linalg.pad_tensor %8 low[0, 0] high[5, 0]  {
        ^bb0(%arg2: index, %arg3: index):  // no predecessors
          linalg.yield %cst : f32
        } : tensor<27x16xf32> to tensor<32x16xf32>
        %13 = linalg.matmul ins(%11, %12 : tensor<64x32xf32>, tensor<32x16xf32>) outs(%10 : tensor<64x16xf32>) -> tensor<64x16xf32>
        %14 = tensor.cast %13 : tensor<64x16xf32> to tensor<?x?xf32>
        flow.dispatch.tensor.store %14, %2, offsets = [%arg0, %arg1], sizes = [%c64, %c16], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:12544x16xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: func @padded_matmul()
// CHECK-DAG: %[[LHS_PADDED:.+]] = memref.alloc() : memref<64x32xf32>
// CHECK-DAG: %[[RHS_PADDED:.+]] = memref.alloc() : memref<32x16xf32>
// CHECK-DAG: %[[C0:.+]] = constant 0.000000e+00 : f32
// CHECK-DAG: %[[LHS:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<12544x27xf32>
// CHECK-DAG: %[[RHS:.+]] = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : memref<27x16xf32>
// CHECK-DAG: %[[DST:.+]] = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : memref<12544x16xf32>
// CHECK-DAG: %[[LHS_V:.+]] = memref.subview %[[LHS]][%{{.*}}, 0] [64, 27] [1, 1]
// CHECK-DAG: %[[RHS_V:.+]] = memref.subview %[[RHS]][0, %{{.*}}] [27, 16] [1, 1]
// CHECK-DAG: %[[DST_V:.+]] = memref.subview %[[DST]][%{{.*}}, %{{.*}}] [64, 16] [1, 1]
//     CHECK: linalg.fill(%[[DST_V]], %[[C0]])
//     CHECK: linalg.fill(%[[LHS_PADDED]], %[[C0]]) : memref<64x32xf32>, f32
//     CHECK: %[[LHS_PADDED_INTER:.+]] = memref.subview %[[LHS_PADDED]][0, 0] [64, 27] [1, 1]
//     CHECK: linalg.copy(%[[LHS_V]], %[[LHS_PADDED_INTER]])
//     CHECK: linalg.fill(%[[RHS_PADDED]], %[[C0]]) : memref<32x16xf32>, f32
//     CHECK: %[[RHS_PADDED_INTER:.+]] = memref.subview %[[RHS_PADDED]][0, 0] [27, 16] [1, 1]
//     CHECK: linalg.copy(%[[RHS_V]], %[[RHS_PADDED_INTER]])
//     CHECK: linalg.matmul ins(%[[LHS_PADDED]], %[[RHS_PADDED]] : memref<64x32xf32>, memref<32x16xf32>)

// -----

func @dot_general_padded() {
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c1 = constant 1 : index
  %cst = constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:?x?xf32>
  %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:?x?xf32>
  %m = hal.interface.load.constant offset = 0 : index
  %n = hal.interface.load.constant offset = 1 : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_y]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_y]
  scf.for %arg0 = %3 to %m step %4 {
    %5 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_x]
    %6 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_x]
    scf.for %arg1 = %5 to %n step %6 {
      %7 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%arg0)[%m]
      %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 2], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x2xf32>
      %9 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%arg1)[%n]
      %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [2, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<2x?xf32>
      %11 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%arg0)[%m]
      %12 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%arg1)[%n]
      %13 = linalg.pad_tensor %8 low[0, 0] high[1, 2]  {
      ^bb0(%arg2: index, %arg3: index):  // no predecessors
        linalg.yield %cst : f32
      } : tensor<?x2xf32> to tensor<4x4xf32>
      %14 = linalg.pad_tensor %10 low[0, 0] high[2, 3]  {
      ^bb0(%arg2: index, %arg3: index):  // no predecessors
        linalg.yield %cst : f32
      } : tensor<2x?xf32> to tensor<4x4xf32>
      %15 = linalg.init_tensor [4, 4] : tensor<4x4xf32>
      %16 = linalg.fill(%15, %cst) : tensor<4x4xf32>, f32 -> tensor<4x4xf32>
      %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%13, %14 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%16 : tensor<4x4xf32>) -> tensor<4x4xf32>
      %18 = subtensor %17[0, 0] [%7, %9] [1, 1] : tensor<4x4xf32> to tensor<?x?xf32>
      flow.dispatch.tensor.store %18, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
    }
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}
//      CHECK: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (4, -d0 + s0)>
//      CHECK: func @dot_general_padded
//   CHECK-DAG:      %[[ALLOC_RET0:.+]] = memref.alloc
//   CHECK-DAG:      %[[ALLOC_ARG1:.+]] = memref.alloc
//   CHECK-DAG:      %[[ALLOC_ARG0:.+]] = memref.alloc
//   CHECK-DAG:  %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0[{{.*}}] : memref<?x?xf32>
//   CHECK-DAG:  %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1[{{.*}}] : memref<?x?xf32>
//   CHECK-DAG:  %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0[{{.*}}] : memref<?x?xf32>
//   CHECK-DAG:  %[[M:.+]] = hal.interface.load.constant offset = 0
//   CHECK-DAG:  %[[N:.+]] = hal.interface.load.constant offset = 1
//       CHECK:  scf.for %[[IV0:.+]] = %{{.+}} to %[[M]]
//       CHECK:    scf.for %[[IV1:.+]] = %{{.+}} to %[[N]]
//   CHECK-DAG:      %[[TILE_M:.+]] = affine.min #[[MAP1]](%[[IV0]])[%[[M]]]
//   CHECK-DAG:      %[[TILE_N:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[N]]]
//   CHECK-DAG:      %[[ARG0_SV:.+]] = memref.subview %[[ARG0]]
//   CHECK-DAG:      %[[ARG1_SV:.+]] = memref.subview %[[ARG1]]
//       CHECK:       linalg.fill(%[[ALLOC_ARG0]]
//       CHECK:      %[[ALLOC_ARG0_SV:.+]] = memref.subview %[[ALLOC_ARG0]]
//       CHECK:       linalg.copy(%[[ARG0_SV]], %[[ALLOC_ARG0_SV]])
//       CHECK:      linalg.fill(%[[ALLOC_ARG1]]
//       CHECK:      linalg.copy(%[[ARG1_SV]]
//       CHECK:      linalg.fill(%[[ALLOC_RET0]]
//       CHECK:      linalg.matmul
//  CHECK-SAME:        ins(%[[ALLOC_ARG0]], %[[ALLOC_ARG1]]
//  CHECK-SAME:        outs(%[[ALLOC_RET0]]
//   CHECK-DAG:      %[[RET0_SV:.+]] = memref.subview %[[RET0]]
//   CHECK-DAG:      %[[ALLOC_RET0_SV:.+]] = memref.subview
//       CHECK:      linalg.copy(%[[ALLOC_RET0_SV]], %[[RET0_SV]])

// -----

func @im2col() {
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  %c112 = constant 112 : index
  %c32 = constant 32 : index
  %c16 = constant 16 : index
  %c4 = constant 4 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c0] : !flow.dispatch.tensor<readonly:1x225x225x8xf32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : !flow.dispatch.tensor<readonly:3x3x8x32xf32>
  %2 = hal.interface.binding.subspan @io::@ret0[%c0] : !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_count_z = hal.interface.workgroup.count[2] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_z]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_z]
  scf.for %arg0 = %3 to %c112 step %4 {
    %5 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_y]
    %6 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_y]
    scf.for %arg1 = %5 to %c112 step %6 {
      %7 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_x]
      %8 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_x]
      scf.for %arg2 = %7 to %c32 step %8 {
        %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
        %10 = affine.min affine_map<(d0) -> (33, d0 * -2 + 225)>(%arg0)
        %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
        %12 = affine.min affine_map<(d0) -> (33, d0 * -2 + 225)>(%arg1)
        %13 = flow.dispatch.tensor.load %0, offsets = [0, %9, %11, 0], sizes = [1, %10, %12, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x225x225x8xf32> -> tensor<1x?x?x8xf32>
        %14 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 8, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x8x32xf32> -> tensor<3x3x8x4xf32>
        %15 = linalg.init_tensor [1, 16, 16, 4] : tensor<1x16x16x4xf32>
        %16 = linalg.fill(%15, %cst) {__internal_linalg_transform__ = "workgroup"} : tensor<1x16x16x4xf32>, f32 -> tensor<1x16x16x4xf32>
        %17 = linalg.init_tensor [1, 16, 16, 3, 3, 8] : tensor<1x16x16x3x3x8xf32>
        %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d3, d2 * 2 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<1x?x?x8xf32>) outs(%17 : tensor<1x16x16x3x3x8xf32>) {
        ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
          linalg.yield %arg3 : f32
        } -> tensor<1x16x16x3x3x8xf32>
        %19 = linalg.tensor_collapse_shape %18 [[0, 1, 2], [3, 4, 5]] : tensor<1x16x16x3x3x8xf32> into tensor<256x72xf32>
        %20 = linalg.tensor_collapse_shape %14 [[0, 1, 2], [3]] : tensor<3x3x8x4xf32> into tensor<72x4xf32>
        %21 = linalg.tensor_collapse_shape %16 [[0, 1, 2], [3]] : tensor<1x16x16x4xf32> into tensor<256x4xf32>
        %22 = linalg.matmul ins(%19, %20 : tensor<256x72xf32>, tensor<72x4xf32>) outs(%21 : tensor<256x4xf32>) -> tensor<256x4xf32>
        %23 = linalg.tensor_expand_shape %22 [[0, 1, 2], [3]] : tensor<256x4xf32> into tensor<1x16x16x4xf32>
        %24 = tensor.cast %23 : tensor<1x16x16x4xf32> to tensor<1x?x?x?xf32>
        flow.dispatch.tensor.store %24, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %c16, %c16, %c4], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
      }
    }
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @im2col
//   CHECK-DAG:  %[[ARG0:.+]] = hal.interface.binding.subspan @io::@arg0
//   CHECK-DAG:  %[[ARG1:.+]] = hal.interface.binding.subspan @io::@arg1
//   CHECK-DAG:  %[[RET0:.+]] = hal.interface.binding.subspan @io::@ret0
//   CHECK-DAG:  %[[ALLOC_ARG0:.+]] = memref.alloc() : memref<1x16x16x3x3x8xf32>
//   CHECK-DAG:  %[[ALLOC_ARG1:.+]] = memref.alloc() : memref<3x3x8x4xf32>
//   CHECK-DAG:  %[[ALLOC_RET0:.+]] = memref.alloc() : memref<1x16x16x4xf32>
//       CHECK:  scf.for
//       CHECK:    scf.for
//       CHECK:      scf.for
//   CHECK-DAG:      %[[ARG0_SV:.+]] = memref.subview %[[ARG0]]
//   CHECK-DAG:      %[[ARG1_SV:.+]] = memref.subview %[[ARG1]]
//   CHECK-DAG:      linalg.copy(%[[ARG1_SV]], %[[ALLOC_ARG1]])
//   CHECK-DAG:      linalg.fill(%[[ALLOC_RET0]]
//       CHECK:      linalg.generic
//  CHECK-SAME:        ins(%[[ARG0_SV]]
//  CHECK-SAME:        outs(%[[ALLOC_ARG0]]
//   CHECK-DAG:      %[[ALLOC_ARG0_RESHAPE:.+]] = linalg.collapse_shape %[[ALLOC_ARG0]]
//   CHECK-DAG:      %[[ALLOC_ARG1_RESHAPE:.+]] = linalg.collapse_shape %[[ALLOC_ARG1]]
//   CHECK-DAG:      %[[ALLOC_RET0_RESHAPE:.+]] = linalg.collapse_shape %[[ALLOC_RET0]]
//       CHECK:      linalg.matmul
//  CHECK-SAME:        ins(%[[ALLOC_ARG0_RESHAPE]], %[[ALLOC_ARG1_RESHAPE]]
//  CHECK-SAME:        outs(%[[ALLOC_RET0_RESHAPE]]
//       CHECK:      %[[RET0_SV:.+]] = memref.subview %[[RET0]]
//       CHECK:      linalg.copy(%[[ALLOC_RET0]], %[[RET0_SV]])

// -----

func @multi_result_reduce() {
  %c0 = constant 0 : index
  %c0_i32 = constant 0 : i32
  %c-2147483648_i32 = constant -2147483648 : i32
  %c2 = constant 2 : index
  %0 = hal.interface.binding.subspan @io::@ro0[%c0] : !flow.dispatch.tensor<readonly:?x?xi32>
  %1 = hal.interface.binding.subspan @io::@ro1[%c0] : !flow.dispatch.tensor<readonly:?x?xi32>
  %2 = hal.interface.binding.subspan @io::@wo0[%c0] : !flow.dispatch.tensor<writeonly:?xi32>
  %3 = hal.interface.binding.subspan @io::@wo1[%c0] : !flow.dispatch.tensor<writeonly:?xi32>
  %d0 = hal.interface.load.constant offset = 0 : index
  %d1 = hal.interface.load.constant offset = 1 : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
  scf.for %arg0 = %4 to %d1 step %5 {
    %6 = affine.min affine_map<(d0)[s0] -> (128, -d0 + s0)>(%arg0)[%d1]
    %7 = flow.dispatch.tensor.load %0, offsets = [0, %arg0], sizes = [%d0, %6], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xi32> -> tensor<?x?xi32>
    %9 = flow.dispatch.tensor.load %1, offsets = [0, %arg0], sizes = [%d0, %6], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xi32> -> tensor<?x?xi32>
    %13 = linalg.init_tensor [%6] : tensor<?xi32>
    %14 = linalg.fill(%13, %c-2147483648_i32) {__internal_linalg_transform__ = "workgroup", lowering.config = {tileSizes = [[128]]}} : tensor<?xi32>, i32 -> tensor<?xi32>
    %17 = linalg.fill(%13, %c0_i32) {__internal_linalg_transform__ = "workgroup", lowering.config = {tileSizes = [[128]]}} : tensor<?xi32>, i32 -> tensor<?xi32>
    %18:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%7, %9 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%14, %17 : tensor<?xi32>, tensor<?xi32>) attrs =  {__internal_linalg_transform__ = "workgroup", lowering.config = {tileSizes = [[128]]}} {
    ^bb0(%arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32):  // no predecessors
      %19 = cmpi sge, %arg1, %arg3 : i32
      %20 = select %19, %arg1, %arg3 : i32
      %21 = cmpi eq, %arg1, %arg3 : i32
      %22 = cmpi slt, %arg2, %arg4 : i32
      %23 = select %22, %arg2, %arg4 : i32
      %24 = select %19, %arg2, %arg4 : i32
      %25 = select %21, %23, %24 : i32
      linalg.yield %20, %25 : i32, i32
    } -> tensor<?xi32>, tensor<?xi32>
    flow.dispatch.tensor.store %18#0, %2, offsets = [%arg0], sizes = [%6], strides = [1] : tensor<?xi32> -> !flow.dispatch.tensor<writeonly:?xi32>
    flow.dispatch.tensor.store %18#1, %3, offsets = [%arg0], sizes = [%6], strides = [1] : tensor<?xi32> -> !flow.dispatch.tensor<writeonly:?xi32>
  }
  return
}
hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ro1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @wo0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  hal.interface.binding @wo1, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}
// CHECK-LABEL: func @multi_result_reduce
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan @io::@ro0
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan @io::@ro1
//   CHECK-DAG:   %[[RET0:.+]] = hal.interface.binding.subspan @io::@wo0
//   CHECK-DAG:   %[[RET1:.+]] = hal.interface.binding.subspan @io::@wo1
//       CHECK:   scf.for
//   CHECK-DAG:     %[[ARG0_SV:.+]] = memref.subview %[[ARG0]]
//   CHECK-DAG:     %[[ARG1_SV:.+]] = memref.subview %[[ARG1]]
//   CHECK-DAG:     %[[RET0_SV:.+]] = memref.subview %[[RET0]]
//   CHECK-DAG:     linalg.fill(%[[RET0_SV]]
//   CHECK-DAG:     %[[RET1_SV:.+]] = memref.subview %[[RET1]]
//   CHECK-DAG:     linalg.fill(%[[RET1_SV]]
//       CHECK:     linalg.generic
//  CHECK-SAME:       ins(%[[ARG0_SV]], %[[ARG1_SV]]
//  CHECK-SAME:       outs(%[[RET0_SV]], %[[RET1_SV]]


// -----

#config0 = {tileSizes = [[64, 64]]}
#config1 = {nativeVectorSize = [4, 4, 4], tileSizes = [[64, 64], [32, 32, 24], [4, 4, 4]]}
#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0) -> (64, -d0 + 250)>
#map2 = affine_map<(d0) -> (64, -d0 + 370)>
#map3 = affine_map<(d0) -> (32, -d0 + 250)>
#map4 = affine_map<(d0) -> (24, -d0 + 144)>
#map5 = affine_map<(d0) -> (32, -d0 + 370)>
#map6 = affine_map<(d0, d1) -> (32, d0 - d1)>
module  {
  func @l1_tiled_matmul() {
    %cst = constant 0.000000e+00 : f32
    %c32 = constant 32 : index
    %c24 = constant 24 : index
    %c144 = constant 144 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c250 = constant 250 : index
    %c370 = constant 370 : index
    %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:250x144xf32>
    %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:144x370xf32>
    %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:250x370xf32>
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
        %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 144], strides = [1, 1] : !flow.dispatch.tensor<readonly:250x144xf32> -> tensor<?x144xf32>
        %9 = affine.min #map2(%arg1)
        %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [144, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:144x370xf32> -> tensor<144x?xf32>
        %11 = linalg.init_tensor [%7, %9] : tensor<?x?xf32>
        %12 = linalg.fill(%11, %cst) {__internal_linalg_transform__ = "workgroup", lowering.config = #config0} : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
        %13 = scf.for %arg2 = %c0 to %c250 step %c32 iter_args(%arg3 = %12) -> (tensor<?x?xf32>) {
          %14 = scf.for %arg4 = %c0 to %c370 step %c32 iter_args(%arg5 = %arg3) -> (tensor<?x?xf32>) {
            %15 = scf.for %arg6 = %c0 to %c144 step %c24 iter_args(%arg7 = %arg5) -> (tensor<?x?xf32>) {
              %16 = affine.min #map3(%arg2)
              %17 = affine.min #map4(%arg6)
              %18 = subtensor %8[%arg2, %arg6] [%16, %17] [1, 1] : tensor<?x144xf32> to tensor<?x?xf32>
              %19 = affine.min #map5(%arg4)
              %20 = subtensor %10[%arg6, %arg4] [%17, %19] [1, 1] : tensor<144x?xf32> to tensor<?x?xf32>
              %21 = memref.dim %arg7, %c0 : tensor<?x?xf32>
              %22 = affine.min #map6(%21, %arg2)
              %23 = memref.dim %arg7, %c1 : tensor<?x?xf32>
              %24 = affine.min #map6(%23, %arg4)
              %25 = subtensor %arg7[%arg2, %arg4] [%22, %24] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %26 = linalg.matmul {__internal_linalg_transform__ = "workgroup_l1_tile", lowering.config = #config1} ins(%18, %20 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%25 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %27 = subtensor_insert %26 into %arg7[%arg2, %arg4] [%22, %24] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
              scf.yield %27 : tensor<?x?xf32>
            }
            scf.yield %15 : tensor<?x?xf32>
          }
          scf.yield %14 : tensor<?x?xf32>
        }
        flow.dispatch.tensor.store %13, %2, offsets = [%arg0, %arg1], sizes = [%7, %9], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:250x370xf32>
      }
    }
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// CHECK-LABEL: l1_tiled_matmul
//    CHECK-DAG: %[[M:.+]] = constant 250 : index
//    CHECK-DAG: %[[N:.+]] = constant 370 : index
//    CHECK-DAG: %[[K:.+]] = constant 144 : index
//    CHECK-DAG: %[[L1_MN_SIZE:.+]] = constant 32 : index
//    CHECK-DAG: %[[L1_K_SIZE:.+]] = constant 24 : index
//    CHECK-DAG: %[[LHS:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_external[%{{.*}}] : memref<250x144xf32>
//    CHECK-DAG: %[[RHS:.+]] = hal.interface.binding.subspan @io::@s0b1_ro_external[%{{.*}}] : memref<144x370xf32>
//    CHECK-DAG: %[[DST:.+]] = hal.interface.binding.subspan @io::@s0b2_xw_external[%{{.*}}] : memref<250x370xf32>
//        CHECK: scf.for %[[WORKGROUP_I:.+]] = %{{.*}} to %[[M]] step %{{.*}} {
//        CHECK:    scf.for %[[WORKGROUP_J:.+]] = %{{.*}} to %[[N]] step %{{.*}} {
//    CHECK-DAG:        %[[WORKGROUP_I_SIZE:.+]] = affine.min #{{.*}}(%[[WORKGROUP_I]])
//    CHECK-DAG:        %[[LHS_WORKGROUP_TILE:.+]] = memref.subview %[[LHS]][%[[WORKGROUP_I]], 0] [%[[WORKGROUP_I_SIZE]], 144] [1, 1] : memref<250x144xf32> to memref<?x144xf32
//    CHECK-DAG:        %[[WORKGROUP_J_SIZE:.+]] = affine.min #{{.*}}(%[[WORKGROUP_J]])
//    CHECK-DAG:        %[[RHS_WORKGROUP_TILE:.+]] = memref.subview %[[RHS]][0, %[[WORKGROUP_J]]] [144, %[[WORKGROUP_J_SIZE]]] [1, 1] : memref<144x370xf32> to memref<144x?xf32
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
