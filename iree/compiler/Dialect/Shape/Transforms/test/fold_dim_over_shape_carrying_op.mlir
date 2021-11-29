// RUN: iree-opt -split-input-file -iree-fold-dim-over-shape-carrying-op -canonicalize %s | IreeFileCheck %s

//      CHECK: func @memrefDim
// CHECK-SAME: (%[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
func @memrefDim(%d0: index, %d1: index) -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %subspan = hal.interface.binding.subspan @io::@s0b0_ro_constant[%c0] : memref<?x7x?xf32>{%d0, %d1}
  %dim0 = memref.dim %subspan, %c0 : memref<?x7x?xf32>
  %dim1 = memref.dim %subspan, %c1 : memref<?x7x?xf32>
  %dim2 = memref.dim %subspan, %c2 : memref<?x7x?xf32>
  // CHECK: %[[C7:.+]] = arith.constant 7 : index
  // CHECK: return %[[DIM0]], %[[C7]], %[[DIM1]]
  return %dim0, %dim1, %dim2 : index, index, index
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer"
}

// -----

//      CHECK: func @tensorDim
// CHECK-SAME: (%{{.+}}: f32, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
func @tensorDim(%value: f32, %d0: index, %d1: index) -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %splat = flow.tensor.splat %value : tensor<?x8x?xf32>{%d0, %d1}
  %dim0 = tensor.dim %splat, %c0 : tensor<?x8x?xf32>
  %dim1 = tensor.dim %splat, %c1 : tensor<?x8x?xf32>
  %dim2 = tensor.dim %splat, %c2 : tensor<?x8x?xf32>
  // CHECK: %[[C8:.+]] = arith.constant 8 : index
  // CHECK: return %[[DIM0]], %[[C8]], %[[DIM1]]
  return %dim0, %dim1, %dim2 : index, index, index
}
