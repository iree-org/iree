// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers -canonicalize %s | IreeFileCheck %s

module  {
  func @pad_cst() {
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %c4 = constant 4 : index
    %c2 = constant 2 : index
    %c5 = constant 5 : index
    %c3 = constant 3 : index
    %0 = hal.interface.load.tensor @io::@arg0, offset = %c0 : tensor<12x4xf32>
    %1 = linalg.pad_tensor %0 low[%c4, %c5] high[%c2, %c3]  {
    ^bb0(%arg0: index, %arg1: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<12x4xf32> to tensor<18x12xf32>
    hal.interface.store.tensor %1, @io::@ret0, offset = %c0 : tensor<18x12xf32>
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}
// CHECK-LABEL: @pad_cst
//   CHECK-DAG: %[[CST:.+]] = constant 0.000000e+00 : f32
//   CHECK-DAG: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<18x12xf32>
//   CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<12x4xf32>
//       CHECK: linalg.fill(%[[OUT]], %[[CST]])
//       CHECK: %[[SUBVIEW:.+]] = memref.subview %[[OUT]][4, 5] [12, 4] [1, 1]
//       CHECK: linalg.copy(%[[IN]], %[[SUBVIEW]])

// -----

module  {
  func @pad_memref() {
    %c0 = constant 0 : index
    %c4 = constant 4 : index
    %c2 = constant 2 : index
    %c5 = constant 5 : index
    %c3 = constant 3 : index
    %0 = hal.interface.load.tensor @io::@arg0, offset = %c0 : tensor<12x4xf32>
    %1 = hal.interface.load.tensor @io::@arg1, offset = %c0 : tensor<f32>
    %2 = tensor.extract %1[] : tensor<f32>
    %3 = linalg.pad_tensor %0 low[%c4, %c5] high[%c2, %c3]  {
    ^bb0(%arg0: index, %arg1: index):  // no predecessors
      linalg.yield %2 : f32
    } : tensor<12x4xf32> to tensor<18x12xf32>
    hal.interface.store.tensor %3, @io::@ret0, offset = %c0 : tensor<18x12xf32>
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}
// CHECK-LABEL: @pad_memref
//   CHECK-DAG: %[[OUT:.+]] = iree.placeholder for "interface buffer" {binding = @io::@ret0} : memref<18x12xf32>
//   CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<12x4xf32>
//   CHECK-DAG: %[[PAD_BUF:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg1} : memref<f32>
//       CHECK: %[[PAD_VAL:.+]] = memref.load %[[PAD_BUF]][] : memref<f32>
//       CHECK: linalg.fill(%[[OUT]], %[[PAD_VAL]])
//       CHECK: %[[SUBVIEW:.+]] = memref.subview %[[OUT]][4, 5] [12, 4] [1, 1]
//       CHECK: linalg.copy(%[[IN]], %[[SUBVIEW]])
