// RUN: iree-opt -iree-linearize-memrefs -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @vector_load_store(
// CHECK-SAME:    %[[ARG0:.*]]: memref<2x3x4xi32>)
// CHECK-DAG:     %[[C6:.*]] = arith.constant 6 : index
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %[[ARG0]] to
// CHECK-SAME:                    offset: [0], sizes: [24], strides: [1] :
// CHECK-SAME:                    memref<2x3x4xi32> to memref<24xi32>
// CHECK:         %[[LOAD:.*]] = vector.load %[[CAST]][%[[C6]]]
// CHECK:         %[[CAST_2:.*]] = memref.reinterpret_cast %[[ARG0]] to
// CHECK-SAME:                    offset: [0], sizes: [24], strides: [1] :
// CHECK-SAME:                    memref<2x3x4xi32> to memref<24xi32>
// CHECK:         vector.store %[[LOAD]], %[[CAST_2]][%[[C6]]]
// CHECK:         return %[[LOAD]]
func.func @vector_load_store(%arg0: memref<2x3x4xi32>) -> vector<2xi32> {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %1 = vector.load %arg0[%c0, %c1, %c2] : memref<2x3x4xi32>, vector<2xi32>
  vector.store %1, %arg0[%c0, %c1, %c2] : memref<2x3x4xi32>, vector<2xi32>
  return %1 : vector<2xi32>
}

// -----

// CHECK-LABEL: @memref_load_store_alloc_dealloc(
// CHECK-DAG:     %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:     %[[C7:.*]] = arith.constant 7 : index
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<24xi32>
// CHECK:         %[[LOAD:.*]] = memref.load %[[ALLOC]][%[[C6]]]
// CHECK:         %[[ALLOCA:.*]] = memref.alloca() : memref<60xi32>
// CHECK:         memref.store %[[LOAD]], %[[ALLOCA]][%[[C7]]] {nontemporal = true} : memref<60xi32>
// CHECK:         memref.dealloc %[[ALLOC]]
// CHECK:         memref.dealloc %[[ALLOCA]]
// CHECK:         return %[[LOAD]]
func.func @memref_load_store_alloc_dealloc() -> i32 {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<2x3x4xi32>
  %1 = memref.load %0[%c0, %c1, %c2] : memref<2x3x4xi32>
  %2 = memref.alloca() : memref<3x4x5xi32>
  memref.store %1, %2[%c0, %c1, %c2] : memref<3x4x5xi32>
  memref.dealloc %0 : memref<2x3x4xi32>
  memref.dealloc %2 : memref<3x4x5xi32>
  return %1 : i32
}

// -----

// CHECK-LABEL: @memref_copy(
// CHECK-SAME:    %[[ARG0:.*]]: memref<2x3x4xi32>,
// CHECK-SAME:    %[[ARG1:.*]]: memref<2x3x4xi32>)
// CHECK:         %[[CAST_1:.*]] = memref.reinterpret_cast %[[ARG0]] to
// CHECK-SAME:                    offset: [0], sizes: [24], strides: [1] :
// CHECK-SAME:                    memref<2x3x4xi32> to memref<24xi32>
// CHECK:         %[[CAST_2:.*]] = memref.reinterpret_cast %[[ARG1]] to
// CHECK-SAME:                    offset: [0], sizes: [24], strides: [1] :
// CHECK-SAME:                    memref<2x3x4xi32> to memref<24xi32>
// CHECK:         memref.copy %[[CAST_1]], %[[CAST_2]]
// CHECK:         return
func.func @memref_copy(%arg0: memref<2x3x4xi32>, %arg1: memref<2x3x4xi32>) {
  memref.copy %arg0, %arg1 : memref<2x3x4xi32> to memref<2x3x4xi32>
  return
}
