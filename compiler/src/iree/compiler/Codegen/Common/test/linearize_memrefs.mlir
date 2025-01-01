// RUN: iree-opt -iree-linearize-memrefs -allow-unregistered-dialect %s | FileCheck %s

//--------------------------------------------------------------------------
//---------------------------- VECTOR OPS ----------------------------------
//--------------------------------------------------------------------------

// CHECK-LABEL: @vector_load_store_static(
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
func.func @vector_load_store_static(%arg0: memref<2x3x4xi32>) -> vector<2x3xi32> {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %1 = vector.load %arg0[%c0, %c1, %c2] : memref<2x3x4xi32>, vector<2x3xi32>
  vector.store %1, %arg0[%c0, %c1, %c2] : memref<2x3x4xi32>, vector<2x3xi32>
  return %1 : vector<2x3xi32>
}

// CHECK-LABEL: @vector_load_store_dynamic(
// CHECK-SAME:    %[[DIM0:.*]]: index, %[[DIM1:.*]]: index, %[[DIM2:.*]]: index, %[[I0:.*]]: index, %[[I1:.*]]: index, %[[I2:.*]]: index)
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[LINEAR_SIZE:.*]] = affine.linearize_index disjoint [%[[DIM0]], %[[C0]], %[[C0]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[ALLOC:.*]] = memref.alloca(%[[LINEAR_SIZE]]) : memref<?xi32>
// CHECK:         %[[EXPAND_SHAPE:.*]] = memref.expand_shape %[[ALLOC]]
// CHECK{LITERAL}:                          [[0, 1, 2]]
// CHECK-SAME:                              output_shape [%[[DIM0]], %[[DIM1]], %[[DIM2]]]
// CHECK-SAME:                              : memref<?xi32> into memref<?x?x?xi32>
// CHECK:         %[[LINEAR_INDEX:.*]] = affine.linearize_index disjoint [%[[I0]], %[[I1]], %[[I2]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[LINEAR_SIZE:.*]] = affine.linearize_index disjoint [%[[DIM0]], %[[C0]], %[[C0]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %[[EXPAND_SHAPE]] to
// CHECK-SAME:                        offset: [0], sizes: [%[[LINEAR_SIZE]]], strides: [1] :
// CHECK-SAME:                        memref<?x?x?xi32> to memref<?xi32>
// CHECK:         %[[LOAD:.*]] = vector.load %[[CAST]][%[[LINEAR_INDEX]]] : memref<?xi32>, vector<2x3xi32>
// CHECK:         %[[LINEAR_INDEX:.*]] = affine.linearize_index disjoint [%[[I0]], %[[I1]], %[[I2]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[LINEAR_SIZE:.*]] = affine.linearize_index disjoint [%[[DIM0]], %[[C0]], %[[C0]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %[[EXPAND_SHAPE]] to
// CHECK-SAME:                        offset: [0], sizes: [%[[LINEAR_SIZE]]], strides: [1] :
// CHECK-SAME:                        memref<?x?x?xi32> to memref<?xi32>
// CHECK:         vector.store %[[LOAD]], %[[CAST]][%[[LINEAR_INDEX]]] : memref<?xi32>, vector<2x3xi32>
// CHECK:         return %[[LOAD]] : vector<2x3xi32>
func.func @vector_load_store_dynamic(%dim0 : index, %dim1: index, %dim2: index, %i0: index, %i1: index, %i2: index) -> vector<2x3xi32> {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloca(%dim0, %dim1, %dim2) : memref<?x?x?xi32>
  %1 = vector.load %alloc[%i0, %i1, %i2] : memref<?x?x?xi32>, vector<2x3xi32>
  vector.store %1, %alloc[%i0, %i1, %i2] : memref<?x?x?xi32>, vector<2x3xi32>
  return %1 : vector<2x3xi32>
}

// -----

//--------------------------------------------------------------------------
//---------------------------- MEMREF OPS ----------------------------------
//--------------------------------------------------------------------------


// CHECK-LABEL: @memref_load_store_alloc_dealloc_static(
// CHECK-DAG:     %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:     %[[C7:.*]] = arith.constant 7 : index
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<24xi32>
// CHECK:         %[[EXPAND_ALLOC:.*]] = memref.expand_shape %[[ALLOC]]
// CHECK{LITERAL}:                       [[0, 1, 2]] output_shape [2, 3, 4]
// CHECK-SAME:                           : memref<24xi32> into memref<2x3x4xi32>
// CHECK:         %[[RESHAPE_ALLOC:.*]] = memref.reinterpret_cast %[[EXPAND_ALLOC]] to offset: [0], sizes: [24], strides: [1]
// CHECK-SAME:                           : memref<2x3x4xi32> to memref<24xi32>
// CHECK:         %[[LOAD:.*]] = memref.load %[[RESHAPE_ALLOC]][%[[C6]]]
// CHECK:         %[[ALLOCA:.*]] = memref.alloca() : memref<60xi32>
// CHECK:         %[[EXPAND_ALLOCA:.*]] = memref.expand_shape %[[ALLOCA]]
// CHECK{LITERAL}:                 [[0, 1, 2]] output_shape [3, 4, 5]
// CHECK-SAME:                           : memref<60xi32> into memref<3x4x5xi32>
// CHECK:         %[[RESHAPE_ALLOCA:.*]] = memref.reinterpret_cast %[[EXPAND_ALLOCA]] to offset: [0], sizes: [60], strides: [1]
// CHECK-SAME:                           : memref<3x4x5xi32> to memref<60xi32>
// CHECK:         memref.store %[[LOAD]], %[[RESHAPE_ALLOCA]][%[[C7]]] {nontemporal = true} : memref<60xi32>
// CHECK:         %[[RESHAPE_ALLOC_2:.*]] = memref.reinterpret_cast %[[EXPAND_ALLOC]]
// CHECK:         memref.dealloc %[[RESHAPE_ALLOC_2]]
// CHECK:         %[[RESHAPE_ALLOCA_2:.*]] = memref.reinterpret_cast %[[EXPAND_ALLOCA]]
// CHECK:         memref.dealloc %[[RESHAPE_ALLOCA_2]]
// CHECK:         return %[[LOAD]]
func.func @memref_load_store_alloc_dealloc_static() -> i32 {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<2x3x4xi32>
  %1 = memref.load %0[%c0, %c1, %c2] : memref<2x3x4xi32>
  %2 = memref.alloca() : memref<3x4x5xi32>
  memref.store %1, %2[%c0, %c1, %c2] {nontemporal = true} : memref<3x4x5xi32>
  memref.dealloc %0 : memref<2x3x4xi32>
  memref.dealloc %2 : memref<3x4x5xi32>
  return %1 : i32
}

// CHECK-LABEL: @memref_load_store_alloc_dealloc_dynamic(
// CHECK-SAME:    %[[DIM0:.*]]: index, %[[DIM1:.*]]: index, %[[DIM2:.*]]: index, %[[I0:.*]]: index, %[[I1:.*]]: index, %[[I2:.*]]: index)
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[LINEAR_SIZE:.*]] = affine.linearize_index disjoint [%[[DIM0]], %[[C0]], %[[C0]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[ALLOC:.*]] = memref.alloca(%[[LINEAR_SIZE]]) : memref<?xf32>
// CHECK:         %[[EXPAND_SHAPE:.*]] = memref.expand_shape %[[ALLOC]]
// CHECK{LITERAL}:                          [[0, 1, 2]]
// CHECK-SAME:                              output_shape [%[[DIM0]], %[[DIM1]], %[[DIM2]]]
// CHECK-SAME:                              : memref<?xf32> into memref<?x?x?xf32>
// CHECK:         %[[LINEAR_INDEX:.*]] = affine.linearize_index disjoint [%[[I0]], %[[I1]], %[[I2]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[LINEAR_SIZE:.*]] = affine.linearize_index disjoint [%[[DIM0]], %[[C0]], %[[C0]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %[[EXPAND_SHAPE]] to
// CHECK-SAME:                        offset: [0], sizes: [%[[LINEAR_SIZE]]], strides: [1] :
// CHECK-SAME:                        memref<?x?x?xf32> to memref<?xf32>
// CHECK:         %[[LOAD:.*]] = memref.load %[[CAST]][%[[LINEAR_INDEX]]] : memref<?xf32>
// CHECK:         %[[LINEAR_INDEX:.*]] = affine.linearize_index disjoint [%[[C2]], %[[C1]], %[[C0]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[LINEAR_SIZE:.*]] = affine.linearize_index disjoint [%[[DIM0]], %[[C0]], %[[C0]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %[[EXPAND_SHAPE]] to
// CHECK-SAME:                        offset: [0], sizes: [%[[LINEAR_SIZE]]], strides: [1] :
// CHECK-SAME:                        memref<?x?x?xf32> to memref<?xf32>
// CHECK:         memref.store %[[LOAD]], %[[CAST]][%[[LINEAR_INDEX]]] : memref<?xf32>
// CHECK:         %[[LINEAR_SIZE:.*]] = affine.linearize_index disjoint [%[[DIM0]], %[[C0]], %[[C0]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %[[EXPAND_SHAPE]] to
// CHECK-SAME:                        offset: [0], sizes: [%[[LINEAR_SIZE]]], strides: [1] :
// CHECK-SAME:                        memref<?x?x?xf32> to memref<?xf32>
// CHECK:         memref.dealloc %[[CAST]] : memref<?xf32>
// CHECK:         return %[[LOAD]] : f32
func.func @memref_load_store_alloc_dealloc_dynamic(%dim0 : index, %dim1: index, %dim2: index, %i0: index, %i1: index, %i2: index) -> f32 {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloca(%dim0, %dim1, %dim2) : memref<?x?x?xf32>
  %1 = memref.load %alloc[%i0, %i1, %i2] : memref<?x?x?xf32>
  memref.store %1, %alloc[%c2, %c1, %c0] : memref<?x?x?xf32>
  memref.dealloc %alloc : memref<?x?x?xf32>
  return %1 : f32
}
// -----

// CHECK-LABEL: @memref_copy_static(
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
func.func @memref_copy_static(%arg0: memref<2x3x4xi32>, %arg1: memref<2x3x4xi32>) {
  memref.copy %arg0, %arg1 : memref<2x3x4xi32> to memref<2x3x4xi32>
  return
}

// CHECK-LABEL: @memref_copy_dynamic(
// CHECK-SAME:    %[[DIM0:.*]]: index, %[[DIM1:.*]]: index, %[[DIM2:.*]]: index, %[[DIM3:.*]]: index, %[[DIM4:.*]]: index, %[[DIM5:.*]]: index)
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[LINEAR_SIZE:.*]] = affine.linearize_index disjoint [%[[DIM0]], %[[C0]], %[[C0]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[ALLOC:.*]] = memref.alloca(%[[LINEAR_SIZE]]) : memref<?xf32>
// CHECK:         %[[EXPAND_SHAPE:.*]] = memref.expand_shape %[[ALLOC]]
// CHECK{LITERAL}:                          [[0, 1, 2]]
// CHECK-SAME:                              output_shape [%[[DIM0]], %[[DIM1]], %[[DIM2]]]
// CHECK-SAME:                              : memref<?xf32> into memref<?x?x?xf32>
// CHECK:         %[[LINEAR_SIZE_1:.*]] = affine.linearize_index disjoint [%[[DIM3]], %[[C0]], %[[C0]]
// CHECK-SAME:                              by (%[[DIM3]], %[[DIM4]], %[[DIM5]]) : index
// CHECK:         %[[ALLOC_1:.*]] = memref.alloca(%[[LINEAR_SIZE_1]]) : memref<?xf32>
// CHECK:         %[[EXPAND_SHAPE_1:.*]] = memref.expand_shape %[[ALLOC]]
// CHECK{LITERAL}:                          [[0, 1, 2]]
// CHECK-SAME:                              output_shape [%[[DIM3]], %[[DIM4]], %[[DIM5]]]
// CHECK-SAME:                              : memref<?xf32> into memref<?x?x?xf32>
// CHECK:         %[[LINEAR_SIZE:.*]] = affine.linearize_index disjoint [%[[DIM0]], %[[C0]], %[[C0]]
// CHECK-SAME:                              by (%[[DIM0]], %[[DIM1]], %[[DIM2]]) : index
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %[[EXPAND_SHAPE]] to
// CHECK-SAME:                        offset: [0], sizes: [%[[LINEAR_SIZE]]], strides: [1] :
// CHECK-SAME:                        memref<?x?x?xf32> to memref<?xf32>
// CHECK:         %[[LINEAR_SIZE_1:.*]] = affine.linearize_index disjoint [%[[DIM3]], %[[C0]], %[[C0]]
// CHECK-SAME:                              by (%[[DIM3]], %[[DIM4]], %[[DIM5]]) : index
// CHECK:         %[[CAST_1:.*]] = memref.reinterpret_cast %[[EXPAND_SHAPE_1]] to
// CHECK-SAME:                        offset: [0], sizes: [%[[LINEAR_SIZE_1]]], strides: [1] :
// CHECK-SAME:                        memref<?x?x?xf32> to memref<?xf32>
// CHECK:         memref.copy %[[CAST]], %[[CAST_1]] : memref<?xf32> to memref<?xf32>
// CHECK:         return
func.func @memref_copy_dynamic(%dim0 : index, %dim1: index, %dim2: index, %dim3 : index, %dim4: index, %dim5: index) {
  %alloc = memref.alloca(%dim0, %dim1, %dim2) : memref<?x?x?xf32>
  %alloc1 = memref.alloca(%dim3, %dim4, %dim5) : memref<?x?x?xf32>
  memref.copy %alloc, %alloc1 : memref<?x?x?xf32> to memref<?x?x?xf32>
  return
}
