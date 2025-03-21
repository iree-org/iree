// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-memrefs))" %s | FileCheck

// TODO: support vector dialect.
// TODO: support stores.
// TODO: test subviews.

// -----

func.func @load_scalar_from_memref(%input: memref<4x8xf32>) -> f32 {
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 2 : index
  %value = memref.load %input[%c0, %c1] : memref<4x8xf32>
  return %value : f32
}
// CHECK-LABEL: func @load_scalar_from_memref

// -----

func.func @load_scalar_from_memref_static_dim(%input: memref<4x8xf32, strided<[8, 12], offset: 100>>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %value = memref.load %input[%c0, %c1] : memref<4x8xf32, strided<[8, 12], offset: 100>>
  return %value : f32
}
// CHECK-LABEL: func @load_scalar_from_memref_static_dim

// -----

func.func @load_scalar_from_memref_static_dim_2(%input: memref<4x8xf32, strided<[8, 12], offset: 100>>, %row: index, %col: index) -> f32 {
  %value = memref.load %input[%row, %col] : memref<4x8xf32, strided<[8, 12], offset: 100>>
  return %value : f32
}
// CHECK-LABEL: func @load_scalar_from_memref_static_dim_2

// -----

func.func @load_scalar_from_memref_dynamic_dim(%input: memref<4x8xf32, strided<[?, ?], offset: ?>>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %value = memref.load %input[%c0, %c1] : memref<4x8xf32, strided<[?, ?], offset: ?>>
  return %value : f32
}
// CHECK-LABEL: func @load_scalar_from_memref_dynamic_dim

// -----

func.func @load_scalar_from_memref_dynamic_dim_2(%input: memref<4x8xf32, strided<[?, ?], offset: ?>>, %row: index, %col: index) -> f32 {
  %value = memref.load %input[%row, %col] : memref<4x8xf32, strided<[?, ?], offset: ?>>
  return %value : f32
}
// CHECK-LABEL: func @load_scalar_from_memref_dynamic_dim_2

// -----

func.func @load_scalar_from_memref_subview(%input: memref<4x8xf32>, %row: index, %col: index) -> memref<1x1xf32, strided<[8, 1], offset: ?>> {
  %subview = memref.subview %input[%row, %col] [1, 1] [1, 1] : memref<4x8xf32> to memref<1x1xf32, strided<[8, 1], offset: ?>>
  return %subview : memref<1x1xf32, strided<[8, 1], offset: ?>>
}
// CHECK-LABEL: func @load_scalar_from_memref_subview

// -----

func.func @store_scalar_from_memref_static_dim_2(%input: memref<4x8xf32, strided<[8, 12], offset: 100>>, %row: index, %col: index, %value: f32) {
  memref.store %value, %input[%row, %col] : memref<4x8xf32, strided<[8, 12], offset: 100>>
  return
}
// CHECK-LABEL: func @store_scalar_from_memref_static_dim_2

// -----

func.func @store_scalar_from_memref_dynamic_dim_2(%input: memref<4x8xf32, strided<[?, ?], offset: ?>>, %row: index, %col: index, %value: f32) {
  memref.store %value, %input[%row, %col] : memref<4x8xf32, strided<[?, ?], offset: ?>>
  return
}
// CHECK-LABEL: func @store_scalar_from_memref_dynamic_dim_2

// -----

func.func @load_vector_from_memref(%input: memref<4x8xf32>) -> vector<8xf32> {
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %value = vector.load %input[%c3, %c6] : memref<4x8xf32>, vector<8xf32>
  return %value : vector<8xf32>
}
// CHECK-LABEL: func @load_vector_from_memref

// -----

func.func @load_vector_from_memref_odd(%input: memref<3x7xi2>) -> vector<3xi2> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %value = vector.load %input[%c1, %c3] : memref<3x7xi2>, vector<3xi2>
  return %value : vector<3xi2>
}
// CHECK-LABEL: func @load_vector_from_memref_odd

// -----

func.func @load_vector_from_memref_dynamic(%input: memref<3x7xi2>, %row: index, %col: index) -> vector<3xi2> {
  %value = vector.load %input[%row, %col] : memref<3x7xi2>, vector<3xi2>
  return %value : vector<3xi2>
}
// CHECK-LABEL: func @load_vector_from_memref_dynamic

// -----

func.func @store_vector_to_memref_odd(%input: memref<3x7xi2>, %value: vector<3xi2>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  vector.store %value, %input[%c1, %c3] : memref<3x7xi2>, vector<3xi2>
  return
}
// CHECK-LABEL: func @store_vector_to_memref_odd

// -----

func.func @store_vector_to_memref_dynamic(%input: memref<3x7xi2>, %value: vector<3xi2>, %row: index, %col: index) {
  vector.store %value, %input[%row, %col] : memref<3x7xi2>, vector<3xi2>
  return
}
// CHECK-LABEL: func @store_vector_to_memref_dynamic
