// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-memrefs))" %s | FileCheck

// TODO: support vector dialect.
// TODO: support stores.
// TODO: test subviews.

// -----

func.func @load_scalar_from_memref(%input: memref<4x8xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %value = memref.load %input[%c0, %c1] : memref<4x8xf32>
  return %value : f32
}

// -----

func.func @load_scalar_from_memref_static_dim(%input: memref<4x8xf32, strided<[8, 12], offset: 100>>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %value = memref.load %input[%c0, %c1] : memref<4x8xf32, strided<[8, 12], offset: 100>>
  return %value : f32
}

// -----

func.func @load_scalar_from_memref_static_dim_2(%input: memref<4x8xf32, strided<[8, 12], offset: 100>>, %row: index, %col: index) -> f32 {
  %value = memref.load %input[%row, %col] : memref<4x8xf32, strided<[8, 12], offset: 100>>
  return %value : f32
}

// -----

func.func @load_scalar_from_memref_dynamic_dim(%input: memref<4x8xf32, strided<[?, ?], offset: ?>>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %value = memref.load %input[%c0, %c1] : memref<4x8xf32, strided<[?, ?], offset: ?>>
  return %value : f32
}

// -----

func.func @load_scalar_from_memref_dynamic_dim_2(%input: memref<4x8xf32, strided<[?, ?], offset: ?>>, %row: index, %col: index) -> f32 {
  %value = memref.load %input[%row, %col] : memref<4x8xf32, strided<[?, ?], offset: ?>>
  return %value : f32
}

// -----

func.func @load_scalar_from_memref(%input: memref<4x8xf32>, %row: index, %col: index) -> memref<1x1xf32, strided<[8, 1], offset: ?>> {
  %subview = memref.subview %input[%row, %col] [1, 1] [1, 1] : memref<4x8xf32> to memref<1x1xf32, strided<[8, 1], offset: ?>>
  return %subview : memref<1x1xf32, strided<[8, 1], offset: ?>>
}

// -----

// TODO: change to support vector.load 
func.func @load_vector_from_memref(%input: memref<4x8xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %value = vector.load %input[%c0, %c0] : memref<4x8xf32>, vector<8xf32>
  return %value : vector<8xf32>
}

// -----

func.func @load_vector_from_memref_odd(%input: memref<3x7xi2>) -> vector<3xi2> {
  %c0 = arith.constant 0 : index
  %value = vector.load %input[%c0, %c0] : memref<3x7xi2>, vector<3xi2>
  return %value : vector<3xi2>
}

// -----

func.func @load_vector_from_memref_dynamic(%input: memref<3x7xi2>, %row: index, %col: index) -> vector<3xi2> {
  %value = vector.load %input[%row, %col] : memref<3x7xi2>, vector<3xi2>
  return %value : vector<3xi2>
}

