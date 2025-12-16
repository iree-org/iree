// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

// Test that canonicalization resolves dim to the dynamic values
// from set_encoding.

#encoding = #iree_encoding.testing<>
func.func @encoding_dim_canonicalize_dynamic(%arg0: tensor<?x?xf32>, %m: index, %n: index, %k: index) -> (index, index, index) {
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  %dim_m = iree_encoding.dim %0[0] : tensor<?x?xf32, #encoding>
  %dim_n = iree_encoding.dim %0[1] : tensor<?x?xf32, #encoding>
  %dim_k = iree_encoding.dim %0[2] : tensor<?x?xf32, #encoding>
  return %dim_m, %dim_n, %dim_k : index, index, index
}
// CHECK-LABEL: func.func @encoding_dim_canonicalize_dynamic
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[M:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[N:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[K:[a-zA-Z0-9]+]]: index
// CHECK:         return %[[M]], %[[N]], %[[K]]

// -----

// Test that canonicalization traces through tensor.cast for dynamic values.

#encoding = #iree_encoding.testing<>
func.func @encoding_dim_canonicalize_through_cast(%arg0: tensor<?x?xf32>, %m: index, %n: index) -> (index, index) {
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  %1 = tensor.cast %0 : tensor<?x?xf32, #encoding> to tensor<4x8xf32, #encoding>
  %dim_m = iree_encoding.dim %1[0] : tensor<4x8xf32, #encoding>
  %dim_n = iree_encoding.dim %1[1] : tensor<4x8xf32, #encoding>
  return %dim_m, %dim_n : index, index
}
// CHECK-LABEL: func.func @encoding_dim_canonicalize_through_cast
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[M:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[N:[a-zA-Z0-9]+]]: index
// CHECK:         return %[[M]], %[[N]]

// -----

// Test that dim is NOT resolved when there's no set_encoding in producer chain.

#encoding = #iree_encoding.testing<>
func.func @encoding_dim_no_set_encoding(%arg0: tensor<?x?xf32, #encoding>) -> index {
  %dim = iree_encoding.dim %arg0[0] : tensor<?x?xf32, #encoding>
  return %dim : index
}
// CHECK-LABEL: func.func @encoding_dim_no_set_encoding
// CHECK:         %[[DIM:.+]] = iree_encoding.dim
// CHECK:         return %[[DIM]]

// -----

// Test that dim traces through linalg.fill (DPS op).
// The ReifyEncodingDimThroughDPS pattern forwards the query to the init operand.

#encoding = #iree_encoding.testing<>
func.func @encoding_dim_through_linalg_fill(%arg0: tensor<?x?xf32>, %m: index, %n: index) -> (index, index) {
  %cst = arith.constant 0.0 : f32
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding>
  %dim_m = iree_encoding.dim %1[0] : tensor<?x?xf32, #encoding>
  %dim_n = iree_encoding.dim %1[1] : tensor<?x?xf32, #encoding>
  return %dim_m, %dim_n : index, index
}
// CHECK-LABEL: func.func @encoding_dim_through_linalg_fill
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[M:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[N:[a-zA-Z0-9]+]]: index
// CHECK:         return %[[M]], %[[N]]

// -----

// Test that dim traces through linalg.generic (DPS op).

#encoding = #iree_encoding.testing<>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @encoding_dim_through_linalg_generic(%arg0: tensor<?x?xf32>, %m: index, %n: index) -> (index, index) {
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  %1 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%0 : tensor<?x?xf32, #encoding>) outs(%0 : tensor<?x?xf32, #encoding>) {
  ^bb0(%in: f32, %out: f32):
    %neg = arith.negf %in : f32
    linalg.yield %neg : f32
  } -> tensor<?x?xf32, #encoding>
  %dim_m = iree_encoding.dim %1[0] : tensor<?x?xf32, #encoding>
  %dim_n = iree_encoding.dim %1[1] : tensor<?x?xf32, #encoding>
  return %dim_m, %dim_n : index, index
}
// CHECK-LABEL: func.func @encoding_dim_through_linalg_generic
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[M:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[N:[a-zA-Z0-9]+]]: index
// CHECK:         return %[[M]], %[[N]]

// -----

// Test that dim traces through a chain of operations:
// set_encoding -> tensor.cast -> linalg.fill

#encoding = #iree_encoding.testing<>
func.func @encoding_dim_through_chain(%arg0: tensor<?x?xf32>, %m: index, %n: index) -> (index, index) {
  %cst = arith.constant 0.0 : f32
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  %1 = tensor.cast %0 : tensor<?x?xf32, #encoding> to tensor<4x8xf32, #encoding>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<4x8xf32, #encoding>) -> tensor<4x8xf32, #encoding>
  %dim_m = iree_encoding.dim %2[0] : tensor<4x8xf32, #encoding>
  %dim_n = iree_encoding.dim %2[1] : tensor<4x8xf32, #encoding>
  return %dim_m, %dim_n : index, index
}
// CHECK-LABEL: func.func @encoding_dim_through_chain
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[M:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[N:[a-zA-Z0-9]+]]: index
// CHECK:         return %[[M]], %[[N]]

// -----

// Test that dim is NOT resolved through DPS op when the producer chain
// doesn't have a set_encoding.

#encoding = #iree_encoding.testing<>
func.func @encoding_dim_through_dps_no_set_encoding(%arg0: tensor<?x?xf32, #encoding>) -> index {
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding>
  %dim = iree_encoding.dim %0[0] : tensor<?x?xf32, #encoding>
  return %dim : index
}
// CHECK-LABEL: func.func @encoding_dim_through_dps_no_set_encoding
// CHECK:         %[[DIM:.+]] = iree_encoding.dim
// CHECK:         return %[[DIM]]
