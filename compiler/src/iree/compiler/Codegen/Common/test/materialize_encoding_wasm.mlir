// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#out = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @simd32(%lhs: tensor<?x?xf32, #lhs>, %rhs: tensor<?x?xf32, #rhs>, %acc: tensor<?x?xf32, #out>) -> tensor<?x?xf32, #out> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "system-wasm-wasm_32", {target_triple = "wasm32-unknown-emscripten", cpu_features = "+simd128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>) outs(%acc : tensor<?x?xf32, #out>) -> tensor<?x?xf32, #out>
  return %result : tensor<?x?xf32, #out>
}
// CHECK-LABEL: func @simd32(
// CHECK-SAME: tensor<?x?x4x1xf32>
// CHECK-SAME: tensor<?x?x8x1xf32>
// CHECK-SAME: tensor<?x?x4x8xf32>
// CHECK: linalg.mmt4d

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#out = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @simd64(%lhs: tensor<?x?xf32, #lhs>, %rhs: tensor<?x?xf32, #rhs>, %acc: tensor<?x?xf32, #out>) -> tensor<?x?xf32, #out> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "system-wasm-wasm_64", {target_triple = "wasm64-unknown-emscripten", cpu_features = "+simd128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>) outs(%acc : tensor<?x?xf32, #out>) -> tensor<?x?xf32, #out>
  return %result : tensor<?x?xf32, #out>
}
// CHECK-LABEL: func @simd64(
// CHECK-SAME: tensor<?x?x4x1xf32>
// CHECK-SAME: tensor<?x?x8x1xf32>
// CHECK-SAME: tensor<?x?x4x8xf32>
// CHECK: linalg.mmt4d

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#out = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @scalar32(%lhs: tensor<?x?xf32, #lhs>, %rhs: tensor<?x?xf32, #rhs>, %acc: tensor<?x?xf32, #out>) -> tensor<?x?xf32, #out> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "system-wasm-wasm_32", {target_triple = "wasm32-unknown-emscripten", cpu_features = "", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>) outs(%acc : tensor<?x?xf32, #out>) -> tensor<?x?xf32, #out>
  return %result : tensor<?x?xf32, #out>
}
// CHECK-LABEL: func @scalar32(
// CHECK-NOT: linalg.mmt4d
// CHECK: linalg.matmul

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#out = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @scalar64(%lhs: tensor<?x?xf32, #lhs>, %rhs: tensor<?x?xf32, #rhs>, %acc: tensor<?x?xf32, #out>) -> tensor<?x?xf32, #out> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "system-wasm-wasm_64", {target_triple = "wasm64-unknown-emscripten", cpu_features = "", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>) outs(%acc : tensor<?x?xf32, #out>) -> tensor<?x?xf32, #out>
  return %result : tensor<?x?xf32, #out>
}
// CHECK-LABEL: func @scalar64(
// CHECK-NOT: linalg.mmt4d
// CHECK: linalg.matmul

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f64, f64, f64], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f64, f64, f64], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#out = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f64, f64, f64], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @f64_32(%lhs: tensor<?x?xf64, #lhs>, %rhs: tensor<?x?xf64, #rhs>, %acc: tensor<?x?xf64, #out>) -> tensor<?x?xf64, #out> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "system-wasm-wasm_32", {target_triple = "wasm32-unknown-emscripten", cpu_features = "+simd128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf64, #lhs>, tensor<?x?xf64, #rhs>) outs(%acc : tensor<?x?xf64, #out>) -> tensor<?x?xf64, #out>
  return %result : tensor<?x?xf64, #out>
}
// CHECK-LABEL: func @f64_32(
// CHECK-NOT: linalg.mmt4d
// CHECK: linalg.matmul

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f64, f64, f64], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f64, f64, f64], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#out = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f64, f64, f64], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @f64_64(%lhs: tensor<?x?xf64, #lhs>, %rhs: tensor<?x?xf64, #rhs>, %acc: tensor<?x?xf64, #out>) -> tensor<?x?xf64, #out> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "system-wasm-wasm_64", {target_triple = "wasm64-unknown-emscripten", cpu_features = "+simd128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf64, #lhs>, tensor<?x?xf64, #rhs>) outs(%acc : tensor<?x?xf64, #out>) -> tensor<?x?xf64, #out>
  return %result : tensor<?x?xf64, #out>
}
// CHECK-LABEL: func @f64_64(
// CHECK-NOT: linalg.mmt4d
// CHECK: linalg.matmul
