// RUN: iree-opt --split-input-file --verify-diagnostics %s

// -----

module {
  func.func @export_config_invalid_type() attributes {
    // expected-error @+1 {{expected workgroup size to have atmost 3 entries}}
    export_config = #iree_codegen.export_config<workgroup_size = [4, 1, 1, 1]>
  } {
    return
  }
}

// -----

func.func @load_from_buffer_invalid_shape(%arg0: memref<5xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{buffer and tensor shapes must be compatible and element types must match}}
  %value = iree_codegen.load_from_buffer %arg0 : memref<5xf32> -> tensor<4xf32>
  return %value : tensor<4xf32>
}

// -----

func.func @load_from_buffer_invalid_element_type(%arg0: memref<4xf32>) -> tensor<4xf16> {
  // expected-error @+1 {{buffer and tensor shapes must be compatible and element types must match}}
  %value = iree_codegen.load_from_buffer %arg0 : memref<4xf32> -> tensor<4xf16>
  return %value : tensor<4xf16>
}

// -----

func.func @store_to_buffer_invalid_shape(%arg0: tensor<4xf32>, %arg1: memref<5xf32>) {
  // expected-error @+1 {{tensor and buffer shapes must be compatible and element types must match}}
  iree_codegen.store_to_buffer %arg0, %arg1 : tensor<4xf32> into memref<5xf32>
  return
}

// -----

func.func @store_to_buffer_invalid_element_type(%arg0: tensor<4xf16>, %arg1: memref<4xf32>) {
  // expected-error @+1 {{tensor and buffer shapes must be compatible and element types must match}}
  iree_codegen.store_to_buffer %arg0, %arg1 : tensor<4xf16> into memref<4xf32>
  return
}

// -----

func.func private @swizzle(%offset: index) -> i32
func.func @symbolic_swizzle_wrong_signature(%arg0: memref<?xf32>) -> memref<?xf32> {
  // expected-error @+1 {{Symbolic swizzle attr defining function requires single index result}}
  %0 = iree_codegen.swizzle_hint %arg0[#iree_codegen.swizzle_impl<@swizzle, 1>] : memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

func.func @symbolic_swizzle_missing_function(%arg0: memref<?xf32>) -> memref<?xf32> {
  // expected-error @+1 {{Symbolic swizzle attr could not find defining function}}
  %0 = iree_codegen.swizzle_hint %arg0[#iree_codegen.swizzle_impl<@swizzle, 1>] : memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

util.global @swizzle : index
func.func @symbolic_swizzle_wrong_kind(%arg0: memref<?xf32>) -> memref<?xf32> {
  // expected-error @+1 {{Symbolic swizzle attr could not find defining function}}
  %0 = iree_codegen.swizzle_hint %arg0[#iree_codegen.swizzle_impl<@swizzle, 1>] : memref<?xf32>
  return %0 : memref<?xf32>
}
