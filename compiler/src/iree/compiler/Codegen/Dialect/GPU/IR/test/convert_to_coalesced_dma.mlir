// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-convert-to-coalesced-dma))" %s | FileCheck %s

// Test converting gather patterns to coalesced DMA operations

// CHECK-LABEL: func.func @basic_gather_pattern
func.func @basic_gather_pattern(%indices: memref<32xindex>,
                                %source: memref<2048xf32>,
                                %dest: memref<128xf32>) {
  // TODO: Add pattern that should be converted to coalesced_gather_dma
  // For now, this is a placeholder test
  // CHECK: return
  return
}

// -----

// Test with different memory spaces
// CHECK-LABEL: func.func @gather_with_memory_spaces
func.func @gather_with_memory_spaces(%indices: memref<32xindex>,
                                     %source: memref<2048xf32, #gpu.address_space<global>>,
                                     %dest: memref<128xf32, #gpu.address_space<workgroup>>) {
  // TODO: Add pattern that involves proper memory spaces for DMA
  // CHECK: return
  return
}

// -----

// Test with nested loops that could become coalesced DMA
// CHECK-LABEL: func.func @nested_loop_gather
func.func @nested_loop_gather(%indices: memref<64xindex>,
                              %source: memref<4096xf32>,
                              %dest: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index

  // TODO: Add scf.forall or scf.for loops that could be converted
  // This might involve a pattern like:
  // scf.forall (%i, %j) in (32, 1) {
  //   ... gather operations ...
  // }

  // CHECK: return
  return
}

// -----

// Test case that should NOT be converted (negative test)
// CHECK-LABEL: func.func @no_conversion_simple_load
func.func @no_conversion_simple_load(%source: memref<1024xf32>,
                                     %dest: memref<1024xf32>) {
  %c0 = arith.constant 0 : index
  %val = memref.load %source[%c0] : memref<1024xf32>
  memref.store %val, %dest[%c0] : memref<1024xf32>

  // CHECK-NOT: iree_gpu.coalesced_gather_dma
  // CHECK: memref.load
  // CHECK: memref.store
  return
}

// -----

// Test with affine operations that might be converted
// CHECK-LABEL: func.func @affine_gather_pattern
func.func @affine_gather_pattern(%indices: memref<32xindex>,
                                 %source: memref<2048xf32>,
                                 %dest: memref<128xf32>) {
  affine.for %i = 0 to 32 {
    // TODO: Add affine.load operations with gather patterns
    // that could be converted to coalesced DMA
  }

  // CHECK: return
  return
}

// -----

// Test with tensor operations that might be converted after bufferization
// CHECK-LABEL: func.func @tensor_gather_pattern
func.func @tensor_gather_pattern(%indices: tensor<32xindex>,
                                 %source: tensor<2048xf32>) -> tensor<128xf32> {
  // TODO: Add tensor.extract or linalg operations that represent gather patterns
  // These would typically be bufferized first, then converted

  %empty = tensor.empty() : tensor<128xf32>

  // Placeholder - actual implementation would have gather-like operations

  // CHECK: return
  return %empty : tensor<128xf32>
}

// -----

// Test with dynamic shapes
// CHECK-LABEL: func.func @dynamic_gather_pattern
func.func @dynamic_gather_pattern(%indices: memref<?xindex>,
                                  %source: memref<?xf32>,
                                  %dest: memref<?xf32>) {
  // TODO: Add patterns with dynamic dimensions
  // The pass should handle or reject these appropriately

  // CHECK: return
  return
}

// -----

// Test with multiple data types
// CHECK-LABEL: func.func @multi_type_gather
func.func @multi_type_gather(%indices: memref<32xindex>,
                             %source_f32: memref<2048xf32>,
                             %source_f16: memref<2048xf16>,
                             %dest_f32: memref<128xf32>,
                             %dest_f16: memref<128xf16>) {
  // TODO: Test conversion with different element types

  // CHECK: return
  return
}

// -----

// Test with strided layouts
// CHECK-LABEL: func.func @strided_gather_pattern
func.func @strided_gather_pattern(%indices: memref<32xindex>,
                                  %source: memref<2048xf32, strided<[2], offset: 0>>,
                                  %dest: memref<128xf32>) {
  // TODO: Add patterns with strided memory layouts

  // CHECK: return
  return
}

// -----

// Test pattern that should create multiple coalesced DMA operations
// CHECK-LABEL: func.func @multiple_gathers
func.func @multiple_gathers(%indices1: memref<32xindex>,
                            %indices2: memref<32xindex>,
                            %source: memref<2048xf32>,
                            %dest1: memref<128xf32>,
                            %dest2: memref<128xf32>) {
  // TODO: Add patterns that would create multiple coalesced_gather_dma ops

  // CHECK: return
  return
}
