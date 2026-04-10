// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-lowering, iree-llvmgpu-legalize-nd-vectors, canonicalize, iree-codegen-vector-transfer-lowering, cse))" --split-input-file %s | FileCheck %s

// Test unrolling of a 2D transfer_gather representing an embedding lookup:
// outer dim is gathered (indices), inner dim is contiguous.

func.func @transfer_gather_unroll_embedding_lookup(
  %source: memref<4096x64xf16>,
  %indices: vector<4xindex>) -> vector<4x64xf16> {
  %cst = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%indices : vector<4xindex>], %cst {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>]
  } : memref<4096x64xf16>, vector<4x64xf16>
  return %out : vector<4x64xf16>
}

// CHECK-LABEL: func.func @transfer_gather_unroll_embedding_lookup
// CHECK-NOT: transfer_gather
// CHECK-COUNT-4: vector.load
// CHECK-NOT: transfer_gather

// -----

// Test unrolling of a masked 2D transfer_gather.
// Same embedding lookup shape but with a mask on the result.

func.func @transfer_gather_unroll_masked(
  %source: memref<4096x64xf16>,
  %indices: vector<4xindex>,
  %mask: vector<4x64xi1>) -> vector<4x64xf16> {
  %cst = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%indices : vector<4xindex>], %cst, %mask {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>,
                     affine_map<(d0, d1)[s0] -> (d0, d1)>]
  } : memref<4096x64xf16>, vector<4x64xf16>, vector<4x64xi1>
  return %out : vector<4x64xf16>
}

// CHECK-LABEL: func.func @transfer_gather_unroll_masked
// CHECK-NOT: transfer_gather
// CHECK-COUNT-4: vector.maskedload
// CHECK-NOT: transfer_gather

// -----

// Test unrolling of a 3D transfer_gather with a transposed 2D index vector.
// The first two output dims (d0=4, d1=8) are both gathered via a single
// index vec of shape 8x4 (note: d1 before d0, i.e. "transposed").
// The inner dim (d2=64) is contiguous.

func.func @transfer_gather_unroll_transposed_index(
  %source: memref<4096x64xf16>,
  %indices: vector<8x4xindex>) -> vector<4x8x64xf16> {
  %cst = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_gather %source[%c0, %c0]
  [%indices : vector<8x4xindex>], %cst {
    indexing_maps = [affine_map<(d0, d1, d2)[s0] -> (s0, d2)>,
                     affine_map<(d0, d1, d2)[s0] -> (d1, d0)>]
  } : memref<4096x64xf16>, vector<4x8x64xf16>
  return %out : vector<4x8x64xf16>
}

// CHECK-LABEL: func.func @transfer_gather_unroll_transposed_index
// CHECK-NOT: transfer_gather
// CHECK-COUNT-32: vector.load
// CHECK-NOT: transfer_gather

// -----

// Test unrolling of a 2D transfer_scatter: outer dim is scattered (indices),
// inner dim is contiguous.

func.func @transfer_scatter_unroll_embedding_write(
  %source: memref<4096x64xf16>,
  %vector: vector<4x64xf16>,
  %indices: vector<4xindex>) {
  %c0 = arith.constant 0 : index
  iree_vector_ext.transfer_scatter %vector into %source[%c0, %c0]
  [%indices : vector<4xindex>] {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>]
  } : vector<4x64xf16>, memref<4096x64xf16>
  return
}

// CHECK-LABEL: func.func @transfer_scatter_unroll_embedding_write
// CHECK-NOT: transfer_scatter
// CHECK-COUNT-4: vector.store {{.+}} : memref<4096x64xf16>, vector<64xf16>
// CHECK-NOT: transfer_scatter

// -----

// Test unrolling of a masked 2D transfer_scatter.

func.func @transfer_scatter_unroll_masked(
  %source: memref<4096x64xf16>,
  %vector: vector<4x64xf16>,
  %indices: vector<4xindex>,
  %mask: vector<4x64xi1>) {
  %c0 = arith.constant 0 : index
  iree_vector_ext.transfer_scatter %vector into %source[%c0, %c0]
  [%indices : vector<4xindex>], %mask {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>,
                     affine_map<(d0, d1)[s0] -> (d0, d1)>]
  } : vector<4x64xf16>, memref<4096x64xf16>, vector<4x64xi1>
  return
}

// CHECK-LABEL: func.func @transfer_scatter_unroll_masked
// CHECK-NOT: transfer_scatter
// CHECK-COUNT-4: vector.maskedstore {{.+}} : memref<4096x64xf16>, vector<64xi1>, vector<64xf16>
// CHECK-NOT: transfer_scatter

// -----

// Test unrolling of a 2D transfer_scatter with tensor semantics.

func.func @transfer_scatter_unroll_tensor(
  %dest: tensor<4096x64xf16>,
  %vector: vector<4x64xf16>,
  %indices: vector<4xindex>) -> tensor<4096x64xf16> {
  %c0 = arith.constant 0 : index
  %out = iree_vector_ext.transfer_scatter %vector into %dest[%c0, %c0]
  [%indices : vector<4xindex>] {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (s0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>]
  } : vector<4x64xf16>, tensor<4096x64xf16> -> tensor<4096x64xf16>
  return %out : tensor<4096x64xf16>
}

// CHECK-LABEL: func.func @transfer_scatter_unroll_tensor
// CHECK-NOT: transfer_scatter
// CHECK-COUNT-4: vector.transfer_write {{.+}} : vector<64xf16>, tensor<4096x64xf16>
// CHECK-NOT: transfer_scatter

// -----

// Test unrolling of a 3D transfer_scatter with a transposed 2D index vector.
// The first two output dims (d0=4, d1=8) are both scattered via a single
// index vec of shape 8x4 (note: d1 before d0, i.e. "transposed").
// The inner dim (d2=64) is contiguous.

func.func @transfer_scatter_unroll_transposed_index(
  %dest: memref<4096x64xf16>,
  %vector: vector<4x8x64xf16>,
  %indices: vector<8x4xindex>) {
  %c0 = arith.constant 0 : index
  iree_vector_ext.transfer_scatter %vector into %dest[%c0, %c0]
  [%indices : vector<8x4xindex>] {
    indexing_maps = [affine_map<(d0, d1, d2)[s0] -> (s0, d2)>,
                     affine_map<(d0, d1, d2)[s0] -> (d1, d0)>]
  } : vector<4x8x64xf16>, memref<4096x64xf16>
  return
}

// CHECK-LABEL: func.func @transfer_scatter_unroll_transposed_index
// CHECK-NOT: transfer_scatter
// CHECK-COUNT-32: vector.store {{.+}} : memref<4096x64xf16>, vector<64xf16>
// CHECK-NOT: transfer_scatter
