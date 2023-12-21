// RUN: iree-dialects-opt --canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: @to_simt_to_simd_fold
func.func @to_simt_to_simd_fold(%simd: vector<64x64xf32>) -> vector<64x64xf32> {
  // Both to_simt and to_simd should be folded away.
  // CHECK-NOT: iree_vector_ext.to_simt
  %simt = iree_vector_ext.to_simt %simd : vector<64x64xf32> -> vector<4x4x4xf32>
  // CHECK-NOT: iree_vector_ext.to_simd
  %simd_out = iree_vector_ext.to_simd %simt : vector<4x4x4xf32> -> vector<64x64xf32> 
  func.return %simd_out : vector<64x64xf32>
}

// -----

// CHECK-LABEL: @to_simd_to_simt_fold
func.func @to_simd_to_simt_fold(%simt: vector<4x4x4xf32>) -> vector<4x4x4xf32> {
  // Both to_simt and to_simd should be folded away.
  // CHECK-NOT: iree_vector_ext.to_simt
  %simd = iree_vector_ext.to_simd %simt : vector<4x4x4xf32> -> vector<64x64xf32>
  // CHECK-NOT: iree_vector_ext.to_simd
  %simt_out = iree_vector_ext.to_simt %simd : vector<64x64xf32> -> vector<4x4x4xf32> 
  func.return %simt_out : vector<4x4x4xf32>
}

// -----
