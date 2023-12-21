// RUN: iree-dialects-opt --canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: @to_simt_to_simd_fold
func.func @to_simt_to_simd_fold(%simd: vector<64x64xf32>) -> vector<64x64xf32> {
  // Both to_simt and to_simd should be dce-ed after folding.
  // CHECK-NOT: iree_vector_ext.to_simt
  %simt = iree_vector_ext.to_simt %simd : vector<64x64xf32> -> vector<4x4x4xf32>
  // CHECK-NOT: iree_vector_ext.to_simd
  %simd_out = iree_vector_ext.to_simd %simt : vector<4x4x4xf32> -> vector<64x64xf32> 
  func.return %simd_out : vector<64x64xf32>
}

// -----

// CHECK-LABEL: @to_simd_to_simt_fold
func.func @to_simd_to_simt_fold(%simt: vector<4x4x4xf32>) -> vector<4x4x4xf32> {
  // Both to_simt and to_simd should be dce-ed after folding.
  // CHECK-NOT: iree_vector_ext.to_simt
  %simd = iree_vector_ext.to_simd %simt : vector<4x4x4xf32> -> vector<64x64xf32>
  // CHECK-NOT: iree_vector_ext.to_simd
  %simt_out = iree_vector_ext.to_simt %simd : vector<64x64xf32> -> vector<4x4x4xf32> 
  func.return %simt_out : vector<4x4x4xf32>
}

// -----

// CHECK-LABEL: @to_simd_to_simt_multi_use
func.func @to_simd_to_simt_multi_use(%simt: vector<4x4x4xf32>) -> (vector<4x4x4xf16>, vector<64x64xf32>) {
  // The to_simd operation should not be dce-ed after folding because it is returned.
  // CHECK: iree_vector_ext.to_simd %[[SIMT:.*]] : vector<4x4x4xf32> -> vector<64x64xf32>
  %simd = iree_vector_ext.to_simd %simt : vector<4x4x4xf32> -> vector<64x64xf32>
  // The to_simt operation should be dce-ed after folding.
  // CHECK-NOT: iree_vector_ext.to_simt
  %simt_out = iree_vector_ext.to_simt %simd : vector<64x64xf32> -> vector<4x4x4xf32> 

  // Check if the folding happened correctly.
  // CHECK: arith.truncf %[[SIMT]]
  %trunced = arith.truncf %simt_out : vector<4x4x4xf32> to vector<4x4x4xf16>

  func.return %trunced, %simd : vector<4x4x4xf16>, vector<64x64xf32>
}

// -----
