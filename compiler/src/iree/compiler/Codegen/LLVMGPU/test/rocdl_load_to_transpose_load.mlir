// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-rocdl-load-to-transpose-load))' %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Tests for hint removal (when transformation doesn't apply)
//===----------------------------------------------------------------------===//

// Test: 1D vector - hints removed but no transformation (wrong vector shape)
// CHECK-LABEL: func.func @hint_removal_1d_vector
func.func @hint_removal_1d_vector(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // CHECK-NOT: iree_codegen.index_hint
  // CHECK: vector.transfer_read
  // CHECK-NOT: amdgpu.transpose_load
  %0 = vector.transfer_read %src[%row, %col], %cst {in_bounds = [true]}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  return %0 : vector<4xf16>
}

// -----

// Test: Orphaned hint (not used) should be dropped
// CHECK-LABEL: func.func @orphaned_hint
func.func @orphaned_hint() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: iree_codegen.index_hint
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %c1 {hint = #iree_gpu.lane_increment<16>} : index
  return
}

// -----

// Test: Global memory - hints removed but no transformation
// CHECK-LABEL: func.func @no_transform_global_memory
func.func @no_transform_global_memory(%src: memref<128x256xf16>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // CHECK-NOT: iree_codegen.index_hint
  // CHECK: vector.transfer_read
  // CHECK-NOT: amdgpu.transpose_load
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Column hint not used for unit dim - no transformation
// CHECK-LABEL: func.func @no_transform_column_hint_unused
func.func @no_transform_column_hint_unused(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // Using constant for column instead of col - NOT valid
  // CHECK-NOT: amdgpu.transpose_load
  // CHECK: vector.transfer_read
  %0 = vector.transfer_read %src[%row, %c5], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Column size != 1 - no transformation
// CHECK-LABEL: func.func @no_transform_column_size_not_1
func.func @no_transform_column_size_not_1(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x2xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // CHECK-NOT: amdgpu.transpose_load
  // CHECK: vector.transfer_read
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x2xf16>
  return %0 : vector<4x2xf16>
}

// -----

// Test: Unsupported element type (f32) - no transformation
// CHECK-LABEL: func.func @no_transform_f32
func.func @no_transform_f32(%src: memref<128x256xf32, #gpu.address_space<workgroup>>) -> vector<4x1xf32> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f32
  // CHECK-NOT: amdgpu.transpose_load
  // CHECK: vector.transfer_read
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<4x1xf32>
  return %0 : vector<4x1xf32>
}

// -----

// Test: Row size not multiple of intrinsic size - no transformation
// CHECK-LABEL: func.func @no_transform_non_multiple_row_size
func.func @no_transform_non_multiple_row_size(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<3x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // CHECK-NOT: amdgpu.transpose_load
  // CHECK: vector.transfer_read
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<3x1xf16>
  return %0 : vector<3x1xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Tests for row index validation - row index must be uniform across 16 lanes
//===----------------------------------------------------------------------===//

// Test: Row index from column hint (lane_increment) - should NOT transform
// CHECK-LABEL: func.func @no_transform_row_from_column_hint
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_row_from_column_hint(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // Row from col (lane_increment - varies across lanes) - INVALID
  %0 = vector.transfer_read %src[%col, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Row index depends on gpu.thread_id directly - should NOT transform
// CHECK-LABEL: func.func @no_transform_row_from_thread_id
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_row_from_thread_id(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // Row from thread_id directly - NOT from hint
  %0 = vector.transfer_read %src[%tid, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Row index depends on gpu.lane_id - should NOT transform
// CHECK-LABEL: func.func @no_transform_row_from_lane_id
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_row_from_lane_id(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %lane = gpu.lane_id
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // Row depends on lane_id
  %0 = vector.transfer_read %src[%lane, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Row index from a function argument - should NOT transform
// CHECK-LABEL: func.func @no_transform_row_from_block_arg
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_row_from_block_arg(%src: memref<128x256xf16, #gpu.address_space<workgroup>>, %row_arg: index) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // Row from function argument - could vary per thread
  %0 = vector.transfer_read %src[%row_arg, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Row index mixes row hint with thread_id - should NOT transform
// CHECK-LABEL: func.func @no_transform_row_mixed_with_thread_id
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_row_mixed_with_thread_id(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // Row is row + thread_id (thread_id varies per lane)
  %row_mixed = arith.addi %row, %tid : index
  %0 = vector.transfer_read %src[%row_mixed, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Tests for successful transformation
//===----------------------------------------------------------------------===//

// Test: Basic f16 4x1 transformation - row from lane_constant, column from lane_increment
// CHECK-LABEL: func.func @transform_basic_f16_4x1
// CHECK-NOT: iree_codegen.index_hint
// CHECK: %[[LOAD:.+]] = amdgpu.transpose_load %{{.*}}[%{{.*}}] : memref<128x256xf16, #gpu.address_space<workgroup>> -> vector<4xf16>
// CHECK: %[[CAST:.+]] = vector.shape_cast %[[LOAD]] : vector<4xf16> to vector<4x1xf16>
// CHECK: return %[[CAST]]
func.func @transform_basic_f16_4x1(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Constant row index, hint column - valid transformation
// CHECK-LABEL: func.func @transform_constant_row
// CHECK-NOT: iree_codegen.index_hint
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @transform_constant_row(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c5 = arith.constant 5 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c5 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // Row from constant, column from col - should transform
  %0 = vector.transfer_read %src[%c5, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Row index is arith.addi of row hint and constant - should transform
// CHECK-LABEL: func.func @transform_row_hint_plus_constant
// CHECK-NOT: iree_codegen.index_hint
// CHECK: gpu.lane_id
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @transform_row_hint_plus_constant(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c10 = arith.constant 10 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c10 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // Row is row + constant offset
  %c5 = arith.constant 5 : index
  %row_plus = arith.addi %row, %c5 : index
  %0 = vector.transfer_read %src[%row_plus, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Row index is arith.muli of row hint and constant - should transform
// CHECK-LABEL: func.func @transform_row_hint_times_constant
// CHECK-NOT: iree_codegen.index_hint
// CHECK: gpu.lane_id
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @transform_row_hint_times_constant(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c10 = arith.constant 10 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c10 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // Row is row * constant
  %c4 = arith.constant 4 : index
  %row_times = arith.muli %row, %c4 : index
  %0 = vector.transfer_read %src[%row_times, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Row index is complex expression: (row + c1) * c2 - should transform
// CHECK-LABEL: func.func @transform_row_complex_expression
// CHECK-NOT: iree_codegen.index_hint
// CHECK: gpu.lane_id
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @transform_row_complex_expression(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c10 = arith.constant 10 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c10 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // Row is (row + 5) * 4
  %c5 = arith.constant 5 : index
  %c4 = arith.constant 4 : index
  %tmp = arith.addi %row, %c5 : index
  %row_expr = arith.muli %tmp, %c4 : index
  %0 = vector.transfer_read %src[%row_expr, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Row index uses affine.apply - should transform
// CHECK-LABEL: func.func @transform_row_affine_apply
// CHECK-NOT: iree_codegen.index_hint
// CHECK: gpu.lane_id
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @transform_row_affine_apply(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c10 = arith.constant 10 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c10 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // Row uses affine.apply on row
  %row_affine = affine.apply affine_map<(d0) -> (d0 * 4 + 8)>(%row)
  %0 = vector.transfer_read %src[%row_affine, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: bf16 4x1 transformation
// CHECK-LABEL: func.func @transform_bf16_4x1
// CHECK: amdgpu.transpose_load
// CHECK-SAME: -> vector<4xbf16>
// CHECK: vector.shape_cast
func.func @transform_bf16_4x1(%src: memref<128x256xbf16, #gpu.address_space<workgroup>>) -> vector<4x1xbf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : bf16
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xbf16, #gpu.address_space<workgroup>>, vector<4x1xbf16>
  return %0 : vector<4x1xbf16>
}

// -----

// Test: i8 8x1 transformation (intrinsic size 8 for 8-bit)
// CHECK-LABEL: func.func @transform_i8_8x1
// CHECK: amdgpu.transpose_load
// CHECK-SAME: -> vector<8xi8>
// CHECK: vector.shape_cast
func.func @transform_i8_8x1(%src: memref<128x256xi8, #gpu.address_space<workgroup>>) -> vector<8x1xi8> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %c0_i8 = arith.constant 0 : i8
  %0 = vector.transfer_read %src[%row, %col], %c0_i8
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xi8, #gpu.address_space<workgroup>>, vector<8x1xi8>
  return %0 : vector<8x1xi8>
}

// -----

// Test: f16 8x1 unrolling (2x transpose_load)
// CHECK-LABEL: func.func @transform_unroll_f16_8x1
// CHECK-NOT: iree_codegen.index_hint
// CHECK: %[[L0:.+]] = amdgpu.transpose_load {{.*}} -> vector<4xf16>
// CHECK: %[[L1:.+]] = amdgpu.transpose_load {{.*}} -> vector<4xf16>
// CHECK: vector.insert_strided_slice %[[L0]], {{.*}} {offsets = [0], strides = [1]}
// CHECK: vector.insert_strided_slice %[[L1]], {{.*}} {offsets = [4], strides = [1]}
// CHECK: vector.shape_cast {{.*}} : vector<8xf16> to vector<8x1xf16>
func.func @transform_unroll_f16_8x1(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<8x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8x1xf16>
  return %0 : vector<8x1xf16>
}

// -----

// Test: 3D memref with multiple row hints - should transform
// CHECK-LABEL: func.func @transform_3d_memref
// CHECK-NOT: iree_codegen.index_hint
// CHECK: amdgpu.transpose_load
func.func @transform_3d_memref(%src: memref<64x128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid_x = gpu.thread_id x
  %tid_y = gpu.thread_id y
  %row0 = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %row1 = iree_codegen.index_hint %tid_y {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid_x {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // row0 and row1 are uniform (lane_constant), col is lane_increment
  %0 = vector.transfer_read %src[%row0, %row1, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>}
       : memref<64x128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Multiple reads from same hints both transform
// CHECK-LABEL: func.func @transform_multiple_reads
// CHECK-NOT: iree_codegen.index_hint
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @transform_multiple_reads(
    %src1: memref<128x256xf16, #gpu.address_space<workgroup>>,
    %src2: memref<128x256xf16, #gpu.address_space<workgroup>>) -> (vector<4x1xf16>, vector<4x1xf16>) {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src1[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  %1 = vector.transfer_read %src2[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0, %1 : vector<4x1xf16>, vector<4x1xf16>
}

// -----

// Test: Mix of transformable and non-transformable reads
// CHECK-LABEL: func.func @mixed_transformable
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
// CHECK: vector.transfer_read
func.func @mixed_transformable(
    %src_lds: memref<128x256xf16, #gpu.address_space<workgroup>>,
    %src_global: memref<128x256xf16>) -> (vector<4x1xf16>, vector<4x1xf16>) {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %tid {hint = #iree_gpu.lane_increment<16>} : index
  %cst = arith.constant 0.0 : f16
  // This should transform (workgroup memory)
  %0 = vector.transfer_read %src_lds[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  // This should NOT transform (global memory)
  %1 = vector.transfer_read %src_global[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16>, vector<4x1xf16>
  return %0, %1 : vector<4x1xf16>, vector<4x1xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Tests for preprocessing: auto-injecting hints from thread_id -> delinearize
//===----------------------------------------------------------------------===//

// Test: Basic 2-element delinearize_index from thread_id gets hints injected
// CHECK-LABEL: func.func @preprocess_delinearize_2d
// CHECK-NOT: iree_codegen.index_hint
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @preprocess_delinearize_2d(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %tid = gpu.thread_id x
  // Delinearize into (4, 16): result #0 should get lane_constant<16>, result #1 should get lane_increment<16>
  %indices:2 = affine.delinearize_index %tid into (4, 16) : index, index
  %cst = arith.constant 0.0 : f16
  // Row is first result (uniform within 16-lane groups), col is second result (varies within 16-lane groups)
  %0 = vector.transfer_read %src[%indices#0, %indices#1], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: 3-element delinearize_index from thread_id gets hints with correct group sizes
// For basis (2, 4, 16): result #0 gets lane_constant<64>, result #1 gets lane_constant<16>, result #2 gets lane_increment<16>
// CHECK-LABEL: func.func @preprocess_delinearize_3d
// CHECK-NOT: iree_codegen.index_hint
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @preprocess_delinearize_3d(%src: memref<64x128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %tid = gpu.thread_id x
  // Delinearize into (2, 4, 16): outer results get lane_constant, final gets lane_increment
  %indices:3 = affine.delinearize_index %tid into (2, 4, 16) : index, index, index
  %cst = arith.constant 0.0 : f16
  // Row indices are first two results (uniform), col is third result (varies)
  %0 = vector.transfer_read %src[%indices#0, %indices#1, %indices#2], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>}
       : memref<64x128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Delinearize from non-thread_id should NOT get hints injected (and should not transform)
// CHECK-LABEL: func.func @preprocess_delinearize_non_threadid
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @preprocess_delinearize_non_threadid(%src: memref<128x256xf16, #gpu.address_space<workgroup>>, %arg: index) -> vector<4x1xf16> {
  // Delinearize from function argument - should NOT get hints
  %indices:2 = affine.delinearize_index %arg into (4, 16) : index, index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%indices#0, %indices#1], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Delinearize results used through arithmetic ops still transform
// CHECK-LABEL: func.func @preprocess_delinearize_with_arith
// CHECK-NOT: iree_codegen.index_hint
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @preprocess_delinearize_with_arith(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %tid = gpu.thread_id x
  %indices:2 = affine.delinearize_index %tid into (4, 16) : index, index
  %cst = arith.constant 0.0 : f16
  // Row is first result + constant offset
  %c5 = arith.constant 5 : index
  %row = arith.addi %indices#0, %c5 : index
  %0 = vector.transfer_read %src[%row, %indices#1], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}
