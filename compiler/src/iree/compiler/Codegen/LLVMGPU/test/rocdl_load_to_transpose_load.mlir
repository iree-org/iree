// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 --pass-pipeline='builtin.module(func.func(iree-rocdl-load-to-transpose-load))' %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Tests for cases where transformation doesn't apply
//===----------------------------------------------------------------------===//

// Test: Global memory - no transformation (only workgroup memory supported)
// CHECK-LABEL: func.func @no_transform_global_memory
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_global_memory(%src: memref<128x256xf16>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<16>) : index
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Column index not from lane_increment hint - no transformation
// CHECK-LABEL: func.func @no_transform_column_hint_unused
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_column_hint_unused(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<16>) : index
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
  %cst = arith.constant 0.0 : f16
  // Using constant for column instead of col - NOT valid
  %0 = vector.transfer_read %src[%row, %c5], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Innermost vector dimension != 1 - no transformation
// CHECK-LABEL: func.func @no_transform_column_size_not_1
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_column_size_not_1(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x2xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<16>) : index
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x2xf16>
  return %0 : vector<4x2xf16>
}

// -----

// Test: Unsupported element type (f32) - no transformation
// CHECK-LABEL: func.func @no_transform_f32
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_f32(%src: memref<128x256xf32, #gpu.address_space<workgroup>>) -> vector<4x1xf32> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<16>) : index
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf32, #gpu.address_space<workgroup>>, vector<4x1xf32>
  return %0 : vector<4x1xf32>
}

// -----

// Test: Row size not multiple of intrinsic size - no transformation
// CHECK-LABEL: func.func @no_transform_non_multiple_row_size
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_non_multiple_row_size(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<3x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<16>) : index
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<3x1xf16>
  return %0 : vector<3x1xf16>
}

// -----

// Test: Column index with lane_increment step > 1 - no transformation
// transpose_load requires column indices to be consecutive (step=1)
// CHECK-LABEL: func.func @no_transform_column_step_not_1
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_column_step_not_1(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<16>) : index
  // Column hint has step=2, which is not supported for transpose_load
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, step = 2, aligned>) : index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Column index with lane_increment group_size < 16 - no transformation
// transpose_load operates on 16-lane groups, so group_size must be >= 16
// CHECK-LABEL: func.func @no_transform_column_group_size_too_small
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_column_group_size_too_small(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<16>) : index
  // Column hint has group_size=8, which is too small for transpose_load
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<8, aligned>) : index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Row index with lane_constant group_size not a multiple of 16 - no transformation
// transpose_load requires row uniformity within 16-lane groups
// CHECK-LABEL: func.func @no_transform_row_group_size_not_multiple
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_row_group_size_not_multiple(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  // Row hint has group_size=20, which is not a multiple of 16
  %row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<20>) : index
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Tests for row index validation - row index must be uniform across 16 lanes
//===----------------------------------------------------------------------===//

// Test: Row index from lane_increment hint (varies across lanes) - should NOT transform
// CHECK-LABEL: func.func @no_transform_row_from_column_hint
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_row_from_column_hint(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<16>) : index
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
  %cst = arith.constant 0.0 : f16
  // Row from col (lane_increment - varies across lanes) - INVALID
  %0 = vector.transfer_read %src[%col, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Row index from block argument (unknown provenance) - should NOT transform
// CHECK-LABEL: func.func @no_transform_row_from_block_arg
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_row_from_block_arg(%src: memref<128x256xf16, #gpu.address_space<workgroup>>, %row_arg: index) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
  %cst = arith.constant 0.0 : f16
  // Row from function argument - could vary per thread
  %0 = vector.transfer_read %src[%row_arg, %col], %cst
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
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG: %[[ROW:.+]] = iree_codegen.index_hint %{{.*}}(#iree_gpu.lane_constant<16>) : index
// CHECK-DAG: %[[COL:.+]] = iree_codegen.index_hint %{{.*}}(#iree_gpu.lane_increment<16, aligned>) : index
// CHECK: %[[LANE_ID:.+]] = gpu.lane_id
// Index arithmetic for transpose load:
//   row_offset = (lane_id % 16) / 4
//   col_base = col - (lane_id % 16) + ((lane_id % 16) % 4) * 4
// CHECK: %[[REM16:.+]] = arith.remui %[[LANE_ID]], %[[C16]] : index
// CHECK: %[[DIV4:.+]] = arith.divui %[[REM16]], %[[C4]] : index
// CHECK: %[[SUB:.+]] = arith.subi %[[COL]], %[[REM16]] : index
// CHECK: %[[REM4:.+]] = arith.remui %[[REM16]], %[[C4]] : index
// CHECK: %[[MUL:.+]] = arith.muli %[[REM4]], %[[C4]] : index
// CHECK: %[[NEW_COL:.+]] = arith.addi %[[SUB]], %[[MUL]] : index
// CHECK: %[[NEW_ROW:.+]] = arith.addi %[[ROW]], %[[DIV4]] : index
// CHECK: %[[LOAD:.+]] = amdgpu.transpose_load %{{.*}}[%[[NEW_ROW]], %[[NEW_COL]]] : memref<128x256xf16, #gpu.address_space<workgroup>> -> vector<4xf16>
// CHECK: %[[CAST:.+]] = vector.shape_cast %[[LOAD]] : vector<4xf16> to vector<4x1xf16>
// CHECK: return %[[CAST]]
func.func @transform_basic_f16_4x1(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<16>) : index
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Row index through arith ops (hint + constant) - should transform
// CHECK-LABEL: func.func @transform_row_hint_plus_constant
// CHECK: gpu.lane_id
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @transform_row_hint_plus_constant(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c10 = arith.constant 10 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c10(#iree_gpu.lane_constant<16>) : index
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
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

// Test: Row index through affine.apply - should transform
// CHECK-LABEL: func.func @transform_row_affine_apply
// CHECK: gpu.lane_id
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @transform_row_affine_apply(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16> {
  %c10 = arith.constant 10 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c10(#iree_gpu.lane_constant<16>) : index
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
  %cst = arith.constant 0.0 : f16
  // Row uses affine.apply on row
  %row_affine = affine.apply affine_map<(d0) -> (d0 * 4 + 8)>(%row)
  %0 = vector.transfer_read %src[%row_affine, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: i8 8x1 transformation (intrinsic size 8 for 8-bit types)
// CHECK-LABEL: func.func @transform_i8_8x1
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG: %[[ROW:.+]] = iree_codegen.index_hint %{{.*}}(#iree_gpu.lane_constant<16>) : index
// CHECK-DAG: %[[COL:.+]] = iree_codegen.index_hint %{{.*}}(#iree_gpu.lane_increment<16, aligned>) : index
// CHECK: %[[LANE_ID:.+]] = gpu.lane_id
// Index arithmetic for i8 transpose load (intrinsic size = 8):
//   row_offset = (lane_id % 16) / 2
//   col_base = col - (lane_id % 16) + ((lane_id % 16) % 2) * 8
// CHECK: %[[REM16:.+]] = arith.remui %[[LANE_ID]], %[[C16]] : index
// CHECK: %[[DIV2:.+]] = arith.divui %[[REM16]], %[[C2]] : index
// CHECK: %[[SUB:.+]] = arith.subi %[[COL]], %[[REM16]] : index
// CHECK: %[[REM2:.+]] = arith.remui %[[REM16]], %[[C2]] : index
// CHECK: %[[MUL:.+]] = arith.muli %[[REM2]], %[[C8]] : index
// CHECK: %[[NEW_COL:.+]] = arith.addi %[[SUB]], %[[MUL]] : index
// CHECK: %[[NEW_ROW:.+]] = arith.addi %[[ROW]], %[[DIV2]] : index
// CHECK: %[[LOAD:.+]] = amdgpu.transpose_load %{{.*}}[%[[NEW_ROW]], %[[NEW_COL]]] : memref<128x256xi8, #gpu.address_space<workgroup>> -> vector<8xi8>
// CHECK: vector.shape_cast %[[LOAD]] : vector<8xi8> to vector<8x1xi8>
func.func @transform_i8_8x1(%src: memref<128x256xi8, #gpu.address_space<workgroup>>) -> vector<8x1xi8> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<16>) : index
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
  %c0_i8 = arith.constant 0 : i8
  %0 = vector.transfer_read %src[%row, %col], %c0_i8
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xi8, #gpu.address_space<workgroup>>, vector<8x1xi8>
  return %0 : vector<8x1xi8>
}

// -----

// Test: f16 8x1 unrolling (2x transpose_load when row size > intrinsic size)
// CHECK-LABEL: func.func @transform_unroll_f16_8x1
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG: %[[ROW:.+]] = iree_codegen.index_hint %{{.*}}(#iree_gpu.lane_constant<16>) : index
// CHECK-DAG: %[[COL:.+]] = iree_codegen.index_hint %{{.*}}(#iree_gpu.lane_increment<16, aligned>) : index
// CHECK: %[[LANE_ID:.+]] = gpu.lane_id
// Index arithmetic computed once for both loads:
// CHECK: %[[REM16:.+]] = arith.remui %[[LANE_ID]], %[[C16]] : index
// CHECK: %[[DIV4:.+]] = arith.divui %[[REM16]], %[[C4]] : index
// CHECK: %[[SUB:.+]] = arith.subi %[[COL]], %[[REM16]] : index
// CHECK: %[[REM4:.+]] = arith.remui %[[REM16]], %[[C4]] : index
// CHECK: %[[MUL:.+]] = arith.muli %[[REM4]], %[[C4]] : index
// CHECK: %[[NEW_COL:.+]] = arith.addi %[[SUB]], %[[MUL]] : index
// First load at row + row_offset:
// CHECK: %[[ROW0:.+]] = arith.addi %[[ROW]], %[[DIV4]] : index
// CHECK: %[[L0:.+]] = amdgpu.transpose_load %{{.*}}[%[[ROW0]], %[[NEW_COL]]] : memref<128x256xf16, #gpu.address_space<workgroup>> -> vector<4xf16>
// Second load at row + row_offset + 4:
// CHECK: %[[ROW_OFFSET:.+]] = arith.addi %[[DIV4]], %[[C4]] : index
// CHECK: %[[ROW1:.+]] = arith.addi %[[ROW]], %[[ROW_OFFSET]] : index
// CHECK: %[[L1:.+]] = amdgpu.transpose_load %{{.*}}[%[[ROW1]], %[[NEW_COL]]] : memref<128x256xf16, #gpu.address_space<workgroup>> -> vector<4xf16>
// CHECK: vector.insert_strided_slice %[[L0]], {{.*}} {offsets = [0], strides = [1]}
// CHECK: vector.insert_strided_slice %[[L1]], {{.*}} {offsets = [4], strides = [1]}
// CHECK: vector.shape_cast {{.*}} : vector<8xf16> to vector<8x1xf16>
func.func @transform_unroll_f16_8x1(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<8x1xf16> {
  %c0 = arith.constant 0 : index
  %tid = gpu.thread_id x
  %row = iree_codegen.index_hint %c0(#iree_gpu.lane_constant<16>) : index
  %col = iree_codegen.index_hint %tid(#iree_gpu.lane_increment<16, aligned>) : index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%row, %col], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<8x1xf16>
  return %0 : vector<8x1xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Tests for preprocessing: seeding hints on thread_id and propagating through delinearize
//===----------------------------------------------------------------------===//

// Test: 3-element delinearize_index from thread_id gets hints with correct group sizes
// With workgroup_size = [128, 1, 1], thread_id x gets lane_increment<128>
// For basis (2, 4, 16): result #0 gets lane_constant<64>, result #1 gets lane_constant<16>, result #2 gets lane_increment<16>
#translation_128 = #iree_codegen.translation_info<pipeline = LLVMGPUDefault workgroup_size = [128, 1, 1]>
// CHECK-LABEL: func.func @preprocess_delinearize_3d
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @preprocess_delinearize_3d(%src: memref<64x128x256xf16, #gpu.address_space<workgroup>>) -> vector<4x1xf16>
    attributes {translation_info = #translation_128} {
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

// Test: Pre-existing aligned lane_increment hint on non-thread_id value with delinearize
// This tests the pattern propagation when hints come from sources other than seeding.
// The hint must have `aligned` for propagation to produce useful hints.
// CHECK-LABEL: func.func @preprocess_delinearize_preexisting_hint
// CHECK: amdgpu.transpose_load
// CHECK: vector.shape_cast
func.func @preprocess_delinearize_preexisting_hint(%src: memref<128x256xf16, #gpu.address_space<workgroup>>, %arg: index) -> vector<4x1xf16> {
  // Pre-existing aligned lane_increment hint on a function argument
  // The pattern should propagate hints through delinearize when aligned=true
  %arg_hinted = iree_codegen.index_hint %arg(#iree_gpu.lane_increment<64, aligned>) : index
  %indices:2 = affine.delinearize_index %arg_hinted into (4, 16) : index, index
  %cst = arith.constant 0.0 : f16
  // After propagation: result #0 gets lane_constant<16>, result #1 gets lane_increment<16, aligned>
  %0 = vector.transfer_read %src[%indices#0, %indices#1], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}

// -----

// Test: Non-aligned lane_increment hint blocks propagation
// When lane_increment does NOT have `aligned`, we cannot safely propagate
// through delinearize because the modulo may cause wrap-around within a lane
// group. The pattern returns failure, leaving the original hints unchanged.
// CHECK-LABEL: func.func @no_transform_delinearize_nonaligned_hint
// CHECK-NOT: amdgpu.transpose_load
// CHECK: vector.transfer_read
func.func @no_transform_delinearize_nonaligned_hint(%src: memref<128x256xf16, #gpu.address_space<workgroup>>, %arg: index) -> vector<4x1xf16> {
  // lane_increment<64> without aligned:
  // - Cannot propagate safely through delinearize
  // - Pattern fails, no hints propagated, blocking transformation
  %arg_hinted = iree_codegen.index_hint %arg(#iree_gpu.lane_increment<64>) : index
  %indices:2 = affine.delinearize_index %arg_hinted into (4, 16) : index, index
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%indices#0, %indices#1], %cst
       {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4x1xf16>
  return %0 : vector<4x1xf16>
}
