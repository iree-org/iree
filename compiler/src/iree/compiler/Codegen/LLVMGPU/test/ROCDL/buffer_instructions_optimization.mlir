// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN: --pass-pipeline="builtin.module(func.func(iree-rocdl-buffer-instructions-optimization, canonicalize, cse))" %s \
// RUN:  | FileCheck %s

// vector.broadcast of a dynamic scalar i1 mask on transfer_read.

func.func @simplify_broadcast_mask_transfer_read(
    %mem : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>,
    %cond : i1) -> vector<8xbf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : bf16
  %mask = vector.broadcast %cond : i1 to vector<8xi1>
  %read = vector.transfer_read %mem[%c0], %cst, %mask
      {in_bounds = [true]}
      : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
  return %read : vector<8xbf16>
}

// CHECK-LABEL: @simplify_broadcast_mask_transfer_read
//  CHECK-SAME:   (%[[MEM:.+]]: memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[COND:.+]]: i1)
//   CHECK-DAG: %[[PAD:.+]] = arith.constant dense<0.000000e+00> : vector<8xbf16>
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[MEM]]
//  CHECK-SAME:   {in_bounds = [true]} : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
//       CHECK: %[[SEL:.+]] = arith.select %[[COND]], %[[READ]], %[[PAD]] : vector<8xbf16>
//       CHECK: return %[[SEL]] : vector<8xbf16>

// -----

// vector.broadcast of a dynamic scalar i1 mask on maskedload.

func.func @simplify_broadcast_mask_maskedload(
    %mem : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>,
    %cond : i1) -> vector<8xbf16> {
  %c0 = arith.constant 0 : index
  %passthru = arith.constant dense<0.0> : vector<8xbf16>
  %mask = vector.broadcast %cond : i1 to vector<8xi1>
  %load = vector.maskedload %mem[%c0], %mask, %passthru
      : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>,
        vector<8xi1>, vector<8xbf16> into vector<8xbf16>
  return %load : vector<8xbf16>
}

// CHECK-LABEL: @simplify_broadcast_mask_maskedload
//  CHECK-SAME:   (%[[MEM:.+]]: memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[COND:.+]]: i1)
//   CHECK-DAG: %[[PT:.+]] = arith.constant dense<0.000000e+00> : vector<8xbf16>
//       CHECK: %[[LOAD:.+]] = vector.load %[[MEM]]
//       CHECK: %[[SEL:.+]] = arith.select %[[COND]], %[[LOAD]], %[[PT]] : vector<8xbf16>
//       CHECK: return %[[SEL]] : vector<8xbf16>

// -----

// Constant true mask on transfer_read -> direct unmasked read.

func.func @simplify_constant_true_transfer_read(
    %mem : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>)
    -> vector<8xbf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : bf16
  %true = arith.constant true
  %mask = vector.broadcast %true : i1 to vector<8xi1>
  %read = vector.transfer_read %mem[%c0], %cst, %mask
      {in_bounds = [true]}
      : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
  return %read : vector<8xbf16>
}

// CHECK-LABEL: @simplify_constant_true_transfer_read
//  CHECK-SAME:   (%[[MEM:.+]]: memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>)
//   CHECK-NOT: arith.select
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[MEM]]
//  CHECK-SAME:   {in_bounds = [true]} : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
//       CHECK: return %[[READ]] : vector<8xbf16>

// -----

// Constant true mask on maskedload -> direct vector.load.

func.func @simplify_constant_true_maskedload(
    %mem : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>)
    -> vector<8xbf16> {
  %c0 = arith.constant 0 : index
  %passthru = arith.constant dense<0.0> : vector<8xbf16>
  %true = arith.constant true
  %mask = vector.broadcast %true : i1 to vector<8xi1>
  %load = vector.maskedload %mem[%c0], %mask, %passthru
      : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>,
        vector<8xi1>, vector<8xbf16> into vector<8xbf16>
  return %load : vector<8xbf16>
}

// CHECK-LABEL: @simplify_constant_true_maskedload
//  CHECK-SAME:   (%[[MEM:.+]]: memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>)
//   CHECK-NOT: arith.select
//   CHECK-NOT: vector.maskedload
//       CHECK: %[[LOAD:.+]] = vector.load %[[MEM]]
//       CHECK: return %[[LOAD]] : vector<8xbf16>

// -----

// Non-fat-buffer memref should not be simplified.

func.func @no_simplify_non_fat_buffer(
    %mem : memref<8xbf16>,
    %cond : i1) -> vector<8xbf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : bf16
  %mask = vector.broadcast %cond : i1 to vector<8xi1>
  %read = vector.transfer_read %mem[%c0], %cst, %mask
      {in_bounds = [true]}
      : memref<8xbf16>, vector<8xbf16>
  return %read : vector<8xbf16>
}

// CHECK-LABEL: @no_simplify_non_fat_buffer
//       CHECK: vector.broadcast
//       CHECK: vector.transfer_read {{.*}} %{{.*}} :
//       CHECK: return

// -----

// transfer_read not in_bounds should not be simplified.

func.func @no_simplify_not_in_bounds(
    %mem : memref<6xbf16, #amdgpu.address_space<fat_raw_buffer>>,
    %cond : i1) -> vector<8xbf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : bf16
  %mask = vector.broadcast %cond : i1 to vector<8xi1>
  %read = vector.transfer_read %mem[%c0], %cst, %mask
      {in_bounds = [false]}
      : memref<6xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
  return %read : vector<8xbf16>
}

// CHECK-LABEL: @no_simplify_not_in_bounds
//       CHECK: vector.broadcast
//       CHECK: vector.transfer_read {{.*}} %{{.*}} :
//       CHECK: return

// -----

// Mask not from vector.broadcast should not be simplified.

func.func @no_simplify_non_broadcast_mask(
    %mem : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>,
    %mask : vector<8xi1>) -> vector<8xbf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : bf16
  %read = vector.transfer_read %mem[%c0], %cst, %mask
      {in_bounds = [true]}
      : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
  return %read : vector<8xbf16>
}

// CHECK-LABEL: @no_simplify_non_broadcast_mask
//       CHECK: vector.transfer_read {{.*}} %{{.*}} :
//       CHECK: return

// -----

// Masked transfer_write should not be simplified (only reads are handled).

func.func @no_simplify_masked_transfer_write(
    %mem : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>,
    %vec : vector<8xbf16>,
    %cond : i1) {
  %c0 = arith.constant 0 : index
  %mask = vector.broadcast %cond : i1 to vector<8xi1>
  vector.transfer_write %vec, %mem[%c0], %mask
      {in_bounds = [true]}
      : vector<8xbf16>, memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>
  return
}

// CHECK-LABEL: @no_simplify_masked_transfer_write
//       CHECK: vector.broadcast
//       CHECK: vector.transfer_write
//       CHECK: return

// -----

// Multi-dimensional transfer_read with broadcast mask should be simplified.

func.func @simplify_broadcast_mask_2d_read(
    %mem : memref<4x8xbf16, #amdgpu.address_space<fat_raw_buffer>>,
    %cond : i1) -> vector<4x8xbf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : bf16
  %mask = vector.broadcast %cond : i1 to vector<4x8xi1>
  %read = vector.transfer_read %mem[%c0, %c0], %cst, %mask
      {in_bounds = [true, true]}
      : memref<4x8xbf16, #amdgpu.address_space<fat_raw_buffer>>,
        vector<4x8xbf16>
  return %read : vector<4x8xbf16>
}

// CHECK-LABEL: @simplify_broadcast_mask_2d_read
//  CHECK-SAME:   (%[[MEM:.+]]: memref<4x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[COND:.+]]: i1)
//   CHECK-DAG: %[[PAD:.+]] = arith.constant dense<0.000000e+00> : vector<4x8xbf16>
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[MEM]]
//  CHECK-SAME:   {in_bounds = [true, true]}
//       CHECK: %[[SEL:.+]] = arith.select %[[COND]], %[[READ]], %[[PAD]] : vector<4x8xbf16>
//       CHECK: return %[[SEL]] : vector<4x8xbf16>

// -----

// Maskedload with non-broadcast mask should not be simplified.

func.func @no_simplify_maskedload_non_broadcast_mask(
    %mem : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>,
    %mask : vector<8xi1>) -> vector<8xbf16> {
  %c0 = arith.constant 0 : index
  %passthru = arith.constant dense<0.0> : vector<8xbf16>
  %load = vector.maskedload %mem[%c0], %mask, %passthru
      : memref<8xbf16, #amdgpu.address_space<fat_raw_buffer>>,
        vector<8xi1>, vector<8xbf16> into vector<8xbf16>
  return %load : vector<8xbf16>
}

// CHECK-LABEL: @no_simplify_maskedload_non_broadcast_mask
//       CHECK: vector.maskedload
//       CHECK: return

// -----

// Maskedload from non-fat-buffer memref should not be simplified.

func.func @no_simplify_maskedload_non_fat_buffer(
    %mem : memref<8xbf16>,
    %cond : i1) -> vector<8xbf16> {
  %c0 = arith.constant 0 : index
  %passthru = arith.constant dense<0.0> : vector<8xbf16>
  %mask = vector.broadcast %cond : i1 to vector<8xi1>
  %load = vector.maskedload %mem[%c0], %mask, %passthru
      : memref<8xbf16>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
  return %load : vector<8xbf16>
}

// CHECK-LABEL: @no_simplify_maskedload_non_fat_buffer
//       CHECK: vector.maskedload
//       CHECK: return

// -----

// Non-identity permutation map on transfer_read should be preserved.
// Use affine_map<(d0, d1) -> (d0)> to read along the first dimension,
// which is not the minor identity and will be printed explicitly.

func.func @simplify_broadcast_mask_permutation_map(
    %mem : memref<8x4xbf16, #amdgpu.address_space<fat_raw_buffer>>,
    %cond : i1) -> vector<8xbf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : bf16
  %mask = vector.broadcast %cond : i1 to vector<8xi1>
  %read = vector.transfer_read %mem[%c0, %c0], %cst, %mask
      {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>}
      : memref<8x4xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xbf16>
  return %read : vector<8xbf16>
}

// CHECK-LABEL: @simplify_broadcast_mask_permutation_map
//  CHECK-SAME:   (%[[MEM:.+]]: memref<8x4xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[COND:.+]]: i1)
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[MEM]]
//  CHECK-SAME:   permutation_map = #map
//       CHECK: arith.select %[[COND]], %[[READ]]
//       CHECK: return
