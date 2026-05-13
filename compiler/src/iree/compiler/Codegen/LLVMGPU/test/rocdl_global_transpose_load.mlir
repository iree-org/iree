// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1201 \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-rocdl-load-to-transpose-load))' \
// RUN:   %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-rocdl-load-to-transpose-load))' \
// RUN:   %s | FileCheck %s --check-prefix=CHECK-NO-RDNA4

// Verify that vector.transfer_read + vector.transpose from flat global memory
// is replaced with amdgpu.global_transpose_load on RDNA4 (gfx1200+), and that
// the write indices are corrected for contiguous K writes.

// -----

// BF16: vector<1x8xbf16> read + [1,0] transpose → global_transpose_load.
// The write indices are recomputed from (N_base, K_single):
//   N_new = N_base + K_single % 8
//   K_new = (K_single floordiv 8) * 8

// CHECK-LABEL: func.func @global_transpose_load_bf16
// CHECK-NOT: vector.transpose
// CHECK:     %[[TR:.*]] = amdgpu.global_transpose_load
// CHECK-SAME:   : memref<128x64xbf16> -> vector<8xbf16>
// CHECK:     %[[CAST:.*]] = vector.shape_cast %[[TR]] : vector<8xbf16> to vector<1x8xbf16>
// CHECK:     vector.transfer_write %[[CAST]]

// CHECK-NO-RDNA4-LABEL: func.func @global_transpose_load_bf16
// CHECK-NO-RDNA4-NOT: amdgpu.global_transpose_load
// CHECK-NO-RDNA4: vector.transpose
func.func @global_transpose_load_bf16(
    %src: memref<128x64xbf16>,
    %dst: memref<8x64xbf16, #gpu.address_space<workgroup>>,
    %n_base: index, %k_single: index) {
  %cst = arith.constant 0.0 : bf16
  %0 = vector.transfer_read %src[%k_single, %n_base], %cst {in_bounds = [true, true]}
      : memref<128x64xbf16>, vector<1x8xbf16>
  %1 = vector.transpose %0, [1, 0] : vector<1x8xbf16> to vector<8x1xbf16>
  vector.transfer_write %1, %dst[%n_base, %k_single] {in_bounds = [true, true]}
      : vector<8x1xbf16>, memref<8x64xbf16, #gpu.address_space<workgroup>>
  return
}

// -----

// F16: same pattern works for f16.
// CHECK-LABEL: func.func @global_transpose_load_f16
// CHECK-NOT: vector.transpose
// CHECK: amdgpu.global_transpose_load {{.*}} -> vector<8xf16>
// CHECK-NO-RDNA4-LABEL: func.func @global_transpose_load_f16
// CHECK-NO-RDNA4-NOT: amdgpu.global_transpose_load
func.func @global_transpose_load_f16(
    %src: memref<128x64xf16>,
    %dst: memref<8x64xf16, #gpu.address_space<workgroup>>,
    %n_base: index, %k_single: index) {
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%k_single, %n_base], %cst {in_bounds = [true, true]}
      : memref<128x64xf16>, vector<1x8xf16>
  %1 = vector.transpose %0, [1, 0] : vector<1x8xf16> to vector<8x1xf16>
  vector.transfer_write %1, %dst[%n_base, %k_single] {in_bounds = [true, true]}
      : vector<8x1xf16>, memref<8x64xf16, #gpu.address_space<workgroup>>
  return
}

// -----

// No transform: source is workgroup memory (not global).
// CHECK-LABEL: func.func @no_transform_workgroup_src
// CHECK-NOT: amdgpu.global_transpose_load
// CHECK: vector.transpose
func.func @no_transform_workgroup_src(
    %src: memref<128x64xbf16, #gpu.address_space<workgroup>>,
    %dst: memref<8x64xbf16, #gpu.address_space<workgroup>>,
    %n_base: index, %k_single: index) {
  %cst = arith.constant 0.0 : bf16
  %0 = vector.transfer_read %src[%k_single, %n_base], %cst {in_bounds = [true, true]}
      : memref<128x64xbf16, #gpu.address_space<workgroup>>, vector<1x8xbf16>
  %1 = vector.transpose %0, [1, 0] : vector<1x8xbf16> to vector<8x1xbf16>
  vector.transfer_write %1, %dst[%n_base, %k_single] {in_bounds = [true, true]}
      : vector<8x1xbf16>, memref<8x64xbf16, #gpu.address_space<workgroup>>
  return
}

// -----

// No transform: wrong read shape (vector<8x1> instead of vector<1x8>).
// CHECK-LABEL: func.func @no_transform_wrong_shape
// CHECK-NOT: amdgpu.global_transpose_load
func.func @no_transform_wrong_shape(
    %src: memref<128x64xbf16>,
    %dst: memref<8x64xbf16, #gpu.address_space<workgroup>>,
    %n_base: index, %k_single: index) {
  %cst = arith.constant 0.0 : bf16
  %0 = vector.transfer_read %src[%k_single, %n_base], %cst {in_bounds = [true, true]}
      : memref<128x64xbf16>, vector<8x1xbf16>
  %1 = vector.transpose %0, [1, 0] : vector<8x1xbf16> to vector<1x8xbf16>
  vector.transfer_write %1, %dst[%n_base, %k_single] {in_bounds = [true, true]}
      : vector<1x8xbf16>, memref<8x64xbf16, #gpu.address_space<workgroup>>
  return
}
