// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1201 \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-rocdl-load-to-transpose-load))' \
// RUN:   %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-rocdl-load-to-transpose-load))' \
// RUN:   %s | FileCheck %s --check-prefix=CHECK-NO-RDNA4

// -----

// BF16 with explicit gpu global address space.

// CHECK-DAG: #[[N_MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1 mod 8)>
// CHECK-DAG: #[[K_MAP:.*]] = affine_map<(d0, d1) -> ((d1 floordiv 8) * 8)>
// CHECK-LABEL: func.func @global_transpose_load_bf16
// CHECK-SAME:    %[[N_BASE:[A-Za-z0-9]+]]: index, %[[K_SINGLE:[A-Za-z0-9]+]]: index
// CHECK-NOT: vector.transpose
// CHECK:     %[[TR:.*]] = amdgpu.global_transpose_load
// CHECK-SAME:   : memref<128x64xbf16, #gpu.address_space<global>> -> vector<8xbf16>
// CHECK:     %[[N_NEW:.*]] = affine.apply #[[N_MAP]](%[[N_BASE]], %[[K_SINGLE]])
// CHECK:     %[[K_NEW:.*]] = affine.apply #[[K_MAP]](%[[N_BASE]], %[[K_SINGLE]])
// CHECK:     %[[CAST:.*]] = vector.shape_cast %[[TR]] : vector<8xbf16> to vector<1x8xbf16>
// CHECK:     vector.transfer_write %[[CAST]], {{.*}}[%[[N_NEW]], %[[K_NEW]]]

// CHECK-NO-RDNA4-LABEL: func.func @global_transpose_load_bf16
// CHECK-NO-RDNA4-NOT: amdgpu.global_transpose_load
// CHECK-NO-RDNA4: vector.transpose
func.func @global_transpose_load_bf16(
    %src: memref<128x64xbf16, #gpu.address_space<global>>,
    %dst: memref<8x64xbf16, #gpu.address_space<workgroup>>,
    %n_base: index, %k_single: index) {
  %cst = arith.constant 0.0 : bf16
  %0 = vector.transfer_read %src[%k_single, %n_base], %cst {in_bounds = [true, true]}
      : memref<128x64xbf16, #gpu.address_space<global>>, vector<1x8xbf16>
  %1 = vector.transpose %0, [1, 0] : vector<1x8xbf16> to vector<8x1xbf16>
  vector.transfer_write %1, %dst[%n_base, %k_single] {in_bounds = [true, true]}
      : vector<8x1xbf16>, memref<8x64xbf16, #gpu.address_space<workgroup>>
  return
}

// -----

// F16 with explicit gpu global address space.

// CHECK-LABEL: func.func @global_transpose_load_f16
// CHECK-NOT: vector.transpose
// CHECK: amdgpu.global_transpose_load {{.*}} -> vector<8xf16>
// CHECK-NO-RDNA4-LABEL: func.func @global_transpose_load_f16
// CHECK-NO-RDNA4-NOT: amdgpu.global_transpose_load
func.func @global_transpose_load_f16(
    %src: memref<128x64xf16, #gpu.address_space<global>>,
    %dst: memref<8x64xf16, #gpu.address_space<workgroup>>,
    %n_base: index, %k_single: index) {
  %cst = arith.constant 0.0 : f16
  %0 = vector.transfer_read %src[%k_single, %n_base], %cst {in_bounds = [true, true]}
      : memref<128x64xf16, #gpu.address_space<global>>, vector<1x8xf16>
  %1 = vector.transpose %0, [1, 0] : vector<1x8xf16> to vector<8x1xf16>
  vector.transfer_write %1, %dst[%n_base, %k_single] {in_bounds = [true, true]}
      : vector<8x1xf16>, memref<8x64xf16, #gpu.address_space<workgroup>>
  return
}

// -----

// HAL storage buffer (descriptor_type) source — cast to global before load.

// CHECK-LABEL: func.func @global_transpose_load_hal_buffer
// CHECK-NOT: vector.transpose
// CHECK: memref.memory_space_cast {{.*}} : memref<128x64xbf16, #hal.descriptor_type<storage_buffer>> to memref<128x64xbf16, #gpu.address_space<global>>
// CHECK: amdgpu.global_transpose_load
// CHECK-NO-RDNA4-LABEL: func.func @global_transpose_load_hal_buffer
// CHECK-NO-RDNA4-NOT: amdgpu.global_transpose_load
func.func @global_transpose_load_hal_buffer(
    %src: memref<128x64xbf16, #hal.descriptor_type<storage_buffer>>,
    %dst: memref<8x64xbf16, #gpu.address_space<workgroup>>,
    %n_base: index, %k_single: index) {
  %cst = arith.constant 0.0 : bf16
  %0 = vector.transfer_read %src[%k_single, %n_base], %cst {in_bounds = [true, true]}
      : memref<128x64xbf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xbf16>
  %1 = vector.transpose %0, [1, 0] : vector<1x8xbf16> to vector<8x1xbf16>
  vector.transfer_write %1, %dst[%n_base, %k_single] {in_bounds = [true, true]}
      : vector<8x1xbf16>, memref<8x64xbf16, #gpu.address_space<workgroup>>
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

// No transform: source has no address space (not a recognized global).
// CHECK-LABEL: func.func @no_transform_no_memspace
// CHECK-NOT: amdgpu.global_transpose_load
// CHECK: vector.transpose
func.func @no_transform_no_memspace(
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

// No transform: wrong read shape (vector<8x1> instead of vector<1x8>).
// CHECK-LABEL: func.func @no_transform_wrong_shape
// CHECK-NOT: amdgpu.global_transpose_load
func.func @no_transform_wrong_shape(
    %src: memref<128x64xbf16, #gpu.address_space<global>>,
    %dst: memref<8x64xbf16, #gpu.address_space<workgroup>>,
    %n_base: index, %k_single: index) {
  %cst = arith.constant 0.0 : bf16
  %0 = vector.transfer_read %src[%k_single, %n_base], %cst {in_bounds = [true, true]}
      : memref<128x64xbf16, #gpu.address_space<global>>, vector<8x1xbf16>
  %1 = vector.transpose %0, [1, 0] : vector<8x1xbf16> to vector<1x8xbf16>
  vector.transfer_write %1, %dst[%n_base, %k_single] {in_bounds = [true, true]}
      : vector<1x8xbf16>, memref<8x64xbf16, #gpu.address_space<workgroup>>
  return
}
