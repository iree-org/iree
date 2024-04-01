// RUN: iree-opt %s | FileCheck %s

module {
  func.func @test_mfma_f16_16x16x16_f32() attributes {
      mma_types = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>} {
    return
  }
}
// CHECK-LABEL: func @test_mfma_f16_16x16x16_f32
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>

module {
  func.func @test_mfma_f16_32x32x8_f32() attributes {
      mma_types = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>} {
    return
  }
}
// CHECK-LABEL: func @test_mfma_f16_32x32x8_f32
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>

module {
  func.func @test_wmma_f16_16x16x16_f32() attributes {
      mma_types = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>} {
    return
  }
}
// CHECK-LABEL: func @test_wmma_f16_16x16x16_f32
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>
