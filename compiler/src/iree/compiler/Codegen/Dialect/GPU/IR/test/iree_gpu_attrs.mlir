// RUN: iree-opt %s | FileCheck %s

module {
  func.func @test_mfma_f16_16x16x16_f32() attributes {
      mma_types = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>} {
    return
  }
}
// CHECK-LABEL: func @test_mfma_f16_16x16x16_f32
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>

module {
  func.func @test_mfma_f16_32x32x8_f32() attributes {
      mma_types = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>} {
    return
  }
}
// CHECK-LABEL: func @test_mfma_f16_32x32x8_f32
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>

module {
  func.func @test_wmma_f16_16x16x16_f32() attributes {
      mma_types = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>} {
    return
  }
}
// CHECK-LABEL: func @test_wmma_f16_16x16x16_f32
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>

module {
  func.func @test_any_lowering_config() attributes {
      lowering_config = #iree_gpu.lowering_config<{workgroup = [16, 16], thread = [0, 4]}>} {
    return
  }
}
// CHECK-LABEL: func @test_any_lowering_config
//  CHECK-SAME:   lowering_config = #iree_gpu.lowering_config<{thread = [0, 4], workgroup = [16, 16]}>
