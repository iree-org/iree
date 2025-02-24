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
  func.func @test_WMMAR3_f16_16x16x16_f32() attributes {
      mma_types = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>} {
    return
  }
}
// CHECK-LABEL: func @test_WMMAR3_f16_16x16x16_f32
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>

module {
  func.func @test_data_tiled_mfma_f32_16x16x4_f32() attributes {
      mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 4, subgroups_m = 2, intrinsics_k = 1>} {
    return
  }
}
// CHECK-LABEL: func @test_data_tiled_mfma_f32_16x16x4_f32
//  CHECK-SAME:   mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 4, subgroups_m = 2>

module {
  func.func @test_WMMAR4_f16_16x16x16_f32() attributes {
      mma_types = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>} {
    return
  }
}
// CHECK-LABEL: func @test_WMMAR4_f16_16x16x16_f32
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>

module {
  func.func @test_data_tiled_mfma_f32_16x16x4_f32() attributes {
      mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 4, subgroups_m = 2, intrinsics_k = 1>} {
    return
  }
}
// CHECK-LABEL: func @test_data_tiled_mfma_f32_16x16x4_f32
//  CHECK-SAME:   mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 4, subgroups_m = 2>

module {
  func.func @test_data_tiled_mfma_f32_16x16x16_f16() attributes {
      mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16, intrinsics_m = 1, subgroups_n = 2, intrinsics_k = 2>} {
    return
  }
}
// CHECK-LABEL: func @test_data_tiled_mfma_f32_16x16x16_f16
//  CHECK-SAME:   mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16, subgroups_n = 2, intrinsics_k = 2>

module {
  func.func @test_data_tiled_mfma_i32_16x16x32_i8() attributes {
      mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8, intrinsics_m = 1, intrinsics_n = 1, intrinsics_k = 1>} {
    return
  }
}
// CHECK-LABEL: func @test_data_tiled_mfma_i32_16x16x32_i8
//  CHECK-SAME:   mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8>

module {
  func.func @test_any_lowering_config() attributes {
      lowering_config = #iree_gpu.lowering_config<{workgroup = [16, 16], thread = [0, 4]}>} {
    return
  }
}
// CHECK-LABEL: func @test_any_lowering_config
//  CHECK-SAME:   lowering_config = #iree_gpu.lowering_config<{thread = [0, 4], workgroup = [16, 16]}>
