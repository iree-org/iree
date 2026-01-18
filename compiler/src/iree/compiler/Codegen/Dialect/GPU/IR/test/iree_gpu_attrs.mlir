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
  func.func @test_col_major_mfma_f16_16x16x16_f32() attributes {
      mma_types = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>} {
    return
  }
}
// CHECK-LABEL: func @test_col_major_mfma_f16_16x16x16_f32
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16, col_major = true>

module {
  func.func @test_col_major_vmfma_f16_16x16x32_f32() attributes {
      mma_types = #iree_gpu.virtual_mma_layout<VMFMA_F32_16x16x32_F16, col_major = true>} {
    return
  }
}
// CHECK-LABEL: func @test_col_major_vmfma_f16_16x16x32_f32
//  CHECK-SAME:   mma_types = #iree_gpu.virtual_mma_layout<VMFMA_F32_16x16x32_F16, col_major = true>

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
      mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 4, subgroups_m = 2, intrinsics_k = 1, operands_interleaving_intrinsics_k = [0, 1]>} {
    return
  }
}
// CHECK-LABEL: func @test_data_tiled_mfma_f32_16x16x4_f32
//  CHECK-SAME:   mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 4, subgroups_m = 2, operands_interleaving_intrinsics_k = [0, 1]>

module {
  func.func @test_WMMAR4_f16_16x16x16_f32() attributes {
      mma_types = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>} {
    return
  }
}
// CHECK-LABEL: func @test_WMMAR4_f16_16x16x16_f32
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>

module {
  func.func @test_WMMA_f32_16x16x4_f32() attributes {
      mma_types = #iree_gpu.mma_layout<WMMA_F32_16x16x4_F32>} {
    return
  }
}
// CHECK-LABEL: func @test_WMMA_f32_16x16x4_f32
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<WMMA_F32_16x16x4_F32>

module {
  func.func @test_WMMA_f32_16x16x32_f16() attributes {
      mma_types = #iree_gpu.mma_layout<WMMA_F32_16x16x32_F16>} {
    return
  }
}
// CHECK-LABEL: func @test_WMMA_f32_16x16x32_f16
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<WMMA_F32_16x16x32_F16>

module {
  func.func @test_WMMA_f32_16x16x64_f8E4M3FN() attributes {
      mma_types = #iree_gpu.mma_layout<WMMA_F32_16x16x64_F8E4M3FN>} {
    return
  }
}
// CHECK-LABEL: func @test_WMMA_f32_16x16x64_f8E4M3FN
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<WMMA_F32_16x16x64_F8E4M3FN>

module {
  func.func @test_WMMA_f32_16x16x128_f8E4M3FN() attributes {
      mma_types = #iree_gpu.mma_layout<WMMA_F32_16x16x128_F8E4M3FN>} {
    return
  }
}
// CHECK-LABEL: func @test_WMMA_f32_16x16x128_f8E4M3FN
//  CHECK-SAME:   mma_types = #iree_gpu.mma_layout<WMMA_F32_16x16x128_F8E4M3FN>

module {
  func.func @test_data_tiled_mfma_f32_16x16x4_f32() attributes {
      mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 4, subgroups_m = 2, intrinsics_k = 1, operands_interleaving_intrinsics_k = [0, 1]>} {
    return
  }
}
// CHECK-LABEL: func @test_data_tiled_mfma_f32_16x16x4_f32
//  CHECK-SAME:   mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 4, subgroups_m = 2, operands_interleaving_intrinsics_k = [0, 1]>

module {
  func.func @test_data_tiled_mfma_f32_16x16x4_f32_subgroups_k() attributes {
      mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 4, subgroups_k = 2, operands_interleaving_intrinsics_k = [0, 1]>} {
    return
  }
}
// CHECK-LABEL: func @test_data_tiled_mfma_f32_16x16x4_f32_subgroups_k
//  CHECK-SAME:   mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 4, subgroups_k = 2, operands_interleaving_intrinsics_k = [0, 1]>


module {
  func.func @test_data_tiled_mfma_f32_16x16x16_f16() attributes {
      mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16, intrinsics_m = 1, subgroups_n = 2, intrinsics_k = 2, operands_interleaving_intrinsics_k = [0, 1]>} {
    return
  }
}
// CHECK-LABEL: func @test_data_tiled_mfma_f32_16x16x16_f16
//  CHECK-SAME:   mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16, subgroups_n = 2, intrinsics_k = 2, operands_interleaving_intrinsics_k = [0, 1]>

module {
  func.func @test_data_tiled_mfma_i32_16x16x32_i8() attributes {
      mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8, intrinsics_m = 1, intrinsics_n = 1, intrinsics_k = 1>} {
    return
  }
}
// CHECK-LABEL: func @test_data_tiled_mfma_i32_16x16x32_i8
//  CHECK-SAME:   mma_types = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8>

module {
  func.func @test_data_tiled_scaled_mfma_F32_32x32x64_B32() attributes {
      mma_types = #iree_gpu.data_tiled_scaled_mma_layout<intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>} {
    return
  }
}
// CHECK-LABEL: func @test_data_tiled_scaled_mfma_F32_32x32x64_B32
//  CHECK-SAME:   mma_types = #iree_gpu.data_tiled_scaled_mma_layout<intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>

module {
  func.func @test_data_tiled_scaled_mfma_F32_16x16x128_B32() attributes {
      mma_types = #iree_gpu.data_tiled_scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32, intrinsics_m = 4, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 4, operands_interleaving_intrinsics_k = [2, 3]>} {
    return
  }
}
// CHECK-LABEL: func @test_data_tiled_scaled_mfma_F32_16x16x128_B32
//  CHECK-SAME:   mma_types = #iree_gpu.data_tiled_scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32, intrinsics_m = 4, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 4, operands_interleaving_intrinsics_k = [2, 3]>

module {
  func.func @test_any_lowering_config() attributes {
      lowering_config = #iree_gpu.lowering_config<{workgroup = [16, 16], thread = [0, 4]}>} {
    return
  }
}
// CHECK-LABEL: func @test_any_lowering_config
//  CHECK-SAME:   lowering_config = #iree_gpu.lowering_config<{thread = [0, 4], workgroup = [16, 16]}>

module {
  func.func @test_lowering_config_reordering() attributes {
      lowering_config = #iree_gpu.lowering_config<{workgroup = [256, 256], workgroup_reordering_strategy = #iree_gpu.conditional_transpose<8, 38>}>} {
    return
  }
}
// CHECK-LABEL: func @test_lowering_config_reordering
//  CHECK-SAME:   lowering_config = #iree_gpu.lowering_config<{workgroup = [256, 256], workgroup_reordering_strategy = #iree_gpu.conditional_transpose<8, 38>}>

module {
  func.func @test_cache_swizzle_promotion() attributes {
      promotion_types = [#iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>]} {
    return
  }
}
// CHECK-LABEL: func @test_cache_swizzle_promotion
//  CHECK-SAME:   promotion_types = [#iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>]

module {
  func.func @test_lane_constant() attributes {
      lane_constant = #iree_gpu.lane_constant<16>} {
    return
  }
}
// CHECK-LABEL: func @test_lane_constant
//  CHECK-SAME:   lane_constant = #iree_gpu.lane_constant<16>

module {
  func.func @test_lane_increment() attributes {
      lane_increment = #iree_gpu.lane_increment<16>} {
    return
  }
}
// CHECK-LABEL: func @test_lane_increment
//  CHECK-SAME:   lane_increment = #iree_gpu.lane_increment<16>

module {
  func.func @test_lane_increment_with_step() attributes {
      lane_increment = #iree_gpu.lane_increment<16, step = 2>} {
    return
  }
}
// CHECK-LABEL: func @test_lane_increment_with_step
//  CHECK-SAME:   lane_increment = #iree_gpu.lane_increment<16, step = 2>
