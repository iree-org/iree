// RUN: iree-opt --iree-codegen-lower-bitcode-ukernels --split-input-file %s | FileCheck %s

// CHECK-LABEL:       @pure_argmax_ukernel_test_with_provider
// CHECK-SAME:          %[[ARG0:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:          %[[ARG1:[a-zA-Z0-9]+]]: tensor<f32>
// CHECK-SAME:          %[[ARG2:[a-zA-Z0-9]+]]: tensor<i64>
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?xf32>
// CHECK:               %[[FALSE:.*]] = arith.constant false
// CHECK:               %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_amdgpu_argmax_f32i64"
// CHECK-SAME:            ins(%[[ARG0]] : tensor<?xf32>)
// CHECK-SAME:            outs(%[[ARG1]], %[[ARG2]] : tensor<f32>, tensor<i64>)
// CHECK-SAME:            (%[[DIM]], %[[FALSE]] : index, i1)
// CHECK-SAME:            fn_def_attrs {vm.import.module = "rocm"}
// CHECK-SAME{LITERAL}:   strided_dims([[], [], []])
// CHECK:               return %[[MICRO_KERNEL]]#1
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.ukernel_provider = #rocm.ukernel_provider}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  func.func @pure_argmax_ukernel_test_with_provider(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<i64>) -> tensor<i64> {
    %cst = arith.constant 0.000000e+00 : f32
    %0:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<?xf32>) outs(%arg1, %arg2 : tensor<f32>, tensor<i64>) attrs =  {iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"iree_uk_amdgpu_argmax_f32i64", bitcode>} {
    ^bb0(%in: f32, %out: f32, %out_0: i64):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i64
      %3 = arith.maximumf %in, %out : f32
      %4 = arith.cmpf ogt, %in, %out : f32
      %5 = arith.select %4, %2, %out_0 : i64
      linalg.yield %3, %5 : f32, i64
    } -> (tensor<f32>, tensor<i64>)
    return %0#1 : tensor<i64>
  }
}

// -----

// CHECK-LABEL:       @argmax_ukernel_test_with_provider
// CHECK-SAME:          %[[ARG0:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:          %[[ARG1:[a-zA-Z0-9]+]]: tensor<f32>
// CHECK-SAME:          %[[ARG2:[a-zA-Z0-9]+]]: tensor<i64>
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?xf32>
// CHECK:               %[[TRUE:.*]] = arith.constant true
// CHECK:               %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_amdgpu_argmax_f32i64"
// CHECK-SAME:            ins(%[[ARG0]] : tensor<?xf32>)
// CHECK-SAME:            outs(%[[ARG1]], %[[ARG2]] : tensor<f32>, tensor<i64>)
// CHECK-SAME:            (%[[DIM]], %[[TRUE]] : index, i1)
// CHECK-SAME:            fn_def_attrs {vm.import.module = "rocm"}
// CHECK-SAME{LITERAL}:   strided_dims([[], [], []])
// CHECK:               return %[[MICRO_KERNEL]]#0, %[[MICRO_KERNEL]]#1
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.ukernel_provider = #rocm.ukernel_provider}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  func.func @argmax_ukernel_test_with_provider(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<i64>) -> (tensor<f32>, tensor<i64>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<?xf32>) outs(%arg1, %arg2 : tensor<f32>, tensor<i64>) attrs =  {iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"iree_uk_amdgpu_argmax_f32i64", bitcode>} {
    ^bb0(%in: f32, %out: f32, %out_0: i64):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i64
      %3 = arith.maximumf %in, %out : f32
      %4 = arith.cmpf ogt, %in, %out : f32
      %5 = arith.select %4, %2, %out_0 : i64
      linalg.yield %3, %5 : f32, i64
    } -> (tensor<f32>, tensor<i64>)
    return %0#0, %0#1 : tensor<f32>, tensor<i64>
  }
}

// -----

// CHECK-LABEL:       @multi_mma_mfma_i32_16x16x32_i8_with_gpu_arch
// CHECK-SAME:          %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x2x8x4x16x2x8xi8>
// CHECK-SAME:          %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x2x4x2x4x16x2x8xi8>
// CHECK-SAME:          %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x1x4x8x2x4x16x4xi32>
// CHECK:               %[[ALLOC:.*]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<8192xi8>
// CHECK:               %[[C1_INDEX:.*]] = arith.constant 1 : index
// CHECK:               %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C1_INDEX]] : tensor<1x2x8x4x16x2x8xi8>
// CHECK:               %[[DIM_CAST:.*]] = arith.index_cast %[[DIM]] : index to i32
// CHECK:               %[[C8:.*]] = arith.constant 8 : i32
// CHECK:               %[[C1:.*]] = arith.constant 1 : i32
// CHECK:               %[[C2:.*]] = arith.constant 2 : i32
// CHECK:               %[[C4:.*]] = arith.constant 4 : i32
// CHECK:               %[[C2_1:.*]] = arith.constant 2 : i32
// CHECK:               %[[C8192:.*]] = arith.constant 8192 : i32
// CHECK:               %[[UK_GENERIC:.*]] = iree_codegen.ukernel.generic "iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8"
// CHECK-SAME:            ins(%[[ARG0]], %[[ARG1]] : tensor<1x2x8x4x16x2x8xi8>, tensor<1x2x4x2x4x16x2x8xi8>)
// CHECK-SAME:            outs(%[[ARG2]] : tensor<1x1x4x8x2x4x16x4xi32>)
// CHECK-SAME:            (%[[ALLOC]], %[[C8192]], %[[DIM_CAST]], %[[C8]], %[[C1]], %[[C2]], %[[C4]], %[[C2_1]] : tensor<8192xi8>, i32, i32, i32, i32, i32, i32, i32)
// CHECK-SAME:            fn_def_attrs {vm.import.module = "rocm"}
// CHECK-SAME{LITERAL}:   strided_dims([[], [], [4], []])
// CHECK:               return %[[UK_GENERIC]] : tensor<1x1x4x8x2x4x16x4xi32>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">, iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, iree_codegen.ukernel_provider = #rocm.ukernel_provider, ukernels = "all"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  func.func @multi_mma_mfma_i32_16x16x32_i8_with_gpu_arch(%arg0: tensor<1x2x8x4x16x2x8xi8>, %arg1: tensor<1x2x4x2x4x16x2x8xi8>, %arg2: tensor<1x1x4x8x2x4x16x4xi32>) -> tensor<1x1x4x8x2x4x16x4xi32> {
    %0 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2) {
      indexing_maps = [#map, #map1, #map2],
      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8", bitcode>,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2, operands_interleaving_intrinsics_k = [0, 1]>,
      semantics = #iree_gpu.mma_semantics<distributed = false, opaque = false>
    } : tensor<1x2x8x4x16x2x8xi8>, tensor<1x2x4x2x4x16x2x8xi8> into tensor<1x1x4x8x2x4x16x4xi32>
    return %0 : tensor<1x1x4x8x2x4x16x4xi32>
  }
}

// -----

// CHECK-LABEL:       @multi_mma_mfma_i32_16x16x32_i8_without_gpu_arch
// CHECK-SAME:          %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x2x8x4x16x2x8xi8>
// CHECK-SAME:          %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x2x4x2x4x16x2x8xi8>
// CHECK-SAME:          %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x1x4x8x2x4x16x4xi32>
// CHECK:               %[[NULL:.*]] = iree_codegen.null_pointer
// CHECK:               %[[C1_INDEX:.*]] = arith.constant 1 : index
// CHECK:               %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C1_INDEX]] : tensor<1x2x8x4x16x2x8xi8>
// CHECK:               %[[DIM_CAST:.*]] = arith.index_cast %[[DIM]] : index to i32
// CHECK:               %[[C8:.*]] = arith.constant 8 : i32
// CHECK:               %[[C1:.*]] = arith.constant 1 : i32
// CHECK:               %[[C2:.*]] = arith.constant 2 : i32
// CHECK:               %[[C4:.*]] = arith.constant 4 : i32
// CHECK:               %[[C2_1:.*]] = arith.constant 2 : i32
// CHECK:               %[[C0:.*]] = arith.constant 0 : i32
// CHECK:               %[[UK_GENERIC:.*]] = iree_codegen.ukernel.generic "iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8"
// CHECK-SAME:            ins(%[[ARG0]], %[[ARG1]] : tensor<1x2x8x4x16x2x8xi8>, tensor<1x2x4x2x4x16x2x8xi8>)
// CHECK-SAME:            outs(%[[ARG2]] : tensor<1x1x4x8x2x4x16x4xi32>)
// CHECK-SAME:            (%[[NULL]], %[[C0]], %[[DIM_CAST]], %[[C8]], %[[C1]], %[[C2]], %[[C4]], %[[C2_1]] : !iree_codegen.null_pointer, i32, i32, i32, i32, i32, i32, i32)
// CHECK-SAME:            fn_def_attrs {vm.import.module = "rocm"}
// CHECK-SAME{LITERAL}:   strided_dims([[], [], [4]])
// CHECK:               return %[[UK_GENERIC]] : tensor<1x1x4x8x2x4x16x4xi32>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.ukernel_provider = #rocm.ukernel_provider, ukernels = "all"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  func.func @multi_mma_mfma_i32_16x16x32_i8_without_gpu_arch(%arg0: tensor<1x2x8x4x16x2x8xi8>, %arg1: tensor<1x2x4x2x4x16x2x8xi8>, %arg2: tensor<1x1x4x8x2x4x16x4xi32>) -> tensor<1x1x4x8x2x4x16x4xi32> {
    %0 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2) {
      indexing_maps = [#map, #map1, #map2],
      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8", bitcode>,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2, operands_interleaving_intrinsics_k = [0, 1]>,
      semantics = #iree_gpu.mma_semantics<distributed = false, opaque = false>
    } : tensor<1x2x8x4x16x2x8xi8>, tensor<1x2x4x2x4x16x2x8xi8> into tensor<1x1x4x8x2x4x16x4xi32>
    return %0 : tensor<1x1x4x8x2x4x16x4xi32>
  }
}

// -----

// CHECK:               %[[UK_GENERIC:.*]] = iree_codegen.ukernel.generic "iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8"
// CHECK-SAME{LITERAL}:   strided_dims([[], [], [2]])
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.ukernel_provider = #rocm.ukernel_provider, ukernels = "all"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  func.func @multi_mma_mfma_i32_16x16x32_i8_intrinsics_equal_to_one(%arg0: tensor<1x1x1x1x1x2x8xi8>, %arg1: tensor<1x1x1x1x1x2x8xi8>, %arg2: tensor<1x1x1x1x1x4xi32>) -> tensor<1x1x1x1x1x4xi32> {
    %0 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2) {
      indexing_maps = [#map, #map1, #map2],
      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8", bitcode>,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8, subgroups_n = 4, intrinsics_k = 2, operands_interleaving_intrinsics_k = [0, 1]>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : tensor<1x1x1x1x1x2x8xi8>, tensor<1x1x1x1x1x2x8xi8> into tensor<1x1x1x1x1x4xi32>
    return %0 : tensor<1x1x1x1x1x4xi32>
  }
}
