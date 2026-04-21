// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --iree-codegen-llvmgpu-rocdl-lowering-pipeline='include-llvm-lowering=false' %s | FileCheck %s

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>],
  flags = Indirect
>
#translation_info = #iree_codegen.translation_info<pipeline =
  #iree_gpu.pipeline<TileAndFuse>
  workgroup_size = [256, 1, 1]
  subgroup_size = 64,
  {
    gpu_pipeline_options = #iree_gpu.pipeline_options<
      prefetch_num_stages = 0,
      no_reduce_shared_memory_bank_conflicts = true>
  }
>
#config = #iree_gpu.lowering_config<{
  workgroup = [1, 1, 0, 0],
  reduction = [0, 0, 1, 1],
  promote_operands = [0, 1]
}>
func.func @data_tiled_scaled_mma_inner_tiled()
  attributes {hal.executable.target = #executable_target_rocm, translation_info = #translation_info} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x1x8x4x4x16x32xf4E2M1FN>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x1x4x2x4x4x16x32xf4E2M1FN>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x8x4x16x4xf8E8M0FNU>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x4x2x4x16x4xf8E8M0FNU>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<9x9x4x8x2x4x16x4xf32>>
  %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [9, 9, 1, 8, 4, 4, 16, 32], strides = [1, 1, 1, 1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x1x8x4x4x16x32xf4E2M1FN>> -> tensor<9x9x1x8x4x4x16x32xf4E2M1FN>
  %6 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0, 0, 0, 0, 0, 0], sizes = [9, 9, 1, 4, 2, 4, 4, 16, 32], strides = [1, 1, 1, 1, 1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x1x4x2x4x4x16x32xf4E2M1FN>> -> tensor<9x9x1x4x2x4x4x16x32xf4E2M1FN>
  %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0, 0, 0, 0], sizes = [9, 9, 8, 4, 16, 4], strides = [1, 1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x8x4x16x4xf8E8M0FNU>> -> tensor<9x9x8x4x16x4xf8E8M0FNU>
  %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0, 0, 0, 0, 0], sizes = [9, 9, 4, 2, 4, 16, 4], strides = [1, 1, 1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x4x2x4x16x4xf8E8M0FNU>> -> tensor<9x9x4x2x4x16x4xf8E8M0FNU>
  %9 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [9, 9, 4, 8, 2, 4, 16, 4], strides = [1, 1, 1, 1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<9x9x4x8x2x4x16x4xf32>> -> tensor<9x9x4x8x2x4x16x4xf32>
  %10 = iree_codegen.inner_tiled ins(%5, %6, %7, %8) outs(%9) {
    lowering_config = #config,
    indexing_maps = [
      affine_map<(m, n, k, kb) -> (m, k, kb)>,
      affine_map<(m, n, k, kb) -> (n, k, kb)>,
      affine_map<(m, n, k, kb) -> (m, k)>,
      affine_map<(m, n, k, kb) -> (n, k)>,
      affine_map<(m, n, k, kb) -> (m, n)>],
    iterator_types = [
      #linalg.iterator_type<parallel>,
      #linalg.iterator_type<parallel>,
      #linalg.iterator_type<reduction>,
      #linalg.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32,
      intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 4, operands_interleaving_intrinsics_k = [2, 3]>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = false>}
    : tensor<9x9x1x8x4x4x16x32xf4E2M1FN>, tensor<9x9x1x4x2x4x4x16x32xf4E2M1FN>, tensor<9x9x8x4x16x4xf8E8M0FNU>, tensor<9x9x4x2x4x16x4xf8E8M0FNU> into tensor<9x9x4x8x2x4x16x4xf32>
  iree_tensor_ext.dispatch.tensor.store %10, %4, offsets = [0, 0, 0, 0, 0, 0, 0, 0], sizes = [9, 9, 4, 8, 2, 4, 16, 4], strides = [1, 1, 1, 1, 1, 1, 1, 1] : tensor<9x9x4x8x2x4x16x4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<9x9x4x8x2x4x16x4xf32>>
  return
}

// CHECK-LABEL: func.func @data_tiled_scaled_mma_inner_tiled()
// CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:  %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:  %[[C9:.+]] = arith.constant 9 : index
// CHECK-DAG:  %[[BINDING_A:.+]] = hal.interface.binding.subspan {{.*}} binding(0)
// CHECK-DAG:  %[[ASSUMED_BINDING_A:.+]] = memref.assume_alignment %[[BINDING_A]]
// CHECK-DAG:  %[[BUFFER_A:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_BINDING_A]]
// CHECK-DAG:  %[[BINDING_A_SCALE:.+]] = hal.interface.binding.subspan {{.*}} binding(2)
// CHECK-DAG:  %[[ASSUMED_BINDING_A_SCALE:.+]] = memref.assume_alignment %[[BINDING_A_SCALE]]
// CHECK-DAG:  %[[BUFFER_A_SCALE:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_BINDING_A_SCALE]]
// CHECK-DAG:  %[[BINDING_B:.+]] = hal.interface.binding.subspan {{.*}} binding(1)
// CHECK-DAG:  %[[ASSUMED_BINDING_B:.+]] = memref.assume_alignment %[[BINDING_B]]
// CHECK-DAG:  %[[BUFFER_B:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_BINDING_B]]
// CHECK-DAG:  %[[BINDING_B_SCALE:.+]] = hal.interface.binding.subspan {{.*}} binding(3)
// CHECK-DAG:  %[[ASSUMED_BINDING_B_SCALE:.+]] = memref.assume_alignment %[[BINDING_B_SCALE]]
// CHECK-DAG:  %[[BUFFER_B_SCALE:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_BINDING_B_SCALE]]
// CHECK-DAG:  %[[BINDING_C:.+]] = hal.interface.binding.subspan {{.*}} binding(4)
// CHECK-DAG:  %[[ASSUMED_BINDING_C:.+]] = memref.assume_alignment %[[BINDING_C]]
// CHECK-DAG:  %[[BUFFER_C:.+]] = amdgpu.fat_raw_buffer_cast %[[ASSUMED_BINDING_C]]
// CHECK-DAG:  %[[A_ALLOC:.+]] = memref.alloc() : memref<1x1x1x8x4x4x16x32xf4E2M1FN, #gpu.address_space<workgroup>>
// CHECK-DAG:  %[[B_ALLOC:.+]] = memref.alloc() : memref<1x1x1x4x2x4x4x16x32xf4E2M1FN, #gpu.address_space<workgroup>>
// CHECK:      gpu.barrier memfence [#gpu.address_space<workgroup>]
// CHECK:      %[[LOOP:.+]]:16 = scf.for {{.*}} %[[C0]] to %[[C9]] step %[[C1]] iter_args(%arg[[#ITER_BASE:]] = {{.*}}) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
// CHECK-DAG:    %[[A_GLOBAL_LOAD:.+]] = vector.transfer_read %[[BUFFER_A]]{{.*}} vector<32xf4E2M1FN>
// CHECK-DAG:    %[[B_GLOBAL_LOAD:.+]] = vector.transfer_read %[[BUFFER_B]]{{.*}} vector<32xf4E2M1FN>
// CHECK-DAG:    vector.transfer_write %[[A_GLOBAL_LOAD]], %[[A_ALLOC]]
// CHECK-DAG:    vector.transfer_write %[[B_GLOBAL_LOAD]], %[[B_ALLOC]]
// CHECK:        gpu.barrier memfence [#gpu.address_space<workgroup>]
// CHECK-DAG:    %[[A_READ:.+]] = vector.transfer_read %[[A_ALLOC]]{{.*}} vector<8x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[B_READ:.+]] = vector.transfer_read %[[B_ALLOC]]{{.*}} vector<2x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[A_SCALE_READ:.+]] = vector.transfer_read %[[BUFFER_A_SCALE]]{{.*}} vector<8x1x1x4xf8E8M0FNU>
// CHECK-DAG:    %[[B_SCALE_READ:.+]] = vector.transfer_read %[[BUFFER_B_SCALE]]{{.*}} vector<2x1x1x4xf8E8M0FNU>
// CHECK-DAG:    %[[A_EXTRACT00:.+]] = vector.extract %[[A_READ]][0, 0, 0, 0] : vector<32xf4E2M1FN> from vector<8x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[A_EXTRACT01:.+]] = vector.extract %[[A_READ]][0, 1, 0, 0] : vector<32xf4E2M1FN> from vector<8x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[A_EXTRACT02:.+]] = vector.extract %[[A_READ]][0, 2, 0, 0] : vector<32xf4E2M1FN> from vector<8x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[A_EXTRACT03:.+]] = vector.extract %[[A_READ]][0, 3, 0, 0] : vector<32xf4E2M1FN> from vector<8x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[A_EXTRACT70:.+]] = vector.extract %[[A_READ]][7, 0, 0, 0] : vector<32xf4E2M1FN> from vector<8x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[A_EXTRACT71:.+]] = vector.extract %[[A_READ]][7, 1, 0, 0] : vector<32xf4E2M1FN> from vector<8x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[A_EXTRACT72:.+]] = vector.extract %[[A_READ]][7, 2, 0, 0] : vector<32xf4E2M1FN> from vector<8x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[A_EXTRACT73:.+]] = vector.extract %[[A_READ]][7, 3, 0, 0] : vector<32xf4E2M1FN> from vector<8x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[B_EXTRACT00:.+]] = vector.extract %[[B_READ]][0, 0, 0, 0] : vector<32xf4E2M1FN> from vector<2x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[B_EXTRACT01:.+]] = vector.extract %[[B_READ]][0, 1, 0, 0] : vector<32xf4E2M1FN> from vector<2x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[B_EXTRACT02:.+]] = vector.extract %[[B_READ]][0, 2, 0, 0] : vector<32xf4E2M1FN> from vector<2x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[B_EXTRACT03:.+]] = vector.extract %[[B_READ]][0, 3, 0, 0] : vector<32xf4E2M1FN> from vector<2x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[B_EXTRACT10:.+]] = vector.extract %[[B_READ]][1, 0, 0, 0] : vector<32xf4E2M1FN> from vector<2x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[B_EXTRACT11:.+]] = vector.extract %[[B_READ]][1, 1, 0, 0] : vector<32xf4E2M1FN> from vector<2x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[B_EXTRACT12:.+]] = vector.extract %[[B_READ]][1, 2, 0, 0] : vector<32xf4E2M1FN> from vector<2x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[B_EXTRACT13:.+]] = vector.extract %[[B_READ]][1, 3, 0, 0] : vector<32xf4E2M1FN> from vector<2x4x1x1x32xf4E2M1FN>
// CHECK-DAG:    %[[A_SCALE_VECTOR0:.+]] = vector.extract_strided_slice {{.*}} {offsets = [0], sizes = [4], strides = [1]} : vector<32xf8E8M0FNU> to vector<4xf8E8M0FNU>
// CHECK-DAG:    %[[A_SCALE_VECTOR7:.+]] = vector.extract_strided_slice {{.*}} {offsets = [28], sizes = [4], strides = [1]} : vector<32xf8E8M0FNU> to vector<4xf8E8M0FNU>
// CHECK-DAG:    %[[B_SCALE_VECTOR0:.+]] = vector.extract_strided_slice {{.*}} {offsets = [0], sizes = [4], strides = [1]} : vector<8xf8E8M0FNU> to vector<4xf8E8M0FNU>
// CHECK-DAG:    %[[B_SCALE_VECTOR1:.+]] = vector.extract_strided_slice {{.*}} {offsets = [4], sizes = [4], strides = [1]} : vector<8xf8E8M0FNU> to vector<4xf8E8M0FNU>
// CHECK-DAG:    %[[C_00_1:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR0]][0] * %[[A_EXTRACT00]]) * (%[[B_SCALE_VECTOR0]][0] * %[[B_EXTRACT00]]) + %arg[[#ITER_BASE]]
// CHECK-DAG:    %[[C_00_2:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR0]][1] * %[[A_EXTRACT01]]) * (%[[B_SCALE_VECTOR0]][1] * %[[B_EXTRACT01]]) + %[[C_00_1]]
// CHECK-DAG:    %[[C_00_3:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR0]][2] * %[[A_EXTRACT02]]) * (%[[B_SCALE_VECTOR0]][2] * %[[B_EXTRACT02]]) + %[[C_00_2]]
// CHECK-DAG:    %[[C_00_4:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR0]][3] * %[[A_EXTRACT03]]) * (%[[B_SCALE_VECTOR0]][3] * %[[B_EXTRACT03]]) + %[[C_00_3]]
// CHECK-DAG:    %[[C_01_1:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR0]][0] * %[[A_EXTRACT00]]) * (%[[B_SCALE_VECTOR1]][0] * %[[B_EXTRACT10]]) + %arg[[#ITER_BASE+1]]
// CHECK-DAG:    %[[C_01_2:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR0]][1] * %[[A_EXTRACT01]]) * (%[[B_SCALE_VECTOR1]][1] * %[[B_EXTRACT11]]) + %[[C_01_1]]
// CHECK-DAG:    %[[C_01_3:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR0]][2] * %[[A_EXTRACT02]]) * (%[[B_SCALE_VECTOR1]][2] * %[[B_EXTRACT12]]) + %[[C_01_2]]
// CHECK-DAG:    %[[C_01_4:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR0]][3] * %[[A_EXTRACT03]]) * (%[[B_SCALE_VECTOR1]][3] * %[[B_EXTRACT13]]) + %[[C_01_3]]
// CHECK-DAG:    %[[C_70_1:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR7]][0] * %[[A_EXTRACT70]]) * (%[[B_SCALE_VECTOR0]][0] * %[[B_EXTRACT00]]) + %arg[[#ITER_BASE+14]]
// CHECK-DAG:    %[[C_70_2:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR7]][1] * %[[A_EXTRACT71]]) * (%[[B_SCALE_VECTOR0]][1] * %[[B_EXTRACT01]]) + %[[C_70_1]]
// CHECK-DAG:    %[[C_70_3:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR7]][2] * %[[A_EXTRACT72]]) * (%[[B_SCALE_VECTOR0]][2] * %[[B_EXTRACT02]]) + %[[C_70_2]]
// CHECK-DAG:    %[[C_70_4:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR7]][3] * %[[A_EXTRACT73]]) * (%[[B_SCALE_VECTOR0]][3] * %[[B_EXTRACT03]]) + %[[C_70_3]]
// CHECK-DAG:    %[[C_71_1:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR7]][0] * %[[A_EXTRACT70]]) * (%[[B_SCALE_VECTOR1]][0] * %[[B_EXTRACT10]]) + %arg[[#ITER_BASE+15]]
// CHECK-DAG:    %[[C_71_2:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR7]][1] * %[[A_EXTRACT71]]) * (%[[B_SCALE_VECTOR1]][1] * %[[B_EXTRACT11]]) + %[[C_71_1]]
// CHECK-DAG:    %[[C_71_3:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR7]][2] * %[[A_EXTRACT72]]) * (%[[B_SCALE_VECTOR1]][2] * %[[B_EXTRACT12]]) + %[[C_71_2]]
// CHECK-DAG:    %[[C_71_4:.+]] = amdgpu.scaled_mfma 16x16x128 (%[[A_SCALE_VECTOR7]][3] * %[[A_EXTRACT73]]) * (%[[B_SCALE_VECTOR1]][3] * %[[B_EXTRACT13]]) + %[[C_71_3]]
// CHECK:        scf.yield
// CHECK:      vector.insert_strided_slice %[[LOOP]]#0, %{{.+}} {offsets = [0, 0, 0, 0, 0]{{.*}}} : vector<4xf32> into vector<8x2x1x1x4xf32>
// CHECK:      vector.insert_strided_slice %[[LOOP]]#1, %{{.+}} {offsets = [0, 1, 0, 0, 0]{{.*}}} : vector<4xf32> into vector<8x2x1x1x4xf32>
// CHECK:      vector.insert_strided_slice %[[LOOP]]#14, %{{.+}} {offsets = [7, 0, 0, 0, 0]{{.*}}} : vector<4xf32> into vector<8x2x1x1x4xf32>
// CHECK:      vector.insert_strided_slice %[[LOOP]]#15, %{{.+}} {offsets = [7, 1, 0, 0, 0]{{.*}}} : vector<4xf32> into vector<8x2x1x1x4xf32>
// CHECK:      vector.transfer_read %[[BUFFER_C]]
// CHECK:      arith.addf
// CHECK:      vector.transfer_write
// CHECK:      iree_codegen.dispatch_config @data_tiled_scaled_mma_inner_tiled workgroup_size = [256, 1, 1] subgroup_size = 64

// -----

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout_f16_transb = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config_f16_transb = #iree_gpu.lowering_config<{
  workgroup = [128, 128, 0],
  reduction = [0, 0, 1],
  subgroup = [4, 4, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>,
  promote_operands = [0, 1]
}>
func.func @matmul_transpose_b_f16()
  attributes {hal.executable.target = #executable_target_rocm, translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_f16_transb) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_f16_transb) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10240x1280xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_f16_transb) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10240x1280xf16>> -> tensor<10240x1280xf16>
  %5 = tensor.empty() : tensor<2048x10240xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
  %7 = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    {lowering_config = #config_f16_transb}
    ins(%3, %4 : tensor<2048x1280xf16>, tensor<10240x1280xf16>)
    outs(%6 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  return
}

//    CHECK-LABEL: func @matmul_transpose_b_f16
//      CHECK-DAG:   %[[ALLOC_A:.+]] = memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[ALLOC_B:.+]] = memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[GLOBAL_A:.+]] = amdgpu.fat_raw_buffer_cast {{.*}} : memref<2048x1280xf16{{.*}}> to memref<2048x1280xf16, #amdgpu.address_space<fat_raw_buffer>>
//      CHECK-DAG:   %[[GLOBAL_B:.+]] = amdgpu.fat_raw_buffer_cast {{.*}} : memref<10240x1280xf16{{.*}}> to memref<10240x1280xf16, #amdgpu.address_space<fat_raw_buffer>>
//      CHECK-DAG:   %[[GLOBAL_C:.+]] = amdgpu.fat_raw_buffer_cast {{.*}} : memref<{{.*}}xf32{{.*}}> to memref<{{.*}}xf32, #amdgpu.address_space<fat_raw_buffer>>
//          CHECK:   affine.delinearize_index {{.*}} into (16, 80)
//          CHECK:   scf.for {{.*}} -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
//      CHECK-DAG:     vector.transfer_read %[[GLOBAL_A]]{{.*}} : memref<2048x1280xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
//      CHECK-DAG:     vector.transfer_read %[[GLOBAL_B]]{{.*}} : memref<10240x1280xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
//          CHECK:     gpu.barrier
//      CHECK-DAG:     vector.transfer_read {{.*}}#gpu.address_space<workgroup>{{.*}} vector<4x1x1x8xf16>
//      CHECK-DAG:     vector.transfer_read {{.*}}#gpu.address_space<workgroup>{{.*}} vector<4x1x1x8xf16>
// CHECK-COUNT-16:     amdgpu.mfma 16x16x32
//      CHECK-DAG:     vector.transfer_write {{.*}} %[[ALLOC_A]]{{.*}} : vector<8xf16>, memref<128x36xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:     vector.transfer_write {{.*}} %[[ALLOC_B]]{{.*}} : vector<8xf16>, memref<128x36xf16, #gpu.address_space<workgroup>>
//          CHECK:     scf.yield
//          CHECK:   }
//          CHECK:   vector.transfer_write {{.*}} %[[GLOBAL_C]]{{.*}} : vector<{{.*}}xf32>, memref<{{.*}}xf32, #amdgpu.address_space<fat_raw_buffer>>
//          CHECK:   iree_codegen.dispatch_config @matmul_transpose_b_f16 workgroup_size = [256, 1, 1] subgroup_size = 64

// -----

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout_i8_transb = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config_i8_transb = #iree_gpu.lowering_config<{
  workgroup = [128, 128, 0],
  reduction = [0, 0, 1],
  subgroup = [4, 4, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x64_I8>,
  promote_operands = [0, 1]
}>

func.func @matmul_transpose_b_i8()
  attributes {hal.executable.target = #executable_target_rocm, translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_i8_transb) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_i8_transb) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10240x1280xi8>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_i8_transb) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xi8>> -> tensor<2048x1280xi8>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10240x1280xi8>> -> tensor<10240x1280xi8>
  %5 = tensor.empty() : tensor<2048x10240xi32>
  %6 = linalg.fill ins(%cst : i32) outs(%5 : tensor<2048x10240xi32>) -> tensor<2048x10240xi32>
  %7 = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    {lowering_config = #config_i8_transb}
    ins(%3, %4 : tensor<2048x1280xi8>, tensor<10240x1280xi8>)
    outs(%6 : tensor<2048x10240xi32>) -> tensor<2048x10240xi32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xi32>>
  return
}

//    CHECK-LABEL: func @matmul_transpose_b_i8
//      CHECK-DAG:   %[[ALLOC_A:.+]] = memref.alloc() : memref<128x72xi8, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[ALLOC_B:.+]] = memref.alloc() : memref<128x72xi8, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[GLOBAL_A:.+]] = amdgpu.fat_raw_buffer_cast {{.*}} : memref<2048x1280xi8{{.*}}> to memref<2048x1280xi8, #amdgpu.address_space<fat_raw_buffer>>
//      CHECK-DAG:   %[[GLOBAL_B:.+]] = amdgpu.fat_raw_buffer_cast {{.*}} : memref<10240x1280xi8{{.*}}> to memref<10240x1280xi8, #amdgpu.address_space<fat_raw_buffer>>
//      CHECK-DAG:   %[[GLOBAL_C:.+]] = amdgpu.fat_raw_buffer_cast {{.*}} : memref<{{.*}}xi32{{.*}}> to memref<{{.*}}xi32, #amdgpu.address_space<fat_raw_buffer>>
//          CHECK:   affine.delinearize_index {{.*}} into (16, 80)
//          CHECK:   scf.for {{.*}} -> (vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>) {
//      CHECK-DAG:     vector.transfer_read %[[GLOBAL_A]]{{.*}} : memref<2048x1280xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
//      CHECK-DAG:     vector.transfer_read %[[GLOBAL_B]]{{.*}} : memref<10240x1280xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
//          CHECK:     gpu.barrier
//      CHECK-DAG:     vector.transfer_read {{.*}}#gpu.address_space<workgroup>{{.*}} vector<4x1x1x16xi8>
//      CHECK-DAG:     vector.transfer_read {{.*}}#gpu.address_space<workgroup>{{.*}} vector<4x1x1x16xi8>
// CHECK-COUNT-16:     amdgpu.mfma 16x16x64
//      CHECK-DAG:     vector.transfer_write {{.*}} %[[ALLOC_A]]{{.*}} : vector<16xi8>, memref<128x72xi8, #gpu.address_space<workgroup>>
//      CHECK-DAG:     vector.transfer_write {{.*}} %[[ALLOC_B]]{{.*}} : vector<16xi8>, memref<128x72xi8, #gpu.address_space<workgroup>>
//          CHECK:     scf.yield
//          CHECK:   }
//          CHECK:   vector.transfer_write {{.*}} %[[GLOBAL_C]]{{.*}} : vector<{{.*}}xi32>, memref<{{.*}}xi32, #amdgpu.address_space<fat_raw_buffer>>
//          CHECK:   iree_codegen.dispatch_config @matmul_transpose_b_i8 workgroup_size = [256, 1, 1] subgroup_size = 64

// -----

// regular matmul (non-transpose-b) with f16 using gfx950 config
#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout_f16 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config_f16 = #iree_gpu.lowering_config<{
  workgroup = [128, 128, 0],
  reduction = [0, 0, 1],
  subgroup = [4, 4, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>,
  promote_operands = [0, 1]
}>

func.func @matmul_f16()
  attributes {hal.executable.target = #executable_target_rocm, translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x10240xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_f16) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 10240], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x10240xf16>> -> tensor<1280x10240xf16>
  %5 = tensor.empty() : tensor<2048x10240xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
  %7 = linalg.matmul
    {lowering_config = #config_f16}
    ins(%3, %4 : tensor<2048x1280xf16>, tensor<1280x10240xf16>)
    outs(%6 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  return
}

//    CHECK-LABEL: func @matmul_f16
//      CHECK-DAG:   %[[ALLOC_B:.+]] = memref.alloc() : memref<32x132xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[ALLOC_A:.+]] = memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[GLOBAL_A:.+]] = amdgpu.fat_raw_buffer_cast {{.*}} : memref<2048x1280xf16{{.*}}> to memref<2048x1280xf16, #amdgpu.address_space<fat_raw_buffer>>
//      CHECK-DAG:   %[[GLOBAL_B:.+]] = amdgpu.fat_raw_buffer_cast {{.*}} : memref<1280x10240xf16{{.*}}> to memref<1280x10240xf16, #amdgpu.address_space<fat_raw_buffer>>
//      CHECK-DAG:   %[[GLOBAL_C:.+]] = amdgpu.fat_raw_buffer_cast {{.*}} : memref<{{.*}}xf32{{.*}}> to memref<{{.*}}xf32, #amdgpu.address_space<fat_raw_buffer>>
//          CHECK:   affine.delinearize_index {{.*}} into (16, 80)
//          CHECK:   scf.for {{.*}} -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
//      CHECK-DAG:     vector.transfer_read %[[GLOBAL_A]]{{.*}} : memref<2048x1280xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
//      CHECK-DAG:     vector.transfer_read %[[GLOBAL_B]]{{.*}} : memref<1280x10240xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
//          CHECK:     gpu.barrier
//          CHECK:     vector.transfer_read {{.*}}#gpu.address_space<workgroup>{{.*}} vector<4x1x1x8xf16>
//  CHECK-COUNT-8:     amdgpu.transpose_load {{.*}}#gpu.address_space<workgroup>{{.*}} -> vector<4xf16>
// CHECK-COUNT-16:     amdgpu.mfma 16x16x32
//      CHECK-DAG:     vector.transfer_write {{.*}} %[[ALLOC_A]]{{.*}} : vector<8xf16>, memref<128x36xf16, #gpu.address_space<workgroup>>
//      CHECK-DAG:     vector.transfer_write {{.*}} %[[ALLOC_B]]{{.*}} : vector<8xf16>, memref<32x132xf16, #gpu.address_space<workgroup>>
//          CHECK:     scf.yield
//          CHECK:   }
//          CHECK:   vector.transfer_write {{.*}} %[[GLOBAL_C]]{{.*}} : vector<{{.*}}xf32>, memref<{{.*}}xf32, #amdgpu.address_space<fat_raw_buffer>>
//          CHECK:   iree_codegen.dispatch_config @matmul_f16 workgroup_size = [256, 1, 1] subgroup_size = 64

// -----

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout_i8 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config_i8 = #iree_gpu.lowering_config<{
  workgroup = [128, 128, 0],
  reduction = [0, 0, 1],
  subgroup = [4, 4, 0],
  mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x64_I8>,
  promote_operands = [0, 1]
}>

func.func @matmul_i8()
  attributes {hal.executable.target = #executable_target_rocm, translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_i8) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_i8) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x10240xi8>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_i8) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xi32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x1280xi8>> -> tensor<2048x1280xi8>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 10240], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x10240xi8>> -> tensor<1280x10240xi8>
  %5 = tensor.empty() : tensor<2048x10240xi32>
  %6 = linalg.fill ins(%cst : i32) outs(%5 : tensor<2048x10240xi32>) -> tensor<2048x10240xi32>
  %7 = linalg.matmul
    {lowering_config = #config_i8}
    ins(%3, %4 : tensor<2048x1280xi8>, tensor<1280x10240xi8>)
    outs(%6 : tensor<2048x10240xi32>) -> tensor<2048x10240xi32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x10240xi32>>
  return
}

//    CHECK-LABEL: func @matmul_i8
//      CHECK-DAG:   %[[ALLOC_B:.+]] = memref.alloc() : memref<64x136xi8, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[ALLOC_A:.+]] = memref.alloc() : memref<128x72xi8, #gpu.address_space<workgroup>>
//      CHECK-DAG:   %[[GLOBAL_A:.+]] = amdgpu.fat_raw_buffer_cast {{.*}} : memref<2048x1280xi8{{.*}}> to memref<2048x1280xi8, #amdgpu.address_space<fat_raw_buffer>>
//      CHECK-DAG:   %[[GLOBAL_B:.+]] = amdgpu.fat_raw_buffer_cast {{.*}} : memref<1280x10240xi8{{.*}}> to memref<1280x10240xi8, #amdgpu.address_space<fat_raw_buffer>>
//      CHECK-DAG:   %[[GLOBAL_C:.+]] = amdgpu.fat_raw_buffer_cast {{.*}} : memref<{{.*}}xi32{{.*}}> to memref<{{.*}}xi32, #amdgpu.address_space<fat_raw_buffer>>
//          CHECK:   affine.delinearize_index {{.*}} into (16, 80)
//          CHECK:   scf.for {{.*}} -> (vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>) {
//      CHECK-DAG:     vector.transfer_read %[[GLOBAL_A]]{{.*}} : memref<2048x1280xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
//      CHECK-DAG:     vector.transfer_read %[[GLOBAL_B]]{{.*}} : memref<1280x10240xi8, #amdgpu.address_space<fat_raw_buffer>>, vector<16xi8>
//          CHECK:     gpu.barrier
//          CHECK:     vector.transfer_read {{.*}}#gpu.address_space<workgroup>{{.*}} vector<4x1x1x16xi8>
//  CHECK-COUNT-8:     amdgpu.transpose_load {{.*}}#gpu.address_space<workgroup>{{.*}} -> vector<8xi8>
// CHECK-COUNT-16:     amdgpu.mfma 16x16x64
//      CHECK-DAG:     vector.transfer_write {{.*}} %[[ALLOC_A]]{{.*}} : vector<16xi8>, memref<128x72xi8, #gpu.address_space<workgroup>>
//      CHECK-DAG:     vector.transfer_write {{.*}} %[[ALLOC_B]]{{.*}} : vector<16xi8>, memref<64x136xi8, #gpu.address_space<workgroup>>
//          CHECK:     scf.yield
//          CHECK:   }
//          CHECK:   vector.transfer_write {{.*}} %[[GLOBAL_C]]{{.*}} : vector<{{.*}}xi32>, memref<{{.*}}xi32, #amdgpu.address_space<fat_raw_buffer>>
//          CHECK:   iree_codegen.dispatch_config @matmul_i8 workgroup_size = [256, 1, 1] subgroup_size = 64

// -----

// DataTiledScaledMMAAttr with unshuffled_operands pipeline test: verifies XOR
// swizzle hints on data operands, software pipelining, and amdgpu.scaled_mfma
// generation.

#executable_target_rocm_pdt = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout_pdt = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>],
  flags = Indirect
>
#translation_info_pdt = #iree_codegen.translation_info<pipeline =
  #iree_gpu.pipeline<TileAndFuse>
  workgroup_size = [64, 1, 1]
  subgroup_size = 64,
  {
    gpu_pipeline_options = #iree_gpu.pipeline_options<
      prefetch_num_stages = 2,
      no_reduce_shared_memory_bank_conflicts = true>
  }
>
#config_pdt = #iree_gpu.lowering_config<{
  workgroup = [1, 1, 0, 0],
  reduction = [0, 0, 1, 1],
  promote_operands = [0, 1, 2, 3],
  promotion_types = [
    #iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config,
      swizzle = #iree_codegen.xor_shuffle<256, 32>>,
    #iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config,
      swizzle = #iree_codegen.xor_shuffle<256, 32>>,
    #iree_gpu.derived_thread_config,
    #iree_gpu.derived_thread_config]
}>
func.func @unshuffled_dt_scaled_mma()
  attributes {hal.executable.target = #executable_target_rocm_pdt, translation_info = #translation_info_pdt} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_pdt) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x1x16x4x32xf4E2M1FN>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_pdt) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x1x16x4x32xf4E2M1FN>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_pdt) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x4x16xf8E8M0FNU>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout_pdt) binding(3) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x4x16xf8E8M0FNU>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout_pdt) binding(4) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<9x9x4x16x4xf32>>
  %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0, 0], sizes = [9, 9, 1, 16, 4, 32], strides = [1, 1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x1x16x4x32xf4E2M1FN>> -> tensor<9x9x1x16x4x32xf4E2M1FN>
  %6 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0, 0, 0], sizes = [9, 9, 1, 16, 4, 32], strides = [1, 1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x1x16x4x32xf4E2M1FN>> -> tensor<9x9x1x16x4x32xf4E2M1FN>
  %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [9, 9, 4, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x4x16xf8E8M0FNU>> -> tensor<9x9x4x16xf8E8M0FNU>
  %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [9, 9, 4, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x4x16xf8E8M0FNU>> -> tensor<9x9x4x16xf8E8M0FNU>
  %9 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0, 0, 0, 0], sizes = [9, 9, 4, 16, 4], strides = [1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<9x9x4x16x4xf32>> -> tensor<9x9x4x16x4xf32>
  %10 = iree_codegen.inner_tiled ins(%5, %6, %7, %8) outs(%9) {
    lowering_config = #config_pdt,
    indexing_maps = [
      affine_map<(m, n, k, kb) -> (m, k, kb)>,
      affine_map<(m, n, k, kb) -> (n, k, kb)>,
      affine_map<(m, n, k, kb) -> (m, k)>,
      affine_map<(m, n, k, kb) -> (n, k)>,
      affine_map<(m, n, k, kb) -> (m, n)>],
    iterator_types = [
      #linalg.iterator_type<parallel>,
      #linalg.iterator_type<parallel>,
      #linalg.iterator_type<reduction>,
      #linalg.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32,
      operands_interleaving_intrinsics_m = [2],
      operands_interleaving_intrinsics_n = [3],
      operands_interleaving_intrinsics_k = [2, 3],
      unshuffled_operands = [0, 1]>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = false>}
    : tensor<9x9x1x16x4x32xf4E2M1FN>, tensor<9x9x1x16x4x32xf4E2M1FN>, tensor<9x9x4x16xf8E8M0FNU>, tensor<9x9x4x16xf8E8M0FNU> into tensor<9x9x4x16x4xf32>
  iree_tensor_ext.dispatch.tensor.store %10, %4, offsets = [0, 0, 0, 0, 0], sizes = [9, 9, 4, 16, 4], strides = [1, 1, 1, 1, 1] : tensor<9x9x4x16x4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<9x9x4x16x4xf32>>
  return
}

// CHECK-LABEL: func.func @unshuffled_dt_scaled_mma()
// CHECK-DAG:  %[[PDT_C0:.+]] = arith.constant 0 : index
// CHECK-DAG:  %[[PDT_C1:.+]] = arith.constant 1 : index
// CHECK-DAG:  %[[PDT_C8:.+]] = arith.constant 8 : index
// CHECK-DAG:  %[[PDT_BUFFER_A:.+]] = amdgpu.fat_raw_buffer_cast %{{.*}} : memref<9x9x1x16x4x32xf4E2M1FN{{.*}}> to memref<9x9x1x16x4x32xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-DAG:  %[[PDT_BUFFER_B:.+]] = amdgpu.fat_raw_buffer_cast %{{.*}} : memref<9x9x1x16x4x32xf4E2M1FN{{.*}}> to memref<9x9x1x16x4x32xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>>
// CHECK-DAG:  %[[PDT_A_ALLOC:.+]] = memref.alloc() {iree_codegen.swizzle = #iree_codegen.xor_shuffle<256, 32>} : memref<2048xf4E2M1FN, #gpu.address_space<workgroup>>
// CHECK-DAG:  %[[PDT_B_ALLOC:.+]] = memref.alloc() {iree_codegen.swizzle = #iree_codegen.xor_shuffle<256, 32>} : memref<2048xf4E2M1FN, #gpu.address_space<workgroup>>
//     CHECK:  gpu.barrier memfence [#gpu.address_space<workgroup>]
//     CHECK:  scf.for {{.*}} %[[PDT_C0]] to %[[PDT_C8]] step %[[PDT_C1]] iter_args({{.*}}) -> (vector<4xf32>)
// CHECK-DAG:    vector.transfer_read %[[PDT_BUFFER_A]]{{.*}} vector<32xf4E2M1FN>
// CHECK-DAG:    vector.transfer_read %[[PDT_BUFFER_B]]{{.*}} vector<32xf4E2M1FN>
//     CHECK:    gpu.barrier
//     CHECK:    amdgpu.scaled_mfma 16x16x128
//     CHECK:    scf.yield
//     CHECK:  }
//     CHECK:  amdgpu.scaled_mfma 16x16x128
