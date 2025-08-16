// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --pass-pipeline='builtin.module(linalg-specialize-generic-ops,iree-llvmgpu-select-lowering-strategy)' %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SPECIALIZED

// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --pass-pipeline='builtin.module(linalg-generalize-named-ops,iree-llvmgpu-select-lowering-strategy)' %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,GENERALIZED


// ============================================================================
// All static dimension matmuls
// ============================================================================

!TA = tensor<32x32xf32>
!TB = tensor<32x32xf32>
!TC = tensor<32x32xf32>
!DTC = !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x32xf32>>

//      CHECK:    #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
// CHECK-SAME:    workgroup_size = [256, 1, 1] subgroup_size = 64, {}>
func.func @matmul_32_32_32(%arg0: !TA, %arg1: !TB, %arg2: !TC, %arg3: !DTC) {
  // Sanity check that generalize/specialize works.
  // GENERALIZED:   linalg.generic
  // SPECIALIZED:   linalg.matmul
  //      CHECK:  #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
  // CHECK-SAME:  promote_operands = [0, 1], reduction = [0, 0, 32], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64,
  // CHECK-SAME:  workgroup = [32, 32, 0]}>
  %0 = linalg.matmul ins(%arg0, %arg1 : !TA, !TB) outs(%arg2 : !TC) -> !TC
  iree_tensor_ext.dispatch.tensor.store %0, %arg3, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !TC -> !DTC
  return
}

// -----

!TA = tensor<4096x4096xf32>
!TB = tensor<4096x4096xf32>
!TC = tensor<4096x4096xf32>
!DTC = !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4096x4096xf32>>
//      CHECK:    #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
// CHECK-SAME:    workgroup_size = [256, 1, 1] subgroup_size = 64, {}>
func.func @matmul_4096_4096_4096(%arg0: !TA, %arg1: !TB, %arg2: !TC, %arg3: !DTC) {
  //      CHECK:  #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
  // CHECK-SAME:  promote_operands = [0, 1], reduction = [0, 0, 16], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64,
  // CHECK-SAME:  workgroup = [64, 128, 0]}>}
  %0 = linalg.matmul ins(%arg0, %arg1 : !TA, !TB) outs(%arg2 : !TC) -> !TC
  iree_tensor_ext.dispatch.tensor.store %0, %arg3, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !TC -> !DTC
  return
}

// -----

!TA = tensor<4096x32xf32>
!TB = tensor<32x4096xf32>
!TC = tensor<4096x4096xf32>
!DTC = !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4096x4096xf32>>
//      CHECK:    #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
// CHECK-SAME:    workgroup_size = [256, 1, 1] subgroup_size = 64, {}>
func.func @matmul_4096_32_4096(%arg0: !TA, %arg1: !TB, %arg2: !TC, %arg3: !DTC) {
  //      CHECK:  #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
  // CHECK-SAME:  promote_operands = [0, 1], reduction = [0, 0, 16], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64,
  // CHECK-SAME:  workgroup = [64, 128, 0]}>}
  %0 = linalg.matmul ins(%arg0, %arg1 : !TA, !TB) outs(%arg2 : !TC) -> !TC
  iree_tensor_ext.dispatch.tensor.store %0, %arg3, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !TC -> !DTC
  return
}

// -----

!TA = tensor<4096x1xf32>
!TB = tensor<1x4096xf32>
!TC = tensor<4096x4096xf32>
!DTC = !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4096x4096xf32>>
//      CHECK:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
// CHECK-SAME:   workgroup_size = [256, 1, 1] subgroup_size = 64
// CHECK-SAME:   {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>
func.func @matmul_4096_1_4096(%arg0: !TA, %arg1: !TB, %arg2: !TC, %arg3: !DTC) {
  //      CHECK: #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
  // CHECK-SAME: padding = [64, 128, 4], promote_operands = [0, 1, 2], reduction = [0, 0, 1], subgroup = [2, 4, 0], workgroup = [64, 128, 0]}
  %0 = linalg.matmul ins(%arg0, %arg1 : !TA, !TB) outs(%arg2 : !TC) -> !TC
  iree_tensor_ext.dispatch.tensor.store %0, %arg3, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !TC -> !DTC
  return
}

// -----

// ============================================================================
// Dynamic M
// ============================================================================

!TA = tensor<?x32xf32>
!TB = tensor<32x32xf32>
!TC = tensor<?x32xf32>
!DTC = !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x32xf32>>
//      CHECK:    #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
// CHECK-SAME:    workgroup_size = [64, 1, 1] subgroup_size = 64>
func.func @matmul_DYN_32_32(%arg0: !TA, %arg1: !TB, %arg2: !TC, %arg3: !DTC, %arg4 : index) {
  // CHECK:       #iree_gpu.lowering_config<{promote_operands = [0, 1], reduction = [0, 0, 4], thread = [1, 1, 0], workgroup = [1, 64, 0]}>
  %0 = linalg.matmul ins(%arg0, %arg1 : !TA, !TB) outs(%arg2 : !TC) -> !TC
  iree_tensor_ext.dispatch.tensor.store %0, %arg3, offsets = [0, 0], sizes = [%arg4, 32], strides = [1, 1] : !TC -> !DTC{%arg4}
  return
}

// -----

// TODO(newling) specialized form should be the same as generalized form.

//      GENERALIZED:         #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
// GENERALIZED-SAME:         workgroup_size = [1024, 1, 1] subgroup_size = 64,
// GENERALIZED-SAME:         {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>

//      SPECIALIZED:         #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
!TA = tensor<?x4096xf32>
!TB = tensor<4096x4096xf32>
!TC = tensor<?x4096xf32>
!DTC = !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x4096xf32>>
func.func @matmul_DYN_4096_4096(%arg0: !TA, %arg1: !TB, %arg2: !TC, %arg3: !DTC, %arg4 : index) {
//      GENERALIZED:         #iree_gpu.lowering_config<{lane_basis =  {{\[}}[1, 1, 64], [0, 1, 2]],
// GENERALIZED-SAME:         partial_reduction = [0, 0, 4096], subgroup_basis =  {{\[}}[1, 1, 16], [0, 1, 2]], thread = [0, 0, 4], workgroup = [1, 1, 0]}
  %0 = linalg.matmul ins(%arg0, %arg1 : !TA, !TB) outs(%arg2 : !TC) -> !TC
  iree_tensor_ext.dispatch.tensor.store %0, %arg3, offsets = [0, 0], sizes = [%arg4, 4096], strides = [1, 1] : !TC -> !DTC{%arg4}
  return
}

// -----

// CHECK:    #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
// CHECK:    #iree_gpu.lowering_config<{promote_operands = [0, 1], reduction = [0, 0, 4], thread = [1, 4, 0], workgroup = [1, 256, 0]}
!TA = tensor<?x32xf32>
!TB = tensor<32x4096xf32>
!TC = tensor<?x4096xf32>
!DTC = !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x4096xf32>>
func.func @matmul_DYN_32_4096(%arg0: !TA, %arg1: !TB, %arg2: !TC, %arg3: !DTC, %arg4 : index) {
  %0 = linalg.matmul ins(%arg0, %arg1 : !TA, !TB) outs(%arg2 : !TC) -> !TC
  iree_tensor_ext.dispatch.tensor.store %0, %arg3, offsets = [0, 0], sizes = [%arg4, 4096], strides = [1, 1] : !TC -> !DTC{%arg4}
  return
}

// -----

// CHECK:    #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
// CHECK:    #iree_gpu.lowering_config<{promote_operands = [0, 1], reduction = [0, 0, 1], thread = [1, 4, 0], workgroup = [1, 256, 0]}
!TA = tensor<?x1xf32>
!TB = tensor<1x4096xf32>
!TC = tensor<?x4096xf32>
!DTC = !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x4096xf32>>
func.func @matmul_DYN_1_4096(%arg0: !TA, %arg1: !TB, %arg2: !TC, %arg3: !DTC, %arg4 : index) {
  %0 = linalg.matmul ins(%arg0, %arg1 : !TA, !TB) outs(%arg2 : !TC) -> !TC
  iree_tensor_ext.dispatch.tensor.store %0, %arg3, offsets = [0, 0], sizes = [%arg4, 4096], strides = [1, 1] : !TC -> !DTC{%arg4}
  return
}

// ============================================================================
// Dynamic K
// ============================================================================

// -----

!TA = tensor<32x?xf32>
!TB = tensor<?x32xf32>
!TC = tensor<32x32xf32>
!DTC = !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x32xf32>>

//      CHECK:    #translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
// CHECK-SAME:    workgroup_size = [64, 16, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false,
// CHECK-SAME:    no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>
func.func @matmul_32_32_DYN(%arg0: !TA, %arg1: !TB, %arg2: !TC, %arg3: !DTC) {
   // CHECK:     #iree_gpu.lowering_config<{reduction = [0, 0, 1], thread = [1, 8, 0], workgroup = [64, 128, 1]}
  %0 = linalg.matmul ins(%arg0, %arg1 : !TA, !TB) outs(%arg2 : !TC) -> !TC
  iree_tensor_ext.dispatch.tensor.store %0, %arg3, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !TC -> !DTC
  return
}

// -----

!TA = tensor<4096x?xf32>
!TB = tensor<?x4096xf32>
!TC = tensor<4096x4096xf32>
!DTC = !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4096x4096xf32>>
//      CHECK:         #translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
// CHECK-SAME:         workgroup_size = [64, 16, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false,
// CHECK-SAME:         no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>
func.func @matmul_4096_4096_DYN(%arg0: !TA, %arg1: !TB, %arg2: !TC, %arg3: !DTC) {
   // CHECK:      #iree_gpu.lowering_config<{reduction = [0, 0, 1], thread = [1, 8, 0], workgroup = [64, 128, 1]}
  %0 = linalg.matmul ins(%arg0, %arg1 : !TA, !TB) outs(%arg2 : !TC) -> !TC
  iree_tensor_ext.dispatch.tensor.store %0, %arg3, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !TC -> !DTC
  return
}

// -----

// ============================================================================
// Dynamic all
// ============================================================================


// TODO(newling) specialized form should follow generalized form.

//     SPECIALIZED:      LLVMGPUWarpReduction
//     GENERALIZED:      LLVMGPUVectorDistribute
// GENERALIZED-SAME:     workgroup_size = [1024, 1, 1] subgroup_size = 64,


!TA = tensor<?x?xf32>
!TB = tensor<?x?xf32>
!TC = tensor<?x?xf32>
!DTC = !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>
func.func @matmul_DYN_1_4096(%arg0: !TA, %arg1: !TB, %arg2: !TC, %arg3: !DTC, %arg4 : index, %arg5 : index, %arg6 : index) {

  //      GENERALIZED:     {lane_basis = {{\[}}[1, 1, 64], [0, 1, 2]],
  // GENERALIZED-SAME:     partial_reduction = [0, 0, 4096], subgroup_basis = {{\[}}[1, 1, 16], [0, 1, 2]],
  // GENERALIZED-SAME:     thread = [0, 0, 4], workgroup = [1, 1, 0]}
  %0 = linalg.matmul ins(%arg0, %arg1 : !TA, !TB) outs(%arg2 : !TC) -> !TC
  iree_tensor_ext.dispatch.tensor.store %0, %arg3, offsets = [0, 0], sizes = [%arg4, %arg5], strides = [1, 1] : !TC -> !DTC{%arg4, %arg5}
  return
}
