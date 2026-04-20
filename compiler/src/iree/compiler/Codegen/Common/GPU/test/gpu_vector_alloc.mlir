// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-vector-alloc))" | FileCheck %s
// RUN: iree-opt %s --split-input-file --iree-codegen-gpu-enable-vector-alloc-swizzle --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-vector-alloc))" | FileCheck %s --check-prefix=SWZ

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],

  subgroup_strides = [1, 1],
  thread_strides   = [0, 0]
>

func.func @test(%vector: vector<16x16xf16>) -> vector<16x16xf16> {
  %out = iree_vector_ext.to_layout %vector to layout(#layout) {shared_memory_conversion} : vector<16x16xf16>
  return %out : vector<16x16xf16>
}


//    CHECK-LABEL: func.func @test
//         CHECK:    gpu.barrier memfence [#gpu.address_space<workgroup>]
//         CHECK:    %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16x16xf16, #gpu.address_space<workgroup>>
//         CHECK:    %[[WRITE:.+]] = vector.transfer_write %{{.*}}, %[[ALLOC]]
//         CHECK:    %[[BAR:.+]]   = iree_gpu.value_barrier %[[WRITE]]
//         CHECK:    %[[READ:.+]]  = vector.transfer_read %[[BAR]]
//         CHECK:    %[[OUT:.+]]   = iree_vector_ext.to_layout %[[READ]]

// With --iree-codegen-gpu-enable-vector-alloc-swizzle the alloc is wrapped in
// a SwizzleHintOp (flat-1D alloc + expand_shape). For reader element_tile =
// [4, 1], accessWidth = max(writer=1, reader=1) = 1, innerDim = 16, maxPhase
// = 16 → swizzle xor_shuffle<16, 1, 16, 1> is created.

//    SWZ-LABEL: func.func @test
//         SWZ:    %[[FLAT:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<256xf16, #gpu.address_space<workgroup>>
//         SWZ:    %[[HINT:.+]] = iree_codegen.swizzle_hint %[[FLAT]][#iree_codegen.xor_shuffle<16, 1, 16, 1>]
//         SWZ:    tensor.expand_shape %[[HINT]]

// -----

// With a writer doing vector<8> stores and reader doing vector<8> reads
// (f16 attention K-operand pattern), accessWidth = 8, maxPhase = 16 → swizzle
// created.

#wide_layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [16, 16],
  element_tile = [1, 8],
  subgroup_strides = [1, 1],
  thread_strides = [16, 1]
>

func.func @test_wide(%vector: vector<16x128xf16>) -> vector<16x128xf16> {
  %out = iree_vector_ext.to_layout %vector to layout(#wide_layout) {shared_memory_conversion} : vector<16x128xf16>
  return %out : vector<16x128xf16>
}

//    SWZ-LABEL: func.func @test_wide
//         SWZ:    %[[FLAT:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<2048xf16, #gpu.address_space<workgroup>>
//         SWZ:    %[[HINT:.+]] = iree_codegen.swizzle_hint %[[FLAT]][#iree_codegen.xor_shuffle<128, 8, 128, 1>]
//         SWZ:    %[[EXP:.+]]  = tensor.expand_shape %[[HINT]]
//         SWZ:    vector.transfer_write %{{.*}}, %[[EXP]]

// -----

// f32 with writer vector<4> and reader scalar (MFMA_F32_16x16x4_F32 pattern):
// accessWidth = max(4, 1) = 4 triggers the cap → shrinks to 2 for bijection.

#f32_in = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [16, 16],
  element_tile = [1, 4],
  subgroup_strides = [1, 1],
  thread_strides = [16, 1]
>

#f32_out = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 16],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 1],
  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

func.func @test_f32(%vector: vector<16x64xf32>) -> vector<16x64xf32> {
  %in = iree_vector_ext.to_layout %vector to layout(#f32_in) : vector<16x64xf32>
  %out = iree_vector_ext.to_layout %in to layout(#f32_out) {shared_memory_conversion} : vector<16x64xf32>
  return %out : vector<16x64xf32>
}

//    SWZ-LABEL: func.func @test_f32
//         SWZ:    iree_codegen.swizzle_hint %{{.*}}[#iree_codegen.xor_shuffle<64, 2, 64, 1>]
