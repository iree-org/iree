// RUN: iree-opt %s --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-vector-alloc))" | FileCheck %s

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>

#consumer_layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],
  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

func.func @promote_global_transfer_read(%src: memref<16x16xf16>) -> vector<16x16xf16>
    attributes {translation_info = #translation} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %read = vector.transfer_read %src[%c0, %c0], %cst : memref<16x16xf16>, vector<16x16xf16>
  %out = iree_vector_ext.to_layout %read to layout(#consumer_layout)
      {shared_memory_conversion = #iree_gpu.derived_thread_config} : vector<16x16xf16>
  return %out : vector<16x16xf16>
}

// CHECK-DAG: #[[$READ_LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [16, 4], element_tile = [1, 4], subgroup_strides = [0, 0], thread_strides = [4, 1]>
// CHECK-DAG: #[[$CONSUMER_LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [4, 16], element_tile = [4, 1], subgroup_strides = [0, 0], thread_strides = [0, 0]>
// CHECK-LABEL: func.func @promote_global_transfer_read
// CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
// CHECK: %[[GLOBAL:.+]] = vector.transfer_read %{{.*}} : memref<16x16xf16>, vector<16x16xf16>
// CHECK: %[[READ_LAYOUT_VALUE:.+]] = iree_vector_ext.to_layout %[[GLOBAL]] to layout(#[[$READ_LAYOUT]]) : vector<16x16xf16>
// CHECK: %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16x16xf16, #gpu.address_space<workgroup>>
// CHECK: %[[WRITE:.+]] = vector.transfer_write %[[READ_LAYOUT_VALUE]], %[[ALLOC]]
// CHECK: %[[BARRIER:.+]] = iree_gpu.value_barrier %[[WRITE]]
// CHECK: %[[LDS_READ:.+]] = vector.transfer_read %[[BARRIER]]
// CHECK: %[[OUT:.+]] = iree_vector_ext.to_layout %[[LDS_READ]] to layout(#[[$CONSUMER_LAYOUT]])
// CHECK-NOT: shared_memory_conversion
// CHECK: return %[[OUT]]

// -----

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>

#consumer_layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],
  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

func.func @promote_global_gather(%src: memref<16x16xf16>,
                                 %indices: vector<16x16xindex>,
                                 %mask: vector<16x16xi1>,
                                 %passthru: vector<16x16xf16>) -> vector<16x16xf16>
    attributes {translation_info = #translation} {
  %c0 = arith.constant 0 : index
  %gather = vector.gather %src[%c0, %c0] [%indices], %mask, %passthru
      : memref<16x16xf16>, vector<16x16xindex>, vector<16x16xi1>, vector<16x16xf16> into vector<16x16xf16>
  %out = iree_vector_ext.to_layout %gather to layout(#consumer_layout)
      {shared_memory_conversion = #iree_gpu.derived_thread_config} : vector<16x16xf16>
  return %out : vector<16x16xf16>
}

// CHECK-DAG: #[[$GATHER_READ_LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [16, 4], element_tile = [1, 4], subgroup_strides = [0, 0], thread_strides = [4, 1]>
// CHECK-DAG: #[[$GATHER_CONSUMER_LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [4, 16], element_tile = [4, 1], subgroup_strides = [0, 0], thread_strides = [0, 0]>
// CHECK-LABEL: func.func @promote_global_gather
// CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
// CHECK: %[[GLOBAL:.+]] = vector.gather %{{.*}} : memref<16x16xf16>, vector<16x16xindex>, vector<16x16xi1>, vector<16x16xf16> into vector<16x16xf16>
// CHECK: %[[READ_LAYOUT_VALUE:.+]] = iree_vector_ext.to_layout %[[GLOBAL]] to layout(#[[$GATHER_READ_LAYOUT]]) : vector<16x16xf16>
// CHECK: %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16x16xf16, #gpu.address_space<workgroup>>
// CHECK: %[[WRITE:.+]] = vector.transfer_write %[[READ_LAYOUT_VALUE]], %[[ALLOC]]
// CHECK: %[[BARRIER:.+]] = iree_gpu.value_barrier %[[WRITE]]
// CHECK: %[[LDS_READ:.+]] = vector.transfer_read %[[BARRIER]]
// CHECK: %[[OUT:.+]] = iree_vector_ext.to_layout %[[LDS_READ]] to layout(#[[$GATHER_CONSUMER_LAYOUT]])
// CHECK-NOT: shared_memory_conversion
// CHECK: return %[[OUT]]

// -----

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>

#consumer_layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],
  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

func.func @promote_global_transfer_read_in_loop(%src: memref<16x16xf16>,
                                                %init: vector<16x16xf16>) -> vector<16x16xf16>
    attributes {translation_info = #translation} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f16
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (vector<16x16xf16>) {
    %read = vector.transfer_read %src[%c0, %c0], %cst : memref<16x16xf16>, vector<16x16xf16>
    %out = iree_vector_ext.to_layout %read to layout(#consumer_layout)
        {shared_memory_conversion = #iree_gpu.derived_thread_config} : vector<16x16xf16>
    scf.yield %out : vector<16x16xf16>
  }
  return %result : vector<16x16xf16>
}

// CHECK-LABEL: func.func @promote_global_transfer_read_in_loop
// CHECK: scf.for
// CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK: iree_gpu.value_barrier
// CHECK: scf.yield

// -----

#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],
  subgroup_strides = [0, 0],
  thread_strides   = [4, 1]
>

#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],
  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

func.func @materialize_layout_conflict(%vector: vector<16x16xf16>) -> vector<16x16xf16> {
  %a = iree_vector_ext.to_layout %vector to layout(#layout_a) : vector<16x16xf16>
  %b = iree_vector_ext.to_layout %a to layout(#layout_b) : vector<16x16xf16>
  return %b : vector<16x16xf16>
}

// CHECK-DAG: #[[$CONFLICT_LAYOUT_A:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [16, 4], element_tile = [1, 4], subgroup_strides = [0, 0], thread_strides = [4, 1]>
// CHECK-DAG: #[[$CONFLICT_LAYOUT_B:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [4, 16], element_tile = [4, 1], subgroup_strides = [0, 0], thread_strides = [0, 0]>
// CHECK-LABEL: func.func @materialize_layout_conflict
// CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
// CHECK: %[[A:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$CONFLICT_LAYOUT_A]]) : vector<16x16xf16>
// CHECK: %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16x16xf16, #gpu.address_space<workgroup>>
// CHECK: %[[WRITE:.+]] = vector.transfer_write %[[A]], %[[ALLOC]]
// CHECK: %[[BARRIER:.+]] = iree_gpu.value_barrier %[[WRITE]]
// CHECK: %[[LDS_READ:.+]] = vector.transfer_read %[[BARRIER]]
// CHECK: %[[B:.+]] = iree_vector_ext.to_layout %[[LDS_READ]] to layout(#[[$CONFLICT_LAYOUT_B]]) : vector<16x16xf16>
// CHECK-NOT: shared_memory_conversion
// CHECK: %[[OUT:.+]] = iree_vector_ext.to_layout %[[B]] to layout(#[[$CONFLICT_LAYOUT_B]]) : vector<16x16xf16>
// CHECK: return %[[OUT]]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],
  subgroup_strides = [0, 0],
  thread_strides   = [4, 1]
>

func.func @invalid_shared_memory_conversion_attr(%vector: vector<16x16xf16>) -> vector<16x16xf16> {
  // expected-error @+1 {{shared_memory_conversion attribute must implement IREE::GPU::PromotionAttr}}
  %out = iree_vector_ext.to_layout %vector to layout(#layout) {shared_memory_conversion = "invalid"} : vector<16x16xf16>
  return %out : vector<16x16xf16>
}

// -----

#gpu_target_dma = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128],
  workgroup_memory_bank_count = 32
>>
#exec_target_dma = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_dma}>
#translation_dma = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>

#consumer_layout_dma = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 64],
  element_tile = [4, 1],
  subgroup_strides = [0, 0],
  thread_strides   = [0, 1]
>

func.func @promote_global_transfer_read_with_async_dma(
    %src: memref<4x64xf16>, %other: vector<4x64xf16>)
    -> vector<4x64xf16>
    attributes {hal.executable.target = #exec_target_dma, translation_info = #translation_dma} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %read = vector.transfer_read %src[%c0, %c0], %cst {in_bounds = [true, true]}
      : memref<4x64xf16>, vector<4x64xf16>
  %mul = arith.mulf %read, %other : vector<4x64xf16>
  %out = iree_vector_ext.to_layout %mul to layout(#consumer_layout_dma)
      {shared_memory_conversion = #iree_gpu.use_global_load_dma} : vector<4x64xf16>
  return %out : vector<4x64xf16>
}

// CHECK-LABEL: func.func @promote_global_transfer_read_with_async_dma
// CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]: memref<4x64xf16>
// CHECK: gpu.barrier
// CHECK: %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<4x64xf16, #gpu.address_space<workgroup>>
// CHECK: %[[DMA:.+]] = iree_gpu.async_dma %[[SRC]]
// CHECK-SAME: to %[[ALLOC]]
// CHECK-SAME: vector<4x64xf16>
// CHECK: %[[BARRIER:.+]] = iree_gpu.value_barrier %[[DMA]]
// CHECK: %[[READ:.+]] = vector.transfer_read %[[BARRIER]]
// CHECK: %[[MUL:.+]] = arith.mulf %[[READ]]
// CHECK: %[[OUT:.+]] = iree_vector_ext.to_layout %[[MUL]]
// CHECK-NOT: shared_memory_conversion
// CHECK: return %[[OUT]]

// -----

#gpu_target_dma_fallback = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128],
  workgroup_memory_bank_count = 32
>>
#exec_target_dma_fallback = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_dma_fallback}>
#translation_dma_fallback = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [256, 1, 1] subgroup_size = 64>

#consumer_layout_dma_fallback = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1],
  batch_tile = [1, 1, 1],
  outer_tile = [1, 1, 1],
  thread_tile = [1, 4, 16],
  element_tile = [1, 4, 1],
  subgroup_strides = [0, 0, 0],
  thread_strides   = [0, 16, 1]
>

func.func @promote_global_transfer_read_with_async_dma_fallback(
    %src: memref<1x16x16xf16>) -> vector<1x16x16xf16>
    attributes {hal.executable.target = #exec_target_dma_fallback, translation_info = #translation_dma_fallback} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %read = vector.transfer_read %src[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]}
      : memref<1x16x16xf16>, vector<1x16x16xf16>
  %out = iree_vector_ext.to_layout %read to layout(#consumer_layout_dma_fallback)
      {shared_memory_conversion = #iree_gpu.use_global_load_dma} : vector<1x16x16xf16>
  return %out : vector<1x16x16xf16>
}

// CHECK-LABEL: func.func @promote_global_transfer_read_with_async_dma_fallback
// CHECK: gpu.barrier
// CHECK: %[[GLOBAL:.+]] = vector.transfer_read
// CHECK-NOT: iree_gpu.async_dma
// CHECK: %[[READ_LAYOUT_VALUE:.+]] = iree_vector_ext.to_layout %[[GLOBAL]]
// CHECK: %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>}
// CHECK: %[[WRITE:.+]] = vector.transfer_write %[[READ_LAYOUT_VALUE]], %[[ALLOC]]
// CHECK: %[[BARRIER:.+]] = iree_gpu.value_barrier %[[WRITE]]
// CHECK: %[[LDS_READ:.+]] = vector.transfer_read %[[BARRIER]]
// CHECK: %[[OUT:.+]] = iree_vector_ext.to_layout %[[LDS_READ]]
// CHECK: return %[[OUT]]

// -----

#gpu_target_dma_swizzle = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  max_load_instruction_bits = 128, subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  dma_sizes = [32, 128],
  workgroup_memory_bank_count = 32
>>
#exec_target_dma_swizzle = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #gpu_target_dma_swizzle}>
#translation_dma_swizzle = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>

#consumer_layout_dma_swizzle = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [1, 8],
  subgroup_strides = [0, 0],
  thread_strides   = [16, 1]
>

func.func @promote_global_transfer_read_with_swizzled_async_dma(
    %src: memref<4x128xf16>) -> vector<4x128xf16>
    attributes {hal.executable.target = #exec_target_dma_swizzle, translation_info = #translation_dma_swizzle} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %read = vector.transfer_read %src[%c0, %c0], %cst {in_bounds = [true, true]}
      : memref<4x128xf16>, vector<4x128xf16>
  %out = iree_vector_ext.to_layout %read to layout(#consumer_layout_dma_swizzle)
      {shared_memory_conversion = #iree_gpu.use_global_load_dma} : vector<4x128xf16>
  return %out : vector<4x128xf16>
}

// CHECK-LABEL: func.func @promote_global_transfer_read_with_swizzled_async_dma
// CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]: memref<4x128xf16>
// CHECK: gpu.barrier
// CHECK: %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<4x128xf16, #gpu.address_space<workgroup>>
// CHECK: %[[HINT:.+]] = iree_codegen.swizzle_hint %[[ALLOC]][#iree_codegen.xor_shuffle<64, 8>]
// CHECK: %[[DMA:.+]] = iree_gpu.async_dma %[[SRC]]
// CHECK-SAME: to %[[HINT]]
// CHECK: %[[BARRIER:.+]] = iree_gpu.value_barrier %[[DMA]]
// CHECK: %[[READ:.+]] = vector.transfer_read %[[BARRIER]]
// CHECK: %[[OUT:.+]] = iree_vector_ext.to_layout %[[READ]]
// CHECK: return %[[OUT]]
