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
