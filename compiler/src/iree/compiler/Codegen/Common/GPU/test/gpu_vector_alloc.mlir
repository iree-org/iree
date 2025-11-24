// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-vector-alloc))" | FileCheck %s

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],

  subgroup_strides = [1, 1],
  thread_strides   = [16, 1]
>

func.func @test(%vector: vector<16x16xf16>) -> vector<16x16xf16> {
  %out = iree_vector_ext.to_layout %vector to layout(#layout) {shared_memory_conversion} : vector<16x16xf16>
  return %out : vector<16x16xf16>
}


//      CHECK: #[[$LAYOUT:.+]] = #iree_vector_ext.nested_layout<{{.*}}thread_strides = [16, 1]>
// CHECK-LABEL: func.func @test
//  CHECK-SAME:   %[[VEC:[a-zA-Z0-9]+]]: vector<16x16xf16>
//       CHECK:   gpu.barrier
//       CHECK:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16x16xf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[WRITE:.+]] = vector.transfer_write %[[VEC]], %[[ALLOC]]{{.*}} : vector<16x16xf16>, tensor<16x16xf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[BAR:.+]] = iree_gpu.value_barrier %[[WRITE]] : tensor<16x16xf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[BAR]]{{.*}} : tensor<16x16xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
//       CHECK:   %[[OUT:.+]] = iree_vector_ext.to_layout %[[READ]] to layout(#[[$LAYOUT]]) : vector<16x16xf16>
//       CHECK:   return %[[OUT]] : vector<16x16xf16>
