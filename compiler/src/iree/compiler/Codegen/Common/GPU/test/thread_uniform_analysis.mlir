// RUN: iree-opt %s \
// RUN: --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-test-thread-uniform-analysis))" \
// RUN:  | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(i)[s0] -> (i + s0)>

// CHECK-LABEL: @uniform_ops
func.func @uniform_ops() {
  // CHECK-NEXT: arith.constant {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: arith.constant {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: hal.interface.workgroup.id[0] {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: hal.interface.workgroup.count[0] {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: hal.interface.workgroup.size[0] {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: arith.muli {{.*}} {thread_uniform_analysis.is_uniform = true} : index
  // CHECK-NEXT: arith.addi {{.*}} {thread_uniform_analysis.is_uniform = true} : index
  // CHECK-NEXT: hal.interface.constant.load {{.*}} {thread_uniform_analysis.is_uniform = true} : index
  // CHECK-NEXT: util.assume.int {{.*}} {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: affine.apply {{.*}} {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: scf.yield
  // CHECK-NEXT: } {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: hal.interface.binding.subspan {{.*}} {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: return
  %c1 = arith.constant 1 : index
  %c8192 = arith.constant 8192 : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_size_x = hal.interface.workgroup.size[0] : index
  %0 = arith.muli %workgroup_id_x, %workgroup_count_x : index
  %1 = arith.addi %0, %c8192 : index
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %3 = util.assume.int %2[<umin = 0, umax = 0>, <umin = 4, umax = 2048, udiv = 4>] : index
  %4 = scf.for %arg0 = %1 to %workgroup_size_x step %c1 iter_args(%arg1 = %3) -> (index) {
    %6 = affine.apply #map(%arg0)[%arg1]
    scf.yield %6 : index
  }
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%4) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>>
  return
}

// CHECK-LABEL: @non_uniform_ops
func.func @non_uniform_ops() {
  // CHECK-NEXT: arith.constant {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: arith.constant {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: gpu.thread_id  x {thread_uniform_analysis.is_uniform = false}
  // CHECK-NEXT: hal.interface.workgroup.count[0] {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: hal.interface.workgroup.size[0] {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: arith.muli {{.*}} {thread_uniform_analysis.is_uniform = false} : index
  // CHECK-NEXT: arith.addi {{.*}} {thread_uniform_analysis.is_uniform = false} : index
  // CHECK-NEXT: hal.interface.constant.load {{.*}} {thread_uniform_analysis.is_uniform = true} : index
  // CHECK-NEXT: util.assume.int {{.*}} {thread_uniform_analysis.is_uniform = true}
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: affine.apply {{.*}} {thread_uniform_analysis.is_uniform = false}
  // CHECK-NEXT: scf.yield
  // CHECK-NEXT: } {thread_uniform_analysis.is_uniform = false}
  // CHECK-NEXT: hal.interface.binding.subspan {{.*}} {thread_uniform_analysis.is_uniform = false}
  // CHECK-NEXT: return
  %c1 = arith.constant 1 : index
  %c8192 = arith.constant 8192 : index
  %thread_id_x = gpu.thread_id  x
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_size_x = hal.interface.workgroup.size[0] : index
  %0 = arith.muli %thread_id_x, %workgroup_count_x : index
  %1 = arith.addi %0, %c8192 : index
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %3 = util.assume.int %2[<umin = 0, umax = 0>, <umin = 4, umax = 2048, udiv = 4>] : index
  %4 = scf.for %arg0 = %1 to %workgroup_size_x step %c1 iter_args(%arg1 = %3) -> (index) {
    %6 = affine.apply #map(%arg0)[%arg1]
    scf.yield %6 : index
  }
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%4) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xf32>>
  return
}
