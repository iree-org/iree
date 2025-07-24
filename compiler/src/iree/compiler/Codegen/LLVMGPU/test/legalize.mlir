// RUN: iree-opt --iree-test-llvmgpu-legalize-ops --split-input-file %s | FileCheck %s

// CHECK: memref.global "private" @__shared_memory__ : memref<16x16xf32, #gpu.address_space<workgroup>>
// CHECK: func.func @allocation
// CHECK:   %[[A:.*]] = memref.get_global @__shared_memory__ : memref<16x16xf32, #gpu.address_space<workgroup>>
// CHECK:   memref.store %{{.*}}, %[[A]][%{{.*}}, %{{.*}}] : memref<16x16xf32, #gpu.address_space<workgroup>>
func.func @allocation(%arg0: f32) {
  %0 = memref.alloc() : memref<16x16xf32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  memref.store %arg0, %0[%c0, %c0] : memref<16x16xf32, #gpu.address_space<workgroup>>
  return
}
