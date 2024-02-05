// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-llvmgpu-pack-shared-memory-alloc),cse)" %s | FileCheck %s

func.func @shared_memory_disjoint() {
  %c0 = arith.constant 0 : index
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %cst_i8 = arith.constant 0 : i8
  %0 = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %1 = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %2 = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %0[%c0] : memref<128xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %1[%c0] : memref<128xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %0[%c0] : memref<128xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %2[%c0] : memref<32xf32, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: shared_memory_disjoint
//   CHECK-NOT:   gpu.barrier
//   CHECK-DAG:   %[[PACKED:.+]] = memref.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   memref.view %[[PACKED]][%[[C0]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[C512:.+]] = arith.constant 512 : index
//       CHECK:   memref.view %[[PACKED]][%[[C512]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
//       CHECK:   nvgpu.device_async_create_group
//       CHECK:   nvgpu.device_async_wait %0 {numGroups = 0 : i32}
//       CHECK:   gpu.barrier
//       CHECK:   memref.view %[[PACKED]][%[[C0]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<32xf32, #gpu.address_space<workgroup>>
