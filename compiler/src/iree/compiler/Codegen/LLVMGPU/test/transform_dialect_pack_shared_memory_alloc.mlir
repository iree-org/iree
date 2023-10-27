// RUN: iree-opt %s --iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

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

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.iree.pack_shared_memory_alloc %0 : (!transform.any_op) -> ()
  transform.iree.apply_cse %0 : !transform.any_op
    transform.yield
  } // @__transform_main
} // module
