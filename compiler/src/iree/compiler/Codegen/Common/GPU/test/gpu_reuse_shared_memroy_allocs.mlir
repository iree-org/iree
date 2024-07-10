// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-reuse-shared-memory-allocs),canonicalize,cse)" %s | FileCheck %s


func.func @shared_memory_disjoint() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %a = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %b = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
  %c = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
  scf.for %arg0 = %c0 to %c128 step %c1 {
    memref.store %cst_f32, %a[%arg0] : memref<128xf32, #gpu.address_space<workgroup>>
    memref.store %cst_f32, %b[%arg0] : memref<256xf32, #gpu.address_space<workgroup>>

  }
  memref.store %cst_f32, %c[%c0] : memref<32xf32, #gpu.address_space<workgroup>>
  return
}
// CHECK-LABEL: func.func @shared_memory_disjoint
//   CHECK-DAG:   %[[C1024:.+]] = arith.constant 1024 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<1536xi8, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_A:.+]] = memref.view %[[ALLOC]][%[[C1024]]][] : memref<1536xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_B:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1536xi8, #gpu.address_space<workgroup>> to memref<256xf32, #gpu.address_space<workgroup>>
//       CHECK:   scf.for
//       CHECK:     memref.store {{.*}} %[[VIEW_A]]{{.*}} : memref<128xf32, #gpu.address_space<workgroup>>
//       CHECK:     memref.store {{.*}} %[[VIEW_B]]{{.*}} : memref<256xf32, #gpu.address_space<workgroup>>
//       CHECK:   }
//       CHECK:   gpu.barrier
//       CHECK:   %[[VIEW_C:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1536xi8, #gpu.address_space<workgroup>> to memref<32xf32, #gpu.address_space<workgroup>>
//       CHECK:   memref.store {{.*}} %[[VIEW_C]]{{.*}} : memref<32xf32, #gpu.address_space<workgroup>>
//       CHECK:   return

// -----

func.func @shared_memory_disjoint_subview() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %a = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %b = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
  %c = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
  %a_subview = memref.subview %a[0] [64] [1] : memref<128xf32, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  %b_subview = memref.subview %b[0] [64] [1] : memref<256xf32, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  %c_subview = memref.subview %c[0] [16] [1] : memref<32xf32, #gpu.address_space<workgroup>> to memref<16xf32, #gpu.address_space<workgroup>>
  scf.for %arg0 = %c0 to %c64 step %c1 {
    memref.store %cst_f32, %a_subview[%arg0] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %cst_f32, %b_subview[%arg0] : memref<64xf32, #gpu.address_space<workgroup>>
  }
  memref.store %cst_f32, %c_subview[%c0] : memref<16xf32, #gpu.address_space<workgroup>>
  return
}
// CHECK-LABEL: func.func @shared_memory_disjoint_subview
//   CHECK-DAG:   %[[C1024:.+]] = arith.constant 1024 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<1536xi8, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_A:.+]] = memref.view %[[ALLOC]][%[[C1024]]][] : memref<1536xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[SUBVIEW_A:.+]] = memref.subview %[[VIEW_A]][0] [64] [1] : memref<128xf32, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_B:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1536xi8, #gpu.address_space<workgroup>> to memref<256xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[SUBVIEW_B:.+]] = memref.subview %[[VIEW_B]][0] [64] [1] : memref<256xf32, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
//       CHECK:   scf.for
//       CHECK:     memref.store {{.*}} %[[SUBVIEW_A]]{{.*}} : memref<64xf32, #gpu.address_space<workgroup>>
//       CHECK:     memref.store {{.*}} %[[SUBVIEW_B]]{{.*}} : memref<64xf32, #gpu.address_space<workgroup>>
//       CHECK:   }
//       CHECK:   gpu.barrier
//       CHECK:   %[[VIEW_C:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1536xi8, #gpu.address_space<workgroup>> to memref<32xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[SUBVIEW_C:.+]] = memref.subview %[[VIEW_C]][0] [16] [1] : memref<32xf32, #gpu.address_space<workgroup>> to memref<16xf32, #gpu.address_space<workgroup>>
//       CHECK:   memref.store {{.*}} %[[SUBVIEW_C]]{{.*}} : memref<16xf32, #gpu.address_space<workgroup>>
//       CHECK:   return

// -----

func.func @shared_memory_joint_subview() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %cst_i8 = arith.constant 0 : i8
  %a = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %b = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
  %c = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
  %a_subview = memref.subview %a[0] [64] [1] : memref<128xf32, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  %b_subview_0 = memref.subview %b[0] [64] [1] : memref<256xf32, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  %b_subview_1 = memref.subview %b_subview_0[0] [32] [1] : memref<64xf32, #gpu.address_space<workgroup>> to memref<32xf32, #gpu.address_space<workgroup>>
  %c_subview = memref.subview %c[0] [16] [1] : memref<32xf32, #gpu.address_space<workgroup>> to memref<16xf32, #gpu.address_space<workgroup>>
  scf.for %arg0 = %c0 to %c64 step %c1 {
    memref.store %cst_f32, %a_subview[%arg0] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %cst_f32, %b_subview_0[%arg0] : memref<64xf32, #gpu.address_space<workgroup>>
  }
  memref.store %cst_f32, %c_subview[%c0] : memref<16xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %b_subview_1[%c0] : memref<32xf32, #gpu.address_space<workgroup>>
  return
}
// CHECK-LABEL: func.func @shared_memory_joint_subview
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[ALLOC_A:.+]] = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[SUBVIEW_A:.+]] = memref.subview %[[ALLOC_A]][0] [64] [1] : memref<128xf32, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[ALLOC_B:.+]] = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[SUBVIEW_B:.+]] = memref.subview %[[ALLOC_B]][0] [64] [1] : memref<256xf32, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
//       CHECK:   scf.for
//       CHECK:     memref.store {{.*}} %[[SUBVIEW_A]]{{.*}} : memref<64xf32, #gpu.address_space<workgroup>>
//       CHECK:     memref.store {{.*}} %[[SUBVIEW_B]]{{.*}} : memref<64xf32, #gpu.address_space<workgroup>>
//       CHECK:   }
//       CHECK:   %[[ALLOC_C:.+]] = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[SUBVIEW_C:.+]] = memref.subview %[[ALLOC_C]][0] [16] [1] : memref<32xf32, #gpu.address_space<workgroup>> to memref<16xf32, #gpu.address_space<workgroup>>
//       CHECK:   memref.store {{.*}} %[[SUBVIEW_C]]{{.*}} : memref<16xf32, #gpu.address_space<workgroup>>
//       CHECK:   return
