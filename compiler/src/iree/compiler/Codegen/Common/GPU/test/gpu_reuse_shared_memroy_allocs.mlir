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
//   CHECK-DAG:   %[[C512:.+]] = arith.constant 512 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<1536xi8, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_A:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1536xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_B:.+]] = memref.view %[[ALLOC]][%[[C512]]][] : memref<1536xi8, #gpu.address_space<workgroup>> to memref<256xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_C:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1536xi8, #gpu.address_space<workgroup>> to memref<32xf32, #gpu.address_space<workgroup>>
//       CHECK:   scf.for
//       CHECK:     memref.store {{.*}} %[[VIEW_A]]{{.*}} : memref<128xf32, #gpu.address_space<workgroup>>
//       CHECK:     memref.store {{.*}} %[[VIEW_B]]{{.*}} : memref<256xf32, #gpu.address_space<workgroup>>
//       CHECK:   }
//       CHECK:   gpu.barrier
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
//   CHECK-DAG:   %[[C512:.+]] = arith.constant 512 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<1536xi8, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_A:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1536xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[SUBVIEW_A:.+]] = memref.subview %[[VIEW_A]][0] [64] [1] : memref<128xf32, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_B:.+]] = memref.view %[[ALLOC]][%[[C512]]][] : memref<1536xi8, #gpu.address_space<workgroup>> to memref<256xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[SUBVIEW_B:.+]] = memref.subview %[[VIEW_B]][0] [64] [1] : memref<256xf32, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_C:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1536xi8, #gpu.address_space<workgroup>> to memref<32xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[SUBVIEW_C:.+]] = memref.subview %[[VIEW_C]][0] [16] [1] : memref<32xf32, #gpu.address_space<workgroup>> to memref<16xf32, #gpu.address_space<workgroup>>
//       CHECK:   scf.for
//       CHECK:     memref.store {{.*}} %[[SUBVIEW_A]]{{.*}} : memref<64xf32, #gpu.address_space<workgroup>>
//       CHECK:     memref.store {{.*}} %[[SUBVIEW_B]]{{.*}} : memref<64xf32, #gpu.address_space<workgroup>>
//       CHECK:   }
//       CHECK:   gpu.barrier
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
//   CHECK-DAG:   %[[ALLOC_C:.+]] = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[SUBVIEW_C:.+]] = memref.subview %[[ALLOC_C]][0] [16] [1] : memref<32xf32, #gpu.address_space<workgroup>> to memref<16xf32, #gpu.address_space<workgroup>>
//       CHECK:   scf.for
//       CHECK:     memref.store {{.*}} %[[SUBVIEW_A]]{{.*}} : memref<64xf32, #gpu.address_space<workgroup>>
//       CHECK:     memref.store {{.*}} %[[SUBVIEW_B]]{{.*}} : memref<64xf32, #gpu.address_space<workgroup>>
//       CHECK:   }
//       CHECK:   memref.store {{.*}} %[[SUBVIEW_C]]{{.*}} : memref<16xf32, #gpu.address_space<workgroup>>
//       CHECK:   return

// -----

func.func @view_like_ops() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %a = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %b = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %c = memref.alloc() : memref<2x64xf32, #gpu.address_space<workgroup>>
  %d = memref.alloc() : memref<512xi8, #gpu.address_space<workgroup>>
  %e = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %f = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %a_subview = memref.subview %a[0] [64] [1] : memref<128xf32, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  %b_expand_shape = memref.expand_shape %b[[0, 1]] output_shape [2, 64] : memref<128xf32, #gpu.address_space<workgroup>> into memref<2x64xf32, #gpu.address_space<workgroup>>
  %c_collapse_shape = memref.collapse_shape %c[[0, 1]] : memref<2x64xf32, #gpu.address_space<workgroup>> into memref<128xf32, #gpu.address_space<workgroup>>
  %d_view = memref.view %d[%c0][] : memref<512xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
  %e_reinterpret_cast = memref.reinterpret_cast %e to offset: [0], sizes: [64], strides: [1] : memref<128xf32, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  %f_cast = memref.cast %f : memref<128xf32, #gpu.address_space<workgroup>> to memref<?xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %a_subview[%c0] : memref<64xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %b_expand_shape[%c0, %c0] : memref<2x64xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %c_collapse_shape[%c0] : memref<128xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %d_view[%c0] : memref<128xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %e_reinterpret_cast[%c0] : memref<64xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %f_cast[%c0] : memref<?xf32, #gpu.address_space<workgroup>>
  return
}
// CHECK-LABEL: func.func @view_like_ops
//       CHECK:   memref.alloc() : memref<512xi8, #gpu.address_space<workgroup>>
//   CHECK-NOT:   memref.alloc

// -----

func.func @select_alloc(%val : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %a = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %b = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %c = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
  %cmp = arith.cmpi eq, %val, %c1 : index
  %a_or_b = arith.select %cmp, %a, %b : memref<128xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %a_or_b[%c0] : memref<128xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %c[%c0] : memref<32xf32, #gpu.address_space<workgroup>>
  return
}
// CHECK-LABEL: func.func @select_alloc
//   CHECK-DAG:   %[[C512:.+]] = arith.constant 512 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[ALLOC:.+]] = memref.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_A:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_B:.+]] = memref.view %[[ALLOC]][%[[C512]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_C:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<32xf32, #gpu.address_space<workgroup>>

// -----

func.func @select_alloc_with_if(%val : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %a = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %b = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %c = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
  %cmp = arith.cmpi eq, %val, %c1 : index
  %a_or_b = scf.if %cmp -> (memref<128xf32, #gpu.address_space<workgroup>>) {
    scf.yield %a : memref<128xf32, #gpu.address_space<workgroup>>
  } else {
    scf.yield %b : memref<128xf32, #gpu.address_space<workgroup>>
  }
  memref.store %cst_f32, %a_or_b[%c0] : memref<128xf32, #gpu.address_space<workgroup>>
  memref.store %cst_f32, %c[%c0] : memref<32xf32, #gpu.address_space<workgroup>>
  return
}
// CHECK-LABEL: func.func @select_alloc_with_if
//   CHECK-DAG:   %[[C512:.+]] = arith.constant 512 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[ALLOC:.+]] = memref.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_A:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_B:.+]] = memref.view %[[ALLOC]][%[[C512]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_C:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<32xf32, #gpu.address_space<workgroup>>

// -----

func.func @alloc_with_if(%val : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %a = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %b = memref.alloc() : memref<128xf32, #gpu.address_space<workgroup>>
  %c = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
  %cmp = arith.cmpi eq, %val, %c1 : index
  scf.if %cmp {
    memref.store %cst_f32, %a[%c0] : memref<128xf32, #gpu.address_space<workgroup>>
  } else {
    memref.store %cst_f32, %b[%c0] : memref<128xf32, #gpu.address_space<workgroup>>
  }
  memref.store %cst_f32, %c[%c0] : memref<32xf32, #gpu.address_space<workgroup>>
  return
}
// CHECK-LABEL: func.func @alloc_with_if
//   CHECK-DAG:   %[[C512:.+]] = arith.constant 512 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[ALLOC:.+]] = memref.alloc() : memref<1024xi8, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_A:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_B:.+]] = memref.view %[[ALLOC]][%[[C512]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
//   CHECK-DAG:   %[[VIEW_C:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<1024xi8, #gpu.address_space<workgroup>> to memref<32xf32, #gpu.address_space<workgroup>>
