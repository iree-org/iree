// RUN: iree-opt --split-input-file -iree-transform-dialect-interpreter -transform-dialect-drop-schedule %s | FileCheck %s

#map = affine_map<(d0) -> (d0, 16)>

func.func @non_entry_bb_allocs() {
  cf.br ^bb1
 ^bb1() :
  %0 = memref.alloc() : memref<16xi32>
  memref.dealloc %0 : memref<16xi32>
  return
}
// CHECK-LABEL: func @non_entry_bb_allocs()
//  CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc() : memref<16xi32>
//  CHECK-NEXT:   memref.dealloc %[[ALLOC]] : memref<16xi32>
//  CHECK-NEXT:   cf.br ^bb1
//  CHECK-NEXT:   ^bb1:
//  CHECK-NEXT:   return

func.func @nested_op_alloc_subview_use_static(%arg0 : index, %o0 : index, %o1 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : i32
  scf.for %iv = %c0 to %arg0 step %c1 {
    %0 = affine.min #map(%iv)
    %1 = memref.alloc() : memref<16x16xi32>
    %2 = memref.subview %1[%o0, %o1][%c1, %0][1, 1] : memref<16x16xi32> to memref<?x?xi32, strided<[?, 1], offset: ?>>
    memref.dealloc %1 : memref<16x16xi32>
    scf.yield
  }
  return
}
// CHECK-LABEL: func @nested_op_alloc_subview_use_static(
//  CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc() : memref<16x16xi32>
//       CHECK:   scf.for
//       CHECK:     %[[SIZE:.+]] = affine.min
//       CHECK:     memref.subview %[[ALLOC]]
//  CHECK-NEXT:   }
//  CHECK-NEXT:   memref.dealloc %[[ALLOC]] : memref<16x16xi32>

func.func @nested_op_alloc_subview_use_dynamic(%arg0 : index, %o0 : index, %o1 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : i32
  scf.for %iv = %c0 to %arg0 step %c1 {
    %0 = affine.min #map(%iv)
    %1 = memref.alloc(%0, %0) : memref<?x?xi32>
    %2 = memref.subview %1[%o0, %o1][%c1, %0][1, 1] : memref<?x?xi32> to memref<?x?xi32, strided<[?, 1], offset: ?>>
    memref.dealloc %1 : memref<?x?xi32>
    scf.yield
  }
  return
}
// CHECK-LABEL: func @nested_op_alloc_subview_use_dynamic(
//  CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc() : memref<16x16xi32>
//       CHECK:   scf.for
//       CHECK:     %[[SIZE:.+]] = affine.min
//       CHECK:     %[[SUBVIEW1:.+]] = memref.subview %[[ALLOC]][0, 0] [%[[SIZE]], %[[SIZE]]] [1, 1]
//       CHECK:     memref.subview %[[SUBVIEW1]]
//  CHECK-NEXT:   }
//  CHECK-NEXT:   memref.dealloc %[[ALLOC]] : memref<16x16xi32>


module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.op<"func.func">
    transform.iree.hoist_static_alloc %func : (!transform.op<"func.func">) -> ()
    transform.yield 
  }
} // module
