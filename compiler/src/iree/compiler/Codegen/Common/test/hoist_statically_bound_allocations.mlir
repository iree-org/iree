// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-hoist-statically-bound-allocations))" %s | FileCheck %s

func.func @non_entry_bb_allocas() {
  cf.br ^bb1
 ^bb1() :
  %0 = memref.alloca() : memref<16xi32>
  return
}
// CHECK-LABEL: func @non_entry_bb_allocas()
//  CHECK-NEXT:   %[[ALLOCA:.+]] = memref.alloca() : memref<16xi32>
//  CHECK-NEXT:   cf.br ^bb1
//  CHECK-NEXT:   ^bb1:
//  CHECK-NEXT:   return

// -----

#map = affine_map<(d0) -> (d0, 16)>
func.func @nested_op_alloca_store_use(%arg0 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : i32
  scf.for %iv = %c0 to %arg0 step %c1 {
    %0 = affine.min #map(%iv)
    %1 = memref.alloca(%0) : memref<?xi32>
    memref.store %c42, %1[%iv] : memref<?xi32>
    scf.yield
  }
  return
}
// CHECK-LABEL: func @nested_op_alloca_store_use(
//  CHECK-NEXT:   %[[ALLOCA:.+]] = memref.alloca() : memref<16xi32>
//       CHECK:   scf.for
//       CHECK:     %[[SIZE:.+]] = affine.min
//       CHECK:     %[[SUBVIEW:.+]] = memref.subview %[[ALLOCA]][0] [%[[SIZE]]] [1]
//       CHECK:     %[[CAST:.+]] = memref.cast %[[SUBVIEW]]
//       CHECK:     memref.store %{{.+}}, %[[CAST]]

// -----

#map = affine_map<(d0) -> (d0, 16)>
func.func @nested_op_alloca_linalg_use(%arg0 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : i32
  scf.for %iv = %c0 to %arg0 step %c1 {
    %0 = affine.min #map(%iv)
    %1 = memref.alloca(%0) : memref<?xi32>
    linalg.fill ins(%c42 : i32) outs(%1 : memref<?xi32>)
    scf.yield
  }
  return
}
// CHECK-LABEL: func @nested_op_alloca_linalg_use(
//  CHECK-NEXT:   %[[ALLOCA:.+]] = memref.alloca() : memref<16xi32>
//       CHECK:   scf.for
//       CHECK:     %[[SIZE:.+]] = affine.min
//       CHECK:     %[[SUBVIEW:.+]] = memref.subview %[[ALLOCA]][0] [%[[SIZE]]] [1]
//       CHECK:     %[[CAST:.+]] = memref.cast %[[SUBVIEW]]
//       CHECK:     linalg.fill
//  CHECK-SAME:         outs(%[[CAST]] :

// -----

#map = affine_map<(d0) -> (d0, 16)>
func.func @nested_op_alloca_subview_use(%arg0 : index, %o0 : index, %o1 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : i32
  scf.for %iv = %c0 to %arg0 step %c1 {
    %0 = affine.min #map(%iv)
    // expected-error @+1 {{all stack allocations need to be hoisted to the entry block of the function}}
    %1 = memref.alloca(%0, %0) : memref<?x?xi32>
    %2 = memref.subview %1[%o0, %o1][%c1, %0][1, 1] : memref<?x?xi32> to memref<?x?xi32, strided<[?, 1], offset: ?>>
    scf.yield
  }
  return
}
// CHECK-LABEL: func @nested_op_alloca_subview_use(
//  CHECK-NEXT:   %[[ALLOCA:.+]] = memref.alloca() : memref<16x16xi32>
//       CHECK:   scf.for
//       CHECK:     %[[SIZE:.+]] = affine.min
//       CHECK:     %[[SUBVIEW1:.+]] = memref.subview %[[ALLOCA]][0, 0] [%[[SIZE]], %[[SIZE]]] [1, 1]
//       CHECK:     %[[CAST:.+]] = memref.cast %[[SUBVIEW1]]
//       CHECK:     memref.subview %[[CAST]]

// -----

func.func @non_entry_bb_allocas_func_return_static() -> memref<16xi32> {
  cf.br ^bb1
 ^bb1() :
  %0 = memref.alloca() : memref<16xi32>
  return %0 : memref<16xi32>
}
// CHECK-LABEL: func @non_entry_bb_allocas_func_return_static()
//  CHECK-NEXT:   %[[ALLOCA:.+]] = memref.alloca() : memref<16xi32>
//  CHECK-NEXT:   cf.br ^bb1
//  CHECK-NEXT:   ^bb1:
//  CHECK-NEXT:   return %[[ALLOCA]]

// -----

func.func @non_entry_bb_allocas_func_return_dynamic() -> memref<?xi32> {
  %c16 = arith.constant 16 : index
  cf.br ^bb1
 ^bb1() :
  %0 = memref.alloca(%c16) : memref<?xi32>
  return %0 : memref<?xi32>
}
// CHECK-LABEL: func @non_entry_bb_allocas_func_return_dynamic()
//       CHECK:   cf.br ^bb1
//  CHECK-NEXT:   ^bb1:
//  CHECK-NEXT:   %[[ALLOCA:.+]] = memref.alloca
//  CHECK-NEXT:   return %[[ALLOCA]]

// -----

#map = affine_map<(d0) -> (d0, 16)>
func.func @nested_op_alloca_yields(%arg0 : index, %arg1 : memref<?xi32>) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : i32
  %3 = scf.for %iv = %c0 to %arg0 step %c1 iter_args(%arg2 = %arg1) -> memref<?xi32> {
    %0 = affine.min #map(%iv)
    %1 = memref.alloca(%0) : memref<?xi32>
    memref.store %c42, %1[%iv] : memref<?xi32>
    scf.yield %1 : memref<?xi32>
  }
  return %3 : memref<?xi32>
}
// CHECK-LABEL: func @nested_op_alloca_yields(
//       CHECK:   scf.for
//       CHECK:     %[[ALLOCA:.+]] = memref.alloca
//       CHECK:     scf.yield %[[ALLOCA]]

// -----

#map = affine_map<(d0) -> (d0, 16)>
func.func @nested_op_alloc_linalg_use(%arg0 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : i32
  scf.for %iv = %c0 to %arg0 step %c1 {
    %0 = affine.min #map(%iv)
    %1 = memref.alloc(%0) : memref<?xi32>
    linalg.fill ins(%c42 : i32) outs(%1 : memref<?xi32>)
    memref.dealloc %1 : memref<?xi32>
    scf.yield
  }
  return
}
// CHECK-LABEL: func @nested_op_alloc_linalg_use(
//  CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc() : memref<16xi32>
//       CHECK:   scf.for
//       CHECK:     %[[SIZE:.+]] = affine.min
//       CHECK:     %[[SUBVIEW:.+]] = memref.subview %[[ALLOC]][0] [%[[SIZE]]] [1]
//       CHECK:     %[[CAST:.+]] = memref.cast %[[SUBVIEW]]
//       CHECK:     linalg.fill
//  CHECK-SAME:         outs(%[[CAST]] :
//       CHECK:   memref.dealloc %[[ALLOC:.+]] : memref<16xi32>
