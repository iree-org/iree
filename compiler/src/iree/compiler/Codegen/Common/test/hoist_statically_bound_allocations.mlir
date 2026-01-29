// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-hoist-statically-bound-allocations{vscale-min=1 vscale-max=16}))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-hoist-statically-bound-allocations{vscale-max=0}))" %s | FileCheck %s --check-prefix=CHECK-UNBOUNDED-VSCALE

// Note: Scalable allocations are not hoisted if vscale is unbounded.

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

func.func @non_entry_bb_scalable_allocas() attributes { vscale_range = #llvm.vscale_range<minRange = 1, maxRange = 16>   } {
  cf.br ^bb1
 ^bb1() :
  %vscale = vector.vscale
  %c4 = arith.constant 4 : index
  %c4_vscale = arith.muli %vscale, %c4 : index
  %0 = memref.alloca(%c4_vscale) : memref<?xi32>
  return
}
// CHECK: #[[$SCALABLE_BOUND_MAP_0:.*]] = affine_map<()[s0] -> (s0 * 4)>
// CHECK-LABEL: func @non_entry_bb_scalable_allocas()
//  CHECK-NEXT:   %[[VSCALE:.+]] = vector.vscale
//  CHECK-NEXT:   %[[C4_VSCALE:.+]] = affine.apply #[[$SCALABLE_BOUND_MAP_0]]()[%[[VSCALE]]]
//  CHECK-NEXT:   %[[ALLOCA:.+]] = memref.alloca(%[[C4_VSCALE]]) : memref<?xi32>
//  CHECK-NEXT:   cf.br ^bb1
//  CHECK-NEXT:   ^bb1:

// CHECK-UNBOUNDED-VSCALE-LABEL: func @non_entry_bb_scalable_allocas()
// CHECK-UNBOUNDED-VSCALE: ^bb1:
// CHECK-UNBOUNDED-VSCALE:   memref.alloca

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

#map = affine_map<(d0)[s0] -> (d0, s0 * 32)>
func.func @nested_op_scalable_alloca_linalg_use(%arg0 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : i32
  %vscale = vector.vscale
  scf.for %iv = %c0 to %arg0 step %c1 {
    %0 = affine.min #map(%iv)[%vscale]
    %1 = memref.alloca(%0) : memref<?xi32>
    linalg.fill ins(%c42 : i32) outs(%1 : memref<?xi32>)
    scf.yield
  }
  return
}
// CHECK: #[[$SCALABLE_BOUND_MAP_1:.*]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL: func @nested_op_scalable_alloca_linalg_use(
//  CHECK-NEXT:   %[[VSCALE:.+]] = vector.vscale
//  CHECK-NEXT:   %[[C32_VSCALE:.+]] = affine.apply #[[$SCALABLE_BOUND_MAP_1]]()[%[[VSCALE]]]
//  CHECK-NEXT:   %[[ALLOCA:.+]] = memref.alloca(%[[C32_VSCALE]]) : memref<?xi32>
//       CHECK:   scf.for
//       CHECK:     %[[SIZE:.+]] = affine.min
//       CHECK:     %[[SUBVIEW:.+]] = memref.subview %[[ALLOCA]][0] [%[[SIZE]]] [1]
//       CHECK:     %[[CAST:.+]] = memref.cast %[[SUBVIEW]]
//       CHECK:     linalg.fill
//  CHECK-SAME:         outs(%[[CAST]] :

// CHECK-UNBOUNDED-VSCALE-LABEL: func @nested_op_scalable_alloca_linalg_use(
//       CHECK-UNBOUNDED-VSCALE: scf.for
//       CHECK-UNBOUNDED-VSCALE:   memref.alloca

// -----

#map = affine_map<(d0) -> (d0, 16)>
func.func @nested_op_alloca_subview_use(%arg0 : index, %o0 : index, %o1 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %arg0 step %c1 {
    %0 = affine.min #map(%iv)
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

// -----

#map = affine_map<(d0)[s0] -> (d0, s0 * 32)>
func.func @nested_op_scalable_alloc_linalg_use(%arg0 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : i32
  %vscale = vector.vscale
  scf.for %iv = %c0 to %arg0 step %c1 {
    %0 = affine.min #map(%iv)[%vscale]
    %1 = memref.alloc(%0) : memref<?xi32>
    linalg.fill ins(%c42 : i32) outs(%1 : memref<?xi32>)
    memref.dealloc %1 : memref<?xi32>
    scf.yield
  }
  return
}
// CHECK: #[[$SCALABLE_BOUND_MAP_2:.*]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-LABEL: func @nested_op_scalable_alloc_linalg_use(
//  CHECK-NEXT:   %[[VSCALE:.+]] = vector.vscale
//  CHECK-NEXT:   %[[C32_VSCALE:.+]] = affine.apply #[[$SCALABLE_BOUND_MAP_2]]()[%[[VSCALE]]]
//  CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc(%[[C32_VSCALE]]) : memref<?xi32>
//       CHECK:   scf.for
//  CHECK-SAME:           {
//       CHECK:     %[[SIZE:.+]] = affine.min
//       CHECK:     %[[SUBVIEW:.+]] = memref.subview %[[ALLOC]][0] [%[[SIZE]]] [1]
//       CHECK:     %[[CAST:.+]] = memref.cast %[[SUBVIEW]]
//       CHECK:     linalg.fill
//  CHECK-SAME:         outs(%[[CAST]] :
//       CHECK:   }
//       CHECK:   memref.dealloc %[[ALLOC:.+]] : memref<?xi32>

// CHECK-UNBOUNDED-VSCALE-LABEL: func @nested_op_scalable_alloc_linalg_use(
//       CHECK-UNBOUNDED-VSCALE: scf.for
//       CHECK-UNBOUNDED-VSCALE:   memref.alloc

// -----

// The yield is the iter_arg itself — dimension trivially preserved. The
// alloca's size comes from memref.dim of the iter_arg, and
// computeAllocationBound traces through the loop to the init value.
func.func @hoist_alloca_yield_iter_arg(%arg0 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %static = memref.alloca() : memref<1x4xf32>
  %sv = memref.subview %static[0, 0][1, %arg0][1, 1]
      : memref<1x4xf32> to memref<1x?xf32, strided<[4, 1]>>
  %init = memref.cast %sv
      : memref<1x?xf32, strided<[4, 1]>> to memref<1x?xf32, strided<[?, ?], offset: ?>>
  %result = scf.for %i = %c0 to %arg0 step %c1
      iter_args(%iter = %init) -> (memref<1x?xf32, strided<[?, ?], offset: ?>>) {
    %dim = memref.dim %iter, %c1 : memref<1x?xf32, strided<[?, ?], offset: ?>>
    %alloca = memref.alloca(%dim) : memref<1x?xf32>
    linalg.fill ins(%cst : f32) outs(%alloca : memref<1x?xf32>)
    scf.yield %iter : memref<1x?xf32, strided<[?, ?], offset: ?>>
  }
  return
}
// CHECK-LABEL: func @hoist_alloca_yield_iter_arg(
//       CHECK:   %[[HOISTED:.+]] = memref.alloca() : memref<1x4xf32>
//       CHECK:   scf.for
//   CHECK-NOT:     memref.alloca(
//       CHECK:     %[[DIM:.+]] = memref.dim
//       CHECK:     %[[SV:.+]] = memref.subview %[[HOISTED]][0, 0] [1, %[[DIM]]] [1, 1]
//       CHECK:     linalg.fill

// -----

// The yield traces through cast and subview to an alloca whose dynamic size at
// dimIndex is memref.dim of the iter_arg (self-referential). This exercises the
// cast/subview walk in the function.
func.func @hoist_alloca_yield_self_ref_subview(%arg0 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %static = memref.alloca() : memref<1x4xf32>
  %sv = memref.subview %static[0, 0][1, %arg0][1, 1]
      : memref<1x4xf32> to memref<1x?xf32, strided<[4, 1]>>
  %init = memref.cast %sv
      : memref<1x?xf32, strided<[4, 1]>> to memref<1x?xf32, strided<[?, ?], offset: ?>>
  %result = scf.for %i = %c0 to %arg0 step %c1
      iter_args(%iter = %init) -> (memref<1x?xf32, strided<[?, ?], offset: ?>>) {
    %dim = memref.dim %iter, %c1 : memref<1x?xf32, strided<[?, ?], offset: ?>>
    %alloca = memref.alloca(%dim) : memref<1x?xf32>
    linalg.fill ins(%cst : f32) outs(%alloca : memref<1x?xf32>)
    %val = memref.load %alloca[%c0, %c0] : memref<1x?xf32>
    // Yield traces: cast → subview → alloca (exercises the walk loop).
    %sub = memref.subview %alloca[0, 0][1, %dim][1, 1]
        : memref<1x?xf32> to memref<1x?xf32, strided<[?, 1]>>
    %cast = memref.cast %sub
        : memref<1x?xf32, strided<[?, 1]>> to memref<1x?xf32, strided<[?, ?], offset: ?>>
    scf.yield %cast : memref<1x?xf32, strided<[?, ?], offset: ?>>
  }
  return
}
// CHECK-LABEL: func @hoist_alloca_yield_self_ref_subview(
//       CHECK:   %[[HOISTED:.+]] = memref.alloca() : memref<1x4xf32>
//       CHECK:   scf.for
//   CHECK-NOT:     memref.alloca(
//       CHECK:     %[[DIM:.+]] = memref.dim
//       CHECK:     %[[SV:.+]] = memref.subview %[[HOISTED]][0, 0] [1, %[[DIM]]] [1, 1]
//       CHECK:     linalg.fill
//       CHECK:     memref.load

// -----

// The yield is an inner scf.for result. The inner loop preserves the dimension
// via the case that yield is iter_arg, and the recursive check verifies the
// inner loop.
func.func @hoist_alloca_yield_nested_loop(%arg0 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %static = memref.alloca() : memref<1x4xf32>
  %sv = memref.subview %static[0, 0][1, %arg0][1, 1]
      : memref<1x4xf32> to memref<1x?xf32, strided<[4, 1]>>
  %init = memref.cast %sv
      : memref<1x?xf32, strided<[4, 1]>> to memref<1x?xf32, strided<[?, ?], offset: ?>>
  %result = scf.for %i = %c0 to %arg0 step %c1
      iter_args(%outer_iter = %init) -> (memref<1x?xf32, strided<[?, ?], offset: ?>>) {
    %inner = scf.for %j = %c0 to %arg0 step %c1
        iter_args(%inner_iter = %outer_iter) -> (memref<1x?xf32, strided<[?, ?], offset: ?>>) {
      %dim = memref.dim %inner_iter, %c1 : memref<1x?xf32, strided<[?, ?], offset: ?>>
      %alloca = memref.alloca(%dim) : memref<1x?xf32>
      linalg.fill ins(%cst : f32) outs(%alloca : memref<1x?xf32>)
      scf.yield %inner_iter : memref<1x?xf32, strided<[?, ?], offset: ?>>
    }
    scf.yield %inner : memref<1x?xf32, strided<[?, ?], offset: ?>>
  }
  return
}
// CHECK-LABEL: func @hoist_alloca_yield_nested_loop(
//       CHECK:   %[[HOISTED:.+]] = memref.alloca() : memref<1x4xf32>
//       CHECK:   scf.for
//       CHECK:     scf.for
//   CHECK-NOT:       memref.alloca(
//       CHECK:       %[[DIM:.+]] = memref.dim
//       CHECK:       %[[SV:.+]] = memref.subview %[[HOISTED]][0, 0] [1, %[[DIM]]] [1, 1]
//       CHECK:       linalg.fill

// -----

// Negative test: the yield uses an alloca sized by a different value (%arg1)
// rather than the iter_arg's dimension, so the dimension is not preserved
// across iterations. The alloca should NOT be hoisted.
func.func @no_hoist_alloca_yield_dim_not_preserved(%arg0 : index, %arg1 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %static = memref.alloca() : memref<1x4xf32>
  %sv = memref.subview %static[0, 0][1, %arg0][1, 1]
      : memref<1x4xf32> to memref<1x?xf32, strided<[4, 1]>>
  %init = memref.cast %sv
      : memref<1x?xf32, strided<[4, 1]>> to memref<1x?xf32, strided<[?, ?], offset: ?>>
  %result = scf.for %iv = %c0 to %arg0 step %c1
      iter_args(%iter = %init) -> (memref<1x?xf32, strided<[?, ?], offset: ?>>) {
    %dim = memref.dim %iter, %c1 : memref<1x?xf32, strided<[?, ?], offset: ?>>
    %inner = memref.alloca(%dim) : memref<1x?xf32>
    linalg.fill ins(%cst : f32) outs(%inner : memref<1x?xf32>)
    // Yield an alloca with a different size — dimension not preserved.
    %other = memref.alloca(%arg1) : memref<1x?xf32>
    %cast = memref.cast %other
        : memref<1x?xf32> to memref<1x?xf32, strided<[?, ?], offset: ?>>
    scf.yield %cast : memref<1x?xf32, strided<[?, ?], offset: ?>>
  }
  return
}
// CHECK-LABEL: func @no_hoist_alloca_yield_dim_not_preserved(
//       CHECK:   scf.for
//       CHECK:     %[[DIM:.+]] = memref.dim
//       CHECK:     memref.alloca(%[[DIM]]) : memref<1x?xf32>
