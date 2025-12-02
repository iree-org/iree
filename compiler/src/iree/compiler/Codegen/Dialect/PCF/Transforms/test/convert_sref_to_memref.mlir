// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-convert-sref-to-memref)" --split-input-file --verify-diagnostics | FileCheck %s

util.func private @convert_generic_with_init(%arg0: memref<?x?xi32, strided<[?, 1]>, 3>) {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32, strided<[?, 1]>, 3>) {
    util.optimization_barrier %ref : !pcf.sref<?x?xi32, #pcf.dummy_scope>
    pcf.return
  }
  util.optimization_barrier %0 : memref<?x?xi32, strided<[?, 1]>, 3>
  util.return
}

// CHECK-LABEL: @convert_generic_with_init
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32, strided<[?, 1]>, 3>
//       CHECK:   pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:     execute[{{.*}}] {
//  CHECK-NEXT:     util.optimization_barrier %[[ARG0]]
//  CHECK-NEXT:     pcf.return
//       CHECK:   util.optimization_barrier %[[ARG0]]

// -----

util.func private @convert_generic_with_alloc(%d0: index, %d1: index, %d2: index) {
  %0:2 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref, %ref_1)[%id: index, %count: index]
         : (!pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?xi32>{%d0}, memref<?x?xi32>{%d1, %d2}) {
    util.optimization_barrier %ref, %ref_1 : !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?x?xi32, #pcf.dummy_scope>
    pcf.return
  }
  util.optimization_barrier %0#0, %0#1 : memref<?xi32>, memref<?x?xi32>
  util.return
}

// CHECK-LABEL: @convert_generic_with_alloc
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG1:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG2:[A-Za-z0-9_]+]]: index
//  CHECK-DAG:    %[[ALLOC:.+]] = memref.alloc(%[[ARG0]]) {alignment = 16 : i64} : memref<?xi32>
//  CHECK-DAG:    %[[ALLOC1:.+]] = memref.alloc(%[[ARG1]], %[[ARG2]]) {alignment = 16 : i64} : memref<?x?xi32>
//       CHECK:   pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:     execute[{{.*}}] {
//  CHECK-NEXT:     util.optimization_barrier %[[ALLOC]], %[[ALLOC1]]
//  CHECK-NEXT:     pcf.return
//       CHECK:   util.optimization_barrier %[[ALLOC]], %[[ALLOC1]]

// -----

util.func private @inline_generic_initializer(%arg0: memref<?x?xi32, strided<[?, 1]>, 3>) {
  %0 = pcf.generic scope(#pcf.dummy_scope)
    initialize {
      %c42 = arith.constant 42 : index
      pcf.yield %c42 : index
    } -> (%i: index)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32, strided<[?, 1]>, 3>) {
    util.optimization_barrier %i, %ref : index, !pcf.sref<?x?xi32, #pcf.dummy_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @inline_generic_initializer
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32, strided<[?, 1]>, 3>
//       CHECK:   pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:     execute[{{.*}}] {
//  CHECK-NEXT:     %[[I:.+]] = arith.constant 42 : index
//  CHECK-NEXT:     util.optimization_barrier %[[I]], %[[ARG0]]
//  CHECK-NEXT:     pcf.return

// -----

util.func private @inline_generic_initializer_with_alloc() {
  pcf.generic scope(#pcf.sequential)
    initialize {
      %c42 = arith.constant 42 : index
      %alloc = pcf.alloc(%c42) : !pcf.sref<?x5xi32, #pcf.sequential>
      pcf.yield %c42, %alloc : index, !pcf.sref<?x5xi32, #pcf.sequential>
    } -> (%i: index, %aref: !pcf.sref<?x5xi32, #pcf.sequential>)
    execute[%id: index, %n: index] {
    util.optimization_barrier %i, %aref : index, !pcf.sref<?x5xi32, #pcf.sequential>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @inline_generic_initializer_with_alloc
//       CHECK:   pcf.generic scope(#pcf.sequential)
//  CHECK-NEXT:     execute[{{.*}}] {
//  CHECK-NEXT:     %[[I:.+]] = arith.constant 42 : index
//  CHECK-NEXT:     %[[ALLOC:.+]] = memref.alloc(%[[I]]) {alignment = 16 : i64} : memref<?x5xi32>
//  CHECK-NEXT:     util.optimization_barrier %[[I]], %[[ALLOC]]
//  CHECK-NEXT:     pcf.return

// -----

util.func private @convert_loop_with_init(%arg0: memref<?x?xi32, strided<[?, 1]>, 3>, %n: index) {
  %0 = pcf.loop scope(#pcf.dummy_scope) count(%n)
    execute(%ref = %arg0)[%count: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32, strided<[?, 1]>, 3>) {
    util.optimization_barrier %ref : !pcf.sref<?x?xi32, #pcf.dummy_scope>
    pcf.return
  }
  util.optimization_barrier %0 : memref<?x?xi32, strided<[?, 1]>, 3>
  util.return
}

// CHECK-LABEL: @convert_loop_with_init
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32, strided<[?, 1]>, 3>
//       CHECK:   pcf.loop scope(#pcf.dummy_scope)
//  CHECK-NEXT:     execute[{{.*}}] {
//  CHECK-NEXT:     util.optimization_barrier %[[ARG0]]
//  CHECK-NEXT:     pcf.return
//       CHECK:   util.optimization_barrier %[[ARG0]]

// -----

util.func private @convert_loop_with_alloc(%d0: index, %d1: index, %d2: index, %n: index) {
  %0:2 = pcf.loop scope(#pcf.dummy_scope) count(%n)
    execute(%ref, %ref_1)[%count: index]
         : (!pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?xi32>{%d0}, memref<?x?xi32>{%d1, %d2}) {
    util.optimization_barrier %ref, %ref_1 : !pcf.sref<?xi32, #pcf.dummy_scope>, !pcf.sref<?x?xi32, #pcf.dummy_scope>
    pcf.return
  }
  util.optimization_barrier %0#0, %0#1 : memref<?xi32>, memref<?x?xi32>
  util.return
}

// CHECK-LABEL: @convert_loop_with_alloc
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG1:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG2:[A-Za-z0-9_]+]]: index
//  CHECK-DAG:    %[[ALLOC:.+]] = memref.alloc(%[[ARG0]]) {alignment = 16 : i64} : memref<?xi32>
//  CHECK-DAG:    %[[ALLOC1:.+]] = memref.alloc(%[[ARG1]], %[[ARG2]]) {alignment = 16 : i64} : memref<?x?xi32>
//       CHECK:   pcf.loop scope(#pcf.dummy_scope)
//  CHECK-NEXT:     execute[{{.*}}] {
//  CHECK-NEXT:     util.optimization_barrier %[[ALLOC]], %[[ALLOC1]]
//  CHECK-NEXT:     pcf.return
//       CHECK:   util.optimization_barrier %[[ALLOC]], %[[ALLOC1]]

// -----

util.func private @convert_memref_write_slice(%arg0: memref<?x?xi32>) {
  pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32>) {
    %src = memref.alloc() : memref<3x4xi32>
    pcf.write_slice %src into %ref[1, 2] [3, 4] [1, 1] : memref<3x4xi32> into !pcf.sref<?x?xi32, #pcf.dummy_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @convert_memref_write_slice
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32>
//       CHECK:   pcf.generic
//  CHECK-NEXT:     execute[{{.*}}] {
//   CHECK-DAG:     %[[SRC:.+]] = memref.alloc()
//   CHECK-DAG:     %[[SV:.+]] = memref.subview %[[ARG0]][1, 2] [3, 4] [1, 1] : memref<?x?xi32> to memref<3x4xi32, strided<[?, 1], offset: ?>>
//       CHECK:     memref.copy %[[SRC]], %[[SV]]
//       CHECK:     pcf.return

// -----

util.func private @convert_vector_write_slice(%arg0: memref<?x?xi32>) {
  pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32>) {
    %src = arith.constant dense<0> : vector<3x4xi32>
    pcf.write_slice %src into %ref[1, 2] [3, 3] [1, 1] : vector<3x4xi32> into !pcf.sref<?x?xi32, #pcf.dummy_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @convert_vector_write_slice
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32>
//       CHECK:   pcf.generic
//  CHECK-NEXT:     execute[{{.*}}] {
//   CHECK-DAG:     %[[SRC:.+]] = arith.constant dense<0> : vector<3x4xi32>
//   CHECK-DAG:     %[[SV:.+]] = memref.subview %[[ARG0]][1, 2] [3, 3] [1, 1] : memref<?x?xi32> to memref<3x3xi32, strided<[?, 1], offset: ?>>
//       CHECK:     vector.transfer_write %[[SRC]], %[[SV]][%c0, %c0] {in_bounds = [true, false]}
//       CHECK:     pcf.return

// -----

util.func private @convert_tensor_write_slice(%arg0: memref<?x?xi32>) {
  pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32>) {
    %src = arith.constant dense<0> : tensor<3x4xi32>
    pcf.write_slice %src into %ref[1, 2] [3, 4] [1, 1] : tensor<3x4xi32> into !pcf.sref<?x?xi32, #pcf.dummy_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @convert_tensor_write_slice
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32>
//       CHECK:   pcf.generic
//  CHECK-NEXT:     execute[{{.*}}] {
//   CHECK-DAG:     %[[SRC:.+]] = arith.constant dense<0> : tensor<3x4xi32>
//   CHECK-DAG:     %[[SV:.+]] = memref.subview %[[ARG0]][1, 2] [3, 4] [1, 1] : memref<?x?xi32> to memref<3x4xi32, strided<[?, 1], offset: ?>>
//       CHECK:     iree_codegen.store_to_buffer %[[SRC]], %[[SV]]
//       CHECK:     pcf.return

// -----

util.func private @convert_for_loop_swap(
    %arg0: memref<?x?xi32, strided<[?, 1]>, 3>,
    %arg1: memref<?x?xi32, strided<[?, 1]>, 3>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0:2 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref0 = %arg0, %ref1 = %arg1)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>, !pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32, strided<[?, 1]>, 3>, memref<?x?xi32, strided<[?, 1]>, 3>) {
    scf.for %i = %c0 to %c10 step %c1 iter_args(%iter0 = %ref0, %iter1 = %ref1)
        -> (!pcf.sref<?x?xi32, #pcf.dummy_scope>, !pcf.sref<?x?xi32, #pcf.dummy_scope>) {
      %b:2 = util.optimization_barrier %iter1, %iter0 : !pcf.sref<?x?xi32, #pcf.dummy_scope>, !pcf.sref<?x?xi32, #pcf.dummy_scope>
      scf.yield %b#0, %b#1 : !pcf.sref<?x?xi32, #pcf.dummy_scope>, !pcf.sref<?x?xi32, #pcf.dummy_scope>
    }
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @convert_for_loop_swap
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32, strided<[?, 1]>, 3>
//  CHECK-SAME:     %[[ARG1:[A-Za-z0-9_]+]]: memref<?x?xi32, strided<[?, 1]>, 3>
//       CHECK:   pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:     execute[{{.*}}] {
//  CHECK-NEXT:     scf.for {{.*}} iter_args(%[[ITER0:.+]] = %[[ARG0]], %[[ITER1:.+]] = %[[ARG1]])
//  CHECK-NEXT:       %[[B:.+]]:2 = util.optimization_barrier %[[ITER1]], %[[ITER0]]
//  CHECK-NEXT:       scf.yield %[[B]]#0, %[[B]]#1
//       CHECK:     pcf.return

// -----

util.func private @convert_for_loop_swap_conflict(
    %arg0: memref<?x?xi32, strided<[3, 1]>, 3>,
    %arg1: memref<?x?xi32, strided<[4, 1]>, 3>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0:2 = pcf.generic scope(#pcf.dummy_scope)
    execute(%ref0 = %arg0, %ref1 = %arg1)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>, !pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32, strided<[3, 1]>, 3>, memref<?x?xi32, strided<[4, 1]>, 3>) {
    scf.for %i = %c0 to %c10 step %c1 iter_args(%iter0 = %ref0, %iter1 = %ref1)
        -> (!pcf.sref<?x?xi32, #pcf.dummy_scope>, !pcf.sref<?x?xi32, #pcf.dummy_scope>) {
      %b:2 = util.optimization_barrier %iter1, %iter0 : !pcf.sref<?x?xi32, #pcf.dummy_scope>, !pcf.sref<?x?xi32, #pcf.dummy_scope>
      scf.yield %b#0, %b#1 : !pcf.sref<?x?xi32, #pcf.dummy_scope>, !pcf.sref<?x?xi32, #pcf.dummy_scope>
    }
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @convert_for_loop_swap_conflict
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32, strided<[3, 1]>, 3>
//  CHECK-SAME:     %[[ARG1:[A-Za-z0-9_]+]]: memref<?x?xi32, strided<[4, 1]>, 3>
//       CHECK:   pcf.generic scope(#pcf.dummy_scope)
//  CHECK-NEXT:     execute[{{.*}}] {
//   CHECK-DAG:     %[[CAST0:.+]] = memref.cast %[[ARG0]] : memref<?x?xi32, strided<[3, 1]>, 3> to memref<?x?xi32, strided<[?, 1]>, 3>
//   CHECK-DAG:     %[[CAST1:.+]] = memref.cast %[[ARG1]] : memref<?x?xi32, strided<[4, 1]>, 3> to memref<?x?xi32, strided<[?, 1]>, 3>
//  CHECK-NEXT:     scf.for {{.*}} iter_args(%[[ITER0:.+]] = %[[CAST0]], %[[ITER1:.+]] = %[[CAST1]])
//  CHECK-NEXT:       %[[B:.+]]:2 = util.optimization_barrier %[[ITER1]], %[[ITER0]]
//  CHECK-NEXT:       scf.yield %[[B]]#0, %[[B]]#1
//       CHECK:     pcf.return

// -----

func.func @convert_alloc(%d0: index) -> !pcf.sref<?x5xi32, #pcf.sequential> {
  %0 = pcf.alloc(%d0) : !pcf.sref<?x5xi32, #pcf.sequential>
  return %0 : !pcf.sref<?x5xi32, #pcf.sequential>
}

// CHECK-LABEL: @convert_alloc
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//       CHECK:   %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {alignment = 16 : i64} : memref<?x5xi32>
//       CHECK:   return %[[ALLOC]] : memref<?x5xi32>

// -----

// expected-error@+1 {{failed to legalize operation}}
func.func @invalid_workgroup_alloc(%d0: index) -> !pcf.sref<?x5xi32, #iree_codegen.workgroup> {
// expected-error@+1 {{failed to get memory space for allocation}}
  %0 = pcf.alloc(%d0) : !pcf.sref<?x5xi32, #iree_codegen.workgroup>
  return %0 : !pcf.sref<?x5xi32, #iree_codegen.workgroup>
}

// -----

util.func private @convert_get_memref(%arg0: memref<?x?xi32, strided<[?, 1]>, 3>, %s0: index, %s1: index) {
  pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32, strided<[?, 1]>, 3>) {
    %view = pcf.get_memref %ref[0, 1] [%s0, %s1] [1, 1] : !pcf.sref<?x?xi32, #pcf.dummy_scope> to memref<?x?xi32, strided<[?, ?], offset: ?>>
    util.optimization_barrier %view : memref<?x?xi32, strided<[?, ?], offset: ?>>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @convert_get_memref
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32, strided<[?, 1]>, 3>
//  CHECK-SAME:     %[[S0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:     %[[S1:[A-Za-z0-9_]+]]: index
//       CHECK:   pcf.generic
//  CHECK-NEXT:     execute[{{.*}}] {
//   CHECK-DAG:     %[[CAST:.+]] = memref.memory_space_cast %[[ARG0]] : memref<?x?xi32, strided<[?, 1]>, 3> to memref<?x?xi32, strided<[?, 1]>>
//   CHECK-DAG:     %[[CAST2:.+]] = memref.cast %[[CAST]] : memref<?x?xi32, strided<[?, 1]>> to memref<?x?xi32, strided<[?, ?], offset: ?>>
//   CHECK-DAG:     %[[SV:.+]] = memref.subview %[[CAST2]][0, 1] [%[[S0]], %[[S1]]] [1, 1] : memref<?x?xi32, strided<[?, ?], offset: ?>> to memref<?x?xi32, strided<[?, ?], offset: ?>>
//       CHECK:     util.optimization_barrier %[[SV]]
//       CHECK:     pcf.return

// -----

util.func private @convert_get_memref_no_space(%arg0: memref<?x?xi32, strided<[?, 1]>>, %s0: index, %s1: index) {
  pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32, strided<[?, 1]>>) {
    %view = pcf.get_memref %ref[0, 1] [%s0, %s1] [1, 1] : !pcf.sref<?x?xi32, #pcf.dummy_scope> to memref<?x?xi32, strided<[?, ?], offset: ?>>
    util.optimization_barrier %view : memref<?x?xi32, strided<[?, ?], offset: ?>>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @convert_get_memref_no_space
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32, strided<[?, 1]>>
//  CHECK-SAME:     %[[S0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:     %[[S1:[A-Za-z0-9_]+]]: index
//       CHECK:   pcf.generic
//  CHECK-NEXT:     execute[{{.*}}] {
//   CHECK-DAG:     %[[CAST:.+]] = memref.cast %[[ARG0]] : memref<?x?xi32, strided<[?, 1]>> to memref<?x?xi32, strided<[?, ?], offset: ?>>
//   CHECK-DAG:     %[[SV:.+]] = memref.subview %[[CAST]][0, 1] [%[[S0]], %[[S1]]] [1, 1] : memref<?x?xi32, strided<[?, ?], offset: ?>> to memref<?x?xi32, strided<[?, ?], offset: ?>>
//       CHECK:     util.optimization_barrier %[[SV]]
//       CHECK:     pcf.return

// -----

util.func private @convert_tensor_read_slice(%arg0: memref<?x?xi32>) {
  pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32>) {
    %dst = pcf.read_slice %ref[1, 2] [3, 4] [1, 1] : !pcf.sref<?x?xi32, #pcf.dummy_scope> to tensor<3x4xi32>
    util.optimization_barrier %dst : tensor<3x4xi32>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @convert_tensor_read_slice
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32>
//       CHECK:   pcf.generic
//  CHECK-NEXT:     execute[{{.*}}] {
//   CHECK-DAG:     %[[SV:.+]] = memref.subview %[[ARG0]][1, 2] [3, 4] [1, 1] : memref<?x?xi32> to memref<3x4xi32, strided<[?, 1], offset: ?>>
//   CHECK-DAG:     %[[DST:.+]] = iree_codegen.load_from_buffer %[[SV]] : memref<3x4xi32, strided<[?, 1], offset: ?>> -> tensor<3x4xi32>
//       CHECK:     util.optimization_barrier %[[DST]]
//       CHECK:     pcf.return

// -----

util.func private @convert_vector_read_slice(%arg0: memref<?x?xi32>) {
  pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32>) {
    %dst = pcf.read_slice %ref[1, 2] [3, 3] [1, 1] : !pcf.sref<?x?xi32, #pcf.dummy_scope> to vector<3x4xi32>
    util.optimization_barrier %dst : vector<3x4xi32>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @convert_vector_read_slice
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32>
//       CHECK:   pcf.generic
//  CHECK-NEXT:     execute[{{.*}}] {
//   CHECK-DAG:     %[[SV:.+]] = memref.subview %[[ARG0]][1, 2] [3, 3] [1, 1] : memref<?x?xi32> to memref<3x3xi32, strided<[?, 1], offset: ?>>
//   CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0 : i32
//       CHECK:     %[[DST:.+]] = vector.transfer_read %[[SV]][%c0, %c0], %[[ZERO]] {in_bounds = [true, false]}
//       CHECK:     util.optimization_barrier %[[DST]]
//       CHECK:     pcf.return

// -----

util.func private @convert_tensor_read_slice_no_tied_init(%dim_0: index, %dim_1: index) {
  pcf.generic scope(#pcf.dummy_scope)
    execute(%ref)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32>{%dim_0, %dim_1}) {
    %dst = pcf.read_slice %ref[1, 2] [3, 4] [1, 1] : !pcf.sref<?x?xi32, #pcf.dummy_scope> to tensor<3x4xi32>
    util.optimization_barrier %dst : tensor<3x4xi32>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @convert_tensor_read_slice_no_tied_init
//  CHECK-SAME:     %[[DIM0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:     %[[DIM1:[A-Za-z0-9_]+]]: index
//   CHECK-DAG:     %[[ALLOC:.+]] = memref.alloc(%[[DIM0]], %[[DIM1]]) {alignment = 16 : i64} : memref<?x?xi32>
//       CHECK:   pcf.generic
//  CHECK-NEXT:     execute[{{.*}}] {
//   CHECK-DAG:     %[[SV:.+]] = memref.subview %[[ALLOC]][1, 2] [3, 4] [1, 1] : memref<?x?xi32> to memref<3x4xi32, strided<[?, 1], offset: ?>>
//   CHECK-DAG:     %[[DST:.+]] = iree_codegen.load_from_buffer %[[SV]] : memref<3x4xi32, strided<[?, 1], offset: ?>> -> tensor<3x4xi32>
//       CHECK:     util.optimization_barrier %[[DST]]
//       CHECK:     pcf.return

// -----

util.func private @convert_vector_read_slice_dynamic_out_of_bounds(%arg0: memref<?x?xi32>, %sz: index) {
  pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32>) {
    %dst = pcf.read_slice %ref[1, 2] [3, %sz] [1, 1] : !pcf.sref<?x?xi32, #pcf.dummy_scope> to vector<3x4xi32>
    util.optimization_barrier %dst : vector<3x4xi32>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @convert_vector_read_slice_dynamic_out_of_bounds
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32>
//  CHECK-SAME:     %[[SZ:[A-Za-z0-9_]+]]: index
//       CHECK:   pcf.generic
//  CHECK-NEXT:     execute[{{.*}}] {
//   CHECK-DAG:     %[[SV:.+]] = memref.subview %[[ARG0]][1, 2] [3, %[[SZ]]] [1, 1] : memref<?x?xi32> to memref<3x?xi32, strided<[?, 1], offset: ?>>
//   CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0 : i32
//       CHECK:     %[[DST:.+]] = vector.transfer_read %[[SV]][%c0, %c0], %[[ZERO]] {in_bounds = [true, false]}
//       CHECK:     util.optimization_barrier %[[DST]]
//       CHECK:     pcf.return

// -----

util.func private @convert_get_memref_dynamic_layout(%arg0: memref<?x?xi32, strided<[?, ?], offset: ?>>) {
  pcf.generic scope(#pcf.dummy_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, #pcf.dummy_scope>)
        -> (memref<?x?xi32, strided<[?, ?], offset: ?>>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %view = pcf.get_memref %ref[%c0, %c0] [%c8, %c16] [1, 1] : !pcf.sref<?x?xi32, #pcf.dummy_scope> to memref<?x?xi32, strided<[?, ?], offset: ?>>
    util.optimization_barrier %view : memref<?x?xi32, strided<[?, ?], offset: ?>>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @convert_get_memref_dynamic_layout
//  CHECK-SAME:     %[[ARG0:[A-Za-z0-9_]+]]: memref<?x?xi32, strided<[?, ?], offset: ?>>
//       CHECK:   pcf.generic
//  CHECK-NEXT:     execute[{{.*}}] {
//       CHECK:     %[[SV:.+]] = memref.subview %[[ARG0]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xi32, strided<[?, ?], offset: ?>> to memref<?x?xi32, strided<[?, ?], offset: ?>>
//       CHECK:     util.optimization_barrier %[[SV]]
//       CHECK:     pcf.return
