// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-lower-structural-pcf, cse)" --split-input-file --verify-diagnostics | FileCheck %s

util.func private @generic() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    util.optimization_barrier %id, %n : index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @generic
//   CHECK-DAG:   %[[I:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[N:.+]] = arith.constant 1 : index
//       CHECK:   scf.execute_region
//  CHECK-NEXT:     util.optimization_barrier %[[I]], %[[N]]
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }

// -----

util.func private @generic_multiple_iterators(%d0: index, %d1: index) {
  pcf.generic scope(#pcf.sequential)
    execute[%id0: index, %id1: index, %n0: index, %n1: index] {
    util.optimization_barrier %id0, %id1, %n0, %n1 : index, index, index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @generic_multiple_iterators
//   CHECK-DAG:   %[[I:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[N:.+]] = arith.constant 1 : index
//       CHECK:   scf.execute_region
//  CHECK-NEXT:     util.optimization_barrier %[[I]], %[[I]], %[[N]], %[[N]]
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }

// -----

util.func private @generic_with_multiple_blocks() {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    cf.br ^bb1
   ^bb1:
    util.optimization_barrier
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @generic_with_multiple_blocks
//       CHECK:   scf.execute_region
//  CHECK-NEXT:     cf.br ^bb1
//  CHECK-NEXT:    ^bb1:  // pred: ^bb0
//  CHECK-NEXT:     util.optimization_barrier
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }

// -----

util.func private @br_cond_return(%cond: i1) {
  pcf.generic scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    pcf.br.cond_return %cond ^bb1
   ^bb1:
    util.optimization_barrier
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @br_cond_return
//  CHECK-SAME:  %[[COND:[A-Za-z0-9_]+]]: i1
//       CHECK:   scf.execute_region
//  CHECK-NEXT:     cf.cond_br %[[COND]], ^bb2, ^bb1

// Dest branch (condition is false).
//  CHECK-NEXT:    ^bb1:  // pred: ^bb0
//  CHECK-NEXT:     util.optimization_barrier
//  CHECK-NEXT:     scf.yield

// Return branch (condition is true).
//  CHECK-NEXT:    ^bb2:  // pred: ^bb0
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }

// -----

util.func private @fail_barrier_build(%cond: i1) {
// expected-error@+1 {{failed to construct requested barrier}}
  pcf.generic sync true scope(#pcf.sequential)
    execute[%id: index, %n: index] {
    pcf.return
  }
  util.return
}

// -----

util.func private @lower_loop(%d0: index, %d1: index) {
  pcf.loop scope(#pcf.sequential) count(%d0, %d1)
    execute[%n0: index, %n1: index] {
    util.optimization_barrier %n0, %n1 : index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @lower_loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9_]+]]: index
//       CHECK:   scf.forall (%[[ID1:.+]], %[[ID0:.+]]) in (%[[ARG1]], %[[ARG0]])
//  CHECK-NEXT:     util.optimization_barrier %[[ID0]], %[[ID1]]

// -----

util.func private @workgroup_ids(%d0: index, %d1: index) {
  pcf.loop scope(#iree_codegen.workgroup_scope) count(%d0, %d1)
    execute[%n0: index, %n1: index] {
    util.optimization_barrier %n0, %n1 : index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @workgroup_ids
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9_]+]]: index
//   CHECK-DAG:   %[[WGCX:.+]] = hal.interface.workgroup.count[0] : index
//   CHECK-DAG:   %[[WGCY:.+]] = hal.interface.workgroup.count[1] : index
//   CHECK-DAG:   %[[WGIX:.+]] = hal.interface.workgroup.id[0] : index
//   CHECK-DAG:   %[[WGIY:.+]] = hal.interface.workgroup.id[1] : index
//       CHECK:   scf.forall (%[[ID0:.+]], %[[ID1:.+]]) = (%[[WGIY]], %[[WGIX]]) to (%[[ARG1]], %[[ARG0]]) step (%[[WGCY]], %[[WGCX]])
//  CHECK-NEXT:     util.optimization_barrier %[[ID1]], %[[ID0]]

// -----

util.func private @linearize_workgroup_ids(%d0: index, %d1: index, %d2: index, %d3: index) {
  pcf.loop scope(#iree_codegen.workgroup_scope<linearize>) count(%d0, %d1, %d2, %d3)
    execute[%n0: index, %n1: index, %n2: index, %n3: index] {
    util.optimization_barrier %n0, %n1, %n2, %n3 : index, index, index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @linearize_workgroup_ids
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG2:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG3:[A-Za-z0-9_]+]]: index
//       CHECK:   %[[MUL0:.+]] = arith.muli %workgroup_count_x, %workgroup_count_y : index
//       CHECK:   %[[MUL1:.+]] = arith.muli %[[MUL0]], %workgroup_count_z : index
//       CHECK:   %[[LINEARIZE:.+]] = affine.linearize_index [%c0, %workgroup_id_z, %workgroup_id_y, %workgroup_id_x]
//  CHECK-SAME:     by (1, %workgroup_count_z, %workgroup_count_y, %workgroup_count_x)
//       CHECK:   scf.forall ({{.*}}) = (%c1, %c1, %c1, %[[LINEARIZE]])
//  CHECK-SAME:     to (%[[ARG3]], %[[ARG2]], %[[ARG1]], %[[ARG0]]) step (%c1, %c1, %c1, %[[MUL1]])
