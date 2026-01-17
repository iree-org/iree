// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-lower-structural-pcf, cse)" --split-input-file --verify-diagnostics | FileCheck %s

util.func private @subgroup_scope_generic() {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    execute[%id: index, %n: index] {
    util.optimization_barrier %id, %n : index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @subgroup_scope_generic
//       CHECK:   %[[ID:.+]] = gpu.subgroup_id
//       CHECK:   %[[N:.+]] = gpu.num_subgroups
//       CHECK:   scf.execute_region
//  CHECK-NEXT:     util.optimization_barrier %[[ID]], %[[N]]
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }

// -----

util.func private @subgroup_scope_loop(%d0: index) {
  pcf.loop scope(#iree_gpu.subgroup_scope) count(%d0)
    execute[%n0: index] {
    util.optimization_barrier %n0 : index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @subgroup_scope_loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: index
//       CHECK:   %[[N:.+]] = gpu.num_subgroups
//       CHECK:   %[[ID:.+]] = gpu.subgroup_id
//       CHECK:   scf.forall (%[[IV:.+]]) = (%[[ID]]) to (%[[ARG0]]) step (%[[N]])
//  CHECK-NEXT:     util.optimization_barrier %[[IV]]

// -----

util.func private @lane_scope_generic() {
  pcf.generic scope(#iree_gpu.lane_scope)
    execute[%id: index, %n: index] {
    util.optimization_barrier %id, %n : index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @lane_scope_generic
//       CHECK:   %[[ID:.+]] = gpu.lane_id
//       CHECK:   %[[N:.+]] = gpu.subgroup_size
//       CHECK:   scf.execute_region
//  CHECK-NEXT:     util.optimization_barrier %[[ID]], %[[N]]
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }

// -----

util.func private @lane_scope_loop(%d0: index) {
  pcf.loop scope(#iree_gpu.lane_scope) count(%d0)
    execute[%n0: index] {
    util.optimization_barrier %n0 : index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @lane_scope_loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: index
//       CHECK:   %[[N:.+]] = gpu.subgroup_size
//       CHECK:   %[[ID:.+]] = gpu.lane_id
//       CHECK:   scf.forall (%[[IV:.+]]) = (%[[ID]]) to (%[[ARG0]]) step (%[[N]])
//  CHECK-NEXT:     util.optimization_barrier %[[IV]]

// -----

util.func private @subgroup_scope_multiple_ids() {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    execute[%id0: index, %id1: index, %n0: index, %n1: index] {
    util.optimization_barrier %id0, %id1, %n0, %n1 : index, index, index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @subgroup_scope_multiple_ids
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[ID:.+]] = gpu.subgroup_id
//   CHECK-DAG:   %[[N:.+]] = gpu.num_subgroups
//       CHECK:   scf.execute_region
//  CHECK-NEXT:     util.optimization_barrier %[[ID]], %[[C0]], %[[N]], %[[C1]]
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }

// -----

util.func private @lane_scope_multiple_ids() {
  pcf.generic scope(#iree_gpu.lane_scope)
    execute[%id0: index, %id1: index, %n0: index, %n1: index] {
    util.optimization_barrier %id0, %id1, %n0, %n1 : index, index, index, index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @lane_scope_multiple_ids
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[ID:.+]] = gpu.lane_id
//   CHECK-DAG:   %[[N:.+]] = gpu.subgroup_size
//       CHECK:   scf.execute_region
//  CHECK-NEXT:     util.optimization_barrier %[[ID]], %[[C0]], %[[N]], %[[C1]]
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }

// -----

// Test barrier for subgroup scope - uses gpu.barrier.
util.func private @subgroup_scope_barrier() {
  pcf.generic sync true scope(#iree_gpu.subgroup_scope)
    execute[%id: index, %n: index] {
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @subgroup_scope_barrier
//       CHECK:   gpu.barrier
//       CHECK:   util.return

// -----

// Test barrier failure for lane scope - lane barriers are not supported.
util.func private @lane_scope_barrier() {
  // expected-error@+1 {{failed to construct requested barrier}}
  pcf.generic sync true scope(#iree_gpu.lane_scope)
    execute[%id: index, %n: index] {
    pcf.return
  }
  util.return
}
