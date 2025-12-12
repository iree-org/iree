// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-convert-sref-to-memref)" --split-input-file --verify-diagnostics | FileCheck %s

// Test that subgroup scope allocations use GPU shared memory (workgroup address space).
util.func private @subgroup_scope_alloc(%d0: index) {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    initialize {
      %alloc = pcf.alloc(%d0) : !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>
      pcf.yield %alloc : !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>
    } -> (%aref: !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>)
    execute[%id: index, %n: index] {
    util.optimization_barrier %aref : !pcf.sref<?x8xi32, #iree_gpu.subgroup_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @subgroup_scope_alloc
//  CHECK-SAME:   %[[D0:[A-Za-z0-9_]+]]: index
//       CHECK:   pcf.generic scope(#iree_gpu.subgroup_scope)
//  CHECK-NEXT:     execute[{{.*}}] {
//  CHECK-NEXT:     %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {alignment = 16 : i64} : memref<?x8xi32, #gpu.address_space<workgroup>>
//  CHECK-NEXT:     util.optimization_barrier %[[ALLOC]]
//  CHECK-NEXT:     pcf.return

// -----

util.func private @unimplemented_lane_scope_alloc(%d0: index) {
  pcf.generic scope(#iree_gpu.lane_scope)
    initialize {
      // expected-error@+2 {{failed to get memory space for allocation}}
      // expected-error@+1 {{failed to legalize operation 'pcf.alloc'}}
      %alloc = pcf.alloc(%d0) : !pcf.sref<?x8xi32, #iree_gpu.lane_scope>
      pcf.yield %alloc : !pcf.sref<?x8xi32, #iree_gpu.lane_scope>
    } -> (%aref: !pcf.sref<?x8xi32, #iree_gpu.lane_scope>)
    execute[%id: index, %n: index] {
    util.optimization_barrier %aref : !pcf.sref<?x8xi32, #iree_gpu.lane_scope>
    pcf.return
  }
  util.return
}
