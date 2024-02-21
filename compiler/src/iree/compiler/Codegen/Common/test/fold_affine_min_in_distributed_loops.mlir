// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-fold-affinemin-in-distributed-loops))" --mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: func.func @loop_distributed_to_workgroup_x
func.func @loop_distributed_to_workgroup_x() -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[C32:.+]] = arith.constant 32 : index
  %c32 = arith.constant 32 : index

  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index

  %0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %c32]
  %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %c32]

  %2 = scf.for %iv = %0 to %c32 step %1 iter_args(%arg = %c0) -> (index) {
    %min = affine.min affine_map<(d0, d1) -> (d0, -d1 + 32)>(%c32, %iv)
    // CHECK: scf.yield %[[C32]]
    scf.yield %min : index
  }
  return %2 : index
}

// -----

// CHECK-LABEL: func.func @loop_distributed_to_workgroup_y
func.func @loop_distributed_to_workgroup_y() -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[C32:.+]] = arith.constant 32 : index
  %c32 = arith.constant 32 : index

  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index

  %0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %c32]
  %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %c32]

  %2 = scf.for %iv = %0 to %c32 step %1 iter_args(%arg = %c0) -> (index) {
    %min = affine.min affine_map<(d0, d1) -> (d0, -d1 + 32)>(%c32, %iv)
    // CHECK: scf.yield %[[C32]]
    scf.yield %min : index
  }
  return %2 : index
}

// -----

// CHECK-LABEL: func.func @loop_distributed_to_workgroup_z
func.func @loop_distributed_to_workgroup_z() -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[C32:.+]] = arith.constant 32 : index
  %c32 = arith.constant 32 : index

  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_count_z = hal.interface.workgroup.count[2] : index

  %0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %c32]
  %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %c32]

  %2 = scf.for %iv = %0 to %c32 step %1 iter_args(%arg = %c0) -> (index) {
    %min = affine.min affine_map<(d0, d1) -> (d0, -d1 + 32)>(%c32, %iv)
    // CHECK: scf.yield %[[C32]]
    scf.yield %min : index
  }
  return %2 : index
}

// -----

// CHECK-LABEL: func.func @loop_distributed_to_workitem_x
func.func @loop_distributed_to_workitem_x() -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[C32:.+]] = arith.constant 32 : index
  %c32 = arith.constant 32 : index

  %workitem_id_x = gpu.thread_id x
  %workitem_count_x = gpu.block_dim x

  %0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workitem_id_x, %c32]
  %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workitem_count_x, %c32]

  %2 = scf.for %iv = %0 to %c32 step %1 iter_args(%arg = %c0) -> (index) {
    %min = affine.min affine_map<(d0, d1) -> (d0, -d1 + 32)>(%c32, %iv)
    // CHECK: scf.yield %[[C32]]
    scf.yield %min : index
  }
  return %2 : index
}

// -----

// CHECK-LABEL: func.func @loop_distributed_to_workitem_y
func.func @loop_distributed_to_workitem_y() -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[C32:.+]] = arith.constant 32 : index
  %c32 = arith.constant 32 : index

  %workitem_id_y = gpu.thread_id y
  %workitem_count_y = gpu.block_dim y

  %0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workitem_id_y, %c32]
  %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workitem_count_y, %c32]

  %2 = scf.for %iv = %0 to %c32 step %1 iter_args(%arg = %c0) -> (index) {
    %min = affine.min affine_map<(d0, d1) -> (d0, -d1 + 32)>(%c32, %iv)
    // CHECK: scf.yield %[[C32]]
    scf.yield %min : index
  }
  return %2 : index
}

// -----

// CHECK-LABEL: func.func @loop_distributed_to_workitem_z
func.func @loop_distributed_to_workitem_z() -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[C32:.+]] = arith.constant 32 : index
  %c32 = arith.constant 32 : index

  %workitem_id_z = gpu.thread_id z
  %workitem_count_z = gpu.block_dim z

  %0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workitem_id_z, %c32]
  %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workitem_count_z, %c32]

  %2 = scf.for %iv = %0 to %c32 step %1 iter_args(%arg = %c0) -> (index) {
    %min = affine.min affine_map<(d0, d1) -> (d0, -d1 + 32)>(%c32, %iv)
    // CHECK: scf.yield %[[C32]]
    scf.yield %min : index
  }
  return %2 : index
}

// -----

// CHECK-LABEL: func.func @cst_folded_into_affine_map
func.func @cst_folded_into_affine_map() -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[C32:.+]] = arith.constant 32 : index
  %c32 = arith.constant 32 : index

  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index

  %0 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
  %1 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]

  %2 = scf.for %iv = %0 to %c32 step %1 iter_args(%arg = %c0) -> (index) {
    %min = affine.min affine_map<(d0) -> (32, -d0 + 32)>(%iv)
    // CHECK: scf.yield %[[C32]]
    scf.yield %min : index
  }
  return %2 : index
}

// -----

// CHECK-LABEL: func.func @affine_apply_folded_into_loop
func.func @affine_apply_folded_into_loop() -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index

  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index

  %2 = scf.for %iv = %workgroup_id_x to %c32 step %workgroup_count_x iter_args(%arg = %c0) -> (index) {
    %min = affine.min affine_map<(d0, d1) -> (d0, -d1 + 32)>(%c1, %iv)
    // CHECK: scf.yield %[[C1]]
    scf.yield %min : index
  }
  return %2 : index
}

// -----

// CHECK-LABEL: func.func @unknown_tile_size
func.func @unknown_tile_size() -> index {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index

  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  // CHECK: %[[SIZE:.+]] = hal.interface.workgroup.size[0] : index
  %workgroup_size_x = hal.interface.workgroup.size[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index

  %0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
  %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]

  // CHECK: scf.for %[[IV:[a-z0-9]+]] =
  %2 = scf.for %iv = %0 to %c32 step %1 iter_args(%arg = %c0) -> (index) {
    // CHECK: affine.min affine_map<(d0, d1) -> (d0, -d1 + 32)>(%[[SIZE]], %[[IV]])
    %min = affine.min affine_map<(d0, d1) -> (d0, -d1 + 32)>(%workgroup_size_x, %iv)
    scf.yield %min : index
  }
  return %2 : index
}

// -----

// CHECK-LABEL: func.func @mismatched_id_count
func.func @mismatched_id_count() -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[C32:.+]] = arith.constant 32 : index
  %c32 = arith.constant 32 : index

  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index

  %0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %c32]
  %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %c32]

  // CHECK: scf.for %[[IV:[a-z0-9]+]] =
  %2 = scf.for %iv = %0 to %c32 step %1 iter_args(%arg = %c0) -> (index) {
    // CHECK: affine.min affine_map<(d0, d1) -> (32, -d1 + 32)>(%[[C32]], %[[IV]])
    %min = affine.min affine_map<(d0, d1) -> (d0, -d1 + 32)>(%c32, %iv)
    scf.yield %min : index
  }
  return %2 : index
}

// -----

// CHECK-LABEL: func.func @min_over_min
func.func @min_over_min() -> index {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c112 = arith.constant 112 : index

  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index

  %0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %c8]
  %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %c8]

  // Untiled loop: lb = 0, ub = 112; with tile size = 8, can perfectly tile
  %2 = scf.for %iv = %0 to %c112 step %1 iter_args(%arg = %c0) -> (index) {
    // Undistributed %iv ranges: [0, 8, 16, 24, ..., 104]
    // So (112 - %iv) ranges: [112, 104, ..., 8]
    // So (225 - %iv * 2) ranges: [225, 209, ..., 17]
    // Therefore, %min0 >= 8, %min1 >= 17
    %min0 = affine.min affine_map<(d0, d1) -> (d0, -d1 + 112)>(%c8, %iv)
    %min1 = affine.min affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>(%min0, %iv)
    // CHECK: %[[C17:.+]] = arith.constant 17 : index
    // CHECK: scf.yield %[[C17]]
    scf.yield %min1 : index
  }
  return %2 : index
}

// -----

// CHECK-LABEL: func.func @cannot_prove_cst_bound
func.func @cannot_prove_cst_bound() -> index {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  // CHECK: %[[C9:.+]] = arith.constant 9 : index
  %c9 = arith.constant 9 : index
  %c112 = arith.constant 112 : index

  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index

  %0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %c8]
  %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %c8]

  // Untiled loop: lb = 0, ub = 112; with tile size = 8, can perfectly tile
  // CHECK: scf.for %[[IV:[a-z0-9]+]] =
  %2 = scf.for %iv = %0 to %c112 step %1 iter_args(%arg = %c0) -> (index) {
    // Undistributed %iv ranges: [0, 8, 16, 24, ..., 104]
    // So (112 - %iv) ranges: [112, 104, ..., 8]
    // CHECK: affine.min affine_map<(d0, d1) -> (9, -d1 + 112)>(%[[C9]], %[[IV]])
    %min = affine.min affine_map<(d0, d1) -> (d0, -d1 + 112)>(%c9, %iv)
    scf.yield %min : index
  }
  return %2 : index
}

// -----

// CHECK-LABEL: func.func @can_prove_symbolic_bound
func.func @can_prove_symbolic_bound() -> index {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c112 = arith.constant 112 : index

  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index

  %0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %c8]
  %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %c8]

  // Untiled loop: lb = 0, ub = 112; with tile size = 8, can perfectly tile
  // CHECK: scf.for %[[IV:[a-z0-9]+]] =
  %2 = scf.for %iv = %0 to %c112 step %1 iter_args(%arg = %c0) -> (index) {
    // Undistributed %iv ranges: [0, 8, 16, 24, ..., 104]
    // So (112 - %iv) ranges: [112, 104, ..., 8]
    // CHECK: %[[MIN:.+]] = affine.apply affine_map<(d0) -> (-d0 + 112)>(%[[IV]])
    %min = affine.min affine_map<(d0, d1) -> (d0, -d1 + 112)>(%c112, %iv)
    // CHECK: scf.yield %[[MIN]]
    scf.yield %min : index
  }
  return %2 : index
}
