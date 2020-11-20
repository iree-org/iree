// RUN: iree-opt -split-input-file -iree-codegen-fold-gpu-procid-uses %s | IreeFileCheck %s

module {
  // CHECK-LABEL: func @fold_block_id_x()
  func @fold_block_id_x() -> index attributes {hal.num_workgroups_fn = @num_workgroups} {
    // CHECK: %[[cst:.+]] = constant 3
    // CHECK: return %[[cst]]
    %0 = "gpu.block_id"() {dimension = "x"} : () -> index
    %1 = affine.min affine_map<()[s0] -> (3, s0 * -2 + 225)>()[%0]
    return %1: index
  }

  // CHECK-LABEL: func @fold_block_id_y()
  func @fold_block_id_y() -> index attributes {hal.num_workgroups_fn = @num_workgroups} {
    // CHECK: %[[cst:.+]] = constant 8
    // CHECK: return %[[cst]]
    %0 = "gpu.block_id"() {dimension = "y"} : () -> index
    %1 = affine.min affine_map<()[s0] -> (8, s0 * -1 + s0 * -1 + s0 * -1 + 131)>()[%0]
    return %1: index
  }

  // CHECK-LABEL: func @fold_block_id_z()
  func @fold_block_id_z() -> index attributes {hal.num_workgroups_fn = @num_workgroups} {
    // CHECK: %[[cst:.+]] = constant 11
    // CHECK: return %[[cst]]
    %0 = "gpu.block_id"() {dimension = "z"} : () -> index
    %1 = affine.min affine_map<()[s0] -> (11, s0 + 15)>()[%0]
    return %1: index
  }

  func @num_workgroups() -> (index, index, index) {
    %x = constant 112: index
    %y = constant 42: index
    %z = constant 1: index
    return %x, %y, %z: index, index, index
  }
}

// -----

// CHECK-LABEL: func @fold_thread_id_x()
func @fold_thread_id_x() -> index attributes {spv.entry_point_abi = {local_size = dense<[8, 2, 1]> : vector<3xi32>}} {
  // CHECK: %[[cst:.+]] = constant 7
  // CHECK: return %[[cst]]
  %0 = "gpu.thread_id"() {dimension = "x"} : () -> index
  %1 = affine.min affine_map<()[s0] -> (7, s0 * -1 + s0 * -1 + 21)>()[%0]
  return %1: index
}

// CHECK-LABEL: func @fold_thread_id_y()
func @fold_thread_id_y() -> index attributes {spv.entry_point_abi = {local_size = dense<[8, 2, 1]> : vector<3xi32>}} {
  // CHECK: %[[cst:.+]] = constant 11
  // CHECK: return %[[cst]]
  %0 = "gpu.thread_id"() {dimension = "y"} : () -> index
  %1 = affine.min affine_map<()[s0] -> (11, s0 * -3 + 14)>()[%0]
  return %1: index
}

// CHECK-LABEL: func @fold_thread_id_z()
func @fold_thread_id_z() -> index attributes {spv.entry_point_abi = {local_size = dense<[8, 2, 1]> : vector<3xi32>}} {
  // CHECK: %[[cst:.+]] = constant 21
  // CHECK: return %[[cst]]
  %0 = "gpu.thread_id"() {dimension = "z"} : () -> index
  %1 = affine.min affine_map<()[s0] -> (21, s0 + (s0 + 21))>()[%0]
  return %1: index
}

// -----

// CHECK-LABEL: func @does_not_fold_mod()
func @does_not_fold_mod() -> index attributes {spv.entry_point_abi = {local_size = dense<[8, 2, 1]> : vector<3xi32>}} {
  // CHECK: affine.min
  %0 = "gpu.thread_id"() {dimension = "z"} : () -> index
  %1 = affine.min affine_map<()[s0] -> (21, s0 mod 5)>()[%0]
  return %1: index
}

// CHECK-LABEL: func @does_not_fold_div()
func @does_not_fold_div() -> index attributes {spv.entry_point_abi = {local_size = dense<[8, 2, 1]> : vector<3xi32>}} {
  // CHECK: affine.min
  %0 = "gpu.thread_id"() {dimension = "z"} : () -> index
  %1 = affine.min affine_map<()[s0] -> (21, s0 ceildiv 5)>()[%0]
  return %1: index
}

// CHECK-LABEL: func @does_not_fold_symbol_mul_symbol()
func @does_not_fold_symbol_mul_symbol() -> index attributes {spv.entry_point_abi = {local_size = dense<[8, 2, 1]> : vector<3xi32>}} {
  // CHECK: affine.min
  %0 = "gpu.thread_id"() {dimension = "z"} : () -> index
  %1 = affine.min affine_map<()[s0] -> (21, s0 * s0)>()[%0]
  return %1: index
}

// CHECK-LABEL: func @does_not_fold_if_cst_not_lower_bound()
func @does_not_fold_if_cst_not_lower_bound() -> index attributes {spv.entry_point_abi = {local_size = dense<[8, 2, 1]> : vector<3xi32>}} {
  // CHECK: affine.min
  %0 = "gpu.thread_id"() {dimension = "x"} : () -> index
  // 5 is in %0's range of [0,7] so we cannot fold the following into 5 or 0.
  %1 = affine.min affine_map<()[s0] -> (5, s0)>()[%0]
  return %1: index
}
