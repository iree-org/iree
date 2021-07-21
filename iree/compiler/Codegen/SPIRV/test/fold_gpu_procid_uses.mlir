// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.variant(iree-spirv-fold-gpu-procid-uses))" %s | IreeFileCheck %s

hal.executable @fold_block_id attributes {sym_visibility = "private"} {
  hal.interface @io {
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @fold_block_id attributes {
      interface = @io,
      ordinal = 0 : index
    } {
    ^bb0(%arg0 : index, %arg1 : index, %arg2 : index):
      %x = constant 112: index
      %y = constant 42: index
      %z = constant 1: index
      hal.return %x, %y, %z: index, index, index
    }
    module {
      func @fold_block_id() -> (index, index, index) {
        %0 = "gpu.block_id"() {dimension = "x"} : () -> index
        %1 = "gpu.block_id"() {dimension = "y"} : () -> index
        %2 = "gpu.block_id"() {dimension = "z"} : () -> index
        %3 = affine.min affine_map<()[s0] -> (3, s0 * -2 + 225)>()[%0]
        %4 = affine.min affine_map<()[s0] -> (8, s0 * -1 + s0 * -1 + s0 * -1 + 131)>()[%2]
        %5 = affine.min affine_map<()[s0] -> (11, s0 + 15)>()[%3]
        return %3, %4, %5: index, index, index
      }
    }
  }
}
// CHECK-LABEL: func @fold_block_id()
//   CHECK-DAG:   %[[C3:.+]] = constant 3
//   CHECK-DAG:   %[[C8:.+]] = constant 8
//   CHECK-DAG:   %[[C11:.+]] = constant 11
//   CHECK-DAG:   return %[[C3]], %[[C8]], %[[C11]]

// -----

hal.executable @fold_interface_workgroup_id attributes {sym_visibility = "private"} {
  hal.interface @io {
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @fold_interface_workgroup_id attributes {
      interface = @io,
      ordinal = 0 : index
    } {
    ^bb0(%arg0 : index, %arg1 : index, %arg2 : index):
      %x = constant 112: index
      %y = constant 42: index
      %z = constant 1: index
      hal.return %x, %y, %z: index, index, index
    }
    module {
      func @fold_interface_workgroup_id() -> (index, index, index) {
        %0 = hal.interface.workgroup.id[0] : index
        %1 = hal.interface.workgroup.id[1] : index
        %2 = hal.interface.workgroup.id[2] : index
        %3 = affine.min affine_map<()[s0] -> (3, s0 * -2 + 225)>()[%0]
        %4 = affine.min affine_map<()[s0] -> (8, s0 * -1 + s0 * -1 + s0 * -1 + 131)>()[%2]
        %5 = affine.min affine_map<()[s0] -> (11, s0 + 15)>()[%3]
        return %3, %4, %5: index, index, index
      }
    }
  }
}
// CHECK-LABEL: func @fold_interface_workgroup_id()
//   CHECK-DAG:   %[[C3:.+]] = constant 3
//   CHECK-DAG:   %[[C8:.+]] = constant 8
//   CHECK-DAG:   %[[C11:.+]] = constant 11
//   CHECK-DAG:   return %[[C3]], %[[C8]], %[[C11]]

// -----

hal.executable @fold_thread_id attributes {sym_visibility = "private"} {
  hal.interface @io {
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @fold_thread_id attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module {
      func @fold_thread_id() -> (index, index, index)
        attributes {spv.entry_point_abi = {local_size = dense<[8, 2, 1]> : vector<3xi32>}} {
        %0 = "gpu.thread_id"() {dimension = "x"} : () -> index
        %1 = "gpu.thread_id"() {dimension = "y"} : () -> index
        %2 = "gpu.thread_id"() {dimension = "z"} : () -> index
        %3 = affine.min affine_map<()[s0] -> (7, s0 * -1 + s0 * -1 + 21)>()[%0]
        %4 = affine.min affine_map<()[s0] -> (11, s0 * -3 + 14)>()[%1]
        %5 = affine.min affine_map<()[s0] -> (21, s0 + (s0 + 21))>()[%2]
        return %3, %4, %5 : index, index, index
      }
    }
  }
}
// CHECK-LABEL: func @fold_thread_id()
//   CHECK-DAG:   %[[C7:.+]] = constant 7
//   CHECK-DAG:   %[[C11:.+]] = constant 11
//   CHECK-DAG:   %[[C21:.+]] = constant 21
//   CHECK-DAG:   return %[[C7]], %[[C11]], %[[C21]]

// -----

hal.executable @does_not_fold_mod attributes {sym_visibility = "private"} {
  hal.interface @io {
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @does_not_fold_mod attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module {
      func @does_not_fold_mod() -> index attributes {spv.entry_point_abi = {local_size = dense<[8, 2, 1]> : vector<3xi32>}} {
        %0 = "gpu.thread_id"() {dimension = "z"} : () -> index
        %1 = affine.min affine_map<()[s0] -> (21, s0 mod 5)>()[%0]
        return %1: index
      }
    }
  }
}
// CHECK-LABEL: func @does_not_fold_mod()
//       CHECK:   affine.min

// -----

hal.executable @does_not_fold_div attributes {sym_visibility = "private"} {
  hal.interface @io {
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @does_not_fold_div attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module {
      func @does_not_fold_div() -> index attributes {spv.entry_point_abi = {local_size = dense<[8, 2, 1]> : vector<3xi32>}} {
        %0 = "gpu.thread_id"() {dimension = "z"} : () -> index
        %1 = affine.min affine_map<()[s0] -> (21, s0 ceildiv 5)>()[%0]
        return %1: index
      }
    }
  }
}
// CHECK-LABEL: func @does_not_fold_div()
//       CHECK:   affine.min

// -----

hal.executable @does_not_fold_symbol_mul_symbol attributes {sym_visibility = "private"} {
  hal.interface @io {
  }
  hal.executable.variant @vulkan, target="vulkan" {
    hal.executable.entry_point @does_not_fold_symbol_mul_symbol attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module {
      func @does_not_fold_symbol_mul_symbol() -> index attributes {spv.entry_point_abi = {local_size = dense<[8, 2, 1]> : vector<3xi32>}} {
        // 5 is in %0's range of [0,7] so we cannot fold the following into 5 or 0.
        %0 = "gpu.thread_id"() {dimension = "z"} : () -> index
        %1 = affine.min affine_map<()[s0] -> (21, s0 * s0)>()[%0]
        return %1: index
      }
    }
  }
}
// CHECK-LABEL: func @does_not_fold_symbol_mul_symbol()
//       CHECK:   affine.min
