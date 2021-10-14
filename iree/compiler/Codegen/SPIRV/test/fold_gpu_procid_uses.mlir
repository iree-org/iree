// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(builtin.module(builtin.func(iree-spirv-fold-gpu-procid-uses))))' %s | IreeFileCheck %s

hal.executable private @fold_block_id  {
  hal.interface @io {
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @fold_block_id attributes {
      interface = @io,
      ordinal = 0 : index
    } {
    ^bb0(%arg0 : index, %arg1 : index, %arg2 : index):
      %x = arith.constant 112: index
      %y = arith.constant 42: index
      %z = arith.constant 1: index
      hal.return %x, %y, %z: index, index, index
    }
    builtin.module {
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
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8
//   CHECK-DAG:   %[[C11:.+]] = arith.constant 11
//   CHECK-DAG:   return %[[C3]], %[[C8]], %[[C11]]

// -----

hal.executable private @fold_interface_workgroup_id  {
  hal.interface @io {
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @fold_interface_workgroup_id attributes {
      interface = @io,
      ordinal = 0 : index
    } {
    ^bb0(%arg0 : index, %arg1 : index, %arg2 : index):
      %x = arith.constant 112: index
      %y = arith.constant 42: index
      %z = arith.constant 1: index
      hal.return %x, %y, %z: index, index, index
    }
    builtin.module {
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
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8
//   CHECK-DAG:   %[[C11:.+]] = arith.constant 11
//   CHECK-DAG:   return %[[C3]], %[[C8]], %[[C11]]

// -----

hal.executable private @fold_thread_id  {
  hal.interface @io {
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @fold_thread_id attributes {
      interface = @io,
      ordinal = 0 : index,
      workgroup_size = [8: index, 2: index, 1: index]
    }
    builtin.module {
      func @fold_thread_id() -> (index, index, index) {
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
//   CHECK-DAG:   %[[C7:.+]] = arith.constant 7
//   CHECK-DAG:   %[[C11:.+]] = arith.constant 11
//   CHECK-DAG:   %[[C21:.+]] = arith.constant 21
//   CHECK-DAG:   return %[[C7]], %[[C11]], %[[C21]]

// -----

hal.executable private @does_not_fold_mod  {
  hal.interface @io {
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @does_not_fold_mod attributes {
      interface = @io,
      ordinal = 0 : index,
      workgroup_size = [8: index, 2: index, 1: index]
    }
    builtin.module {
      func @does_not_fold_mod() -> index {
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

hal.executable private @does_not_fold_div  {
  hal.interface @io {
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @does_not_fold_div attributes {
      interface = @io,
      ordinal = 0 : index,
      workgroup_size = [8: index, 2: index, 1: index]
    }
    builtin.module {
      func @does_not_fold_div() -> index {
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

hal.executable private @does_not_fold_symbol_mul_symbol  {
  hal.interface @io {
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @does_not_fold_symbol_mul_symbol attributes {
      interface = @io,
      ordinal = 0 : index,
      workgroup_size = [8: index, 2: index, 1: index]
    }
    builtin.module {
      func @does_not_fold_symbol_mul_symbol() -> index {
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
