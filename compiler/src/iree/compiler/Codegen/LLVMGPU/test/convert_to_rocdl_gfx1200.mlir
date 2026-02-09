// RUN: iree-opt --iree-gpu-test-target=gfx1200 --iree-convert-to-rocdl %s | FileCheck %s

module {
  func.func @global_subgroup_barrier() {
    iree_gpu.global_subgroup_barrier
    return
  }
}

// CHECK-LABEL: llvm.func @global_subgroup_barrier
//       CHECK:   rocdl.s.barrier.signal id = -1
//       CHECK:   rocdl.s.barrier.wait id = -1
