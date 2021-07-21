// RUN: iree-opt -iree-llvmgpu-remove-single-iteration-loop %s | IreeFileCheck %s

// CHECK-LABEL: func @dispatch_0()
func @dispatch_0() attributes {llvmgpu_workgroup_size = dense<[64, 1, 1]> : vector<3xi64>} {
  %c2 = constant 2 : index
  %c256 = constant 256 : index
  //     CHECK: %[[C250:.+]] = constant 250 : index
  %c250 = constant 250 : index
  %tidx = "gpu.thread_id"() {dimension = "x"} : () -> index
  %tidy = "gpu.thread_id"() {dimension = "y"} : () -> index
  // CHECK-NOT: scf.for
  //     CHECK: gpu.barrier
  scf.for %arg3 = %tidy to %c2 step %c2 {
    %0 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%tidx]
    scf.for %arg4 = %0 to %c256 step %c256 {
       gpu.barrier
    }
  }
  // The inner loop doesn't always execute once so it cannot be removed.
  //     CHECK: scf.for %{{.*}} = %{{.*}} to %[[C250]] step %[[C250]] {
  //     CHECK:   gpu.barrier
  scf.for %arg3 = %tidy to %c2 step %c2 {
    %0 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%tidx]
    scf.for %arg4 = %0 to %c250 step %c250 {
       gpu.barrier
    }
  }
  return
}
