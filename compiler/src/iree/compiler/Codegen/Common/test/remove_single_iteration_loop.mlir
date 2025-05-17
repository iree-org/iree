// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-codegen-remove-single-iteration-loop))' %s | FileCheck %s

// CHECK-LABEL: func.func @thread_tile_loop()
func.func @thread_tile_loop() {
  %c2 = arith.constant 2 : index
  %c256 = arith.constant 256 : index
  //     CHECK: %[[C250:.+]] = arith.constant 250 : index
  %c250 = arith.constant 250 : index
  %tidx = gpu.thread_id x upper_bound 64
  %tidy = gpu.thread_id y upper_bound 1
  // CHECK-NOT: scf.for
  //     CHECK: gpu.barrier
  scf.for %arg3 = %tidy to %c2 step %c2 {
    %0 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%tidx]
    scf.for %arg4 = %0 to %c256 step %c256 {
       gpu.barrier
    }
  }
  // The inner loop doesn't always execute once so it cannot be removed.
  //     CHECK: scf.for %{{.*}} = %{{.*}} to %[[C250]] step %[[C250]]
  //     CHECK:   gpu.barrier
  scf.for %arg3 = %tidy to %c2 step %c2 {
    %0 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%tidx]
    scf.for %arg4 = %0 to %c250 step %c250 {
       gpu.barrier
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @workgroup_tile_loop()
func.func @workgroup_tile_loop() {
  %c2048 = arith.constant 2048 : index
  %workgroup_id_x = hal.interface.workgroup.id[0] upper_bound 64 : index
  %workgroup_count_x = arith.constant 64 : index
  %idx = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
  %countx = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
  // CHECK-NOT: scf.for
  //     CHECK: gpu.barrier
  scf.for %arg0 = %idx to %c2048 step %countx {
    gpu.barrier
  }
  return
}

// -----

// CHECK-LABEL: func.func @workgroup_tile_loop_negative()
func.func @workgroup_tile_loop_negative() {
  %c2048 = arith.constant 2048 : index
  %workgroup_id_x = hal.interface.workgroup.id[0] upper_bound 2147483647 : index
  %workgroup_count_x = hal.interface.workgroup.count[0] upper_bound 2147483647 : index
  %idx = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
  %countx = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
  //     CHECK: scf.for
  //     CHECK: gpu.barrier
  scf.for %arg0 = %idx to %c2048 step %countx {
    gpu.barrier
  }
  return
}

// -----

// CHECK-LABEL: func.func @both_workgroup_and_workitem()
//   CHECK-NOT:   scf.for
//       CHECK:   gpu.barrier
func.func @both_workgroup_and_workitem() {
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c112 = arith.constant 112 : index
  %workgroup_id_x = hal.interface.workgroup.id[0] upper_bound 1 : index
  // Any hal.interface.workgroup.count op in a function like this should have
  // have been -iree-codegen-propagate-dispatch-size-bounds 'd away before
  // this pass is called.
  %workgroup_count_x = arith.constant 1 : index
  %workgroup_id_y = hal.interface.workgroup.id[1] upper_bound 14 : index
  %workgroup_count_y = arith.constant 14 : index
  %workgroup_id_z = hal.interface.workgroup.id[2] upper_bound 112 : index
  %workgroup_count_z = arith.constant 112 : index
  scf.for %arg0 = %workgroup_id_z to %c112 step %workgroup_count_z {
    %4 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
    %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
    scf.for %arg1 = %4 to %c112 step %5 {
      %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
      %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
      scf.for %arg2 = %6 to %c32 step %7 {

        // Additional loops distributed to workitems.
        %18 = gpu.thread_id y upper_bound 2
        %19 = arith.constant 2 : index
        %20 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%18]
        %21 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%19]
        scf.for %arg3 = %20 to %c8 step %21 {
          %22 = gpu.thread_id x upper_bound 8
          %23 = arith.constant 8 : index
          %24 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%22]
          %25 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%23]
          scf.for %arg4 = %24 to %c32 step %25 {
            gpu.barrier
          }
        }

      }
    }
  }
  return
}

// -----

#map0 = affine_map<()[s0] -> (s0 ceildiv 4)>
#map1 = affine_map<()[s0] -> (s0 * 4)>
#map2 = affine_map<()[s0, s1] -> (-((s0 * -4 + 4) mod (s1 * 4)) + 4)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0)>
func.func @simple_mul(%0: memref<4xf32>, %1: memref<4xf32>, %2: memref<4xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %3 = affine.apply #map1()[%workgroup_id_x]
  %4 = affine.apply #map1()[%workgroup_count_x]
  %5 = affine.apply #map2()[%workgroup_id_x, %workgroup_count_x]
  scf.for %arg0 = %3 to %5 step %4 {
    %6 = memref.subview %2[%arg0] [4] [1] : memref<4xf32> to memref<4xf32, #map3>
    %7 = memref.subview %0[%arg0] [4] [1] : memref<4xf32> to memref<4xf32, #map3>
    %8 = memref.subview %1[%arg0] [4] [1] : memref<4xf32> to memref<4xf32, #map3>
    %9 = vector.transfer_read %7[%c0], %cst {in_bounds = [true]} : memref<4xf32, #map3>, vector<4xf32>
    %10 = vector.transfer_read %8[%c0], %cst {in_bounds = [true]} : memref<4xf32, #map3>, vector<4xf32>
    %11 = arith.mulf %9, %10 : vector<4xf32>
    vector.transfer_write %11, %6[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, #map3>
  }
  scf.for %arg0 = %5 to %c4 step %4 {
    %6 = memref.subview %2[%arg0] [4] [1] : memref<4xf32> to memref<4xf32, #map3>
    %7 = memref.subview %0[%arg0] [4] [1] : memref<4xf32> to memref<4xf32, #map3>
    %8 = memref.subview %1[%arg0] [4] [1] : memref<4xf32> to memref<4xf32, #map3>
    %9 = vector.transfer_read %7[%c0], %cst {in_bounds = [true]} : memref<4xf32, #map3>, vector<4xf32>
    %10 = vector.transfer_read %8[%c0], %cst {in_bounds = [true]} : memref<4xf32, #map3>, vector<4xf32>
    %11 = arith.mulf %9, %10 : vector<4xf32>
    vector.transfer_write %11, %6[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, #map3>
  }
  return
}
// CHECK-LABEL: func.func @simple_mul
// CHECK:         scf.for
// CHECK:         scf.for

// -----

// CHECK-LABEL: func.func @delinearize_linearize()
func.func @delinearize_linearize() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  // CHECK: %[[C3:.+]] = arith.constant 3 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %tidx = gpu.thread_id x upper_bound 128
  %ids:2 = affine.delinearize_index %tidx into (4, 32) : index, index
  // CHECK-NOT: scf.for
  //     CHECK: gpu.barrier
  scf.for %arg3 = %ids#0 to %c4 step %c4 {
    %0 = affine.linearize_index [%ids#1, %c0] by (32, 2) : index
    scf.for %arg4 = %0 to %c64 step %c64 {
       gpu.barrier
    }
  }
  // The loop loop doesn't always execute once so it cannot be removed.
  //     CHECK: scf.for %{{.*}} = %{{.*}} to %[[C3]] step %{{.*}}
  //     CHECK:   gpu.barrier
  scf.for %arg3 = %ids#0 to %c3 step %c4 {
    gpu.barrier
  }
  // ValueBoundsOpInterface will also work on an arith.muli
  // CHECK-NOT: scf.for
  //     CHECK: gpu.barrier
  scf.for %arg3 = %ids#0 to %c4 step %c4 {
    %0 = arith.muli %ids#1, %c2 : index
    scf.for %arg4 = %0 to %c64 step %c64 {
       gpu.barrier
    }
  }

  return
}

// -----

// Test used as a proxy for a ValueBoundsOpInterface implementation

// CHECK-LABEL: func.func @workgroup_id
func.func @workgroup_id() {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %workgroup_id_x = hal.interface.workgroup.id[0] upper_bound 8 : index
  // CHECK-NOT: scf.for
  //     CHECK: gpu.barrier
  scf.for %arg3 = %workgroup_id_x to %c8 step %c8 {
       gpu.barrier
  }
  return
}

// -----

// CHECK-LABEL: func.func @argument_with_assume
func.func @argument_with_assume(%arg_index : index) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %arg = util.assume.int %arg_index[<umin=0, umax=4>, <umin=4, umax=7>] : index
  %ordinal = iree_tensor_ext.dispatch.workload.ordinal %arg, 0 : index
  // CHECK-NOT: scf.for
  //     CHECK: gpu.barrier
  scf.for %arg3 = %ordinal to %c8 step %c8 {
       gpu.barrier
  }
  return
}
