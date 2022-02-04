// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(builtin.module(builtin.func(iree-codegen-remove-single-iteration-loop))))' %s | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

// CHECK-LABEL: func @dispatch_0()
hal.executable private @dispatch_0  {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @dispatch_0 layout(#executable_layout) attributes {
      workgroup_size = [64: index, 1: index, 1:index]
    }
    builtin.module {
      builtin.func @dispatch_0() {
        %c2 = arith.constant 2 : index
        %c256 = arith.constant 256 : index
        //     CHECK: %[[C250:.+]] = arith.constant 250 : index
        %c250 = arith.constant 250 : index
        %tidx = gpu.thread_id x
        %tidy = gpu.thread_id y
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
    }
  }
}

// -----

#executable_layout = #hal.executable.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

// CHECK-LABEL: func @workgroup_tile_loop()
#translation = #iree_codegen.translation.info<"LLVMGPUDistribute", workload_per_wg = [32]>
hal.executable private @workgroup_tile_loop  {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @workgroup_tile_loop layout(#executable_layout) attributes {
      translation.info = #translation
    }
    builtin.module {
      builtin.func @workgroup_tile_loop() {
        %c2048 = arith.constant 2048 : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %idx = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
        %countx = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
        // CHECK-NOT: scf.for
        //     CHECK: gpu.barrier
        scf.for %arg0 = %idx to %c2048 step %countx {
          gpu.barrier
        }
        return
      }
    }
  }
}

// -----

#executable_layout = #hal.executable.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

// CHECK-LABEL: func @workgroup_tile_loop_negative()
#translation = #iree_codegen.translation.info<"LLVMGPUDistribute", workload_per_wg = [16]>
hal.executable private @workgroup_tile_loop_negative  {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @workgroup_tile_loop_negative layout(#executable_layout) attributes {
      translation.info = #translation
    }
    builtin.module {
      builtin.func @workgroup_tile_loop_negative() {
        %c2048 = arith.constant 2048 : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %idx = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
        %countx = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
        //     CHECK: scf.for
        //     CHECK: gpu.barrier
        scf.for %arg0 = %idx to %c2048 step %countx {
          gpu.barrier
        }
        return
      }
    }
  }
}

// -----

#executable_layout = #hal.executable.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

// CHECK-LABEL: func @both_workgroup_and_workitem()
//   CHECK-NOT:   scf.for
//       CHECK:   gpu.barrier
#translation = #iree_codegen.translation.info<"LLVMGPUDistribute", workload_per_wg = [32, 8, 1]>
hal.executable private @both_workgroup_and_workitem  {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @both_workgroup_and_workitem layout(#executable_layout) attributes {
      translation.info = #translation,
      workgroup_size = [8: index, 2: index, 1: index]
    }
    builtin.module {
      builtin.func @both_workgroup_and_workitem() {
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c112 = arith.constant 112 : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c112 step %workgroup_count_z {
          %4 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_id_y]
          %5 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%workgroup_count_y]
          scf.for %arg1 = %4 to %c112 step %5 {
            %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
            %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
            scf.for %arg2 = %6 to %c32 step %7 {

              // Additional loops distributed to workitems.
              %18 = gpu.thread_id y
              %19 = gpu.block_dim y
              %20 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%18]
              %21 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%19]
              scf.for %arg3 = %20 to %c8 step %21 {
                %22 = gpu.thread_id x
                %23 = gpu.block_dim x
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
    }
  }
}
