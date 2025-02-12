// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-remove-single-iteration-loop)))))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1]>
// CHECK-LABEL: func.func @dispatch_0()
hal.executable private @dispatch_0  {
  hal.executable.variant @cuda target(#hal.executable.target<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @dispatch_0 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @dispatch_0() attributes {translation_info = #translation_info} {
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
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK-LABEL: func.func @workgroup_tile_loop()
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUDistribute workgroup_size = [32, 1, 1]>
hal.executable private @workgroup_tile_loop  {
  hal.executable.variant @cuda target(#hal.executable.target<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @workgroup_tile_loop layout(#pipeline_layout) {
    ^bb0(%arg0 : !hal.device, %arg1 : index):
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      hal.return %c64, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @workgroup_tile_loop()  attributes {translation_info = #translation}  {
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
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK-LABEL: func.func @workgroup_tile_loop_negative()
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUDistribute>
hal.executable private @workgroup_tile_loop_negative  {
  hal.executable.variant @cuda target(#hal.executable.target<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @workgroup_tile_loop_negative layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1 : index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<(d0) -> (d0 ceildiv 16)>(%arg1)
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @workgroup_tile_loop_negative() attributes {translation_info = #translation} {
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
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK-LABEL: func.func @both_workgroup_and_workitem()
//   CHECK-NOT:   scf.for
//       CHECK:   gpu.barrier
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUDistribute workgroup_size = [8, 2, 1]>
hal.executable private @both_workgroup_and_workitem  {
  hal.executable.variant @cuda target(#hal.executable.target<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @both_workgroup_and_workitem layout(#pipeline_layout) {
    ^bb0(%arg0 : !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %c112 = arith.constant 112: index
      hal.return %c1, %c14, %c112 : index, index, index
    }
    builtin.module {
      func.func @both_workgroup_and_workitem() attributes {translation_info = #translation} {
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c112 = arith.constant 112 : index
        %workgroup_id_x = hal.interface.workgroup.id[0] upper_bound 1 : index
        // Any hal.interface.workgroup.count op in a function liket his should have
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
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
#map0 = affine_map<()[s0] -> (s0 ceildiv 4)>
#map1 = affine_map<()[s0] -> (s0 * 4)>
#map2 = affine_map<()[s0, s1] -> (-((s0 * -4 + 4) mod (s1 * 4)) + 4)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0)>
hal.executable private @simple_mul {
  hal.executable.variant public @variant target(#hal.executable.target<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export public @simple_mul ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply #map0()[%arg1]
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @simple_mul() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<4xf32>
        memref.assume_alignment %0, 64 : memref<4xf32>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<4xf32>
        memref.assume_alignment %1, 64 : memref<4xf32>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<4xf32>
        memref.assume_alignment %2, 64 : memref<4xf32>
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
    }
  }
}

// CHECK-LABEL: func.func @simple_mul
// CHECK:         scf.for
// CHECK:         scf.for

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 1, 1]>
// CHECK-LABEL: func.func @delinearize_linearize()
hal.executable private @delinearize_linearize {
  hal.executable.variant @rocm_hsaco_fb target(#hal.executable.target<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @delinearize_linearize layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @delinearize_linearize() attributes {translation_info = #translation_info} {
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
    }
  }
}

// -----

// Test used as a proxy for a ValueBoundsOpInterface implementation

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 1, 1]>
// CHECK-LABEL: func.func @workgroup_id
hal.executable private @workgroup_id {
  hal.executable.variant @rocm_hsaco_fb target(#hal.executable.target<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @workgroup_id layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c8 = arith.constant 8 : index
      hal.return %c8, %c8, %c8 : index, index, index
    }
    builtin.module {
      func.func @workgroup_id() attributes {translation_info = #translation_info} {
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
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 1, 1]>
// CHECK-LABEL: func.func @argument_with_assume
hal.executable private @argument_with_assume {
  hal.executable.variant @rocm_hsaco_fb target(#hal.executable.target<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @workgroup_id layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @argument_with_assume() attributes {translation_info = #translation_info} {
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %arg_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %arg_index = arith.index_cast %arg_i32 : i32 to index
        %arg = util.assume.int %arg_index[<umin=0, umax=4>, <umin=4, umax=7>] : index
        // CHECK-NOT: scf.for
        //     CHECK: gpu.barrier
        scf.for %arg3 = %arg to %c8 step %c8 {
             gpu.barrier
        }
        return
      }
    }
  }
}
