// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-remove-single-iteration-loop)))))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

// CHECK-LABEL: func.func @dispatch_0()
hal.executable private @dispatch_0  {
  hal.executable.variant @cuda target(#hal.executable.target<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @dispatch_0 layout(#pipeline_layout) attributes {
      workgroup_size = [64: index, 1: index, 1:index]
    } {
    ^bb0(%arg0: !hal.device) :
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @dispatch_0() {
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

// CHECK-LABEL: func.func @workgroup_tile_loop()
#translation = #iree_codegen.translation_info<LLVMGPUDistribute>
hal.executable private @workgroup_tile_loop  {
  hal.executable.variant @cuda target(#hal.executable.target<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @workgroup_tile_loop layout(#pipeline_layout) attributes {
      translation_info = #translation
    } {
    ^bb0(%arg0 : !hal.device, %arg1 : index):
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      hal.return %c64, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @workgroup_tile_loop() {
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

// CHECK-LABEL: func.func @workgroup_tile_loop_negative()
#translation = #iree_codegen.translation_info<LLVMGPUDistribute>
hal.executable private @workgroup_tile_loop_negative  {
  hal.executable.variant @cuda target(#hal.executable.target<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @workgroup_tile_loop_negative layout(#pipeline_layout) attributes {
      translation_info = #translation
    } {
    ^bb0(%arg0: !hal.device, %arg1 : index):
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<(d0) -> (d0 ceildiv 16)>(%arg1)
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @workgroup_tile_loop_negative() {
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

// CHECK-LABEL: func.func @both_workgroup_and_workitem()
//   CHECK-NOT:   scf.for
//       CHECK:   gpu.barrier
#translation = #iree_codegen.translation_info<LLVMGPUDistribute>
hal.executable private @both_workgroup_and_workitem  {
  hal.executable.variant @cuda target(#hal.executable.target<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @both_workgroup_and_workitem layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [8: index, 2: index, 1: index]
    } {
    ^bb0(%arg0 : !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %c112 = arith.constant 112: index
      hal.return %c1, %c14, %c112 : index, index, index
    }
    builtin.module {
      func.func @both_workgroup_and_workitem() {
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

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[4], [4], [0]]>
#device_target_cpu = #hal.device.target<"llvm-cpu", {executable_targets = [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>]}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>, #hal.descriptor_set.binding<1, storage_buffer>, #hal.descriptor_set.binding<2, storage_buffer>]>]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#map0 = affine_map<()[s0] -> (s0 ceildiv 4)>
#map1 = affine_map<()[s0] -> (s0 * 4)>
#map2 = affine_map<()[s0, s1] -> (-((s0 * -4 + 4) mod (s1 * 4)) + 4)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {hal.device.targets = [#device_target_cpu]} {
  hal.executable private @simple_mul {
    hal.executable.variant public @embedded_elf_x86_64 target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>) {
      hal.executable.export public @simple_mul ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
        %c1 = arith.constant 1 : index
        %0 = affine.apply #map0()[%arg1]
        hal.return %0, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @simple_mul() {
          %cst = arith.constant 0.000000e+00 : f32
          %c4 = arith.constant 4 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<4xf32>
          memref.assume_alignment %0, 64 : memref<4xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<4xf32>
          memref.assume_alignment %1, 64 : memref<4xf32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<4xf32>
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
}

// CHECK-LABEL: func.func @simple_mul
// CHECK:         scf.for
// CHECK:         scf.for
