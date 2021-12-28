// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true}))' -cse -canonicalize -split-input-file %s | IreeFileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors  {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-unknown-unknown-eabi-elf"
  }> {
    hal.executable.entry_point @matmul_tensors layout(#executable_layout)
    builtin.module {
      func @matmul_tensors() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %M = hal.interface.constant.load[0] : index
        %N = hal.interface.constant.load[1] : index
        %K = hal.interface.constant.load[2] : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:?x?xf32>{%M, %K}
        %2 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:?x?xf32>{%K, %N}
        %4 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) : !flow.dispatch.tensor<readonly:?x?xf32>{%M, %N}
        %6 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(3) : !flow.dispatch.tensor<writeonly:?x?xf32>{%M, %N}
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_size_y, %workgroup_id_y]
        %9 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_size_y, %workgroup_count_y]
        scf.for %arg0 = %8 to %M step %9 {
          %10 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_size_x, %workgroup_id_x]
          %11 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_size_x, %workgroup_count_x]
          scf.for %arg1 = %10 to %N step %11 {
            %12 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %N]
            %13 = flow.dispatch.tensor.load %0, offsets=[%arg0, 0], sizes=[%12, %K], strides=[1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
            %14 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %M]
            %15 = flow.dispatch.tensor.load %2, offsets=[0, %arg1], sizes=[%K, %14], strides=[1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
            %16 = flow.dispatch.tensor.load %4, offsets=[%arg0, %arg1], sizes=[%12, %14], strides=[1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
            %17 = linalg.matmul ins(%13, %15 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
            flow.dispatch.tensor.store %17, %6, offsets=[%arg0, %arg1], sizes=[%12, %14], strides=[1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[], [16, 4, 64], [4, 4, 4]{{\]}}, native_vector_size = [4, 4, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUTileFuseAndVectorize", workload_per_wg = [64, 64]>
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//      CHECK: hal.executable.entry_point public @matmul_tensors
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//      CHECK:    hal.return %[[D0]], %[[D1]], %[[C1]] : index, index, index
//      CHECK: linalg.matmul
// CHECK-SAME:   lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @add_no_config  {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.entry_point @add_no_config layout(#executable_layout)
    builtin.module {
      func @add_no_config() {
        %c0 = arith.constant 0 : index
        %dim0 = hal.interface.constant.load[0] : index
        %dim1 = hal.interface.constant.load[1] : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:?x?xf32>{%dim0, %dim1}
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:?xf32>{%dim1}
        %2 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:?x?xf32>{%dim0, %dim1}
        %3 = flow.dispatch.tensor.load %0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
        %4 = flow.dispatch.tensor.load %1, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:?xf32> -> tensor<?xf32>
        %5 = linalg.init_tensor [%dim0, %dim1] : tensor<?x?xf32>
        %6 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%3, %4 : tensor<?x?xf32>, tensor<?xf32>) outs(%5 : tensor<?x?xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [], sizes = [], strides = [] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
        return
      }
    }
  }
}

//      CHECK:  #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = []>
//      CHECK:  hal.executable private @add_no_config
//      CHECK:  hal.executable.entry_point public @add_no_config
// CHECK-SAME:      translation.info = #[[TRANSLATION]]

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @add4D  {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.entry_point @add4D layout(#executable_layout)
    builtin.module {
      func @add4D() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.constant.load[2] : index
        %3 = hal.interface.constant.load[3] : index
        %4 = hal.interface.constant.load[4] : index
        %5 = hal.interface.constant.load[5] : index
        %6 = hal.interface.constant.load[6] : index
        %7 = hal.interface.constant.load[7] : index
        %8 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) alignment(32) : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%0, %1, %2, %3}
        %9 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) alignment(32) : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%4, %5, %6, %7}
        %10 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) alignment(32) : !flow.dispatch.tensor<writeonly:?x?x?x?xf32>{%0, %1, %2, %3}
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %11 = affine.apply #map0()[%workgroup_id_z, %workgroup_size_z]
        %12 = affine.apply #map0()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %11 to %1 step %12 {
          %13 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
          %14 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %13 to %2 step %14 {
            %15 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
            %16 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %15 to %3 step %16 {
              %17 = affine.min #map1(%arg0)[%1, %workgroup_size_z]
              %18 = affine.min #map1(%arg1)[%2, %workgroup_size_y]
              %19 = affine.min #map1(%arg2)[%3, %workgroup_size_x]
              %20 = flow.dispatch.tensor.load %8, offsets = [0, %arg0, %arg1, %arg2], sizes = [%0, %17, %18, %19], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%0, %1, %2, %3} -> tensor<?x?x?x?xf32>
              %21 = flow.dispatch.tensor.load %9, offsets = [0, %arg0, %arg1, %arg2], sizes = [%4, %17, %18, %19], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%4, %5, %6, %7} -> tensor<?x?x?x?xf32>
              %22 = linalg.init_tensor [%0, %17, %18, %19] : tensor<?x?x?x?xf32>
              %23 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %21 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%22 : tensor<?x?x?x?xf32>) {
              ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
                %24 = arith.addf %arg3, %arg4 : f32
                linalg.yield %24 : f32
              } -> tensor<?x?x?x?xf32>
              flow.dispatch.tensor.store %23, %10, offsets = [0, %arg0, %arg1, %arg2], sizes = [%0, %17, %18, %19], strides = [1, 1, 1, 1] : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?x?x?xf32>{%0, %1, %2, %3}
            }
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64, 64, 64]>
//      CHECK: hal.executable.entry_point public @add4D
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   (%[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]+]]: index)
//  CHECK-DAG:    %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:    %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//  CHECK-DAG:    %[[D2:.+]] = affine.apply #[[MAP0]]()[%[[ARG2]]]
//      CHECK:    hal.return %[[D0]], %[[D1]], %[[D2]] : index, index, index
//      CHECK: linalg.generic

// -----

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @batch_matmul_tensors {
  hal.executable.variant @llvm, target = <"llvm", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-unknown-unknown-eabi-elf"
  }> {
    hal.executable.entry_point @batch_matmul_tensors layout(#executable_layout)
    builtin.module {
      func @batch_matmul_tensors() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.constant.load[2] : index
        %3 = hal.interface.constant.load[3] : index
        %4 = hal.interface.constant.load[4] : index
        %5 = hal.interface.constant.load[5] : index
        %6 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) alignment(32) : !flow.dispatch.tensor<readonly:?x?x?xf32>{%0, %1, %2}
        %7 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) alignment(32) : !flow.dispatch.tensor<readonly:?x?x?xf32>{%3, %4, %5}
        %8 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) alignment(32) : !flow.dispatch.tensor<writeonly:?x?x?xf32>{%0, %1, %5}
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %9 = affine.apply #map0()[%workgroup_id_z, %workgroup_size_z]
        %10 = affine.apply #map0()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %9 to %0 step %10 {
          %11 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
          %12 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %11 to %1 step %12 {
            %13 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
            %14 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %13 to %5 step %14 {
              %15 = affine.min #map1(%arg0)[%0, %workgroup_size_z]
              %16 = affine.min #map1(%arg1)[%1, %workgroup_size_y]
              %17 = flow.dispatch.tensor.load %6, offsets = [%arg0, %arg1, 0], sizes = [%15, %16, %2], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:?x?x?xf32>{%0, %1, %2} -> tensor<?x?x?xf32>
              %18 = affine.min #map1(%arg2)[%5, %workgroup_size_x]
              %19 = flow.dispatch.tensor.load %7, offsets = [%arg0, 0, %arg2], sizes = [%15, %4, %18], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:?x?x?xf32>{%3, %4, %5} -> tensor<?x?x?xf32>
              %20 = affine.min #map2(%arg0)[%0, %workgroup_size_z]
              %21 = affine.min #map2(%arg1)[%1, %workgroup_size_y]
              %22 = affine.min #map2(%arg2)[%5, %workgroup_size_x]
              %23 = linalg.init_tensor [%20, %21, %22] : tensor<?x?x?xf32>
              %24 = linalg.fill(%cst, %23) : f32, tensor<?x?x?xf32> -> tensor<?x?x?xf32>
              %25 = linalg.batch_matmul ins(%17, %19 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%24 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
              flow.dispatch.tensor.store %25, %8, offsets = [%arg0, %arg1, %arg2], sizes = [%15, %16, %18], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?x?xf32>{%0, %1, %5}
            }
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[], [1, 16, 4, 64], [1, 4, 4, 4]{{\]}}, native_vector_size = [1, 4, 4, 4]>
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUTileFuseAndVectorize", workload_per_wg = [64, 64, 1]>
//      CHECK: hal.executable.entry_point public @batch_matmul_tensors
// CHECK-NEXT: (%[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: index)
//  CHECK-DAG:  %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:  %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//      CHECK:  hal.return %[[D0]], %[[D1]], %[[ARG2]]
//      CHECK:  linalg.batch_matmul
// CHECK-SAME:    lowering.config = #[[CONFIG]]

// -----

#compilation = #iree_codegen.compilation.info<
    #iree_codegen.lowering.config<tile_sizes = [[], [32, 32, 32], [4, 4, 4]], native_vector_size = [4, 4, 4]>,
    #iree_codegen.translation.info<"CPUTensorToVectors", workload_per_wg = [32, 32]>,
    workgroup_size = []>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_matmul_tensors  {
  hal.executable.variant @system_elf_x86_64, target = <"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @preset_config layout(#executable_layout)
    builtin.module {
      builtin.func @preset_config() {
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c128 = arith.constant 128 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:128x256xf32>
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:256x512xf32>
        %2 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:128x512xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c128 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c512 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 128)>(%arg0)[%workgroup_size_y]
            %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x256xf32> -> tensor<?x256xf32>
            %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 512)>(%arg1)[%workgroup_size_x]
            %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [256, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x512xf32> -> tensor<256x?xf32>
            %11 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 128)>(%arg0)[%workgroup_size_y]
            %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 512)>(%arg1)[%workgroup_size_x]
            %13 = affine.min affine_map<(d0)[s0] -> (-d0 + 128, s0)>(%arg0)[%workgroup_size_y]
            %14 = affine.min affine_map<(d0)[s0] -> (-d0 + 512, s0)>(%arg1)[%workgroup_size_x]
            %15 = linalg.init_tensor [%13, %14] : tensor<?x?xf32>
            %16 = linalg.fill(%cst, %15) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
            %17 = linalg.matmul {
                 __internal_linalg_transform__ = "workgroup",
                 compilation.info = #compilation}
                 ins(%8, %10 : tensor<?x256xf32>, tensor<256x?xf32>)
                 outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
            flow.dispatch.tensor.store %17, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:128x512xf32>
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[], [32, 32, 32], [4, 4, 4]{{\]}}, native_vector_size = [4, 4, 4]>
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 32)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUTensorToVectors", workload_per_wg = [32, 32]>
//      CHECK: hal.executable.entry_point
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:     %[[NWG_X:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:     %[[NWG_Y:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//      CHECK:     return %[[NWG_X]], %[[NWG_Y]], %[[C1]]
//      CHECK: builtin.module
//      CHECK:   func @preset_config
//  CHECK-DAG:     %[[WGID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:     %[[WGCOUNT_X:.+]] = hal.interface.workgroup.count[0]
//  CHECK-DAG:     %[[WGID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:     %[[WGCOUNT_Y:.+]] = hal.interface.workgroup.count[1]
//      CHECK:     %[[LB_Y:.+]] = affine.apply #[[MAP1]]()[%[[WGID_Y]]]
//      CHECK:     %[[STEP_Y:.+]] = affine.apply #[[MAP1]]()[%[[WGCOUNT_Y]]]
//      CHECK:     scf.for %[[IV0:.+]] = %[[LB_Y]] to %{{.+}} step %[[STEP_Y]]
//      CHECK:       %[[LB_X:.+]] = affine.apply #[[MAP1]]()[%[[WGID_X]]]
//      CHECK:       %[[STEP_X:.+]] = affine.apply #[[MAP1]]()[%[[WGCOUNT_X]]]
//      CHECK:       scf.for %[[IV1:.+]] = %[[LB_X]] to %{{.+}} step %[[STEP_X]]
//      CHECK:         linalg.matmul
// CHECK-SAME:             lowering.config = #[[CONFIG]]
// CHECK-SAME:             ins(%{{.+}}, %{{.+}} : tensor<32x256xf32>, tensor<256x32xf32>)
// CHECK-SAME:             outs(%{{.+}} : tensor<32x32xf32>)

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @tensor_insert {
  hal.executable.variant @system_elf_x86_64, target = <"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @tensor_insert_slice layout(#executable_layout)
    builtin.module {
      builtin.func @tensor_insert_slice() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:?x?xi32>
        %1 = hal.interface.constant.load[0] : index
        %2 = hal.interface.constant.load[1] : index
        %3 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<writeonly:?x?xi32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %4 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_id_y]
        %5 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_y, %workgroup_count_y]
        %d0 = hal.interface.constant.load[2] : index
        %d1 = hal.interface.constant.load[2] : index
        scf.for %arg0 = %4 to %d0 step %5 {
          %6 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg0)[%workgroup_size_y, %d0]
          %7 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_id_x]
          %8 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_x, %workgroup_count_x]
          scf.for %arg1 = %7 to %d1 step %8 {
            %9 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg1)[%workgroup_size_x, %d1]
            %10 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1], sizes = [%6, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xi32> -> tensor<?x?xi32>
            %11 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg0)[%1]
            %12 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg1)[%2]
            flow.dispatch.tensor.store %10, %3, offsets = [%11, %12], sizes = [%6, %9], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:?x?xi32>
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64, 64]>
//      CHECK: hal.executable.entry_point public @tensor_insert_slice
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[NWGSX:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-DAG:   %[[NWGSY:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//      CHECK:   hal.return %[[NWGSX]], %[[NWGSY]], %[[C1]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_1d_fft_stage2  {
  hal.executable.variant @system_elf_x86_64, target = <"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @static_1d_fft_stage2 layout(#executable_layout)
    builtin.module {
      builtin.func @static_1d_fft_stage2() {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readwrite:32xf32>
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readwrite:32xf32>
        %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:32xf32> -> tensor<32xf32>
        %3 = flow.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:32xf32> -> tensor<32xf32>
        %4:2 = iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
        flow.dispatch.tensor.store %4#0, %0, offsets = [], sizes = [], strides = [] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:32xf32>
        flow.dispatch.tensor.store %4#1, %1, offsets = [], sizes = [], strides = [] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:32xf32>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[64]{{\]}}, native_vector_size = []>
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64]>
//       CHECK: hal.executable.entry_point public @static_1d_fft_stage2
//  CHECK-SAME:   translation.info = #[[TRANSLATION]]
//  CHECK-NEXT: ^{{.+}}(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index):
//  CHECK-NEXT:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-NEXT:   %[[T0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-NEXT:   hal.return %[[T0]], %[[C1]], %[[C1]]

//       CHECK: func @static_1d_fft_stage2()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_3d_fft_stage3  {
  hal.executable.variant @system_elf_x86_64, target = <"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @static_3d_fft_stage3 layout(#executable_layout)
    builtin.module {
      builtin.func @static_3d_fft_stage3() {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant dense<[1.000000e+00, 0.707106769, 6.12323426E-17, -0.707106769]> : tensor<4xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -0.707106769, -1.000000e+00, -0.707106769]> : tensor<4xf32>
        %0 = bufferization.to_memref %cst_0 : memref<4xf32>
        %1 = bufferization.to_memref %cst : memref<4xf32>
        %2 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : memref<64x128x32xf32>
        %3 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : memref<64x128x32xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %lb_z = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %step_z = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %workgroup_id_z to %c64 step %workgroup_count_z {
          %lb_y = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %step_y = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %workgroup_id_y to %c128 step %workgroup_count_y {
            %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %4 to %c32 step %5 {
              %6 = memref.subview %2[%arg0, %arg1, %arg2] [1, 1, 4] [1, 1, 1] : memref<64x128x32xf32> to memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>
              %7 = memref.cast %6 : memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>
              %8 = memref.subview %3[%arg0, %arg1, %arg2] [1, 1, 4] [1, 1, 1] : memref<64x128x32xf32> to memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>
              %9 = memref.cast %8 : memref<1x1x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>
              iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"}
                ins(%c3, %1, %0 : index, memref<4xf32>, memref<4xf32>)
                outs(%7, %9 : memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>, memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 4096 + s0 + d1 * 32 + d2)>>)
            }
          }
        }
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[64, 64, 64]{{\]}}, native_vector_size = []>
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64, 64, 64]>
//       CHECK: hal.executable.entry_point public @static_3d_fft_stage3
//  CHECK-SAME:   translation.info = #[[TRANSLATION]]
//  CHECK-NEXT: ^{{.+}}(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index):
//  CHECK-NEXT:   %[[T0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//  CHECK-NEXT:   %[[T1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//  CHECK-NEXT:   %[[T2:.+]] = affine.apply #[[MAP0]]()[%[[ARG2]]]
//  CHECK-NEXT:   hal.return %[[T0]], %[[T1]], %[[T2]]

//       CHECK: func @static_3d_fft_stage3()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @outs_fusion {
  hal.executable.variant @system_elf_x86_64, target = <"llvm", "system-elf-x86_64"> {
    hal.executable.entry_point @outs_fusion_fn layout(#executable_layout)
    builtin.module {
      builtin.func @outs_fusion_fn() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.0 : f32
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:?x?xf32>
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:?x?xf32>
        %2 = hal.interface.constant.load[0] : index
        %3 = hal.interface.constant.load[1] : index
        %4 = hal.interface.constant.load[2] : index
        %5 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:?x?xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %lb_y = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%workgroup_id_y)[%workgroup_size_y]
        %step_y = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%workgroup_count_y)[%workgroup_size_y]
        %lb_x = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%workgroup_id_x)[%workgroup_size_x]
        %step_x = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%workgroup_count_x)[%workgroup_size_x]
        scf.for %iv0 = %lb_y to %2 step %step_y {
          scf.for %iv1 = %lb_x to %3 step %step_x {
            %tile_m = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%iv0)[%workgroup_size_y, %2]
            %tile_n = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%iv1)[%workgroup_size_x, %3]
            %init = linalg.init_tensor[%tile_m, %tile_n] : tensor<?x?xf32>
            %fill = linalg.generic {
                indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
                iterator_types = ["parallel", "parallel"]}
                outs(%init : tensor<?x?xf32>) {
                ^bb0(%arg0: f32):
                  linalg.yield %cst : f32
                } -> tensor<?x?xf32>
            %lhs = flow.dispatch.tensor.load %0, offsets = [%iv0, 0], sizes = [%tile_m, %4], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
            %rhs = flow.dispatch.tensor.load %0, offsets = [0, %iv1], sizes = [%4, %tile_n], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32> -> tensor<?x?xf32>
            %gemm = linalg.generic {
                indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                 affine_map<(d0, d1, d2) -> (d2, d1)>,
                                 affine_map<(d0, d1, d2) -> (d0, d1)>],
                iterator_types = ["parallel", "parallel", "reduction"]}
                ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
                outs(%fill : tensor<?x?xf32>) {
                ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
                  %6 = arith.mulf %arg0, %arg1 : f32
                  %7 = arith.addf %6, %arg2 : f32
                  linalg.yield %6 : f32
                } -> tensor<?x?xf32>
            flow.dispatch.tensor.store %gemm, %5, offsets = [%iv0, %iv1], sizes = [%tile_m, %tile_n], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>
          }
        }
        return
      }
    }
  }
}

//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64, 64]>
//      CHECK: hal.executable.entry_point public @outs_fusion_fn
// CHECK-SAME:   translation.info = #[[TRANSLATION]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv {
  hal.executable.variant public @system_elf_x86_64, target = <"llvm", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }> {
    hal.executable.entry_point public @conv layout(#executable_layout)
    builtin.module {
      func @conv() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : index
        %1 = hal.interface.constant.load[1] : index
        %2 = hal.interface.constant.load[2] : index
        %3 = hal.interface.constant.load[3] : index
        %4 = hal.interface.constant.load[4] : index
        %5 = hal.interface.constant.load[5] : index
        %6 = hal.interface.constant.load[6] : index
        %7 = hal.interface.constant.load[7] : index
        %8 = hal.interface.constant.load[8] : index
        %9 = hal.interface.constant.load[9] : index
        %10 = hal.interface.constant.load[10] : index
        %11 = hal.interface.constant.load[11] : index
        %12 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%0, %1, %2, %3}
        %13 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readwrite:?x?x?x?xf32>{%4, %5, %6, %7}
        %14 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) : !flow.dispatch.tensor<readonly:?x?x?x?xf32>{%8, %9, %10, %11}
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %15 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %16 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %15 to %5 step %16 {
          %17 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %18 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %17 to %6 step %18 {
            %19 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %20 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %19 to %3 step %20 {
              %21 = affine.min affine_map<(d0)[s0, s1, s2] -> (s0 + s2 - 1, -d0 + s0 + s1)>(%arg0)[%0, %5, %workgroup_size_z]
              %22 = affine.min affine_map<(d0)[s0, s1, s2] -> (s0 + s2 - 1, -d0 + s0 + s1)>(%arg1)[%1, %6, %workgroup_size_y]
              %23 = flow.dispatch.tensor.load %14, offsets = [0, %arg0, %arg1, 0], sizes = [%8, %21, %22, %11], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:?x?x?x?xf32> -> tensor<?x?x?x?xf32>
              %24 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg2)[%3, %workgroup_size_x]
              %25 = flow.dispatch.tensor.load %12, offsets = [0, 0, 0, %arg2], sizes = [%0, %1, %2, %24], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:?x?x?x?xf32> -> tensor<?x?x?x?xf32>
              %26 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg0)[%5, %workgroup_size_z]
              %27 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg1)[%6, %workgroup_size_y]
              %28 = affine.min affine_map<(d0)[s0, s1] -> (s1, -d0 + s0)>(%arg2)[%3, %workgroup_size_x]
              %29 = flow.dispatch.tensor.load %13, offsets = [0, %arg0, %arg1, %arg2], sizes = [%4, %26, %27, %28], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:?x?x?x?xf32> -> tensor<?x?x?x?xf32>
              %30 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%23, %25 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%29 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
              flow.dispatch.tensor.store %30, %13, offsets = [0, %arg0, %arg1, %arg2], sizes = [%4, %26, %27, %28], strides = [1, 1, 1, 1] : tensor<?x?x?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?x?x?xf32>
            }
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64, 64, 64]>
//      CHECK: hal.executable.entry_point public @conv
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index, %[[ARG2:[a-zA-Z0-9]+]]: index)
//  CHECK-DAG:     %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]
//  CHECK-DAG:     %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]
//  CHECK-DAG:     %[[D2:.+]] = affine.apply #[[MAP0]]()[%[[ARG2]]
//      CHECK:     hal.return %[[D0]], %[[D1]], %[[D2]]
//      CHECK:     linalg.conv_2d_nhwc_hwcf
//  CHECK-NOT:       lowering.config

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv_static {
  hal.executable.variant public @system_elf_x86_64, target = <"llvm", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 64 : index,
    target_triple = "x86_64-pc-linux-gnu"
  }> {
    hal.executable.entry_point public @conv_static layout(#executable_layout)
    builtin.module {
      func @conv_static() {
        %cst = arith.constant 0.000000e+00 : f32
        %c80 = arith.constant 80 : index
        %c96 = arith.constant 96 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:1x161x161x96xf32>
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:3x3x96xf32>
        %2 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:1x80x80x96xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %3 to %c80 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c80 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c96 step %8 {
              %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %10 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 163)>(%arg0)[%workgroup_size_z]
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 163)>(%arg1)[%workgroup_size_y]
              %13 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 96)>(%arg2)[%workgroup_size_x]
              %14 = flow.dispatch.tensor.load %0, offsets = [0, %9, %11, %arg2], sizes = [1, %10, %12, %13], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x161x161x96xf32> -> tensor<1x?x?x?xf32>
              %15 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 96)>(%arg2)[%workgroup_size_x]
              %16 = flow.dispatch.tensor.load %1, offsets = [0, 0, %arg2], sizes = [3, 3, %15], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x96xf32> -> tensor<3x3x?xf32>
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 80)>(%arg0)[%workgroup_size_z]
              %18 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 80)>(%arg1)[%workgroup_size_y]
              %19 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 96)>(%arg2)[%workgroup_size_x]
              %20 = affine.min affine_map<(d0)[s0] -> (-d0 + 80, s0)>(%arg0)[%workgroup_size_z]
              %21 = affine.min affine_map<(d0)[s0] -> (-d0 + 80, s0)>(%arg1)[%workgroup_size_y]
              %22 = affine.min affine_map<(d0)[s0] -> (-d0 + 96, s0)>(%arg2)[%workgroup_size_x]
              %23 = linalg.init_tensor [1, %20, %21, %22] : tensor<1x?x?x?xf32>
              %24 = linalg.fill(%cst, %23) {__internal_linalg_transform__ = "workgroup"} : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
              %25 = linalg.depthwise_conv_2d_nhwc_hwc {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%14, %16 : tensor<1x?x?x?xf32>, tensor<3x3x?xf32>) outs(%24 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %25, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %17, %18, %19], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x80x80x96xf32>
            }
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64, 64, 32]>
//      CHECK: hal.executable.entry_point public @conv_static
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index, %[[ARG2:[a-zA-Z0-9]+]]: index)
//  CHECK-DAG:     %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]
//  CHECK-DAG:     %[[D1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]
//  CHECK-DAG:     %[[D2:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]]
//      CHECK:     hal.return %[[D0]], %[[D1]], %[[D2]]
//      CHECK:     linalg.depthwise_conv_2d_nhwc_hwc
//  CHECK-NOT:       lowering.config

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @generic_static {
  hal.executable.variant public @system_elf_x86_64, target = <"llvm", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 64 : index,
    target_triple = "x86_64-pc-linux-gnu"
  }> {
    hal.executable.entry_point public @generic_static layout(#executable_layout)
    builtin.module {
      func @generic_static() {
        %c16 = arith.constant 16 : index
        %c96 = arith.constant 96 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:96x16xf32>
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<writeonly:16x96xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %2 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %2 to %c16 step %3 {
          %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %4 to %c96 step %5 {
            %6 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 96)>(%arg1)[%workgroup_size_x]
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 16)>(%arg0)[%workgroup_size_y]
            %8 = flow.dispatch.tensor.load %0, offsets = [%arg1, %arg0], sizes = [%6, %7], strides = [1, 1] : !flow.dispatch.tensor<readonly:96x16xf32> -> tensor<?x?xf32>
            %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 16)>(%arg0)[%workgroup_size_y]
            %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 96)>(%arg1)[%workgroup_size_x]
            %11 = linalg.init_tensor [%9, %10] : tensor<?x?xf32>
            %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<?x?xf32>) outs(%11 : tensor<?x?xf32>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
            ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
              linalg.yield %arg2 : f32
            } -> tensor<?x?xf32>
            flow.dispatch.tensor.store %12, %1, offsets = [%arg0, %arg1], sizes = [%9, %10], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:16x96xf32>
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [32, 8]>
//      CHECK: hal.executable.entry_point public @generic_static
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index, %[[ARG2:[a-zA-Z0-9]+]]: index)
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:     %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]
//  CHECK-DAG:     %[[D1:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]]
//      CHECK:     hal.return %[[D0]], %[[D1]], %[[C1]]
//      CHECK:     linalg.generic
//  CHECK-NOT:       lowering.config

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_static {
  hal.executable.variant public @system_elf_arm_64, target = <"llvm", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }> {
    hal.executable.entry_point public @matmul_static layout(#executable_layout)
    builtin.module {
      func @matmul_static() {
        %cst = arith.constant 0.000000e+00 : f32
        %c196 = arith.constant 196 : index
        %c40 = arith.constant 40 : index
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c28 = arith.constant 28 : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:196x240xf32>
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:240x40xf32>
        %2 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:196x40xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c196 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c40 step %6 {
            %7 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [28, 240], strides = [1, 1] : !flow.dispatch.tensor<readonly:196x240xf32> -> tensor<28x240xf32>
            %8 = tensor.cast %7 : tensor<28x240xf32> to tensor<?x240xf32>
            %9 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [240, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:240x40xf32> -> tensor<240x8xf32>
            %10 = tensor.cast %9 : tensor<240x8xf32> to tensor<240x?xf32>
            %11 = linalg.init_tensor [%c28, %c8] : tensor<?x?xf32>
            %12 = linalg.fill(%cst, %11) {__internal_linalg_transform__ = "workgroup"} : f32, tensor<?x?xf32> -> tensor<?x?xf32>
            %13 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%8, %10 : tensor<?x240xf32>, tensor<240x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
            flow.dispatch.tensor.store %13, %2, offsets = [%arg0, %arg1], sizes = [%c28, %c8], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:196x40xf32>
          }
        }
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[], [4, 4, 60], [4, 4, 4]{{\]}}, native_vector_size = [4, 4, 4]>
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 28)>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUTileFuseAndVectorize", workload_per_wg = [8, 28]>
//       CHECK: hal.executable.entry_point public @matmul_static
//  CHECK-SAME:     translation.info = #[[TRANSLATION]]
//  CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index, %[[ARG2:[a-zA-Z0-9]+]]: index)
//   CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//   CHECK-DAG:     %[[D1:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]]]
//       CHECK:     hal.return %[[D0]], %[[D1]], %[[C1]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @restrict_num_workgroups {
  hal.executable.variant public @system_elf_arm_64, target = <"llvm", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }> {
    hal.executable.entry_point public @restrict_num_workgroups layout(#executable_layout)
    builtin.module {
      func @restrict_num_workgroups() {
        %cst = arith.constant 0.000000e+00 : f32
        %c7 = arith.constant 7 : index
        %c576 = arith.constant 576 : index
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c2 = arith.constant 2 : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:1x11x11x576xf32>
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:5x5x576xf32>
        %2 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:1x7x7x576xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %3 to %c7 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c7 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c576 step %8 {
              %9 = affine.min affine_map<(d0) -> (6, -d0 + 12)>(%arg0)
              %10 = affine.min affine_map<(d0) -> (11, -d0 + 12)>(%arg1)
              %11 = flow.dispatch.tensor.load %0, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %9, %10, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x11x11x576xf32> -> tensor<1x?x?x64xf32>
              %12 = tensor.cast %11 : tensor<1x?x?x64xf32> to tensor<1x?x?x?xf32>
              %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, %arg2], sizes = [5, 5, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:5x5x576xf32> -> tensor<5x5x64xf32>
              %14 = tensor.cast %13 : tensor<5x5x64xf32> to tensor<5x5x?xf32>
              %15 = affine.min affine_map<(d0) -> (2, -d0 + 7)>(%arg0)
              %16 = affine.min affine_map<(d0) -> (-d0 + 7, 2)>(%arg0)
              %17 = linalg.init_tensor [1, %16, %c7, %c64] : tensor<1x?x?x?xf32>
              %18 = linalg.fill(%cst, %17) {__internal_linalg_transform__ = "workgroup"} : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
              %19 = linalg.depthwise_conv_2d_nhwc_hwc {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%12, %14 : tensor<1x?x?x?xf32>, tensor<5x5x?xf32>) outs(%18 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %19, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %15, %c7, %c64], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x7x7x576xf32>
            }
          }
        }
        return
      }
    }
  }
}

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64, 8, 4]>
//       CHECK: hal.executable.entry_point public @restrict_num_workgroups
//  CHECK-SAME:     translation.info = #[[TRANSLATION]]
//  CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index, %[[ARG1:[a-zA-Z0-9]+]]: index, %[[ARG2:[a-zA-Z0-9]+]]: index)
//   CHECK-DAG:     %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
//   CHECK-DAG:     %[[D1:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]]]
//   CHECK-DAG:     %[[D2:.+]] = affine.apply #[[MAP2]]()[%[[ARG2]]]
//       CHECK:     hal.return %[[D0]], %[[D1]], %[[D2]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 3, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @test_exp_0 {
  hal.executable.variant public @system_elf_arm_64, target = <"llvm", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }> {
    hal.executable.entry_point public @test_exp_0 layout(#executable_layout)
    builtin.module {
      func @test_exp_0() {
        %c0 = arith.constant 0 : index
        %size = hal.interface.workgroup.size[0] : index
        %count = hal.interface.workgroup.count[0] : index
        %id = hal.interface.workgroup.id[0] : index
        %lb = hal.interface.constant.load[0] : index
        %ub = hal.interface.constant.load[1] : index
        %step = hal.interface.constant.load[2] : index
        %read = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : memref<?xf32>{%ub}
        %write = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : memref<?xf32>{%ub}
        %offset = affine.apply affine_map<(d0)[s0,s1] -> (d0 + s0 * s1)>(%lb)[%id, %size]
        %stride = affine.apply affine_map<(d0)[s0,s1] -> (d0 * s0 * s1)>(%step)[%count, %size]
        scf.for %iv = %offset to %ub step %stride {
          %val = memref.load %read[%iv] : memref<?xf32>
          memref.store %val, %write[%iv] : memref<?xf32>
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64]>
//      CHECK: hal.executable.entry_point public @test_exp_0
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:     %[[D0:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:     hal.return %[[D0]], %[[C1]], %[[C1]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 3, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @test_exp_1 {
  hal.executable.variant public @system_elf_arm_64, target = <"llvm", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }> {
    hal.executable.entry_point public @test_exp_1 layout(#executable_layout)
    builtin.module {
      func @test_exp_1() {
        %c0 = arith.constant 0 : index
        %size = hal.interface.workgroup.size[0] : index
        %count = hal.interface.workgroup.count[0] : index
        %id = hal.interface.workgroup.id[0] : index
        %lb = hal.interface.constant.load[0] : index
        %ub = hal.interface.constant.load[1] : index
        %step = hal.interface.constant.load[2] : index
        %read = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : memref<?xf32>{%ub}
        %write = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : memref<?xf32>{%ub}
        %offset = affine.apply affine_map<()[s0,s1] -> (5 + s0 * s1)>()[%id, %size]
        %stride = affine.apply affine_map<(d0)[s0,s1] -> (s0 * d0 * s1)>(%step)[%count, %size]
        scf.for %iv = %offset to %ub step %stride {
          %val = memref.load %read[%iv] : memref<?xf32>
          memref.store %val, %write[%iv] : memref<?xf32>
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECk-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64]>
//      CHECK: hal.executable.entry_point public @test_exp_1
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:     %[[D0:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:     hal.return %[[D0]], %[[C1]], %[[C1]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 3, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @test_exp_2 {
  hal.executable.variant public @system_elf_arm_64, target = <"llvm", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }> {
    hal.executable.entry_point public @test_exp_3 layout(#executable_layout)
    builtin.module {
      func @test_exp_3() {
        %c0 = arith.constant 0 : index
        %size = hal.interface.workgroup.size[0] : index
        %count = hal.interface.workgroup.count[0] : index
        %id = hal.interface.workgroup.id[0] : index
        %lb = hal.interface.constant.load[0] : index
        %ub = hal.interface.constant.load[1] : index
        %step = hal.interface.constant.load[2] : index
        %read = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : memref<?xf32>{%ub}
        %write = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : memref<?xf32>{%ub}
        %offset = affine.apply affine_map<(d0)[s0,s1] -> (d0 + s0 * s1)>(%lb)[%id, %size]
        %stride = affine.apply affine_map<()[s0,s1] -> (5 * s0 * s1)>()[%count, %size]
        scf.for %iv = %offset to %ub step %stride {
          %val = memref.load %read[%iv] : memref<?xf32>
          memref.store %val, %write[%iv] : memref<?xf32>
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64]>
//      CHECK: hal.executable.entry_point public @test_exp_3
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:     %[[D0:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:     hal.return %[[D0]], %[[C1]], %[[C1]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 3, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @test_exp_3 {
  hal.executable.variant public @system_elf_arm_64, target = <"llvm", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }> {
    hal.executable.entry_point public @test_exp_4 layout(#executable_layout)
    builtin.module {
      func @test_exp_4() {
        %c0 = arith.constant 0 : index
        %size = hal.interface.workgroup.size[0] : index
        %count = hal.interface.workgroup.count[0] : index
        %id = hal.interface.workgroup.id[0] : index
        %lb = hal.interface.constant.load[0] : index
        %ub = hal.interface.constant.load[1] : index
        %step = hal.interface.constant.load[2] : index
        %read = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : memref<?xf32>{%ub}
        %write = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : memref<?xf32>{%ub}
        %offset = affine.apply affine_map<(d0)[s0,s1] -> (s0 * s1 + d0)>(%lb)[%id, %size]
        %stride = affine.apply affine_map<()[s0,s1] -> (s0 * 5 * s1)>()[%count, %size]
        scf.for %iv = %offset to %ub step %stride {
          %val = memref.load %read[%iv] : memref<?xf32>
          memref.store %val, %write[%iv] : memref<?xf32>
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64]>
//      CHECK: hal.executable.entry_point public @test_exp_4
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:     %[[D0:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:     hal.return %[[D0]], %[[C1]], %[[C1]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 3, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @test_exp_4 {
  hal.executable.variant public @system_elf_arm_64, target = <"llvm", "system-elf-arm_64", {
    data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-linux-android30"
  }> {
    hal.executable.entry_point public @test_exp_5 layout(#executable_layout)
    builtin.module {
      func @test_exp_5() {
        %c0 = arith.constant 0 : index
        %size = hal.interface.workgroup.size[0] : index
        %count = hal.interface.workgroup.count[0] : index
        %id = hal.interface.workgroup.id[0] : index
        %lb = hal.interface.constant.load[0] : index
        %ub = hal.interface.constant.load[1] : index
        %step = hal.interface.constant.load[2] : index
        %read = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : memref<?xf32>{%ub}
        %write = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : memref<?xf32>{%ub}
        %offset = affine.apply affine_map<()[s0,s1] -> (s0 * s1 + 5)>()[%id, %size]
        %stride = affine.apply affine_map<()[s0,s1] -> (s0 * s1 * 5)>()[%count, %size]
        scf.for %iv = %offset to %ub step %stride {
          %val = memref.load %read[%iv] : memref<?xf32>
          memref.store %val, %write[%iv] : memref<?xf32>
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUDefault", workload_per_wg = [64]>
//      CHECK: hal.executable.entry_point public @test_exp_5
// CHECK-SAME:     translation.info = #[[TRANSLATION]]
// CHECK-NEXT:   ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:     %[[D0:.+]] = affine.apply #[[MAP]]()[%[[ARG0]]]
//      CHECK:     hal.return %[[D0]], %[[C1]], %[[C1]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_x86  {
  hal.executable.variant public @embedded_elf_x86_64, target = #hal.executable.target<
    "llvm",
    "embedded-elf-x86_64", {
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 16 : index,
      target_triple = "x86_64-unknown-unknown-eabi-elf"
    }> {
    hal.executable.entry_point public @matmul_x86 layout(#executable_layout)
    builtin.module {
      func @matmul_x86() {
        %c128 = arith.constant 128 : index
        %c384 = arith.constant 384 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(0) : !flow.dispatch.tensor<readonly:384x512xf32>
        %1 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(1) : !flow.dispatch.tensor<readonly:512x128xf32>
        %2 = hal.interface.binding.subspan type(storage_buffer) set(0) binding(2) : !flow.dispatch.tensor<writeonly:384x128xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c384 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c128 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 384)>(%arg0)[%workgroup_size_y]
            %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:384x512xf32> -> tensor<?x512xf32>
            %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 128)>(%arg1)[%workgroup_size_x]
            %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [512, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:512x128xf32> -> tensor<512x?xf32>
            %11 = affine.min affine_map<(d0)[s0] -> (-d0 + 384, s0)>(%arg0)[%workgroup_size_y]
            %12 = affine.min affine_map<(d0)[s0] -> (-d0 + 128, s0)>(%arg1)[%workgroup_size_x]
            %13 = linalg.init_tensor [%11, %12] : tensor<?x?xf32>
            %14 = linalg.fill(%cst, %13) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
            %15 = linalg.matmul ins(%8, %10 : tensor<?x512xf32>, tensor<512x?xf32>) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
            flow.dispatch.tensor.store %15, %2, offsets = [%arg0, %arg1], sizes = [%7, %9], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:384x128xf32>
          }
        }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"CPUTileFuseAndVectorize", workload_per_wg = [64, 64]>
//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_codegen.lowering.config<tile_sizes = [{{\[}}], [8, 8, 8], [1, 4, 4]], native_vector_size = [1, 4, 4]>
//  CHECK:       linalg.matmul {lowering.config = #[[CONFIG]]}
