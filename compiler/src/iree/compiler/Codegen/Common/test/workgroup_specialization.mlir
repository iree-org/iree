// RUN: iree-opt --iree-codegen-enable-workgroup-specialization --pass-pipeline='hal.executable(hal.executable.variant(iree-codegen-workgroup-specialization)), canonicalize, cse' --split-input-file %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [16, 4, 0], [0, 0, 64]]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0) -> (-d0 + 123, 64)>
#map2 = affine_map<(d0) -> (-d0 + 789, 64)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer>, <2, storage_buffer>, <3, storage_buffer>]>]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
module {
  hal.executable private @matmul_tensors {
    hal.executable.variant public @llvm, target = #executable_target_embedded_elf_arm_64_ {
      hal.executable.export public @matmul_tensors layout(#pipeline_layout) attributes {translation_info = #translation} {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c13 = arith.constant 13 : index
        hal.return %c13, %c2, %c1 : index, index, index
      }
      builtin.module {
        func.func @matmul_tensors() {
          %c123 = arith.constant 123 : index
          %c789 = arith.constant 789 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:123x456xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:456x789xf32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:123x789xf32>
          %workgroup_id_x = hal.interface.workgroup.id[0] : index
          %workgroup_count_x = hal.interface.workgroup.count[0] : index
          %workgroup_id_y = hal.interface.workgroup.id[1] : index
          %workgroup_count_y = hal.interface.workgroup.count[1] : index
          %3 = affine.apply #map0()[%workgroup_id_y]
          %4 = affine.apply #map0()[%workgroup_count_y]
          scf.for %arg0 = %3 to %c123 step %4 {
            %5 = affine.min #map1(%arg0)
            %6 = affine.apply #map0()[%workgroup_id_x]
            %7 = affine.apply #map0()[%workgroup_count_x]
            scf.for %arg1 = %6 to %c789 step %7 {
              %8 = affine.min #map2(%arg1)
              %9 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%5, 456], strides = [1, 1] : !flow.dispatch.tensor<readonly:123x456xf32> -> tensor<?x456xf32>
              %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [456, %8], strides = [1, 1] : !flow.dispatch.tensor<readonly:456x789xf32> -> tensor<456x?xf32>
              %11 = linalg.init_tensor [%5, %8] : tensor<?x?xf32>
              %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %13 = linalg.matmul {lowering_config = #config} ins(%9, %10 : tensor<?x456xf32>, tensor<456x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
              flow.dispatch.tensor.store %13, %2, offsets = [%arg0, %arg1], sizes = [%5, %8], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:123x789xf32>
            }
          }
          return
        }
      }
    }
  }
}

// CHECK: func.func @matmul_tensors()
// CHECK: %[[CMP0:.+]] = arith.cmpi eq, %{{.+}}, %c64 : index
// CHECK: %[[CMP1:.+]] = arith.cmpi eq, %{{.+}}, %c64 : index
// CHECK: %[[COND:.+]] = arith.andi %[[CMP0]], %[[CMP1]] : i1
// CHECK: scf.if %[[COND]] {
// CHECK:   scf.for
// CHECK:     scf.for 
// CHECK:       linalg.matmul
// CHECK-SAME:                ins(%{{.+}}, %{{.+}} : tensor<64x456xf32>, tensor<456x64xf32>) outs(%{{.+}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK: } else {
// CHECK:   scf.for
// CHECK:     scf.for 
// CHECK:       linalg.matmul
// CHECK-SAME:                ins(%{{.+}}, %{{.+}} : tensor<?x456xf32>, tensor<456x?xf32>) outs(%{{.+}} : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [16, 4, 0], [0, 0, 64]]>
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0) -> (-d0 + 123, 64)>
#map2 = affine_map<(d0) -> (-d0 + 789, 64)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer>, <2, storage_buffer>, <3, storage_buffer>]>]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
module {
  hal.executable private @add_tensors {
    hal.executable.variant public @llvm, target = #executable_target_embedded_elf_arm_64_ {
      hal.executable.export public @add_tensors layout(#pipeline_layout) attributes {translation_info = #translation} {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c13 = arith.constant 13 : index
        hal.return %c13, %c2, %c1 : index, index, index
      }
      builtin.module {
        func.func @add_tensors() {
          %c123 = arith.constant 123 : index
          %c789 = arith.constant 789 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:123x789xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:123x789xf32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:123x789xf32>
          %workgroup_id_x = hal.interface.workgroup.id[0] : index
          %workgroup_count_x = hal.interface.workgroup.count[0] : index
          %workgroup_id_y = hal.interface.workgroup.id[1] : index
          %workgroup_count_y = hal.interface.workgroup.count[1] : index
          %3 = affine.apply #map0()[%workgroup_id_y]
          %4 = affine.apply #map0()[%workgroup_count_y]
          scf.for %arg0 = %3 to %c123 step %4 {
            %5 = affine.min #map1(%arg0)
            %6 = affine.apply #map0()[%workgroup_id_x]
            %7 = affine.apply #map0()[%workgroup_count_x]
            scf.for %arg1 = %6 to %c789 step %7 {
              %8 = affine.min #map2(%arg1)
              %9 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1], sizes = [%5, %8], strides = [1, 1] : !flow.dispatch.tensor<readonly:123x789xf32> -> tensor<?x?xf32>
              %10 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1], sizes = [%5, %8], strides = [1, 1] : !flow.dispatch.tensor<readonly:123x789xf32> -> tensor<?x?xf32>
              %11 = linalg.init_tensor [%5, %8] : tensor<?x?xf32>
              %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %13 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%9, %10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) attrs =  {lowering_config = #config} {
              ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
                %14 = arith.addf %arg2, %arg3 : f32
                linalg.yield %14 : f32
              } -> tensor<?x?xf32>
              flow.dispatch.tensor.store %13, %2, offsets = [%arg0, %arg1], sizes = [%5, %8], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:123x789xf32>
            }
          }
          return
        }
      }
    }
  }
}

// CHECK: func.func @add_tensors()
// CHECK: %[[CMP0:.+]] = arith.cmpi eq, %{{.+}}, %c64 : index
// CHECK: %[[CMP1:.+]] = arith.cmpi eq, %{{.+}}, %c64 : index
// CHECK: %[[COND:.+]] = arith.andi %[[CMP0]], %[[CMP1]] : i1
// CHECK: scf.if %[[COND]] {
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       linalg.generic
// CHECK-SAME:                ins(%{{.+}}, %{{.+}} : tensor<64x64xf32>, tensor<64x64xf32>) outs(%{{.+}} : tensor<64x64xf32>)
// CHECK: } else {
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       linalg.generic
// CHECK-SAME:                ins(%{{.+}}, %{{.+}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.+}} : tensor<?x?xf32>)
