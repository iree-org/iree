// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true}))' %s | IreeFileCheck %s

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm", "embedded-elf-arm_64", {data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-unknown-unknown-eabi-elf"}>
#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map3 = affine_map<(d0)[s0] -> (s0, -d0 + 96)>
#map4 = affine_map<(d0)[s0] -> (s0, -d0 + 128)>
hal.executable private @mmt4d_384x384x512_4x1x4_dispatch_0 {
    hal.interface public @io {
      hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding public @s0b2_rw_external, set=0, binding=2, type="StorageBuffer", access="Read|Write"
    }
    hal.executable.variant public @embedded_elf_arm_64, target = #executable_target_embedded_elf_arm_64_ {
      hal.executable.entry_point public @mmt4d_384x384x512_4x1x4_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
      builtin.module  {
        func @mmt4d_384x384x512_4x1x4_dispatch_0() {
          %c0 = arith.constant 0 : index
          %c96 = arith.constant 96 : index
          %c128 = arith.constant 128 : index
          %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:96x384x4x1xf32>
          %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:128x384x4x1xf32>
          %2 = hal.interface.binding.subspan @io::@s0b2_rw_external[%c0] : !flow.dispatch.tensor<readwrite:96x128x4x4xf32>
          %workgroup_size_x = hal.interface.workgroup.size[0] : index
          %workgroup_size_y = hal.interface.workgroup.size[1] : index
          %workgroup_id_x = hal.interface.workgroup.id[0] : index
          %workgroup_count_x = hal.interface.workgroup.count[0] : index
          %workgroup_id_y = hal.interface.workgroup.id[1] : index
          %workgroup_count_y = hal.interface.workgroup.count[1] : index
          %3 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
          %4 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg0 = %3 to %c96 step %4 {
            %5 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
            %6 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg1 = %5 to %c128 step %6 {
              %7 = affine.min #map3(%arg0)[%workgroup_size_y]
              %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0, 0, 0], sizes = [%7, 384, 4, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:96x384x4x1xf32> -> tensor<?x384x4x1xf32>
              %9 = affine.min #map4(%arg1)[%workgroup_size_x]
              %10 = flow.dispatch.tensor.load %1, offsets = [%arg1, 0, 0, 0], sizes = [%9, 384, 4, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:128x384x4x1xf32> -> tensor<?x384x4x1xf32>
              %11 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1, 0, 0], sizes = [%7, %9, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:96x128x4x4xf32> -> tensor<?x?x4x4xf32>
              %12 = linalg.mmt4d {__internal_linalg_transform__ = "workgroup"} ins(%8, %10 : tensor<?x384x4x1xf32>, tensor<?x384x4x1xf32>) outs(%11 : tensor<?x?x4x4xf32>) -> tensor<?x?x4x4xf32>
              flow.dispatch.tensor.store %12, %2, offsets = [%arg0, %arg1, 0, 0], sizes = [%7, %9, 4, 4], strides = [1, 1, 1, 1] : tensor<?x?x4x4xf32> -> !flow.dispatch.tensor<readwrite:96x128x4x4xf32>
            }
          }
          return
        }
        hal.interface private @io {
          hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
          hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
          hal.interface.binding public @s0b2_rw_external, set=0, binding=2, type="StorageBuffer", access="Read|Write"
        }
      }
    }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[48, 32], [1, 1, 1, 4, 4, 1], [1, 1, 1, 4, 4, 1]{{\]}}, native_vector_size = [1, 1, 1, 4, 4, 1]
//      CHECK: func @mmt4d_384x384x512_4x1x4_dispatch_0()
//      CHECK:   linalg.mmt4d
// CHECK-SAME:     lowering.config = #[[CONFIG]]
