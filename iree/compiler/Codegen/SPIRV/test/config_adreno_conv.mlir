// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | IreeFileCheck %s

// Conv - large OC - distribute to only one workgroup dimension.

hal.executable @conv_112x112x512 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 1024 : i32,
        max_compute_workgroup_size = dense<[1024, 1024, 64]> : vector<3xi32>,
        subgroup_size = 64 : i32}>
    }> {
    hal.executable.entry_point public @conv_112x112x512 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @conv_112x112x512() {
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c112 = arith.constant 112 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:1x225x225x3xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:3x3x3x512xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:1x112x112x512xf32>
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
        scf.for %arg0 = %3 to %c112 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c112 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c512 step %8 {
              %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %10 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>(%arg0)[%workgroup_size_z]
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>(%arg1)[%workgroup_size_y]
              %13 = flow.dispatch.tensor.load %0, offsets = [0, %9, %11, 0], sizes = [1, %10, %12, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x225x225x3xf32> -> tensor<1x?x?x3xf32>
              %14 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 512)>(%arg2)[%workgroup_size_x]
              %15 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 3, %14], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x3x512xf32> -> tensor<3x3x3x?xf32>
              %16 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg0)[%workgroup_size_z]
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_y]
              %18 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 512)>(%arg2)[%workgroup_size_x]
              %19 = affine.min affine_map<(d0)[s0] -> (-d0 + 112, s0)>(%arg0)[%workgroup_size_z]
              %20 = affine.min affine_map<(d0)[s0] -> (-d0 + 112, s0)>(%arg1)[%workgroup_size_y]
              %21 = affine.min affine_map<(d0)[s0] -> (-d0 + 512, s0)>(%arg2)[%workgroup_size_x]
              %22 = linalg.init_tensor [1, %19, %20, %21] : tensor<1x?x?x?xf32>
              %23 = linalg.fill(%cst, %22) : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
              %24 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%13, %15 : tensor<1x?x?x3xf32>, tensor<3x3x3x?xf32>) outs(%23 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %24, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %16, %17, %18], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x512xf32>
            }
          }
        }
        return
      }
      hal.interface private @io {
        hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[0, 1, 8, 256], [0, 1, 8, 4], [0, 0, 0, 0, 1, 1, 4]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [256, 8, 1]>
//  CHECK-DAG: #[[MAP_X:.+]] = affine_map<()[s0] -> (s0 ceildiv 256)
//  CHECK-DAG: #[[MAP_Y:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: hal.executable.entry_point public @conv_112x112x512
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [64 : index, 1 : index, 1 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %[[Z:.+]]: index):
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP_X]]()[%[[X]]]
// CHECK-NEXT:   %[[Y_COUNT:.+]] = affine.apply #[[MAP_Y]]()[%[[Y]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y_COUNT]], %[[Z]]

//      CHECK: func @conv_112x112x512()
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

// Conv - medium OC/OW/OH - distribute to two workgroup dimensions.

hal.executable @conv_112x112x32 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 1024 : i32,
        max_compute_workgroup_size = dense<[1024, 1024, 64]> : vector<3xi32>,
        subgroup_size = 64 : i32}>
    }> {
    hal.executable.entry_point public @conv_112x112x32 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @conv_112x112x32() {
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c112 = arith.constant 112 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:1x225x225x3xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:3x3x3x32xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
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
        scf.for %arg0 = %3 to %c112 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c112 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c32 step %8 {
              %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %10 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>(%arg0)[%workgroup_size_z]
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 225)>(%arg1)[%workgroup_size_y]
              %13 = flow.dispatch.tensor.load %0, offsets = [0, %9, %11, 0], sizes = [1, %10, %12, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x225x225x3xf32> -> tensor<1x?x?x3xf32>
              %14 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %15 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 3, %14], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x3x32xf32> -> tensor<3x3x3x?xf32>
              %16 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg0)[%workgroup_size_z]
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_y]
              %18 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %19 = affine.min affine_map<(d0)[s0] -> (-d0 + 112, s0)>(%arg0)[%workgroup_size_z]
              %20 = affine.min affine_map<(d0)[s0] -> (-d0 + 112, s0)>(%arg1)[%workgroup_size_y]
              %21 = affine.min affine_map<(d0)[s0] -> (-d0 + 32, s0)>(%arg2)[%workgroup_size_x]
              %22 = linalg.init_tensor [1, %19, %20, %21] : tensor<1x?x?x?xf32>
              %23 = linalg.fill(%cst, %22) : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
              %24 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%13, %15 : tensor<1x?x?x3xf32>, tensor<3x3x3x?xf32>) outs(%23 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %24, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %16, %17, %18], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
            }
          }
        }
        return
      }
      hal.interface private @io {
        hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[0, 4, 16, 32], [0, 4, 2, 4], [0, 0, 0, 0, 1, 1, 4]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [32, 16, 4]>
//  CHECK-DAG: #[[MAP_X:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)
//  CHECK-DAG: #[[MAP_Y:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//  CHECK-DAG: #[[MAP_Z:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: hal.executable.entry_point public @conv_112x112x32
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [8 : index, 8 : index, 1 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %[[Z:.+]]: index):
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP_X]]()[%[[X]]]
// CHECK-NEXT:   %[[Y_COUNT:.+]] = affine.apply #[[MAP_Y]]()[%[[Y]]]
// CHECK-NEXT:   %[[Z_COUNT:.+]] = affine.apply #[[MAP_Z]]()[%[[Z]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y_COUNT]], %[[Z_COUNT]]

//      CHECK: func @conv_112x112x32()
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

// Conv - small OC/OW/OH - distribute to all three workgroup dimensions.

hal.executable @conv_16x16x16 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 1024 : i32,
        max_compute_workgroup_size = dense<[1024, 1024, 64]> : vector<3xi32>,
        subgroup_size = 64 : i32}>
    }> {
    hal.executable.entry_point public @conv_16x16x16 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @conv_16x16x16() {
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:1x33x33x3xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:3x3x3x16xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:1x16x16x16xf32>
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
        scf.for %arg0 = %3 to %c16 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c16 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c16 step %8 {
              %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %10 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 33)>(%arg0)[%workgroup_size_z]
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 33)>(%arg1)[%workgroup_size_y]
              %13 = flow.dispatch.tensor.load %0, offsets = [0, %9, %11, 0], sizes = [1, %10, %12, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x33x33x3xf32> -> tensor<1x?x?x3xf32>
              %14 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 16)>(%arg2)[%workgroup_size_x]
              %15 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 3, %14], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x3x16xf32> -> tensor<3x3x3x?xf32>
              %16 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 16)>(%arg0)[%workgroup_size_z]
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 16)>(%arg1)[%workgroup_size_y]
              %18 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 16)>(%arg2)[%workgroup_size_x]
              %19 = affine.min affine_map<(d0)[s0] -> (-d0 + 16, s0)>(%arg0)[%workgroup_size_z]
              %20 = affine.min affine_map<(d0)[s0] -> (-d0 + 16, s0)>(%arg1)[%workgroup_size_y]
              %21 = affine.min affine_map<(d0)[s0] -> (-d0 + 16, s0)>(%arg2)[%workgroup_size_x]
              %22 = linalg.init_tensor [1, %19, %20, %21] : tensor<1x?x?x?xf32>
              %23 = linalg.fill(%cst, %22) : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
              %24 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%13, %15 : tensor<1x?x?x3xf32>, tensor<3x3x3x?xf32>) outs(%23 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %24, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %16, %17, %18], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x16x16x16xf32>
            }
          }
        }
        return
      }
      hal.interface private @io {
        hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[0, 8, 8, 16], [0, 2, 2, 4], [0, 0, 0, 0, 1, 1, 4]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [16, 8, 8]>
//  CHECK-DAG: #[[MAP_X:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)
//  CHECK-DAG: #[[MAP_YZ:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//      CHECK: hal.executable.entry_point public @conv_16x16x16
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [4 : index, 4 : index, 4 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %[[Z:.+]]: index):
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP_X]]()[%[[X]]]
// CHECK-NEXT:   %[[Y_COUNT:.+]] = affine.apply #[[MAP_YZ]]()[%[[Y]]]
// CHECK-NEXT:   %[[Z_COUNT:.+]] = affine.apply #[[MAP_YZ]]()[%[[Z]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y_COUNT]], %[[Z_COUNT]]

//      CHECK: func @conv_16x16x16()
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

// Depthwise conv - small OC/OW/OH - distribute to all three workgroup dimensions.

hal.executable @dwconv_28x28x144 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 1024 : i32,
        max_compute_workgroup_size = dense<[1024, 1024, 64]> : vector<3xi32>,
        subgroup_size = 64 : i32}>
    }> {
    hal.executable.entry_point public @dwconv_28x28x144 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @dwconv_28x28x144() {
        %c0 = arith.constant 0 : index
        %c144 = arith.constant 144 : index
        %c28 = arith.constant 28 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:1x57x57x144xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:3x3x144xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:1x28x28x144xf32>
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
        scf.for %arg0 = %3 to %c28 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c28 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c144 step %8 {
              %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %10 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 57)>(%arg0)[%workgroup_size_z]
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 57)>(%arg1)[%workgroup_size_y]
              %13 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 144)>(%arg2)[%workgroup_size_x]
              %14 = flow.dispatch.tensor.load %0, offsets = [0, %9, %11, %arg2], sizes = [1, %10, %12, %13], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x57x57x144xf32> -> tensor<1x?x?x?xf32>
              %15 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 144)>(%arg2)[%workgroup_size_x]
              %16 = flow.dispatch.tensor.load %1, offsets = [0, 0, %arg2], sizes = [3, 3, %15], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x144xf32> -> tensor<3x3x?xf32>
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 28)>(%arg0)[%workgroup_size_z]
              %18 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 28)>(%arg1)[%workgroup_size_y]
              %19 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 144)>(%arg2)[%workgroup_size_x]
              %20 = affine.min affine_map<(d0)[s0] -> (-d0 + 28, s0)>(%arg0)[%workgroup_size_z]
              %21 = affine.min affine_map<(d0)[s0] -> (-d0 + 28, s0)>(%arg1)[%workgroup_size_y]
              %22 = affine.min affine_map<(d0)[s0] -> (-d0 + 144, s0)>(%arg2)[%workgroup_size_x]
              %23 = linalg.init_tensor [1, %20, %21, %22] : tensor<1x?x?x?xf32>
              %24 = linalg.fill(%cst, %23) : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
              %25 = linalg.depthwise_conv_2d_nhwc_hwc {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%14, %16 : tensor<1x?x?x?xf32>, tensor<3x3x?xf32>) outs(%24 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %25, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %17, %18, %19], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x28x28x144xf32>
            }
          }
        }
        return
      }
      hal.interface private @io {
        hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[0, 4, 4, 16], [0, 1, 1, 4], [0, 0, 0, 0, 1, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [16, 4, 4]>
//  CHECK-DAG: #[[MAP_X:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)
//  CHECK-DAG: #[[MAP_YZ:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: hal.executable.entry_point public @dwconv_28x28x144
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [4 : index, 4 : index, 4 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %[[Z:.+]]: index):
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP_X]]()[%[[X]]]
// CHECK-NEXT:   %[[Y_COUNT:.+]] = affine.apply #[[MAP_YZ]]()[%[[Y]]]
// CHECK-NEXT:   %[[Z_COUNT:.+]] = affine.apply #[[MAP_YZ]]()[%[[Z]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y_COUNT]], %[[Z_COUNT]]

//      CHECK: func @dwconv_28x28x144()
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

// Depthwise conv - tiny OC/OW/OH - starving the GPU.

hal.executable @dwconv_4x4x8 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 1024 : i32,
        max_compute_workgroup_size = dense<[1024, 1024, 64]> : vector<3xi32>,
        subgroup_size = 64 : i32}>
    }> {
    hal.executable.entry_point public @dwconv_4x4x8 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @dwconv_4x4x8() {
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:1x9x9x8xf32>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:3x3x8xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:1x4x4x8xf32>
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
        scf.for %arg0 = %3 to %c4 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c4 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c8 step %8 {
              %9 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %10 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 9)>(%arg0)[%workgroup_size_z]
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = affine.min affine_map<(d0)[s0] -> (s0 * 2 + 1, d0 * -2 + 9)>(%arg1)[%workgroup_size_y]
              %13 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 8)>(%arg2)[%workgroup_size_x]
              %14 = flow.dispatch.tensor.load %0, offsets = [0, %9, %11, %arg2], sizes = [1, %10, %12, %13], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x9x9x8xf32> -> tensor<1x?x?x?xf32>
              %15 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 8)>(%arg2)[%workgroup_size_x]
              %16 = flow.dispatch.tensor.load %1, offsets = [0, 0, %arg2], sizes = [3, 3, %15], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x8xf32> -> tensor<3x3x?xf32>
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%arg0)[%workgroup_size_z]
              %18 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%arg1)[%workgroup_size_y]
              %19 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 8)>(%arg2)[%workgroup_size_x]
              %20 = affine.min affine_map<(d0)[s0] -> (-d0 + 4, s0)>(%arg0)[%workgroup_size_z]
              %21 = affine.min affine_map<(d0)[s0] -> (-d0 + 4, s0)>(%arg1)[%workgroup_size_y]
              %22 = affine.min affine_map<(d0)[s0] -> (-d0 + 8, s0)>(%arg2)[%workgroup_size_x]
              %23 = linalg.init_tensor [1, %20, %21, %22] : tensor<1x?x?x?xf32>
              %24 = linalg.fill(%cst, %23) : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
              %25 = linalg.depthwise_conv_2d_nhwc_hwc {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%14, %16 : tensor<1x?x?x?xf32>, tensor<3x3x?xf32>) outs(%24 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %25, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %17, %18, %19], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x4x4x8xf32>
            }
          }
        }
        return
      }
      hal.interface private @io {
        hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer"
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[0, 4, 4, 8], [0, 1, 1, 4], [0, 0, 0, 0, 1, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVVectorize", workload_per_wg = [8, 4, 4]>
//  CHECK-DAG: #[[MAP_X:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)
//  CHECK-DAG: #[[MAP_YZ:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: hal.executable.entry_point public @dwconv_4x4x8
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [2 : index, 4 : index, 4 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %[[Z:.+]]: index):
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP_X]]()[%[[X]]]
// CHECK-NEXT:   %[[Y_COUNT:.+]] = affine.apply #[[MAP_YZ]]()[%[[Y]]]
// CHECK-NEXT:   %[[Z_COUNT:.+]] = affine.apply #[[MAP_YZ]]()[%[[Z]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y_COUNT]], %[[Z_COUNT]]

//      CHECK: func @dwconv_4x4x8()
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:     lowering.config = #[[CONFIG]]
