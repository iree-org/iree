// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | IreeFileCheck %s

// Odd K that forbids vectorization.

hal.executable @batch_matmul_1x3x32 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant public @vulkan_spirv_fb, target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 16384 : i32,
        max_compute_workgroup_invocations = 128 : i32,
        max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>,
        subgroup_size = 4 : i32}>
    }> {
    hal.executable.entry_point public @batch_matmul_1x3x32 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @batch_matmul_1x3x32() {
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c3 = arith.constant 3 : index
        %c1 = arith.constant 1 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:1x3x3xf32>
        %1 = hal.interface.binding.subspan @io::@s0b0_ro_constant[%c0] : !flow.dispatch.tensor<readonly:1x3x32xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:1x3x32xf32>
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
        scf.for %arg0 = %3 to %c1 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c3 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c32 step %8 {
              %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1)>(%arg0)[%workgroup_size_z]
              %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 3)>(%arg1)[%workgroup_size_y]
              %11 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1, 0], sizes = [%9, %10, 3], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:1x3x3xf32> -> tensor<?x?x3xf32>
              %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1)>(%arg0)[%workgroup_size_z]
              %13 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %14 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0, %arg2], sizes = [%12, 3, %13], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:1x3x32xf32> -> tensor<?x3x?xf32>
              %15 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 1)>(%arg0)[%workgroup_size_z]
              %16 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 3)>(%arg1)[%workgroup_size_y]
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %18 = affine.min affine_map<(d0)[s0] -> (-d0 + 1, s0)>(%arg0)[%workgroup_size_z]
              %19 = affine.min affine_map<(d0)[s0] -> (-d0 + 3, s0)>(%arg1)[%workgroup_size_y]
              %20 = affine.min affine_map<(d0)[s0] -> (-d0 + 32, s0)>(%arg2)[%workgroup_size_x]
              %21 = linalg.init_tensor [%18, %19, %20] : tensor<?x?x?xf32>
              %22 = linalg.fill(%cst, %21) : f32, tensor<?x?x?xf32> -> tensor<?x?x?xf32>
              %23 = linalg.batch_matmul {__internal_linalg_transform__ = "workgroup"} ins(%11, %14 : tensor<?x?x3xf32>, tensor<?x3x?xf32>) outs(%22 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
              flow.dispatch.tensor.store %23, %2, offsets = [%arg0, %arg1, %arg2], sizes = [%15, %16, %17], strides = [1, 1, 1] : tensor<?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x3x32xf32>
            }
          }
        }
        return
      }
      hal.interface private @io {
        hal.interface.binding public @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 1, 4], [1, 1, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [4, 1, 1]>
//      CHECK: hal.executable.entry_point public @batch_matmul_1x3x32
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [4 : index, 1 : index, 1 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %[[Z:.+]]: index):
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP]]()[%[[X]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y]], %[[Z]]

//      CHECK: func @batch_matmul_1x3x32()
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

// Non-16 / non-32 bit types cannot be vectorized right now.

hal.executable private @matmul_64x16 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant public @vulkan_spirv_fb, target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 16384 : i32,
        max_compute_workgroup_invocations = 128 : i32,
        max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>,
        subgroup_size = 4 : i32}>
  }> {
    hal.executable.entry_point public @matmul_64x16 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @matmul_64x16() {
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:64x32xi8>
        %1 = hal.interface.binding.subspan @io::@s0b0_ro_constant[%c0] : !flow.dispatch.tensor<readonly:32x16xi8>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:64x16xi32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c64 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c16 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 64)>(%arg0)[%workgroup_size_y]
            %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:64x32xi8> -> tensor<?x32xi8>
            %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 16)>(%arg1)[%workgroup_size_x]
            %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [32, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:32x16xi8> -> tensor<32x?xi8>
            %11 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 64)>(%arg0)[%workgroup_size_y]
            %12 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 16)>(%arg1)[%workgroup_size_x]
            %13 = affine.min affine_map<(d0)[s0] -> (-d0 + 64, s0)>(%arg0)[%workgroup_size_y]
            %14 = affine.min affine_map<(d0)[s0] -> (-d0 + 16, s0)>(%arg1)[%workgroup_size_x]
            %15 = linalg.init_tensor [%13, %14] : tensor<?x?xi32>
            %16 = linalg.fill(%c0_i32, %15) : i32, tensor<?x?xi32> -> tensor<?x?xi32>
            %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%8, %10 : tensor<?x32xi8>, tensor<32x?xi8>) outs(%16 : tensor<?x?xi32>) -> tensor<?x?xi32>
            flow.dispatch.tensor.store %17, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:64x16xi32>
          }
        }
        return
      }
      hal.interface private @io {
        hal.interface.binding public @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 4], [1, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = [4, 1]>
//      CHECK: hal.executable.entry_point public @matmul_64x16
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [4 : index, 1 : index, 1 : index]
// CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %{{.+}}: index):
// CHECK-NEXT:   %[[ONE:.+]] = arith.constant 1 : index
// CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply #[[MAP]]()[%[[X]]]
// CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y]], %[[ONE]]

//      CHECK: func @matmul_64x16()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering.config = #[[CONFIG]]
