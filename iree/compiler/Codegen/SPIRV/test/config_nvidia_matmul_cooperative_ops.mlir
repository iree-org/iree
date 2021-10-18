// RUN: iree-opt -split-input-file -mlir-print-local-scope -pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | IreeFileCheck %s

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 256)>
#map2 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map3 = affine_map<(d0)[s0] -> (-d0 + 256, s0)>
#map4 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>

hal.executable public @matmul_256x1024x128_div_sub {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding public @s0b2_ro_external, set=0, binding=2, type="StorageBuffer", access="Read"
    hal.interface.binding public @s0b3_ro_external, set=0, binding=3, type="StorageBuffer", access="Read"
    hal.interface.binding public @s0b4_xw_external, set=0, binding=4, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.5,
          [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixNV],
          [SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>, NVIDIA:DiscreteGPU,
          {cooperative_matrix_properties_nv = [
            {a_type = i8, b_type = i8, c_type = i32, k_size = 32 : i32,
             m_size = 8 : i32, n_size = 8 : i32, result_type = i32, scope = 3 : i32},
            {a_type = f16, b_type = f16, c_type = f16, k_size = 16 : i32,
             m_size = 16 : i32, n_size = 16 : i32, result_type = f16,
             scope = 3 : i32},
            {a_type = f16, b_type = f16, c_type = f32, k_size = 16 : i32,
             m_size = 16 : i32, n_size = 16 : i32, result_type = f32,
             scope = 3 : i32}],
           max_compute_shared_memory_size = 49152 : i32,
           max_compute_workgroup_invocations = 1024 : i32,
           max_compute_workgroup_size = dense<[2147483647, 65535, 65535]> : vector<3xi32>,
           subgroup_size = 32 : i32}>}> {
    hal.executable.entry_point public @matmul_256x1024x128_div_sub attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @matmul_256x1024x128_div_sub() {
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:256x1024xf16>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:256x1024xf16>
        %2 = hal.interface.binding.subspan @io::@s0b2_ro_external[%c0] : !flow.dispatch.tensor<readonly:256x128xf16>
        %3 = hal.interface.binding.subspan @io::@s0b3_ro_external[%c0] : !flow.dispatch.tensor<readonly:128x1024xf16>
        %4 = hal.interface.binding.subspan @io::@s0b4_xw_external[%c0] : !flow.dispatch.tensor<writeonly:256x1024xf16>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %5 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
        %6 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %5 to %c256 step %6 {
          %7 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
          %8 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %7 to %c1024 step %8 {
            %9 = affine.min #map1(%arg0)[%workgroup_size_y]
            %10 = affine.min #map2(%arg1)[%workgroup_size_x]
            %11 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1], sizes = [%9, %10], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x1024xf16> -> tensor<?x?xf16>
            %12 = affine.min #map1(%arg0)[%workgroup_size_y]
            %13 = affine.min #map2(%arg1)[%workgroup_size_x]
            %14 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1], sizes = [%12, %13], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x1024xf16> -> tensor<?x?xf16>
            %15 = affine.min #map1(%arg0)[%workgroup_size_y]
            %16 = affine.min #map2(%arg1)[%workgroup_size_x]
            %17 = linalg.init_tensor [%15, %16] : tensor<?x?xf16>
            %18 = affine.min #map3(%arg0)[%workgroup_size_y]
            %19 = flow.dispatch.tensor.load %2, offsets = [%arg0, 0], sizes = [%18, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x128xf16> -> tensor<?x128xf16>
            %20 = affine.min #map4(%arg1)[%workgroup_size_x]
            %21 = flow.dispatch.tensor.load %3, offsets = [0, %arg1], sizes = [128, %20], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x1024xf16> -> tensor<128x?xf16>
            %22 = affine.min #map3(%arg0)[%workgroup_size_y]
            %23 = affine.min #map4(%arg1)[%workgroup_size_x]
            %24 = linalg.init_tensor [%22, %23] : tensor<?x?xf16>
            %25 = linalg.fill(%cst, %24) : f16, tensor<?x?xf16> -> tensor<?x?xf16>
            %26 = linalg.matmul ins(%19, %21 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
            %27 = linalg.generic {indexing_maps = [#map5, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"]}
              ins(%26, %11, %14 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>)
              outs(%17 : tensor<?x?xf16>)
              attrs =  {__internal_linalg_transform__ = "workgroup"} {
            ^bb0(%arg2: f16, %arg3: f16, %arg4: f16, %arg5: f16):  // no predecessors
              %28 = arith.divf %arg2, %arg3 : f16
              %29 = arith.subf %28, %arg4 : f16
              linalg.yield %29 : f16
            } -> tensor<?x?xf16>
            flow.dispatch.tensor.store %27, %4, offsets = [%arg0, %arg1], sizes = [%15, %16], strides = [1, 1] : tensor<?x?xf16> -> !flow.dispatch.tensor<writeonly:256x1024xf16>
          }
        }
        return
      }
      hal.interface private @io {
        hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding public @s0b2_ro_external, set=0, binding=2, type="StorageBuffer", access="Read"
        hal.interface.binding public @s0b3_ro_external, set=0, binding=3, type="StorageBuffer", access="Read"
        hal.interface.binding public @s0b4_xw_external, set=0, binding=4, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//          CHECK-LABEL: hal.executable.entry_point public @matmul_256x1024x128_div_sub
//           CHECK-SAME:   translation.info = {passPipeline = "SPIRVVectorizeToCooperativeOps", workloadPerWorkgroup = [16, 16]}
//           CHECK-SAME:   workgroup_size = [32 : index, 1 : index, 1 : index]
//           CHECK-NEXT: ^{{.+}}(%[[X:.+]]: index, %[[Y:.+]]: index, %{{.+}}: index):
//           CHECK-NEXT:   %[[C1:.+]] = arith.constant 1 : index
//           CHECK-NEXT:   %[[X_COUNT:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%[[X]]]
//           CHECK-NEXT:   %[[Y_COUNT:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 16)>()[%[[Y]]]
//           CHECK-NEXT:   hal.return %[[X_COUNT]], %[[Y_COUNT]], %[[C1]]

//                CHECK: func @matmul_256x1024x128_div_sub()
//                CHECK:   linalg.matmul
//  CHECK-SAME{LITERAL}:     lowering.config = {tileSizes = [[16, 16, 16], [16, 16, 16]]}

// -----

// Small K - not supported by cooperative matrix.

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 256)>
#map2 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map3 = affine_map<(d0)[s0] -> (-d0 + 256, s0)>
#map4 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>

hal.executable public @matmul_256x1024x8 {
  hal.interface public @io {
    hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.5,
          [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixNV],
          [SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>, NVIDIA:DiscreteGPU,
          {cooperative_matrix_properties_nv = [
            {a_type = i8, b_type = i8, c_type = i32, k_size = 32 : i32,
             m_size = 8 : i32, n_size = 8 : i32, result_type = i32, scope = 3 : i32},
            {a_type = f16, b_type = f16, c_type = f16, k_size = 16 : i32,
             m_size = 16 : i32, n_size = 16 : i32, result_type = f16,
             scope = 3 : i32},
            {a_type = f16, b_type = f16, c_type = f32, k_size = 16 : i32,
             m_size = 16 : i32, n_size = 16 : i32, result_type = f32,
             scope = 3 : i32}],
           max_compute_shared_memory_size = 49152 : i32,
           max_compute_workgroup_invocations = 1024 : i32,
           max_compute_workgroup_size = dense<[2147483647, 65535, 65535]> : vector<3xi32>,
           subgroup_size = 32 : i32}>}> {
    hal.executable.entry_point public @matmul_256x1024x8 attributes {interface = @io, ordinal = 0 : index}
    builtin.module  {
      func @matmul_256x1024x8() {
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : !flow.dispatch.tensor<readonly:256x8xf16>
        %1 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:8x1024xf16>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:256x1024xf16>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply #map0()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply #map0()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c256 step %4 {
          %5 = affine.apply #map0()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply #map0()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c1024 step %6 {
            %7 = affine.min #map1(%arg0)[%workgroup_size_y]
            %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x8xf16> -> tensor<?x8xf16>
            %9 = affine.min #map2(%arg1)[%workgroup_size_x]
            %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [8, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:8x1024xf16> -> tensor<8x?xf16>
            %11 = affine.min #map1(%arg0)[%workgroup_size_y]
            %12 = affine.min #map2(%arg1)[%workgroup_size_x]
            %13 = affine.min #map3(%arg0)[%workgroup_size_y]
            %14 = affine.min #map4(%arg1)[%workgroup_size_x]
            %15 = linalg.init_tensor [%13, %14] : tensor<?x?xf16>
            %16 = linalg.fill(%cst, %15) : f16, tensor<?x?xf16> -> tensor<?x?xf16>
            %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%8, %10 : tensor<?x8xf16>, tensor<8x?xf16>) outs(%16 : tensor<?x?xf16>) -> tensor<?x?xf16>
            flow.dispatch.tensor.store %17, %2, offsets = [%arg0, %arg1], sizes = [%11, %12], strides = [1, 1] : tensor<?x?xf16> -> !flow.dispatch.tensor<writeonly:256x1024xf16>
          }
        }
        return
      }
      hal.interface private @io {
        hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// CHECK-LABEL: hal.executable.entry_point public @matmul_256x1024x8
//  CHECK-SAME:   passPipeline = "SPIRVVectorize"
