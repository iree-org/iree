// RUN: iree-opt --split-input-file --pass-pipeline='hal.executable(hal.executable.variant(iree-codegen-linalg-to-spirv-pipeline))' %s | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>

hal.executable public @matmul_256x1024x128_div_sub {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spv.target_env = #spv.target_env<
      #spv.vce<v1.5,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixNV],
      [SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spv.resource_limits<
        cooperative_matrix_properties_nv = [
          #spv.coop_matrix_props<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, scope = 3 : i32>,
          #spv.coop_matrix_props<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, scope = 3 : i32>,
          #spv.coop_matrix_props<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, scope = 3 : i32>
        ],
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [2147483647, 65535, 65535],
        subgroup_size = 32>
       >}> {
    hal.executable.export public @matmul_256x1024x128_div_sub layout(#executable_layout)
    builtin.module  {
      func.func @matmul_256x1024x128_div_sub() {
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:256x1024xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:256x1024xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:256x128xf16>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:128x1024xf16>
        %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : !flow.dispatch.tensor<writeonly:256x1024xf16>
        %11 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x1024xf16> -> tensor<256x1024xf16>
        %14 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x1024xf16> -> tensor<256x1024xf16>
        %17 = linalg.init_tensor [256, 1024] : tensor<256x1024xf16>
        %19 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x128xf16> -> tensor<256x128xf16>
        %21 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [128, 1204], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x1024xf16> -> tensor<128x1024xf16>
        %24 = linalg.init_tensor [256, 1024] : tensor<256x1024xf16>
        %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %26 = linalg.matmul ins(%19, %21 : tensor<256x128xf16>, tensor<128x1024xf16>) outs(%25 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %27 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
          ins(%26, %11, %14 : tensor<256x1024xf16>, tensor<256x1024xf16>, tensor<256x1024xf16>)
          outs(%17 : tensor<256x1024xf16>) {
        ^bb0(%arg2: f16, %arg3: f16, %arg4: f16, %arg5: f16):
          %28 = arith.divf %arg2, %arg3 : f16
          %29 = arith.subf %28, %arg4 : f16
          linalg.yield %29 : f16
        } -> tensor<256x1024xf16>
        flow.dispatch.tensor.store %27, %4, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : tensor<256x1024xf16> -> !flow.dispatch.tensor<writeonly:256x1024xf16>
        return
      }
    }
  }
}

//    CHECK-LABEL: spv.func @matmul_256x1024x128_div_sub
//      CHECK-DAG:   %[[COL_MAJOR:.+]] = spv.Constant false
//      CHECK-DAG:   %[[C128:.+]] = spv.Constant 128 : i32
//      CHECK-DAG:   %[[C1024:.+]] = spv.Constant 1024 : i32
//  CHECK-COUNT-8:   spv.CooperativeMatrixLoadNV %{{.+}}, %[[C128]], %[[COL_MAJOR]]
//  CHECK-COUNT-8:   spv.CooperativeMatrixLoadNV %{{.+}}, %[[C1024]], %[[COL_MAJOR]]
//  CHECK-COUNT-8:   spv.CooperativeMatrixMulAddNV
//          CHECK:   %[[ELEMENTWISE1:.+]] = spv.CooperativeMatrixLoadNV %{{.+}}, %[[C1024]], %[[COL_MAJOR]]
//          CHECK:   %[[ELEMENTWISE2:.+]] = spv.CooperativeMatrixLoadNV %{{.+}}, %[[C1024]], %[[COL_MAJOR]]
//          CHECK:   %[[DIV:.+]] = spv.FDiv %{{.+}}, %[[ELEMENTWISE1]] : !spv.coopmatrix<16x16xf16, Subgroup>
//          CHECK:   %[[SUB:.+]] = spv.FSub %[[DIV]], %[[ELEMENTWISE2]] : !spv.coopmatrix<16x16xf16, Subgroup>
//          CHECK:   spv.CooperativeMatrixStoreNV %{{.+}}, %[[SUB]], %[[C1024]], %[[COL_MAJOR]]
