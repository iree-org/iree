// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-spirv-configuration-pipeline, iree-codegen-linalg-to-spirv-pipeline, canonicalize, cse)))' \
// RUN:   %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>

hal.executable public @matmul_256x1024x128_div_exp {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [2147483647, 65535, 65535],
        subgroup_size = 32>
       >}>) {
    hal.executable.export public @matmul_256x1024x128_div_exp layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module  {
      func.func @matmul_256x1024x128_div_exp() {
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x128xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x1024xf16>>
        %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
        %11 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>> -> tensor<256x1024xf16>
        %14 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>> -> tensor<256x1024xf16>
        %17 = tensor.empty() : tensor<256x1024xf16>
        %19 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128xf16>> -> tensor<256x128xf16>
        %21 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [128, 1204], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x1024xf16>> -> tensor<128x1024xf16>
        %24 = tensor.empty() : tensor<256x1024xf16>
        %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %26 = linalg.matmul ins(%19, %21 : tensor<256x128xf16>, tensor<128x1024xf16>) outs(%25 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %27 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
          ins(%26, %11, %14 : tensor<256x1024xf16>, tensor<256x1024xf16>, tensor<256x1024xf16>)
          outs(%17 : tensor<256x1024xf16>) {
        ^bb0(%arg2: f16, %arg3: f16, %arg4: f16, %arg5: f16):
          %28 = arith.divf %arg2, %arg3 : f16
          // spirv.GL.FAbs is not permitted to use cooperative matrix types per the spec.
          %29 = math.absf %28 : f16
          linalg.yield %29 : f16
        } -> tensor<256x1024xf16>
        flow.dispatch.tensor.store %27, %4, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : tensor<256x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
        return
      }
    }
  }
}

//   CHECK-LABEL: spirv.module Logical GLSL450

// With bank conflict reduction, the allocations get padded
//     A matrix gets padded from 256 -> 320
//     B matrix gets padded from 256 -> 288
//     C matrix gets padded from 512 -> 576
//
// This updates the strides in the corresponding cooperative matrix
// loads/stores as well (e.g. stride of 4 to 5 for loads of A).
//
// multi-buffering then doubles the shared memory usage for A and B.
//         CHECK:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<576 x vector<4xf32>>)>, Workgroup>
//         CHECK:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<640 x vector<4xf32>>)>, Workgroup>
//         CHECK:   spirv.GlobalVariable @[[C_MEM:.+]] : !spirv.ptr<!spirv.struct<(!spirv.array<576 x vector<4xf32>>)>, Workgroup>

//         CHECK:   spirv.func @matmul_256x1024x128_div_exp

//     CHECK-DAG:     %[[C5:.+]] = spirv.Constant 5 : i32
//     CHECK-DAG:     %[[C9:.+]] = spirv.Constant 9 : i32
//     CHECK-DAG:     %[[C32:.+]] = spirv.Constant 32 : i32
//     CHECK-DAG:     %[[C128:.+]] = spirv.Constant 128 : i32
//     CHECK-DAG:     %[[F0:.+]] = spirv.Constant 0.000000e+00 : f16
//         CHECK:     %{{.+}} = spirv.CompositeConstruct %[[F0]] : (f16) -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>

//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
//         CHECK:     %[[LOCAL_VAR0:.+]] = spirv.Variable : !spirv.ptr<!spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>, Function>
//         CHECK:     %[[LOCAL_VAR1:.+]] = spirv.Variable : !spirv.ptr<!spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>, Function>
//         CHECK:     %[[LOCAL_VAR2:.+]] = spirv.Variable : !spirv.ptr<!spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>, Function>
//         CHECK:     %[[LOCAL_VAR3:.+]] = spirv.Variable : !spirv.ptr<!spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>, Function>
//         CHECK:     spirv.mlir.loop
//         CHECK:       %[[LD0:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
//         CHECK:       %[[LD1:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>

//         CHECK:       %[[LD2:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
//         CHECK:       %[[LD3:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
//         CHECK:       %[[LD4:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C9]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>
//         CHECK:       %[[LD5:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C9]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>

//         CHECK:       %[[LD6:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C9]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>
//         CHECK:       %[[LD7:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C9]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>

//         CHECK:       %[[MA0:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD0]], %[[LD4]], %{{.+}}
//         CHECK:       %[[MA1:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD1]], %[[LD6]], %[[MA0]]
//         CHECK:       %[[MA2:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD0]], %[[LD5]], %{{.+}}
//         CHECK:       %[[MA3:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD1]], %[[LD7]], %[[MA2]]
//         CHECK:       %[[MA4:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD2]], %[[LD4]], %{{.+}}
//         CHECK:       %[[MA5:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD3]], %[[LD6]], %[[MA4]]
//         CHECK:       %[[MA6:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD2]], %[[LD5]], %{{.+}}
//         CHECK:       %[[MA7:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD3]], %[[LD7]], %[[MA6]]

//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

//         CHECK:       spirv.Store "Function" %[[LOCAL_VAR0]], %[[MA1]]
//         CHECK:       spirv.Store "Function" %[[LOCAL_VAR1]], %[[MA3]]
//         CHECK:       spirv.Store "Function" %[[LOCAL_VAR2]], %[[MA5]]
//         CHECK:       spirv.Store "Function" %[[LOCAL_VAR3]], %[[MA7]]

//         CHECK:       spirv.mlir.merge

//         CHECK:     %[[LD_FN0:.+]] = spirv.Load "Function" %[[LOCAL_VAR3]] : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
//         CHECK:     %[[LD_FN1:.+]] = spirv.Load "Function" %[[LOCAL_VAR2]] : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
//         CHECK:     %[[LD_FN2:.+]] = spirv.Load "Function" %[[LOCAL_VAR1]] : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
//         CHECK:     %[[LD_FN3:.+]] = spirv.Load "Function" %[[LOCAL_VAR0]] : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>

//         CHECK:     %[[LD0:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
//         CHECK:     %[[LD1:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>

//         CHECK:     %[[LD2:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
//         CHECK:     %[[LD3:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
//         CHECK:     %[[LD4:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C9]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>
//         CHECK:     %[[LD5:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C9]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>

//         CHECK:     %[[LD6:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C9]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>
//         CHECK:     %[[LD7:.+]] = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C9]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>

//         CHECK:     %[[MA0:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD0]], %[[LD4]], %{{.+}}
//         CHECK:     %[[MA1:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD1]], %[[LD6]], %[[MA0]]
//         CHECK:     %[[MA2:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD0]], %[[LD5]], %{{.+}}
//         CHECK:     %[[MA3:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD1]], %[[LD7]], %[[MA2]]
//         CHECK:     %[[MA4:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD2]], %[[LD4]], %{{.+}}
//         CHECK:     %[[MA5:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD3]], %[[LD6]], %[[MA4]]
//         CHECK:     %[[MA6:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD2]], %[[LD5]], %{{.+}}
//         CHECK:     %[[MA7:.+]] = spirv.KHR.CooperativeMatrixMulAdd %[[LD3]], %[[LD7]], %[[MA6]]

//         CHECK:     %[[AC:.+]] = spirv.AccessChain %[[C_MEM]]
//         CHECK:     spirv.KHR.CooperativeMatrixStore %[[AC]], %[[MA7]], %[[C9]], <RowMajor>
//         CHECK:     %[[AC:.+]] = spirv.AccessChain %[[C_MEM]]
//         CHECK:     spirv.KHR.CooperativeMatrixStore %[[AC]], %[[MA5]], %[[C9]], <RowMajor>
//         CHECK:     %[[AC:.+]] = spirv.AccessChain %[[C_MEM]]
//         CHECK:     spirv.KHR.CooperativeMatrixStore %[[AC]], %[[MA3]], %[[C9]], <RowMajor>
//         CHECK:     %[[AC:.+]] = spirv.AccessChain %[[C_MEM]]
//         CHECK:     spirv.KHR.CooperativeMatrixStore %[[AC]], %[[MA1]], %[[C9]], <RowMajor>

//         CHECK:     spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
//         CHECK:     spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Load "Workgroup" %{{.+}} : vector<4xf32>
// CHECK-COUNT-2:     spirv.FDiv %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:     spirv.GL.FAbs %{{.+}} : vector<4xf16>
//         CHECK:     spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Load "Workgroup" %{{.+}} : vector<4xf32>
// CHECK-COUNT-2:     spirv.FDiv %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:     spirv.GL.FAbs %{{.+}} : vector<4xf16>
//         CHECK:     spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Load "Workgroup" %{{.+}} : vector<4xf32>
// CHECK-COUNT-2:     spirv.FDiv %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:     spirv.GL.FAbs %{{.+}} : vector<4xf16>
//         CHECK:     spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Load "Workgroup" %{{.+}} : vector<4xf32>
// CHECK-COUNT-2:     spirv.FDiv %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:     spirv.GL.FAbs %{{.+}} : vector<4xf16>
//         CHECK:     spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable public @batch_matmul_16x128x256x512_div {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [2147483647, 65535, 65535],
        subgroup_size = 32>
       >}>) {
    hal.executable.export public @batch_matmul_16x128x256x512_div layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @batch_matmul_16x128x256x512_div() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x128x512xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x512x256xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x128x256xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x128x256xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 128, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x128x512xf16>> -> tensor<16x128x512xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 512, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x256xf16>> -> tensor<16x512x256xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [16, 128, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x128x256xf16>> -> tensor<16x128x256xf16>
        %7 = tensor.empty() : tensor<16x128x256xf16>
        %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<16x128x256xf16>) -> tensor<16x128x256xf16>
        %9 = linalg.batch_matmul ins(%4, %5 : tensor<16x128x512xf16>, tensor<16x512x256xf16>) outs(%8 : tensor<16x128x256xf16>) -> tensor<16x128x256xf16>
        %10 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%9, %6 : tensor<16x128x256xf16>, tensor<16x128x256xf16>) outs(%7 : tensor<16x128x256xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %11 = arith.divf %in, %in_0 : f16
          linalg.yield %11 : f16
        } -> tensor<16x128x256xf16>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0, 0], sizes = [16, 128, 256], strides = [1, 1, 1] : tensor<16x128x256xf16> -> !flow.dispatch.tensor<writeonly:tensor<16x128x256xf16>>
        return
      }
    }
  }
}

//   CHECK-LABEL: spirv.module Logical GLSL450

//         CHECK:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<576 x vector<4xf32>>)>, Workgroup>
//         CHECK:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<640 x vector<4xf32>>)>, Workgroup>

//         CHECK:   spirv.func @batch_matmul_16x128x256x512_div

//     CHECK-DAG:     %[[C5:.+]] = spirv.Constant 5 : i32
//     CHECK-DAG:     %[[C9:.+]] = spirv.Constant 9 : i32
//     CHECK-DAG:     %[[C32:.+]] = spirv.Constant 32 : i32
//     CHECK-DAG:     %[[F0:.+]] = spirv.Constant 0.000000e+00 : f16
//         CHECK:     %{{.+}} = spirv.CompositeConstruct %[[F0]] : (f16) -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>

//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

// CHECK-COUNT-4:     %{{.+}} = spirv.Variable : !spirv.ptr<!spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>, Function>
//         CHECK:     spirv.mlir.loop
// CHECK-COUNT-4:       %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
// CHECK-COUNT-4:       %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C9]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>

// CHECK-COUNT-8:       %{{.+}} = spirv.KHR.CooperativeMatrixMulAdd %{{.+}}, %{{.+}}, %{{.+}}

//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
// CHECK-COUNT-4:       spirv.Store "Function" %{{.+}}, %{{.+}}
//         CHECK:       spirv.mlir.merge

// CHECK-COUNT-4:     %{{.+}} = spirv.Load "Function" %{{.+}} : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>

// CHECK-COUNT-4:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
// CHECK-COUNT-4:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C9]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>
// CHECK-COUNT-8:     %{{.+}} = spirv.KHR.CooperativeMatrixMulAdd %{{.+}}, %{{.+}}, %{{.+}}

// CHECK-COUNT-4:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C32]], <RowMajor> : !spirv.ptr<vector<4xf32>, StorageBuffer>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
// CHECK-COUNT-4:     %{{.+}} = spirv.FDiv %{{.+}}, %{{.+}} : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
// CHECK-COUNT-4:     spirv.KHR.CooperativeMatrixStore %{{.+}}, %{{.+}}, %[[C32]], <RowMajor>


// -----

// Small matmul that each subgroup only handles one tile

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>

hal.executable public @matmul_32x32x32_div {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [2147483647, 65535, 65535],
        subgroup_size = 32>
       >}>) {
    hal.executable.export public @matmul_32x32x32_div layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module  {
      func.func @matmul_32x32x32_div() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<32x32xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<32x32xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<32x32xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x32xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x32xf16>> -> tensor<32x32xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x32xf16>> -> tensor<32x32xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x32xf16>> -> tensor<32x32xf16>
        %7 = tensor.empty() : tensor<32x32xf16>
        %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<32x32xf16>) -> tensor<32x32xf16>
        %9 = linalg.matmul ins(%4, %5 : tensor<32x32xf16>, tensor<32x32xf16>) outs(%8 : tensor<32x32xf16>) -> tensor<32x32xf16>
        %10 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
        ins(%9, %6 : tensor<32x32xf16>, tensor<32x32xf16>) outs(%7 : tensor<32x32xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %11 = arith.divf %in, %in_0 : f16
          linalg.yield %11 : f16
        } -> tensor<32x32xf16>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : tensor<32x32xf16> -> !flow.dispatch.tensor<writeonly:tensor<32x32xf16>>
        return
      }
    }
  }
}

//   CHECK-LABEL: spirv.module Logical GLSL450
// CHECK-COUNT-4: spirv.KHR.CooperativeMatrixLoad
// CHECK-COUNT-2: spirv.KHR.CooperativeMatrixMulAdd
//         CHECK: spirv.KHR.CooperativeMatrixLoad
//         CHECK: spirv.FDiv %{{.+}}, %{{.+}} : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
//         CHECK: spirv.KHR.CooperativeMatrixStore

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

hal.executable public @generic_batch_matmul_32x128x512x64 {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>,
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [2147483647, 65535, 65535],
        subgroup_size = 32>
       >}>) {
    hal.executable.export public @generic_batch_matmul_32x128x512x64 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module  {
      func.func @generic_batch_matmul_32x128x512x64() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<32x128x64xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x512xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x128x512xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 128, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128x64xf16>> -> tensor<32x128x64xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x512xf16>> -> tensor<64x512xf16>
        %5 = tensor.empty() : tensor<32x128x512xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<32x128x512xf16>) -> tensor<32x128x512xf16>
        %7 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%3, %4 : tensor<32x128x64xf16>, tensor<64x512xf16>) outs(%6 : tensor<32x128x512xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<32x128x512xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [32, 128, 512], strides = [1, 1, 1] : tensor<32x128x512xf16> -> !flow.dispatch.tensor<writeonly:tensor<32x128x512xf16>>
        return
      }
    }
  }
}

// With pipelining + multi-buffering the loop here gets completely unrolled.
//   CHECK-LABEL: spirv.module Logical GLSL450

//         CHECK:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<576 x vector<4xf32>>)>, Workgroup>
//         CHECK:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<640 x vector<4xf32>>)>, Workgroup>

//         CHECK:   spirv.func @generic_batch_matmul_32x128x512x64

//     CHECK-DAG:     %[[C5:.+]] = spirv.Constant 5 : i32
//     CHECK-DAG:     %[[C9:.+]] = spirv.Constant 9 : i32
//     CHECK-DAG:     %[[C64:.+]] = spirv.Constant 64 : i32
//     CHECK-DAG:     %[[C256:.+]] = spirv.Constant 256 : i32
//     CHECK-DAG:     %[[F0:.+]] = spirv.Constant 0.000000e+00 : f16
//         CHECK:     %{{.+}} = spirv.CompositeConstruct %[[F0]] : (f16) -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>

//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

// CHECK-COUNT-4:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
// CHECK-COUNT-4:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C9]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>

// CHECK-COUNT-8:     %{{.+}} = spirv.KHR.CooperativeMatrixMulAdd %{{.+}}, %{{.+}}, %{{.+}}

//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

// CHECK-COUNT-4:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
// CHECK-COUNT-4:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C9]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>

// CHECK-COUNT-8:     %{{.+}} = spirv.KHR.CooperativeMatrixMulAdd %{{.+}}, %{{.+}}, %{{.+}}

// CHECK-COUNT-4:     spirv.KHR.CooperativeMatrixStore %{{.+}}, %{{.+}}, %[[C64]], <RowMajor>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>

hal.executable public @matmul_256x1024x128_div_exp {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>
       >}>) {
    hal.executable.export public @matmul_256x1024x128_div_exp layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module  {
      func.func @matmul_256x1024x128_div_exp() {
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x128xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x1024xf16>>
        %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
        %11 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>> -> tensor<256x1024xf16>
        %14 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>> -> tensor<256x1024xf16>
        %17 = tensor.empty() : tensor<256x1024xf16>
        %19 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [256, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128xf16>> -> tensor<256x128xf16>
        %21 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [128, 1204], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x1024xf16>> -> tensor<128x1024xf16>
        %24 = tensor.empty() : tensor<256x1024xf16>
        %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %26 = linalg.matmul ins(%19, %21 : tensor<256x128xf16>, tensor<128x1024xf16>) outs(%25 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %27 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
          ins(%26, %11, %14 : tensor<256x1024xf16>, tensor<256x1024xf16>, tensor<256x1024xf16>)
          outs(%17 : tensor<256x1024xf16>) {
        ^bb0(%arg2: f16, %arg3: f16, %arg4: f16, %arg5: f16):
          %28 = arith.divf %arg2, %arg3 : f16
          // spirv.GL.FAbs is not permitted to use cooperative matrix types per the spec.
          %29 = math.absf %28 : f16
          linalg.yield %29 : f16
        } -> tensor<256x1024xf16>
        flow.dispatch.tensor.store %27, %4, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : tensor<256x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
        return
      }
    }
  }
}

//   CHECK-LABEL: spirv.module Logical GLSL450

//     CHECK-DAG:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<1088 x vector<4xf32>>)>, Workgroup>
//     CHECK-DAG:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<640 x vector<4xf32>>)>, Workgroup>
//     CHECK-DAG:   spirv.GlobalVariable @[[C_MEM:.+]] : !spirv.ptr<!spirv.struct<(!spirv.array<1088 x vector<4xf32>>)>, Workgroup>

//         CHECK:   spirv.func @matmul_256x1024x128_div_exp

//     CHECK-DAG:     %[[C5:.+]] = spirv.Constant 5 : i32
//     CHECK-DAG:     %[[C17:.+]] = spirv.Constant 17 : i32
//     CHECK-DAG:     %[[C32:.+]] = spirv.Constant 32 : i32
//     CHECK-DAG:     %[[C128:.+]] = spirv.Constant 128 : i32
//     CHECK-DAG:     %[[F0:.+]] = spirv.Constant 0.000000e+00 : f16
//         CHECK:     %{{.+}} = spirv.CompositeConstruct %[[F0]] : (f16) -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>

//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

// CHECK-COUNT-8:     %{{.+}} = spirv.Variable : !spirv.ptr<!spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>, Function>
//         CHECK:     spirv.mlir.loop
// CHECK-COUNT-4:       %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
// CHECK-COUNT-8:       %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C17]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>

//CHECK-COUNT-16:       %{{.+}} = spirv.KHR.CooperativeMatrixMulAdd %{{.+}}, %{{.+}}, %{{.+}}

//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
// CHECK-COUNT-8:       spirv.Store "Function" %{{.+}}, %{{.+}}
//         CHECK:       spirv.mlir.merge

// CHECK-COUNT-8:     %{{.+}} = spirv.Load "Function" %{{.+}} : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
// CHECK-COUNT-4:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
// CHECK-COUNT-8:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C17]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>

//CHECK-COUNT-16:     %{{.+}} = spirv.KHR.CooperativeMatrixMulAdd %{{.+}}, %{{.+}}, %{{.+}}

// CHECK-COUNT-8:     spirv.KHR.CooperativeMatrixStore %{{.+}}, %{{.+}}, %[[C17]], <RowMajor>

//         CHECK:     spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
// CHECK-COUNT-2:     spirv.FDiv %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:     spirv.GL.FAbs %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:     spirv.FDiv %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:     spirv.GL.FAbs %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:     spirv.FDiv %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:     spirv.GL.FAbs %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:     spirv.FDiv %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:     spirv.GL.FAbs %{{.+}} : vector<4xf16>
//         CHECK:     spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable public @batch_matmul_16x128x256x512_div {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>
       >}>) {
    hal.executable.export public @batch_matmul_16x128x256x512_div layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @batch_matmul_16x128x256x512_div() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x128x512xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x512x256xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x128x256xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x128x256xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 128, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x128x512xf16>> -> tensor<16x128x512xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 512, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x256xf16>> -> tensor<16x512x256xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [16, 128, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x128x256xf16>> -> tensor<16x128x256xf16>
        %7 = tensor.empty() : tensor<16x128x256xf16>
        %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<16x128x256xf16>) -> tensor<16x128x256xf16>
        %9 = linalg.batch_matmul ins(%4, %5 : tensor<16x128x512xf16>, tensor<16x512x256xf16>) outs(%8 : tensor<16x128x256xf16>) -> tensor<16x128x256xf16>
        %10 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%9, %6 : tensor<16x128x256xf16>, tensor<16x128x256xf16>) outs(%7 : tensor<16x128x256xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %11 = arith.divf %in, %in_0 : f16
          linalg.yield %11 : f16
        } -> tensor<16x128x256xf16>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0, 0], sizes = [16, 128, 256], strides = [1, 1, 1] : tensor<16x128x256xf16> -> !flow.dispatch.tensor<writeonly:tensor<16x128x256xf16>>
        return
      }
    }
  }
}

//   CHECK-LABEL: spirv.module Logical GLSL450

//     CHECK-DAG:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<1088 x vector<4xf32>>)>, Workgroup>
//     CHECK-DAG:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<640 x vector<4xf32>>)>, Workgroup>

//         CHECK:   spirv.func @batch_matmul_16x128x256x512_div

//     CHECK-DAG:     %[[C5:.+]] = spirv.Constant 5 : i32
//     CHECK-DAG:     %[[C17:.+]] = spirv.Constant 17 : i32
//     CHECK-DAG:     %[[C32:.+]] = spirv.Constant 32 : i32
//     CHECK-DAG:     %[[F0:.+]] = spirv.Constant 0.000000e+00 : f16
//         CHECK:     %{{.+}} = spirv.CompositeConstruct %[[F0]] : (f16) -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>

//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

// CHECK-COUNT-4:     %{{.+}} = spirv.Variable : !spirv.ptr<!spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>, Function>
//         CHECK:     spirv.mlir.loop
// CHECK-COUNT-4:       %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
// CHECK-COUNT-4:       %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C17]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>

// CHECK-COUNT-8:       %{{.+}} = spirv.KHR.CooperativeMatrixMulAdd %{{.+}}, %{{.+}}, %{{.+}}

//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:       spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:       spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
// CHECK-COUNT-4:       spirv.Store "Function" %{{.+}}, %{{.+}}
//         CHECK:       spirv.mlir.merge

// CHECK-COUNT-8:     %{{.+}} = spirv.Load "Function" %{{.+}} : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>

// CHECK-COUNT-4:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
// CHECK-COUNT-4:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C17]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>

// CHECK-COUNT-8:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C32]], <RowMajor> : !spirv.ptr<vector<4xf32>, StorageBuffer>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
// CHECK-COUNT-8:     %{{.+}} = spirv.FDiv %{{.+}}, %{{.+}} : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
// CHECK-COUNT-8:     spirv.KHR.CooperativeMatrixStore %{{.+}}, %{{.+}}, %[[C32]], <RowMajor>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

hal.executable public @generic_batch_matmul_32x128x512x64 {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>
       >}>) {
    hal.executable.export public @generic_batch_matmul_32x128x512x64 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module  {
      func.func @generic_batch_matmul_32x128x512x64() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<32x128x64xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x512xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x128x512xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 128, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128x64xf16>> -> tensor<32x128x64xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x512xf16>> -> tensor<64x512xf16>
        %5 = tensor.empty() : tensor<32x128x512xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<32x128x512xf16>) -> tensor<32x128x512xf16>
        %7 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%3, %4 : tensor<32x128x64xf16>, tensor<64x512xf16>) outs(%6 : tensor<32x128x512xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<32x128x512xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [32, 128, 512], strides = [1, 1, 1] : tensor<32x128x512xf16> -> !flow.dispatch.tensor<writeonly:tensor<32x128x512xf16>>
        return
      }
    }
  }
}

//   CHECK-LABEL: spirv.module Logical GLSL450

//     CHECK-DAG:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<1088 x vector<4xf32>>)>, Workgroup>
//     CHECK-DAG:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<640 x vector<4xf32>>)>, Workgroup>

//         CHECK:   spirv.func @generic_batch_matmul_32x128x512x64

//     CHECK-DAG:     %[[C5:.+]] = spirv.Constant 5 : i32
//     CHECK-DAG:     %[[C17:.+]] = spirv.Constant 17 : i32
//     CHECK-DAG:     %[[C64:.+]] = spirv.Constant 64 : i32
//     CHECK-DAG:     %[[C256:.+]] = spirv.Constant 256 : i32
//     CHECK-DAG:     %[[F0:.+]] = spirv.Constant 0.000000e+00 : f16
//         CHECK:     %{{.+}} = spirv.CompositeConstruct %[[F0]] : (f16) -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>

//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

// CHECK-COUNT-4:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
// CHECK-COUNT-8:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C17]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>
//CHECK-COUNT-16:     %{{.+}} = spirv.KHR.CooperativeMatrixMulAdd %{{.+}}, %{{.+}}, %{{.+}}

//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     %{{.+}} = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//         CHECK:     spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//         CHECK:     spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

// CHECK-COUNT-4:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C5]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixA>
// CHECK-COUNT-8:     %{{.+}} = spirv.KHR.CooperativeMatrixLoad %{{.+}}, %[[C17]], <RowMajor> : !spirv.ptr<vector<4xf32>, Workgroup>, i32 -> !spirv.coopmatrix<16x16xf16, Subgroup, MatrixB>
//CHECK-COUNT-16:     %{{.+}} = spirv.KHR.CooperativeMatrixMulAdd %{{.+}}, %{{.+}}, %{{.+}}

// CHECK-COUNT-8:     spirv.KHR.CooperativeMatrixStore %{{.+}}, %{{.+}}, %[[C64]], <RowMajor>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

#compilation = #iree_codegen.compilation_info<
    lowering_config  = <tile_sizes = [[1, 64, 64], [1, 16, 64], [0, 0, 0, 16], [1, 16, 16, 16]]>,
    translation_info = <SPIRVCooperativeMatrixVectorize, {pipeline_depth = 1, store_stage = 1}>,
    workgroup_size = [32, 4, 1], subgroup_size = 32>

hal.executable public @batch_matmul_f16_16x4096x4096x64_truncf_mulf {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>
       >}>) {
    hal.executable.export public @batch_matmul_f16_16x4096x4096x64_truncf_mulf layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module  {
      func.func @batch_matmul_f16_16x4096x4096x64_truncf_mulf() {
        %cst = arith.constant 0.158113882 : f32
        %cst_0 = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x4096x64xf16>>
        %7 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x64x4096xf16>>
        %8 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x4096x4096xf16>>
        %9 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0], sizes = [16, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x4096x64xf16>> -> tensor<16x4096x64xf16>
        %10 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0], sizes = [16, 64, 4096], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x64x4096xf16>> -> tensor<16x64x4096xf16>
        %11 = tensor.empty() : tensor<16x4096x4096xf16>
        %12 = linalg.fill ins(%cst_0 : f16) outs(%11 : tensor<16x4096x4096xf16>) -> tensor<16x4096x4096xf16>
        %13 = linalg.batch_matmul {compilation_info = #compilation}
          ins(%9, %10 : tensor<16x4096x64xf16>, tensor<16x64x4096xf16>)
          outs(%12 : tensor<16x4096x4096xf16>) -> tensor<16x4096x4096xf16>
        %14 = linalg.generic {
              indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
              iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%13 : tensor<16x4096x4096xf16>) outs(%11 : tensor<16x4096x4096xf16>) {
        ^bb0(%in: f16, %out: f16):
          %15 = arith.truncf %cst : f32 to f16
          %16 = arith.mulf %in, %15 : f16
          linalg.yield %16 : f16
        } -> tensor<16x4096x4096xf16>
        flow.dispatch.tensor.store %14, %8, offsets = [0, 0, 0], sizes = [16, 4096, 4096], strides = [1, 1, 1] : tensor<16x4096x4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<16x4096x4096xf16>>
        return
      }
    }
  }
}

//   CHECK-LABEL: spirv.module Logical GLSL450

//     CHECK-NOT:   spirv.GlobalVariable {{.+}} Workgroup
// CHECK-COUNT-2:   spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<{{.+}}>)>, Workgroup>
//     CHECK-NOT:   spirv.GlobalVariable {{.+}} Workgroup

//         CHECK:   spirv.func @batch_matmul_f16_16x4096x4096x64_truncf_mulf

//     CHECK-DAG:     %[[C512:.+]] = spirv.Constant 512 : i32
//     CHECK-DAG:     %[[SCALAR:.+]] = spirv.Constant 0.158113882 : f32


// CHECK-COUNT-4:     %{{.+}} = spirv.Load "Function" %{{.+}} : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>
//         CHECK:     %[[CONVERT:.+]] = spirv.FConvert %[[SCALAR]] : f32 to f16
// CHECK-COUNT-4:     %{{.+}} = spirv.MatrixTimesScalar %{{.+}}, %[[CONVERT]] : !spirv.coopmatrix<16x16xf16, Subgroup, MatrixAcc>, f16

// CHECK-COUNT-4:     spirv.KHR.CooperativeMatrixStore %{{.+}}, %{{.+}}, %[[C512]], <RowMajor>
