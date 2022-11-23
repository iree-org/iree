// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass)))' %s | FileCheck %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3,
    [Shader, GroupNonUniform, GroupNonUniformShuffle], [SPV_KHR_storage_buffer_storage_class]>, Unknown:Unknown,
    #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 32, cooperative_matrix_properties_nv = []>>}>

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>

hal.executable @warp_reduction_dispatch {
  hal.executable.variant public @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
    hal.executable.export public @warp_reduction_dispatch ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @warp_reduction_dispatch() {
        %c0 = arith.constant 0 : index
        %c10240 = arith.constant 10240 : index
        %cst = arith.constant 1.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x10240xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<512xf32>>
        %5 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 10240], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<512x10240xf32>> -> tensor<512x10240xf32>
        %8 = tensor.empty() : tensor<512xf32>
        %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<512xf32>) -> tensor<512xf32>
        %10 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]}
            ins(%5 : tensor<512x10240xf32>) outs(%9 : tensor<512xf32>) {
          ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
            %11 = arith.addf %arg1, %arg2 : f32
            linalg.yield %11 : f32
          } -> tensor<512xf32>
        flow.dispatch.tensor.store %10, %1, offsets = [0], sizes = [512], strides = [1]
            : tensor<512xf32> -> !flow.dispatch.tensor<writeonly:tensor<512xf32>>
        return
      }
    }
  }
}

//   CHECK-LABEL:  func.func @warp_reduction_dispatch
//     CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//     CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : i32
//     CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : i32
//     CHECK-DAG:    %[[C4:.+]] = arith.constant 4 : i32
//     CHECK-DAG:    %[[C8:.+]] = arith.constant 8 : i32
//     CHECK-DAG:    %[[C16:.+]] = arith.constant 16 : i32
//     CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : i32
//     CHECK-DAG:    %[[C4I:.+]] = arith.constant 4 : index
//     CHECK-DAG:    %[[C32I:.+]] = arith.constant 32 : index
//     CHECK-DAG:    %[[C512:.+]] = arith.constant 512 : index
//     CHECK-DAG:    %[[C10240:.+]] = arith.constant 10240 : index
//     CHECK-DAG:    %[[IDENTITY:.+]] = arith.constant 0.000000e+00 : f32
//     CHECK-DAG:    %[[CF:.+]] = arith.constant 1.000000e+00 : f32
//     CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
//     CHECK-DAG:    %[[TID:.+]] = gpu.thread_id  x
//         CHECK:    %[[R0:.+]] = scf.for %{{.*}} = %[[C0]] to %[[C10240]] step %[[C512]] iter_args(%[[A0:.+]] = %[[CST]]) -> (vector<4xf32>) {
//         CHECK:      %[[V:.+]] = vector.transfer_read {{.*}} {in_bounds = [true]} : memref<512x10240xf32>, vector<4xf32>
//         CHECK:      %[[A1:.+]] = arith.addf %[[V]], %[[A0]] : vector<4xf32>
//         CHECK:      scf.yield %[[A1]] : vector<4xf32>
//         CHECK:    }
//         CHECK:    %[[R1:.+]] = vector.reduction <add>, %[[R0]] : vector<4xf32> into f32
//         CHECK:    %[[S0:.+]], %{{.*}} = gpu.shuffle  xor %[[R1]], %[[C1]], %[[C32]] : f32
//         CHECK:    %[[R2:.+]] = arith.addf %[[R1]], %[[S0]] : f32
//         CHECK:    %[[S1:.+]], %{{.*}} = gpu.shuffle  xor %[[R2]], %[[C2]], %[[C32]] : f32
//         CHECK:    %[[R3:.+]] = arith.addf %[[R2]], %[[S1]] : f32
//         CHECK:    %[[S2:.+]], %{{.*}} = gpu.shuffle  xor %[[R3]], %[[C4]], %[[C32]] : f32
//         CHECK:    %[[R4:.+]] = arith.addf %[[R3]], %[[S2]] : f32
//         CHECK:    %[[S3:.+]], %{{.*}} = gpu.shuffle  xor %[[R4]], %[[C8]], %[[C32]] : f32
//         CHECK:    %[[R5:.+]] = arith.addf %[[R4]], %[[S3]] : f32
//         CHECK:    %[[S4:.+]], %{{.*}} = gpu.shuffle  xor %[[R5]], %[[C16]], %[[C32]] : f32
//         CHECK:    %[[R6:.+]] = arith.addf %[[R5]], %[[S4]] : f32
//         CHECK:    %[[ALLOC:.+]] = memref.alloc() : memref<4xf32, 3>
//         CHECK:    %[[WID:.+]] = arith.divui %{{.*}}, %{{.*}} : index
//         CHECK:    %[[LANE_ID:.*]] = arith.remui %[[TID]], %[[C32I]] : index
//         CHECK:    %[[LANE0:.*]] = arith.cmpi eq, %[[LANE_ID]], %[[C0]] : index
//         CHECK:    scf.if %[[LANE0]] { 
//         CHECK:      memref.store %[[R6]], %[[ALLOC]][%[[WID]]] : memref<4xf32, 3>
//         CHECK:    }
//         CHECK:    gpu.barrier
//         CHECK:    %[[LOAD_VAL:.+]] = memref.load %[[ALLOC]][%[[LANE_ID]]] : memref<4xf32, 3>
//         CHECK:    %[[USE_IDENTITY:.+]] = arith.cmpi sge, %[[LANE_ID]], %[[C4I]] : index
//         CHECK:    %[[LANE_VAL:.+]] = arith.select %[[USE_IDENTITY]], %[[IDENTITY]], %[[LOAD_VAL]] : f32
//         CHECK:    %[[S5:.+]], %{{.*}} = gpu.shuffle  xor %[[LANE_VAL]], %[[C1]], %[[C32]] : f32
//         CHECK:    %[[R7:.+]] = arith.addf %[[LANE_VAL]], %[[S5]] : f32
//         CHECK:    %[[S6:.+]], %{{.*}} = gpu.shuffle  xor %[[R7]], %[[C2]], %[[C32]] : f32
//         CHECK:    %[[R8:.+]] = arith.addf %[[R7]], %[[S6]] : f32
//         CHECK:    %[[S7:.+]], %{{.*}} = gpu.shuffle  xor %[[R8]], %[[C4]], %[[C32]] : f32
//         CHECK:    %[[R9:.+]] = arith.addf %[[R8]], %[[S7]] : f32
//         CHECK:    %[[S8:.+]], %{{.*}} = gpu.shuffle  xor %[[R9]], %[[C8]], %[[C32]] : f32
//         CHECK:    %[[R10:.+]] = arith.addf %[[R9]], %[[S8]] : f32
//         CHECK:    %[[S9:.+]], %{{.*}} = gpu.shuffle  xor %[[R10]], %[[C16]], %[[C32]] : f32
//         CHECK:    %[[R11:.+]] = arith.addf %[[R10]], %[[S9]] : f32
//         CHECK:    %[[R12:.+]] = arith.addf %[[R11]], %[[CF]] : f32
//         CHECK:    %[[R13:.+]] = vector.splat %[[R12]] : vector<1xf32>
//         CHECK:    %[[TID0:.+]] = arith.cmpi eq, %[[TID]], %[[C0]] : index
//         CHECK:    scf.if %[[TID0]] {
//         CHECK:      vector.transfer_write %[[R13]], %{{.*}}[%{{.*}}] {in_bounds = [true]} : vector<1xf32>, memref<512xf32>
//         CHECK:    }
