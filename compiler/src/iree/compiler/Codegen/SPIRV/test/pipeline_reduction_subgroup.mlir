// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-linalg-to-spirv-pipeline)))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @subgroup_reduce {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformShuffle], []>, ARM:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
       subgroup_size = 16>>
    }>) {
    hal.executable.export public @subgroup_reduce ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @subgroup_reduce() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x512xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x512xf32>> -> tensor<2x512xf32>
        %3 = tensor.empty() : tensor<2xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2xf32>) -> tensor<2xf32>
        %5 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%2 : tensor<2x512xf32>) outs(%4 : tensor<2xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %6 = arith.addf %arg1, %arg0 : f32
          linalg.yield %6 : f32
        } -> tensor<2xf32>
        flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: spirv.func @subgroup_reduce()

// CHECK-DAG:   %[[C0:.+]] = spirv.Constant 0 : i32
// CHECK-DAG:   %[[C1:.+]] = spirv.Constant 1 : i32
// CHECK-DAG:   %[[C2:.+]] = spirv.Constant 2 : i32
// CHECK-DAG:   %[[C4:.+]] = spirv.Constant 4 : i32
// CHECK-DAG:   %[[C8:.+]] = spirv.Constant 8 : i32
// CHECK-DAG:   %[[F0:.+]] = spirv.Constant 0.000000e+00 : f32
// CHECK-DAG:   %[[FV0:.+]] = spirv.Constant dense<0.000000e+00> : vector<4xf32>

// CHECK:   %[[LD:.+]] = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
// CHECK:   %[[ADDV0:.+]] = spirv.FAdd %[[LD]], %[[FV0]] : vector<4xf32>

// CHECK:   %[[E0:.+]] = spirv.CompositeExtract %[[ADDV0]][0 : i32] : vector<4xf32>
// CHECK:   %[[E1:.+]] = spirv.CompositeExtract %[[ADDV0]][1 : i32] : vector<4xf32>
// CHECK:   %[[E2:.+]] = spirv.CompositeExtract %[[ADDV0]][2 : i32] : vector<4xf32>
// CHECK:   %[[E3:.+]] = spirv.CompositeExtract %[[ADDV0]][3 : i32] : vector<4xf32>

// CHECK:   %[[ADD0:.+]] = spirv.FAdd %[[E0]], %[[E1]] : f32
// CHECK:   %[[ADD1:.+]] = spirv.FAdd %[[ADD0]], %[[E2]] : f32
// CHECK:   %[[ADD2:.+]] = spirv.FAdd %[[ADD1]], %[[E3]] : f32

// CHECK:   %[[S0:.+]] = spirv.GroupNonUniformShuffleXor <Subgroup> %[[ADD2]], %[[C1]] : f32, i32
// CHECK:   %[[ADD3:.+]] = spirv.FAdd %[[ADD2]], %[[S0]] : f32
// CHECK:   %[[S1:.+]] = spirv.GroupNonUniformShuffleXor <Subgroup> %[[ADD3]], %[[C2]] : f32, i32
// CHECK:   %[[ADD4:.+]] = spirv.FAdd %[[ADD3]], %[[S1]] : f32
// CHECK:   %[[S2:.+]] = spirv.GroupNonUniformShuffleXor <Subgroup> %[[ADD4]], %[[C4]] : f32, i32
// CHECK:   %[[ADD5:.+]] = spirv.FAdd %[[ADD4]], %[[S2]] : f32
// CHECK:   %[[S3:.+]] = spirv.GroupNonUniformShuffleXor <Subgroup> %[[ADD5]], %[[C8]] : f32, i32
// CHECK:   %[[ADD6:.+]] = spirv.FAdd %[[ADD5]], %[[S3]] : f32

// CHECK:   spirv.Store "Workgroup" %{{.+}}, %[[ADD6]] : f32

// CHECK:   spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

// CHECK:   %[[LOAD_VAL:.+]] = spirv.Load "Workgroup" {{.+}} : f32
// CHECK:   %[[S4:.+]] = spirv.GroupNonUniformShuffleXor <Subgroup> %[[LOAD_VAL]], %[[C1]] : f32, i32
// CHECK:   %[[ADD7:.+]] = spirv.FAdd %[[LOAD_VAL]], %[[S4]] : f32
// CHECK:   %[[S5:.+]] = spirv.GroupNonUniformShuffleXor <Subgroup> %[[ADD7]], %[[C2]] : f32, i32
// CHECK:   %[[ADD8:.+]] = spirv.FAdd %[[ADD7]], %[[S5]] : f32
// CHECK:   %[[S6:.+]] = spirv.GroupNonUniformShuffleXor <Subgroup> %[[ADD8]], %[[C4]] : f32, i32
// CHECK:   %[[ADD9:.+]] = spirv.FAdd %[[ADD8]], %[[S6]] : f32
// CHECK:   %[[S7:.+]] = spirv.GroupNonUniformShuffle <Subgroup> %[[ADD9]], %[[C0]] : f32, i32
// CHECK:   %[[ADD10:.+]] = spirv.FAdd %[[S7]], %[[F0]] : f32

// CHECK:   %[[EQ:.+]] = spirv.IEqual %{{.+}}, %[[C0]] : i32
// CHECK:   spirv.mlir.selection {
// CHECK:     spirv.BranchConditional %[[EQ]], ^bb1, ^bb2
// CHECK:   ^bb1:
// CHECK:     spirv.Store "StorageBuffer" %{{.+}}, %[[ADD10]] : f32
// CHECK:     spirv.Branch ^bb2
// CHECK:   ^bb2:
// CHECK:     spirv.mlir.merge
// CHECK:   }
// CHECK:   spirv.Return

// CHECK: spirv.ExecutionMode @{{.+}} "LocalSize", 128, 1, 1

// -----

// Check the case of no GroupNonUniformShuffle capability.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @subgroup_reduce {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, ARM:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
       subgroup_size = 16>>
    }>) {
    hal.executable.export public @subgroup_reduce ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @subgroup_reduce() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x512xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x512xf32>> -> tensor<2x512xf32>
        %3 = tensor.empty() : tensor<2xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2xf32>) -> tensor<2xf32>
        %5 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%2 : tensor<2x512xf32>) outs(%4 : tensor<2xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %6 = arith.addf %arg1, %arg0 : f32
          linalg.yield %6 : f32
        } -> tensor<2xf32>
        flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2xf32>>
        return
      }
    }
  }
}

// CHECK-NOT: spirv.GroupNonUniformShuffleXor
