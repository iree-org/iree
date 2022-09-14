// RUN: iree-opt --split-input-file --pass-pipeline='hal.executable(hal.executable.variant(iree-codegen-linalg-to-spirv-pipeline))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @subgroup_reduce {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader, GroupNonUniformShuffle], []>, ARM:IntegratedGPU, #spv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
       subgroup_size = 16>>
    }> {
    hal.executable.export public @subgroup_reduce ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @subgroup_reduce() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:2x512xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:2xf32>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:2x512xf32> -> tensor<2x512xf32>
        %3 = linalg.init_tensor [2] : tensor<2xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2xf32>) -> tensor<2xf32>
        %5 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%2 : tensor<2x512xf32>) outs(%4 : tensor<2xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %6 = arith.addf %arg1, %arg0 : f32
          linalg.yield %6 : f32
        } -> tensor<2xf32>
        flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:2xf32>
        return
      }
    }
  }
}

// CHECK-LABEL: spv.func @subgroup_reduce()

// CHECK-DAG:   %[[C0:.+]] = spv.Constant 0 : i32
// CHECK-DAG:   %[[C1:.+]] = spv.Constant 1 : i32
// CHECK-DAG:   %[[C2:.+]] = spv.Constant 2 : i32
// CHECK-DAG:   %[[C4:.+]] = spv.Constant 4 : i32
// CHECK-DAG:   %[[C8:.+]] = spv.Constant 8 : i32
// CHECK-DAG:   %[[F0:.+]] = spv.Constant 0.000000e+00 : f32

// CHECK:   %[[LD:.+]] = spv.Load "StorageBuffer" %{{.+}} : vector<4xf32>

// CHECK:   %[[E0:.+]] = spv.CompositeExtract %[[LD]][0 : i32] : vector<4xf32>
// CHECK:   %[[E1:.+]] = spv.CompositeExtract %[[LD]][1 : i32] : vector<4xf32>
// CHECK:   %[[E2:.+]] = spv.CompositeExtract %[[LD]][2 : i32] : vector<4xf32>
// CHECK:   %[[E3:.+]] = spv.CompositeExtract %[[LD]][3 : i32] : vector<4xf32>

// CHECK:   %[[ADD0:.+]] = spv.FAdd %[[E0]], %[[E1]] : f32
// CHECK:   %[[ADD1:.+]] = spv.FAdd %[[ADD0]], %[[E2]] : f32
// CHECK:   %[[ADD2:.+]] = spv.FAdd %[[ADD1]], %[[E3]] : f32

// CHECK:   %[[S0:.+]] = spv.GroupNonUniformShuffleXor <Subgroup> %[[ADD2]], %[[C1]] : f32, i32
// CHECK:   %[[ADD3:.+]] = spv.FAdd %[[ADD2]], %[[S0]] : f32
// CHECK:   %[[S1:.+]] = spv.GroupNonUniformShuffleXor <Subgroup> %[[ADD3]], %[[C2]] : f32, i32
// CHECK:   %[[ADD4:.+]] = spv.FAdd %[[ADD3]], %[[S1]] : f32
// CHECK:   %[[S2:.+]] = spv.GroupNonUniformShuffleXor <Subgroup> %[[ADD4]], %[[C4]] : f32, i32
// CHECK:   %[[ADD5:.+]] = spv.FAdd %[[ADD4]], %[[S2]] : f32
// CHECK:   %[[S3:.+]] = spv.GroupNonUniformShuffleXor <Subgroup> %[[ADD5]], %[[C8]] : f32, i32
// CHECK:   %[[ADD6:.+]] = spv.FAdd %[[ADD5]], %[[S3]] : f32

// CHECK:   spv.Store "Workgroup" %{{.+}}, %[[ADD6]] : f32

// CHECK:   spv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

// CHECK-COUNT-8:   spv.Load "Workgroup" %{{.+}} : f32
// CHECK-COUNT-7:   spv.FAdd %{{.+}}, %{{.+}} : f32
//         CHECK:   spv.FAdd %{{.+}}, %[[F0]] : f32

// CHECK:   %[[EQ:.+]] = spv.IEqual %{{.+}}, %[[C0]] : i32
// CHECK:   spv.mlir.selection {
// CHECK:     spv.BranchConditional %[[EQ]], ^bb1, ^bb2
// CHECK:   ^bb1:
// CHECK:     spv.Store "StorageBuffer" %{{.+}}, %{{.+}} : f32
// CHECK:     spv.Branch ^bb2
// CHECK:   ^bb2:
// CHECK:     spv.mlir.merge
// CHECK:   }
// CHECK:   spv.Return

// CHECK: spv.ExecutionMode @{{.+}} "LocalSize", 128, 1, 1

// -----

// Check the case of no GroupNonUniformShuffle capability.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @subgroup_reduce {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, ARM:IntegratedGPU, #spv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
       subgroup_size = 16>>
    }> {
    hal.executable.export public @subgroup_reduce ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @subgroup_reduce() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:2x512xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:2xf32>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:2x512xf32> -> tensor<2x512xf32>
        %3 = linalg.init_tensor [2] : tensor<2xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2xf32>) -> tensor<2xf32>
        %5 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%2 : tensor<2x512xf32>) outs(%4 : tensor<2xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %6 = arith.addf %arg1, %arg0 : f32
          linalg.yield %6 : f32
        } -> tensor<2xf32>
        flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:2xf32>
        return
      }
    }
  }
}

// CHECK-NOT: spv.GroupNonUniformShuffleXor
