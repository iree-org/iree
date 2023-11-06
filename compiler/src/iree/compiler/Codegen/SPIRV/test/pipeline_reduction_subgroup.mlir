// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-spirv-configuration-pipeline, iree-codegen-linalg-to-spirv-pipeline)))' %s | FileCheck %s

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

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer, ReadOnly>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable private @dynamic_softmax {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.6,
        [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, GroupNonUniformShuffle],
        [SPV_KHR_16bit_storage]>, api=Vulkan, Unknown:DiscreteGPU, #spirv.resource_limits<
          max_compute_shared_memory_size = 65536,
          max_compute_workgroup_invocations = 1024,
          max_compute_workgroup_size = [1024, 1024, 1024],
          subgroup_size = 64>>
    }>) {
    hal.executable.export public @dynamic_softmax ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @dynamic_softmax() {
        %c32_i64 = arith.constant 32 : i64
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32 
        %2 = arith.extui %0 : i32 to i64
        %3 = arith.extui %1 : i32 to i64
        %4 = arith.shli %3, %c32_i64 : i64 
        %5 = arith.ori %2, %4 : i64
        %6 = arith.index_castui %5 : i64 to index
        %7 = flow.dispatch.workload.ordinal %6, 0 : index
        %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x?xf16>>{%7}
        %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x?xf16>>{%7}
        %10 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [32, %7], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x?xf16>>{%7} -> tensor<32x?xf16>
        %11 = tensor.empty(%7) : tensor<32x?xf16> 
        %12 = linalg.softmax dimension(1) ins(%10 : tensor<32x?xf16>) outs(%11 : tensor<32x?xf16>) -> tensor<32x?xf16>
        flow.dispatch.tensor.store %12, %9, offsets = [0, 0], sizes = [32, %7], strides = [1, 1] : tensor<32x?xf16> -> !flow.dispatch.tensor<writeonly:tensor<32x?xf16>>{%7}
        return
      }
    }
  }
}

// CHECK-LABEL: spirv.func @dynamic_softmax
// CHECK-DAG:     %[[F16_MIN:.+]] = spirv.Constant 0xFC00 : f16
// CHECK-DAG:     %[[C0_F16_LD_PAD:.+]] = spirv.Constant 0.000000e+00 : f16
// CHECK-DAG:     %[[C0_I32:.+]] = spirv.Constant 0 : i32

// Do the first local reduction.
// CHECK:         spirv.Store "Workgroup" %{{.*}}, %[[F16_MIN]] : f16
// CHECK:         spirv.mlir.loop

// Masked load of the accumulator from workgroup memory.
// CHECK:           %[[MASK_CHECK:.+]] = spirv.SGreaterThan %{{.*}}, %[[C0_I32]] : i32
// CHECK:           %[[MLOAD_ACC:.+]] = spirv.Variable : !spirv.ptr<f16, Function>
// CHECK:           spirv.mlir.selection {
// CHECK:             spirv.BranchConditional %[[MASK_CHECK]], ^bb1, ^bb2
// CHECK:           ^bb1:  // pred: ^bb0
// CHECK:             %[[IT_ARG:.+]] = spirv.Load "Workgroup" %{{.*}} : f16
// CHECK:             spirv.Store "Function" %[[MLOAD_ACC]], %[[IT_ARG]] : f16
// CHECK:             spirv.Branch ^bb3
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             spirv.Store "Function" %[[MLOAD_ACC]], %[[C0_F16_LD_PAD]] : f16
// CHECK:             spirv.Branch ^bb3
// CHECK:           ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK:             spirv.mlir.merge
// CHECK:           %[[ACC:.+]] = spirv.Load "Function" %[[MLOAD_ACC]] : f16

// Masked load of the next local data element.
// CHECK:           %[[MLOAD_DATA:.+]] = spirv.Variable : !spirv.ptr<f16, Function>
// CHECK:           spirv.mlir.selection {
// CHECK:             spirv.BranchConditional %101, ^bb1, ^bb2
// CHECK:           ^bb1:  // pred: ^bb0
// CHECK:             %[[DATA_PTR:.+]] = spirv.AccessChain %{{.*}} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f16, stride=2> [0])>, StorageBuffer>, i32, i32
// CHECK:             %[[DATA:.+]] = spirv.Load "StorageBuffer" %[[DATA_PTR]] : f16
// CHECK:             spirv.Store "Function" %[[MLOAD_DATA]], %[[DATA]] : f16
// CHECK:             spirv.Branch ^bb3
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             spirv.Store "Function" %[[MLOAD_DATA]], %[[C0_F16_LD_PAD]] : f16
// CHECK:             spirv.Branch ^bb3
// CHECK:           ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK:             spirv.mlir.merge
// CHECK:           %[[NEW_DATA:.+]] = spirv.Load "Function" %104 : f16

// CHECK:           %106 = spirv.GL.FMax %[[NEW_DATA]], %[[ACC]] : f16

// Masked store back to the accumulator.
// CHECK:           spirv.mlir.selection
// CHECK:             spirv.BranchConditional %{{.*}}, ^bb1, ^bb2
// CHECK:           ^bb1:  // pred: ^bb0
// CHECK:             spirv.Store "Workgroup" {{.*}} : f16
// CHECK:             spirv.Branch ^bb2
// CHECK:           ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:             spirv.mlir.merge
// CHECK:           spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
// CHECK:         ^bb3:  // pred: ^bb1
// CHECK:           spirv.mlir.merge

// Finish the first reduction.
// CHECK:         spirv.Load "Workgroup" %{{.*}} : f16
// CHECK-COUNT-6: spirv.GroupNonUniformShuffleXor <Subgroup> {{.*}} : i32, i32

// CHECK:         spirv.Store "Workgroup" %{{.*}}, %[[C0_F16_LD_PAD]] : f16
// CHECK:         spirv.mlir.loop
// CHECK:           spirv.mlir.selection
// CHECK:             spirv.Load "StorageBuffer" %{{.*}} : f16
// CHECK:             spirv.Store "Function" %{{.*}}, %{{.*}} : f16
// CHECK:             spirv.Branch
// CHECK:             spirv.Store "Function" %{{.*}}, %[[C0_F16_LD_PAD]] : f16
// CHECK:           spirv.mlir.selection
// CHECK:             %136 = spirv.Load "Workgroup" %6 : f16
// CHECK:             spirv.Store "Function" %{{.*}}, %{{.*}} : f16
// CHECK:             spirv.Branch
// CHECK:             spirv.Store "Function" %{{.*}}, %[[C0_F16_LD_PAD]] : f16
// CHECK:           spirv.mlir.selection
// CHECK:             spirv.Store "Workgroup" {{.*}} : f16
// CHECK:             spirv.Branch
// CHECK:             spirv.mlir.merge
// CHECK:           spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
// CHECK:           spirv.mlir.merge

// Finish the second reduction
// CHECK-COUNT-6: spirv.GroupNonUniformShuffleXor <Subgroup> {{.*}} : i32, i32

// Store the result back to global memory in a loop.
// CHECK:         spirv.mlir.loop
// CHECK:           spirv.mlir.selection {
// CHECK:             %[[OUT_PTR:.+]] = spirv.AccessChain {{.*}} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f16, stride=2> [0])>, StorageBuffer>, i32, i32
// CHECK:             spirv.Store "StorageBuffer" %[[OUT_PTR]], %{{.*}} : f16
// CHECK:             spirv.Branch
// CHECK:             spirv.mlir.merge
// CHECK:           spirv.mlir.merge
// CHECK:         spirv.Return
