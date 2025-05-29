// RUN: iree-opt --split-input-file --iree-gpu-test-target=valhall1 --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-spirv-configuration-pipeline), iree-codegen-linalg-to-spirv-pipeline)))' %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=vp_android_baseline_2022@vulkan --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-spirv-configuration-pipeline), iree-codegen-linalg-to-spirv-pipeline)))' %s | FileCheck %s --check-prefix=NOSHUFFLE

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @subgroup_reduce {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @subgroup_reduce ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @subgroup_reduce() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x512xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2xf32>>
        %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x512xf32>> -> tensor<2x512xf32>
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
        iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2xf32>>
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
// CHECK-DAG:   %[[F0:.+]] = spirv.Constant 0.000000e+00 : f32
// CHECK-DAG:   %[[FV0:.+]] = spirv.Constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:   %[[FV1:.+]] = spirv.Constant dense<1.000000e+00> : vector<4xf32>

// CHECK:   %[[LD:.+]] = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
// CHECK:   %[[ADDV0:.+]] = spirv.FAdd %[[LD]], %[[FV0]] : vector<4xf32>
// CHECK:   %[[ADD2:.+]] = spirv.Dot %[[ADDV0]], %[[FV1]] : vector<4xf32> -> f32

// CHECK:   %[[S0:.+]] = spirv.GroupNonUniformFAdd <Subgroup> <Reduce> %[[ADD2:.+]] : f32 -> f32

// CHECK:   spirv.Store "Workgroup" %{{.+}}, %[[S0]] : f32

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

// NOSHUFFLE-LABEL: spirv.func @subgroup_reduce()
// NOSHUFFLE-NOT: spirv.GroupNonUniformShuffleXor

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @softmax{
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @softmax ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @softmax() {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 1.000000e+00 : f32
        %c786432 = arith.constant 786432 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c786432) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1536x128xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1536x128xf32>>
        %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1536, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1536x128xf32>> -> tensor<1536x128xf32>
        %3 = tensor.empty() : tensor<1536xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1536xf32>) -> tensor<1536xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<1536x128xf32>) outs(%4 : tensor<1536xf32>) {
        ^bb0(%in: f32, %out: f32):
          %8 = arith.addf %in, %out : f32
          linalg.yield %8 : f32
        } -> tensor<1536xf32>
        %6 = tensor.empty() : tensor<1536x128xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %5 : tensor<1536x128xf32>, tensor<1536xf32>) outs(%6 : tensor<1536x128xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.divf %cst_0, %in_1 : f32
          %9 = arith.mulf %in, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<1536x128xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %1, offsets = [0, 0], sizes = [1536, 128], strides = [1, 1] : tensor<1536x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1536x128xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: spirv.func @softmax()
//       CHECK:   %{{.*}} = spirv.FAdd {{.*}} : vector<4xf32>
