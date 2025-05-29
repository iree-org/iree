// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_80 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target, fold-memref-alias-ops, canonicalize, cse)))))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
hal.executable @transpose_dispatch_0 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @transpose_dispatch_0_generic_4096x4096 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_dispatch_0_generic_4096x4096() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>> -> tensor<4096x4096xf32>
        %3 = tensor.empty() : tensor<4096x4096xf32>
        %4 = linalg.generic {indexing_maps = [ affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<4096x4096xf32>) outs(%3 : tensor<4096x4096xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          linalg.yield %arg0 : f32
        } -> tensor<4096x4096xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : tensor<4096x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL:  hal.executable public @transpose_dispatch_0
//   CHECK-DAG:  %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[TX:.*]] = gpu.thread_id  x
//   CHECK-DAG:  %[[TY:.*]] = gpu.thread_id  y
//   CHECK-DAG:  %[[ALLOC:.*]] = memref.alloc() : memref<32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:  %[[D0_BINDING:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(0) alignment(64) offset(%[[C0]]) : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:  %[[D0:.+]] = memref.assume_alignment %[[D0_BINDING]], 64 : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:  %[[D1_BINDING:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(1) alignment(64) offset(%[[C0]]) : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:  %[[D1:.+]] = memref.assume_alignment %[[D1_BINDING]], 64 : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:  gpu.barrier
//       CHECK:  %[[D2:.*]] = affine.apply #{{.*}}()[%{{.+}}, %[[TY]]]
//       CHECK:  %[[D3:.*]] = affine.apply #{{.*}}()[%{{.+}}, %[[TX]]]
//       CHECK:  %[[D4:.*]] = vector.transfer_read %[[D0]][%[[D2]], %[[D3]]], %[[CST]] {in_bounds = [true, true]} : memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>, vector<1x4xf32>
//       CHECK:  %[[D5:.*]] = affine.apply #{{.*}}()[%[[TX]]]
//       CHECK:  vector.transfer_write %[[D4]], %[[ALLOC]][%[[TY]], %[[D5]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:  gpu.barrier
//       CHECK:  %[[D6:.*]] = vector.transfer_read %[[ALLOC]][%[[D5]], %[[TY]]], %[[CST]] {in_bounds = [true, true]} : memref<32x33xf32, #gpu.address_space<workgroup>>, vector<4x1xf32>
//       CHECK:  %[[D7:.*]] = vector.shape_cast %[[D6]] : vector<4x1xf32> to vector<4xf32>
//       CHECK:  %[[D8:.*]] = affine.apply #{{.*}}()[%{{.+}}, %[[TY]]]
//       CHECK:  %[[D9:.*]] = affine.apply #{{.*}}()[%{{.+}}, %[[TX]]]
//       CHECK:  vector.transfer_write %[[D7]], %[[D1]][%[[D8]], %[[D9]]] {in_bounds = [true]} : vector<4xf32>, memref<4096x4096xf32, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
hal.executable @transpose_single_operand_dispatch_0_generic_768x2048 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @transpose_single_operand_dispatch_0_generic_768x2048 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_single_operand_dispatch_0_generic_768x2048() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x768xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<768x2048xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<768x2048xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 768], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x768xf32>> -> tensor<2048x768xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [768, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<768x2048xf32>> -> tensor<768x2048xf32>
        %5 = tensor.empty() : tensor<768x2048xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<2048x768xf32>, tensor<768x2048xf32>) outs(%5 : tensor<768x2048xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<768x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [768, 2048], strides = [1, 1] : tensor<768x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<768x2048xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL:  hal.executable public @transpose_single_operand_dispatch_0_generic_768x2048
//       CHECK:  %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[TX:.*]] = gpu.thread_id  x
//       CHECK:  %[[TY:.*]] = gpu.thread_id  y
//       CHECK:  %[[ALLOC:.*]] = memref.alloc() : memref<32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:  %[[D0_BINDING:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(0) alignment(64) offset(%[[C0]]) : memref<2048x768xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:  %[[D0:.+]] = memref.assume_alignment %[[D0_BINDING]], 64 : memref<2048x768xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:  %[[D1_BINDING:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(1) alignment(64) offset(%[[C0]]) : memref<768x2048xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:  %[[D1:.+]] = memref.assume_alignment %[[D1_BINDING]], 64 : memref<768x2048xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:  %[[D2_BINDING:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(2) alignment(64) offset(%[[C0]]) : memref<768x2048xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:  %[[D2:.+]] = memref.assume_alignment %[[D2_BINDING]], 64 : memref<768x2048xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:  gpu.barrier
//       CHECK:  %[[D3:.*]] = affine.apply #{{.*}}()[%[[TX]]]
//       CHECK:  %[[D4:.*]] = affine.apply #{{.*}}()[%{{.*}}, %[[TY]]]
//       CHECK:  %[[D5:.*]] = affine.apply #{{.*}}()[%{{.*}}, %[[TX]]]
//       CHECK:  %[[D6:.*]] = vector.transfer_read %[[D0]][%[[D4]], %[[D5]]], %[[CST]] {in_bounds = [true, true]} : memref<2048x768xf32, #hal.descriptor_type<storage_buffer>>, vector<1x4xf32>
//       CHECK:  vector.transfer_write %[[D6]], %[[ALLOC]][%[[TY]], %[[D3]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:  gpu.barrier
//       CHECK:  %[[D7:.*]] = vector.transfer_read %[[ALLOC]][%[[D3]], %[[TY]]], %[[CST]] {in_bounds = [true, true]} : memref<32x33xf32, #gpu.address_space<workgroup>>, vector<4x1xf32>
//       CHECK:  %[[D8:.*]] = arith.addi %[[TY]], %{{.*}}
//       CHECK:  %[[D9:.*]] = arith.addi %[[D3]], %{{.*}}
//       CHECK:  %[[D10:.*]] = vector.transfer_read %[[D1]][%[[D8]], %[[D9]]], %[[CST]] {in_bounds = [true]} : memref<768x2048xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//       CHECK:  %[[D11:.*]] = vector.shape_cast %[[D7]] : vector<4x1xf32> to vector<4xf32>
//       CHECK:  %[[D12:.*]] = arith.addf %[[D11]], %[[D10]] : vector<4xf32>
//       CHECK:  %[[D13:.*]] = affine.apply #{{.*}}()[%{{.*}}, %[[TY]]]
//       CHECK:  %[[D14:.*]] = affine.apply #{{.*}}()[%{{.*}}, %[[TX]]]
//       CHECK:  vector.transfer_write %[[D12]], %[[D2]][%[[D13]], %[[D14]]] {in_bounds = [true]} : vector<4xf32>, memref<768x2048xf32, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
hal.executable @transpose_3d_no_dispatch_0_generic_768x2048x1024 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @transpose_3d_no_dispatch_0_generic_768x2048x1024 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_3d_no_dispatch_0_generic_768x2048x1024() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x768x1024xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<768x2048x1024xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<768x2048x1024xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2048, 768, 1024], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x768x1024xf32>> -> tensor<2048x768x1024xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [768, 2048, 1024], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<768x2048x1024xf32>> -> tensor<768x2048x1024xf32>
        %5 = tensor.empty() : tensor<768x2048x1024xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<2048x768x1024xf32>, tensor<768x2048x1024xf32>) outs(%5 : tensor<768x2048x1024xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<768x2048x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [768, 2048, 1024], strides = [1, 1, 1] : tensor<768x2048x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<768x2048x1024xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL:   hal.executable public @transpose_3d_no_dispatch_0_generic_768x2048x1024 {
//   CHECK-NOT:   gpu.barrier
//   CHECK-NOT:   memref.alloc
//       CHECK:   return

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
hal.executable @transpose_3d_yes_dispatch_0_generic_10x768x2048 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @transpose_3d_yes_dispatch_0_generic_10x768x2048 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_3d_yes_dispatch_0_generic_10x768x2048() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x2048x768xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x768x2048xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<10x768x2048xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [10, 2048, 768], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x2048x768xf32>> -> tensor<10x2048x768xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x768x2048xf32>> -> tensor<10x768x2048xf32>
        %5 = tensor.empty() : tensor<10x768x2048xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<10x2048x768xf32>, tensor<10x768x2048xf32>) outs(%5 : tensor<10x768x2048xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<10x768x2048xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : tensor<10x768x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<10x768x2048xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL:   hal.executable public @transpose_3d_yes_dispatch_0_generic_10x768x2048 {
//   CHECK-DAG:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[TX:.*]] = gpu.thread_id  x
//       CHECK:   %[[TY:.*]] = gpu.thread_id  y
//       CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<1x32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[D0_BINDING:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(0) alignment(64) offset(%[[C0]]) : memref<10x2048x768xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[D0:.+]] = memref.assume_alignment %[[D0_BINDING]], 64 : memref<10x2048x768xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[D1_BINDING:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(1) alignment(64) offset(%[[C0]]) : memref<10x768x2048xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[D1:.+]] = memref.assume_alignment %[[D1_BINDING]], 64 : memref<10x768x2048xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[D2_BINDING:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(2) alignment(64) offset(%[[C0]]) : memref<10x768x2048xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[D2:.+]] = memref.assume_alignment %[[D2_BINDING]], 64 : memref<10x768x2048xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   gpu.barrier
//       CHECK:   %[[D3:.*]] = affine.apply #{{.*}}()[%[[TX]]]
//       CHECK:   %[[D4:.*]] = affine.apply #{{.*}}()[%{{.*}}, %[[TY]]]
//       CHECK:   %[[D5:.*]] = affine.apply #{{.*}}()[%{{.*}}, %[[TX]]]
//       CHECK:   %[[D6:.*]] = vector.transfer_read %[[D0]][%{{.*}}, %[[D4]], %[[D5]]], %[[CST]] {in_bounds = [true, true, true]} : memref<10x2048x768xf32, #hal.descriptor_type<storage_buffer>>, vector<1x1x4xf32>
//       CHECK:   vector.transfer_write %[[D6]], %[[ALLOC]][%[[C0]], %[[TY]], %[[D3]]] {in_bounds = [true, true, true]} : vector<1x1x4xf32>, memref<1x32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:   gpu.barrier
//       CHECK:   %[[D7:.*]] = vector.transfer_read %[[ALLOC]][%[[C0]], %[[D3]], %[[TY]]], %[[CST]] {in_bounds = [true, true]} : memref<1x32x33xf32, #gpu.address_space<workgroup>>, vector<4x1xf32>
//       CHECK:   %[[D8:.*]] = arith.addi %[[TY]], %{{.*}}
//       CHECK:   %[[D9:.*]] = arith.addi %[[D3]], %{{.*}}
//       CHECK:   %[[D10:.*]] = vector.transfer_read %[[D1]][%{{.*}}, %[[D8]], %[[D9]]], %[[CST]] {in_bounds = [true]} : memref<10x768x2048xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//       CHECK:   %[[D11:.*]] = vector.shape_cast %[[D7]] : vector<4x1xf32> to vector<4xf32>
//       CHECK:   %[[D12:.*]] = arith.addf %[[D11]], %[[D10]] : vector<4xf32>
//       CHECK:   %[[D13:.*]] = affine.apply #{{.*}}()[%{{.*}}, %[[TY]]]
//       CHECK:   %[[D14:.*]] = affine.apply #{{.*}}()[%{{.*}}, %[[TX]]]
//       CHECK:   vector.transfer_write %[[D12]], %[[D2]][%{{.*}}, %[[D13]], %[[D14]]] {in_bounds = [true]} : vector<4xf32>, memref<10x768x2048xf32, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
hal.executable @transpose_3d_trans_out_dispatch_0_generic_10x2048x768 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @transpose_3d_trans_out_dispatch_0_generic_10x2048x768 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_3d_trans_out_dispatch_0_generic_10x2048x768() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x768x2048xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x768x2048xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<10x2048x768xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x768x2048xf32>> -> tensor<10x768x2048xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x768x2048xf32>> -> tensor<10x768x2048xf32>
        %5 = tensor.empty() : tensor<10x2048x768xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<10x768x2048xf32>, tensor<10x768x2048xf32>) outs(%5 : tensor<10x2048x768xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<10x2048x768xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [10, 2048, 768], strides = [1, 1, 1] : tensor<10x2048x768xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<10x2048x768xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL:   hal.executable public @transpose_3d_trans_out_dispatch_0_generic_10x2048x768 {
//   CHECK-DAG:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[TX:.*]] = gpu.thread_id  x
//       CHECK:   %[[TY:.*]] = gpu.thread_id  y
//       CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<1x32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[ALLOC1:.*]] = memref.alloc() : memref<1x32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[D0_BINDING:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(0) alignment(64) offset(%[[C0]]) : memref<10x768x2048xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[D0:.+]] = memref.assume_alignment %[[D0_BINDING]], 64 : memref<10x768x2048xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[D1_BINDING:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(1) alignment(64) offset(%[[C0]]) : memref<10x768x2048xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[D1:.+]] = memref.assume_alignment %[[D1_BINDING]], 64 : memref<10x768x2048xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[D2_BINDING:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(2) alignment(64) offset(%[[C0]]) : memref<10x2048x768xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   %[[D2:.+]] = memref.assume_alignment %[[D2_BINDING]], 64 : memref<10x2048x768xf32, #hal.descriptor_type<storage_buffer>>
//       CHECK:   gpu.barrier
//       CHECK:   %[[D3:.*]] = affine.apply #{{.*}}()[%{{.*}}, %[[TY]]]
//       CHECK:   %[[D4:.*]] = affine.apply #{{.*}}()[%{{.*}}, %[[TX]]]
//       CHECK:   %[[D5:.*]] = vector.transfer_read %[[D0]][%{{.*}}, %[[D3]], %[[D4]]], %[[CST]] {in_bounds = [true, true, true]} : memref<10x768x2048xf32, #hal.descriptor_type<storage_buffer>>, vector<1x1x4xf32>
//       CHECK:   %[[D6:.*]] = affine.apply #{{.*}}()[%[[TX]]]
//       CHECK:   vector.transfer_write %[[D5]], %[[ALLOC1]][%[[C0]], %[[TY]], %[[D6]]] {in_bounds = [true, true, true]} : vector<1x1x4xf32>, memref<1x32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[D7:.*]] = vector.transfer_read %[[D1]][%{{.*}}, %[[D3]], %[[D4]]], %[[CST]] {in_bounds = [true, true, true]} : memref<10x768x2048xf32, #hal.descriptor_type<storage_buffer>>, vector<1x1x4xf32>
//       CHECK:   vector.transfer_write %[[D7]], %[[ALLOC]][%[[C0]], %[[TY]], %[[D6]]] {in_bounds = [true, true, true]} : vector<1x1x4xf32>, memref<1x32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:   gpu.barrier
//       CHECK:   %[[D8:.*]] = vector.transfer_read %[[ALLOC1]][%[[C0]], %[[D6]], %[[TY]]], %[[CST]] {in_bounds = [true, true]} : memref<1x32x33xf32, #gpu.address_space<workgroup>>, vector<4x1xf32>
//       CHECK:   %[[D9:.*]] = vector.transfer_read %[[ALLOC]][%[[C0]], %[[D6]], %[[TY]]], %[[CST]] {in_bounds = [true, true]} : memref<1x32x33xf32, #gpu.address_space<workgroup>>, vector<4x1xf32>
//       CHECK:   %[[D10:.*]] = arith.addf %[[D8]], %[[D9]] : vector<4x1xf32>
//       CHECK:   %[[D11:.*]] = vector.shape_cast %[[D10]] : vector<4x1xf32> to vector<4xf32>
//       CHECK:   %[[D12:.*]] = affine.apply #{{.*}}()[%{{.*}}, %[[TY]]]
//       CHECK:   %[[D13:.*]] = affine.apply #{{.*}}()[%{{.*}}, %[[TX]]]
//       CHECK:   vector.transfer_write %[[D11]], %[[D2]][%{{.*}}, %[[D12]], %[[D13]]] {in_bounds = [true]} : vector<4xf32>, memref<10x2048x768xf32, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
hal.executable @transpose_3d_diff_dispatch_0_generic_10x768x2048 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @transpose_3d_diff_dispatch_0_generic_10x768x2048 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_3d_diff_dispatch_0_generic_10x768x2048() {
      %c256 = arith.constant 256 : index
      %c10 = arith.constant 10 : index
      %c768 = arith.constant 768 : index
      %c2048 = arith.constant 2048 : index
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x2048x768xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x768x10xf32>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<10x768x2048xf32>>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %workgroup_count_y = hal.interface.workgroup.count[1] : index
      %workgroup_id_z = hal.interface.workgroup.id[2] : index
      %workgroup_count_z = hal.interface.workgroup.count[2] : index
      scf.for %arg0 = %workgroup_id_z to %c10 step %workgroup_count_z {
          scf.for %arg1 = %workgroup_id_y to %c768 step %workgroup_count_y {
          %3 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_id_x]
          %4 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_count_x]
          scf.for %arg2 = %3 to %c2048 step %4 {
              %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [%arg0, %arg2, %arg1], sizes = [1, %c256, 1], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x2048x768xf32>> -> tensor<1x?x1xf32>
              %6 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [%arg2, %arg1, %arg0], sizes = [%c256, 1, 1], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x768x10xf32>> -> tensor<?x1x1xf32>
              %7 = tensor.empty() : tensor<1x1x256xf32>
              %8 = tensor.cast %5 : tensor<1x?x1xf32> to tensor<1x256x1xf32>
              %9 = tensor.cast %6 : tensor<?x1x1xf32> to tensor<256x1x1xf32>
              %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %9 : tensor<1x256x1xf32>, tensor<256x1x1xf32>) outs(%7 : tensor<1x1x256xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 256]]>} {
              ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
              %12 = arith.addf %arg3, %arg4 : f32
              linalg.yield %12 : f32
              } -> tensor<1x1x256xf32>
              %11 = tensor.cast %10 : tensor<1x1x256xf32> to tensor<1x1x?xf32>
              iree_tensor_ext.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1, %arg2], sizes = [1, 1, %c256], strides = [1, 1, 1] : tensor<1x1x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<10x768x2048xf32>>
          }
          }
      }
      return
      }
    }
  }
}

// CHECK-LABEL:   hal.executable public @transpose_3d_diff_dispatch_0_generic_10x768x2048 {
//   CHECK-NOT:   gpu.barrier
//   CHECK-NOT:   memref.alloc
//       CHECK:   return
