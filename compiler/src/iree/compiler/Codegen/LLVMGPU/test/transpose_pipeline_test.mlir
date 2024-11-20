// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_80 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target, fold-memref-alias-ops, canonicalize, cse)))))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
hal.executable @transpose_dispatch_0 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @transpose_dispatch_0_generic_4096x4096 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_dispatch_0_generic_4096x4096() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4096x4096xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x4096xf32>> -> tensor<4096x4096xf32>
        %3 = tensor.empty() : tensor<4096x4096xf32>
        %4 = linalg.generic {indexing_maps = [ affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<4096x4096xf32>) outs(%3 : tensor<4096x4096xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          linalg.yield %arg0 : f32
        } -> tensor<4096x4096xf32>
        flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : tensor<4096x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL:   func @transpose_dispatch_0
//       CHECK:   %[[A:.*]] = memref.alloc() : memref<32x34xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[B0:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//       CHECK:   %[[B1:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//       CHECK:   gpu.barrier
//       CHECK:   %[[GR0:.*]] = vector.transfer_read %[[B0]]{{.*}} vector<4xf32>
//       CHECK:   vector.transfer_write %[[GR0]], %[[A]]{{.*}} : vector<4xf32>
//       CHECK:   gpu.barrier
//       CHECK:   %[[SR:.*]] = vector.transfer_read %[[A]]{{.*}} vector<4x1xf32>
//       CHECK:   %[[SC:.*]] = vector.shape_cast %[[SR]] : vector<4x1xf32> to vector<4xf32>
//       CHECK:   vector.transfer_write %[[SC]], %[[B1]]{{.*}} : vector<4xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
hal.executable @transpose_single_operand_dispatch_0_generic_768x2048 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @transpose_single_operand_dispatch_0_generic_768x2048 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_single_operand_dispatch_0_generic_768x2048() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2048x768xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<768x2048xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<768x2048xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x768xf32>> -> tensor<2048x768xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [768, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<768x2048xf32>> -> tensor<768x2048xf32>
        %5 = tensor.empty() : tensor<768x2048xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<2048x768xf32>, tensor<768x2048xf32>) outs(%5 : tensor<768x2048xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<768x2048xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [768, 2048], strides = [1, 1] : tensor<768x2048xf32> -> !flow.dispatch.tensor<writeonly:tensor<768x2048xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL:   func @transpose_single_operand_dispatch_0_generic_768x2048 
//       CHECK:   %[[A:.*]] = memref.alloc() : memref<32x34xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[B0:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//       CHECK:   %[[B1:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//       CHECK:   %[[B2:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//       CHECK:   gpu.barrier
//       CHECK:   %[[GR0:.*]] = vector.transfer_read %[[B0]]{{.*}} vector<4xf32>
//       CHECK:   vector.transfer_write %[[GR0]], %[[A]]{{.*}} : vector<4xf32>
//       CHECK:   gpu.barrier
//       CHECK:   %[[SR:.*]] = vector.transfer_read %[[A]]{{.*}} vector<4x1xf32>
//       CHECK:   %[[GR1:.*]] = vector.transfer_read %[[B1]]{{.*}} vector<4xf32>
//       CHECK:   %[[SC:.*]] = vector.shape_cast %[[SR]] : vector<4x1xf32> to vector<4xf32>
//       CHECK:   %[[ADD:.*]] = arith.addf %[[SC]], %[[GR1]] : vector<4xf32>
//       CHECK:   vector.transfer_write %[[ADD]], %[[B2]]{{.*}} : vector<4xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
hal.executable @transpose_3d_no_dispatch_0_generic_768x2048x1024 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @transpose_3d_no_dispatch_0_generic_768x2048x1024 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_3d_no_dispatch_0_generic_768x2048x1024() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2048x768x1024xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<768x2048x1024xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<768x2048x1024xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2048, 768, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x768x1024xf32>> -> tensor<2048x768x1024xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [768, 2048, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<768x2048x1024xf32>> -> tensor<768x2048x1024xf32>
        %5 = tensor.empty() : tensor<768x2048x1024xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<2048x768x1024xf32>, tensor<768x2048x1024xf32>) outs(%5 : tensor<768x2048x1024xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<768x2048x1024xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [768, 2048, 1024], strides = [1, 1, 1] : tensor<768x2048x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<768x2048x1024xf32>>
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
    hal.executable.export public @transpose_3d_yes_dispatch_0_generic_10x768x2048 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_3d_yes_dispatch_0_generic_10x768x2048() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x2048x768xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x768x2048xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<10x768x2048xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [10, 2048, 768], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x2048x768xf32>> -> tensor<10x2048x768xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x768x2048xf32>> -> tensor<10x768x2048xf32>
        %5 = tensor.empty() : tensor<10x768x2048xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<10x2048x768xf32>, tensor<10x768x2048xf32>) outs(%5 : tensor<10x768x2048xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<10x768x2048xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : tensor<10x768x2048xf32> -> !flow.dispatch.tensor<writeonly:tensor<10x768x2048xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL:   func @transpose_3d_yes_dispatch_0_generic_10x768x2048 
//       CHECK:   %[[A:.*]] = memref.alloc() : memref<1x32x34xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[B0:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//       CHECK:   %[[B1:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//       CHECK:   %[[B2:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//       CHECK:   gpu.barrier
//       CHECK:   %[[GR0:.*]] = vector.transfer_read %[[B0]]{{.*}} vector<4xf32>
//       CHECK:   vector.transfer_write %[[GR0]], %[[A]]{{.*}} : vector<4xf32>
//       CHECK:   gpu.barrier
//       CHECK:   %[[SR:.*]] = vector.transfer_read %[[A]]{{.*}} vector<4x1xf32>
//       CHECK:   %[[GR1:.*]] = vector.transfer_read %[[B1]]{{.*}} vector<4xf32>
//       CHECK:   %[[SC:.*]] = vector.shape_cast %[[SR]] : vector<4x1xf32> to vector<4xf32>
//       CHECK:   %[[ADD:.*]] = arith.addf %[[SC]], %[[GR1]] : vector<4xf32>
//       CHECK:   vector.transfer_write %[[ADD]], %[[B2]]{{.*}} : vector<4xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
hal.executable @transpose_3d_trans_out_dispatch_0_generic_10x2048x768 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @transpose_3d_trans_out_dispatch_0_generic_10x2048x768 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_3d_trans_out_dispatch_0_generic_10x2048x768() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x768x2048xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x768x2048xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<10x2048x768xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x768x2048xf32>> -> tensor<10x768x2048xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x768x2048xf32>> -> tensor<10x768x2048xf32>
        %5 = tensor.empty() : tensor<10x2048x768xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<10x768x2048xf32>, tensor<10x768x2048xf32>) outs(%5 : tensor<10x2048x768xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %7 = arith.addf %arg0, %arg1 : f32
          linalg.yield %7 : f32
        } -> tensor<10x2048x768xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [10, 2048, 768], strides = [1, 1, 1] : tensor<10x2048x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<10x2048x768xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL:   func @transpose_3d_trans_out_dispatch_0_generic_10x2048x768
//       CHECK:   %[[A0:.*]] = memref.alloc() : memref<1x32x34xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[A1:.*]] = memref.alloc() : memref<1x32x34xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[B0:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//       CHECK:   %[[B1:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//       CHECK:   %[[B2:.*]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//       CHECK:   gpu.barrier
//       CHECK:   %[[GR0:.*]] = vector.transfer_read %[[B0]]{{.*}} vector<4xf32>
//       CHECK:   vector.transfer_write %[[GR0]], %[[A0]]{{.*}} : vector<4xf32>
//       CHECK:   %[[GR1:.*]] = vector.transfer_read %[[B1]]{{.*}} vector<4xf32>
//       CHECK:   vector.transfer_write %[[GR1]], %[[A1]]{{.*}} : vector<4xf32>
//       CHECK:   gpu.barrier
//       CHECK:   %[[SR0:.*]] = vector.transfer_read %[[A0]]{{.*}} vector<4x1xf32>
//       CHECK:   %[[SR1:.*]] = vector.transfer_read %[[A1]]{{.*}} vector<4x1xf32>
//       CHECK:   %[[ADD:.*]] = arith.addf %[[SR0]], %[[SR1]] : vector<4x1xf32>
//       CHECK:   %[[SC:.*]] = vector.shape_cast %[[ADD]] : vector<4x1xf32> to vector<4xf32>
//       CHECK:   vector.transfer_write %[[SC]], %[[B2]]{{.*}} : vector<4xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
hal.executable @transpose_3d_diff_dispatch_0_generic_10x768x2048 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @transpose_3d_diff_dispatch_0_generic_10x768x2048 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @transpose_3d_diff_dispatch_0_generic_10x768x2048() {
      %c256 = arith.constant 256 : index
      %c10 = arith.constant 10 : index
      %c768 = arith.constant 768 : index
      %c2048 = arith.constant 2048 : index
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x2048x768xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2048x768x10xf32>>
      %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<10x768x2048xf32>>
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
              %5 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg2, %arg1], sizes = [1, %c256, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x2048x768xf32>> -> tensor<1x?x1xf32>
              %6 = flow.dispatch.tensor.load %1, offsets = [%arg2, %arg1, %arg0], sizes = [%c256, 1, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x768x10xf32>> -> tensor<?x1x1xf32>
              %7 = tensor.empty() : tensor<1x1x256xf32>
              %8 = tensor.cast %5 : tensor<1x?x1xf32> to tensor<1x256x1xf32>
              %9 = tensor.cast %6 : tensor<?x1x1xf32> to tensor<256x1x1xf32>
              %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %9 : tensor<1x256x1xf32>, tensor<256x1x1xf32>) outs(%7 : tensor<1x1x256xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 256]]>} {
              ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
              %12 = arith.addf %arg3, %arg4 : f32
              linalg.yield %12 : f32
              } -> tensor<1x1x256xf32>
              %11 = tensor.cast %10 : tensor<1x1x256xf32> to tensor<1x1x?xf32>
              flow.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1, %arg2], sizes = [1, 1, %c256], strides = [1, 1, 1] : tensor<1x1x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<10x768x2048xf32>>
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
