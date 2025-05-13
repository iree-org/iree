// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_60 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-decompose-softmax), iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target)))))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @warp_reduction_dispatch {
hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @warp_reduction_dispatch layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @warp_reduction_dispatch() {
      %c0 = arith.constant 0 : index
      %c10240 = arith.constant 10240 : index
      %cst = arith.constant 1.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x10240xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512xf32>>
      %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 10240], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x10240xf32>> -> tensor<512x10240xf32>
      %8 = tensor.empty() : tensor<512xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<512xf32>) -> tensor<512xf32>
      %10 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]}
          ins(%5 : tensor<512x10240xf32>) outs(%9 : tensor<512xf32>) {
        ^bb0(%arg1: f32, %arg2: f32):
          %11 = arith.addf %arg1, %arg2 : f32
          linalg.yield %11 : f32
        } -> tensor<512xf32>
      iree_tensor_ext.dispatch.tensor.store %10, %1, offsets = [0], sizes = [512], strides = [1]
          : tensor<512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512xf32>>
      return
    }
  }
}
}

//         CHECK: #[[TRANSLATION_INFO:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 32
//         CHECK:  func.func @warp_reduction_dispatch()
//    CHECK-SAME:      translation_info = #[[TRANSLATION_INFO]]
//     CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4xf32>
//     CHECK-DAG:    %[[TID:.+]] = gpu.thread_id  x
//         CHECK:    %[[R0:.+]] = scf.for %{{.*}} = %c0 to %c10240 step %c1024 iter_args(%[[A0:.+]] = %[[CST]]) -> (vector<1x1x4xf32>) {
//         CHECK:      %[[V:.+]] = vector.transfer_read {{.*}} : memref<512x10240xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//         CHECK:      %[[STRIDED:.+]] = vector.insert_strided_slice %[[V]], {{.*}} : vector<4xf32> into vector<1x1x4xf32>
//         CHECK:      %[[ADD:.+]] = arith.addf %[[STRIDED]], %[[A0]] : vector<1x1x4xf32>
//         CHECK:      scf.yield %[[ADD]] : vector<1x1x4xf32>
//         CHECK:    }
//         CHECK:    gpu.subgroup_reduce  add {{.*}} cluster(size = 32) : (f32) -> f32
//         CHECK:    %[[ALLOC:.+]] = memref.alloc() : memref<10xf32, #gpu.address_space<workgroup>>
//         CHECK:    %[[SVIEW:.+]] = memref.subview {{.*}} : memref<10xf32, #gpu.address_space<workgroup>> to memref<8xf32, strided<[1]>, #gpu.address_space<workgroup>>
//         CHECK:    vector.transfer_write %{{.*}}, %[[SVIEW]]{{.*}} : vector<1xf32>, memref<8xf32, strided<[1]>, #gpu.address_space<workgroup>>
//         CHECK:    gpu.barrier
//         CHECK:    vector.transfer_read %[[SVIEW]]{{.*}} : memref<8xf32,
//         CHECK:    gpu.subgroup_reduce  add {{.*}} cluster(size = 8) : (f32) -> f32
//         CHECK:    vector.transfer_write {{.*}} : vector<f32>, memref<512xf32, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @warp_reduction_broadcast_dispatch {
hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @warp_reduction_broadcast_dispatch layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @warp_reduction_broadcast_dispatch() {
      %c0 = arith.constant 0 : index
      %c10240 = arith.constant 10240 : index
      %cst_0 = arith.constant 3.840000e+02 : f32
      %cst = arith.constant 1.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x10240xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x10240xf32>>
      %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1]
          : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x10240xf32>> -> tensor<512x10240xf32>
      %8 = tensor.empty() : tensor<512xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<512xf32>) -> tensor<512xf32>
      %10 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]}
          ins(%5 : tensor<512x10240xf32>) outs(%9 : tensor<512xf32>) {
        ^bb0(%arg1: f32, %arg2: f32):
          %11 = arith.addf %arg1, %arg2 : f32
          linalg.yield %11 : f32
        } -> tensor<512xf32>
      %i = tensor.empty() : tensor<512x10240xf32>
      %11 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%10 : tensor<512xf32>) outs(%i : tensor<512x10240xf32>) {
          ^bb0(%arg0: f32, %arg1: f32):
            %12 = arith.divf %arg0, %cst_0 : f32
            linalg.yield %12 : f32
          } -> tensor<512x10240xf32>
      iree_tensor_ext.dispatch.tensor.store %11, %1, offsets = [0, 0], sizes = [512, 10240], strides = [1, 1]
          : tensor<512x10240xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x10240xf32>>
      return
    }
  }
}
}

//         CHECK: #[[TRANSLATION_INFO:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 32
//         CHECK:  func.func @warp_reduction_broadcast_dispatch()
//    CHECK-SAME:      translation_info = #[[TRANSLATION_INFO]]
//         CHECK:    scf.for {{.*}} -> (vector<1x1x4xf32>) {
//         CHECK:      vector.transfer_read {{.*}} : memref<512x10240xf32,
//         CHECK:      arith.addf {{.*}} : vector<1x1x4xf32>
//         CHECK:      scf.yield
//         CHECK:    gpu.subgroup_reduce
//         CHECK:    vector.transfer_write {{.*}} : vector<1xf32
//         CHECK:    gpu.subgroup_reduce
//         CHECK:    arith.divf {{.*}} : vector<1x1x4xf32>
//         CHECK:    vector.transfer_write {{.*}} : vector<4xf32>, {{.*}}
//         CHECK:    return

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @softmax {
hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @softmax layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @softmax() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -3.40282347E+38 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %cst_1 = arith.constant 1.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<12x128x40960xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
      %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<12x128x40960xf32>> -> tensor<12x128x40960xf32>
      %3 = tensor.empty() : tensor<12x128x40960xf32>
      %4 = linalg.softmax dimension(2) ins(%2 : tensor<12x128x40960xf32>) outs(%3 : tensor<12x128x40960xf32>) -> tensor<12x128x40960xf32>
      iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : tensor<12x128x40960xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
      return
    }
  }
}
}

//         CHECK: #[[TRANSLATION_INFO:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [1024, 1, 1] subgroup_size = 32
//         CHECK:  func.func @softmax()
//    CHECK-SAME:      translation_info = #[[TRANSLATION_INFO]]
//         CHECK:    scf.for {{.*}} -> (vector<1x1x4xf32>) {
//         CHECK:      vector.transfer_read {{.*}} : memref<12x128x40960xf32,
//         CHECK:      arith.maxnumf {{.*}} : vector<1x1x4xf32>
//         CHECK:      scf.yield
//         CHECK:    vector.multi_reduction <maxnumf>
//         CHECK:    gpu.subgroup_reduce  maxnumf
//         CHECK:    vector.transfer_write
//         CHECK:    gpu.barrier
//         CHECK:    gpu.subgroup_reduce  maxnumf
//         CHECK:    vector.broadcast %{{.*}} : f32 to vector<1x1x4xf32>
//         CHECK:    scf.for {{.*}} -> (vector<1x1x4xf32>) {
//         CHECK:      vector.transfer_read
//         CHECK:      arith.subf
//         CHECK:      math.exp
//         CHECK:      arith.addf
//         CHECK:      scf.yield
//         CHECK:    vector.multi_reduction <add>
//         CHECK:    gpu.subgroup_reduce  add
//         CHECK:    vector.transfer_write
//         CHECK:    gpu.barrier
//         CHECK:    vector.transfer_read
//         CHECK:    gpu.subgroup_reduce  add
//         CHECK:    vector.broadcast
//         CHECK:    scf.for
//         CHECK:      vector.transfer_read
//         CHECK:      arith.subf
//         CHECK:      math.exp
//         CHECK:      arith.divf
//         CHECK:      vector.transfer_write
//         CHECK:    }
//         CHECK:    return

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @small_reduction {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @small_reduction ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @small_reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x13xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf32>>
      %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 13], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x13xf32>> -> tensor<1024x13xf32>
      %3 = tensor.empty() : tensor<1024xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1024xf32>) -> tensor<1024xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<1024x13xf32>) outs(%4 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<1024xf32>
      iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [1024], strides = [1] : tensor<1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf32>>
      return
    }
  }
}
}

// Small reduction computes the whole reduction on a single thread.
//   CHECK-LABEL: func.func @small_reduction
//         CHECK:   %[[READ:.+]] = vector.transfer_read {{.*}} #hal.descriptor_type<storage_buffer>>, vector<4x13xf32>
//         CHECK:   vector.multi_reduction <add>, %[[READ]], {{.*}} : vector<4x13xf32> to vector<4xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @group_reduction {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @group_reduction ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x64xf32>) outs(%4 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<8xf32>
      iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_reduction
//       CHECK:    gpu.subgroup_reduce  add {{.*}} cluster(size = 32) : (f32) -> f32

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @group_elementwise_reduction_elementwise {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @group_elementwise_reduction_elementwise ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_elementwise_reduction_elementwise() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x64xf32>) outs(%4 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = arith.addf %in, %in : f32
        %8 = arith.addf %7, %7 : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      } -> tensor<8xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5 : tensor<8xf32>) outs(%3 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = math.sqrt %in : f32
        linalg.yield %7 : f32
      } -> tensor<8xf32>
      iree_tensor_ext.dispatch.tensor.store %6, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_elementwise_reduction_elementwise
//       CHECK:    gpu.subgroup_reduce  add {{.*}} cluster(size = 32) : (f32) -> f32

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @group_reduction_larger {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @group_reduction_larger ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction_larger() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<33x1024xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<33xf32>>
      %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [33, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<33x1024xf32>> -> tensor<33x1024xf32>
      %3 = tensor.empty() : tensor<33xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<33xf32>) -> tensor<33xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<33x1024xf32>) outs(%4 : tensor<33xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<33xf32>
      iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [33], strides = [1] : tensor<33xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<33xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_reduction_larger
//   CHECK:    gpu.subgroup_reduce  add {{.*}} cluster(size = 32) : (f32) -> f32
//   CHECK:    gpu.subgroup_reduce  add {{.*}}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @group_reduction_1d {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @group_reduction_1d ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction_1d() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<f32>>
      %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
      %3 = tensor.empty() : tensor<f32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%2 : tensor<64xf32>) outs(%4 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<f32>
      iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<f32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_reduction_1d
//   CHECK:    gpu.subgroup_reduce  add {{.*}} cluster(size = 32) : (f32) -> f32

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @group_elementwise_reduction_elementwise_4d {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @group_elementwise_reduction_elementwise_4d ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_elementwise_reduction_elementwise_4d() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4x8x64xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x4x8xf32>>
      %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 4, 8, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4x8x64xf32>> -> tensor<2x4x8x64xf32>
      %3 = tensor.empty() : tensor<2x4x8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                           iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2 : tensor<2x4x8x64xf32>) outs(%4 : tensor<2x4x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = arith.addf %in, %in : f32
        %8 = arith.addf %7, %7 : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      } -> tensor<2x4x8xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                           iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<2x4x8xf32>) outs(%3 : tensor<2x4x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = math.sqrt %in : f32
        linalg.yield %7 : f32
      } -> tensor<2x4x8xf32>
      iree_tensor_ext.dispatch.tensor.store %6, %1, offsets = [0, 0, 0], sizes = [2, 4, 8], strides = [1, 1, 1] : tensor<2x4x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x4x8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_elementwise_reduction_elementwise_4d
//   CHECK:    gpu.subgroup_reduce  add {{.*}} cluster(size = 32) : (f32) -> f32

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @i4_dequant_matvec {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export public @i4_dequant_matvec ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @i4_dequant_matvec() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32x128xi4>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x128xf16>>
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf16>>
        %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4096, 32, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32x128xi4>> -> tensor<4096x32x128xi4>
        %6 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32xf16>> -> tensor<4096x32xf16>
        %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4096, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32xf16>> -> tensor<4096x32xf16>
        %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x128xf16>> -> tensor<32x128xf16>
        %9 = tensor.empty() : tensor<4096xf16>
        %10 = tensor.empty() : tensor<4096x32x128xf16>
        %11 = linalg.fill ins(%cst : f16) outs(%9 : tensor<4096xf16>) -> tensor<4096xf16>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6, %7 : tensor<4096x32x128xi4>, tensor<4096x32xf16>, tensor<4096x32xf16>) outs(%10 : tensor<4096x32x128xf16>) {
        ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
          %14 = arith.extui %in : i4 to i32
          %15 = arith.uitofp %14 : i32 to f16
          %16 = arith.subf %15, %in_1 : f16
          %17 = arith.mulf %16, %in_0 : f16
          linalg.yield %17 : f16
        } -> tensor<4096x32x128xf16>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%8, %12 : tensor<32x128xf16>, tensor<4096x32x128xf16>) outs(%11 : tensor<4096xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %14 = arith.mulf %in, %in_0 : f16
          %15 = arith.addf %14, %out : f16
          linalg.yield %15 : f16
        } -> tensor<4096xf16>
        iree_tensor_ext.dispatch.tensor.store %13, %4, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf16>>
        return
      }
    }
  }
}

//   CHECK-LABEL: func.func @i4_dequant_matvec()
//     CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//     CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//     CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//     CHECK-DAG:   %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4xf16>
//         CHECK:   scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%{{.*}} = %[[CST]]) -> (vector<1x1x4xf16>)
//         CHECK:     arith.mulf %{{.*}}, %{{.*}} : vector<1x1x4xf16>
//         CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<1x1x4xf16>

//         CHECK:   vector.multi_reduction <add>, %{{.*}}, %{{.*}} [0, 1, 2] : vector<1x1x4xf16> to f16
