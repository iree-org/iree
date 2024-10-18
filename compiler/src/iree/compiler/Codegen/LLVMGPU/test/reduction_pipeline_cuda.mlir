// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_60 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-decompose-softmax), iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target)))))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @warp_reduction_dispatch {
hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export @warp_reduction_dispatch layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @warp_reduction_dispatch() {
      %c0 = arith.constant 0 : index
      %c10240 = arith.constant 10240 : index
      %cst = arith.constant 1.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<512x10240xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<writeonly:tensor<512xf32>>
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

//         CHECK: #[[TRANSLATION_INFO:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction workgroup_size = [256, 1, 1] subgroup_size = 32>
//         CHECK:  func.func @warp_reduction_dispatch()
//    CHECK-SAME:      translation_info = #[[TRANSLATION_INFO]]
//     CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
//     CHECK-DAG:    %[[TID:.+]] = gpu.thread_id  x
//         CHECK:    %[[R0:.+]] = scf.for %{{.*}} = %c0 to %c10240 step %c1024 iter_args(%[[A0:.+]] = %[[CST]]) -> (vector<4xf32>) {
//         CHECK:      %[[V:.+]] = vector.transfer_read {{.*}} : memref<512x10240xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//         CHECK:      %[[ADD:.+]] = arith.addf %[[V]], %[[A0]] : vector<4xf32>
//         CHECK:      scf.yield %[[ADD]] : vector<4xf32>
//         CHECK:    }
// CHECK-COUNT-5:    gpu.shuffle  xor {{.*}} : f32
//         CHECK:    %[[ALLOC:.+]] = memref.alloc() : memref<8xf32, #gpu.address_space<workgroup>>
//         CHECK:    scf.if %{{.*}} {
//         CHECK:      memref.store %{{.*}}, %[[ALLOC]]{{.*}} : memref<8xf32, #gpu.address_space<workgroup>>
//         CHECK:    }
//         CHECK:    gpu.barrier
//         CHECK:    memref.load %[[ALLOC]]{{.*}} : memref<8xf32, #gpu.address_space<workgroup>>
// CHECK-COUNT-3:    gpu.shuffle  xor {{.*}} : f32
//         CHECK:    gpu.shuffle  idx {{.*}} : f32
//         CHECK:    scf.if %{{.*}} {
//         CHECK:      vector.transfer_write {{.*}} : vector<1xf32>, memref<512xf32, #hal.descriptor_type<storage_buffer>>
//         CHECK:    }

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @warp_reduction_broadcast_dispatch {
hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export @warp_reduction_broadcast_dispatch layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @warp_reduction_broadcast_dispatch() {
      %c0 = arith.constant 0 : index
      %c10240 = arith.constant 10240 : index
      %cst_0 = arith.constant 3.840000e+02 : f32
      %cst = arith.constant 1.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<512x10240xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<writeonly:tensor<512x10240xf32>>
      %5 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:tensor<512x10240xf32>> -> tensor<512x10240xf32>
      %8 = tensor.empty() : tensor<512xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<512xf32>) -> tensor<512xf32>
      %10 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]}
          ins(%5 : tensor<512x10240xf32>) outs(%9 : tensor<512xf32>) {
        ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
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
      flow.dispatch.tensor.store %11, %1, offsets = [0, 0], sizes = [512, 10240], strides = [1, 1]
          : tensor<512x10240xf32> -> !flow.dispatch.tensor<writeonly:tensor<512x10240xf32>>
      return
    }
  }
}
}

//         CHECK: #[[TRANSLATION_INFO:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction workgroup_size = [256, 1, 1] subgroup_size = 32>
//         CHECK:  func.func @warp_reduction_broadcast_dispatch()
//    CHECK-SAME:      translation_info = #[[TRANSLATION_INFO]]
//         CHECK:    scf.for {{.*}} -> (vector<4xf32>) {
//         CHECK:      vector.transfer_read {{.*}} : memref<512x10240xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//         CHECK:      arith.addf {{.*}} : vector<4xf32>
//         CHECK:      scf.yield
// CHECK-COUNT-5:    gpu.shuffle  xor
//         CHECK:    scf.if
//         CHECK:      memref.store {{.*}} : memref<8xf32, #gpu.address_space<workgroup>>
//         CHECK:    }
// CHECK-COUNT-3:    gpu.shuffle  xor
//         CHECK:    gpu.shuffle  idx
//         CHECK:    arith.divf {{.*}} : vector<4xf32>
//         CHECK:    scf.for
//         CHECK:      vector.transfer_write {{.*}} : vector<4xf32>, memref<512x10240xf32, #hal.descriptor_type<storage_buffer>>
//         CHECK:    }
//         CHECK:    return

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @softmax {
hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export @softmax layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @softmax() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -3.40282347E+38 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %cst_1 = arith.constant 1.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<12x128x40960xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<12x128x40960xf32>> -> tensor<12x128x40960xf32>
      %3 = tensor.empty() : tensor<12x128x40960xf32>
      %4 = linalg.softmax dimension(2) ins(%2 : tensor<12x128x40960xf32>) outs(%3 : tensor<12x128x40960xf32>) -> tensor<12x128x40960xf32>
      flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : tensor<12x128x40960xf32> -> !flow.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
      return
    }
  }
}
}

//         CHECK: #[[TRANSLATION_INFO:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction workgroup_size = [1024, 1, 1] subgroup_size = 32>
//         CHECK:  func.func @softmax()
//    CHECK-SAME:      translation_info = #[[TRANSLATION_INFO]]
//         CHECK:    scf.for {{.*}} -> (vector<4xf32>) {
//         CHECK:      vector.transfer_read {{.*}} : memref<12x128x40960xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//         CHECK:      arith.maxnumf {{.*}} : vector<4xf32>
//         CHECK:      scf.yield
//         CHECK:    vector.reduction <maxnumf>, %{{.*}} : vector<4xf32> into f32
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxnumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxnumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxnumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxnumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxnumf
//         CHECK:    arith.remui
//         CHECK:    scf.if
//         CHECK:      memref.store {{.*}} : memref<32xf32, #gpu.address_space<workgroup>>
//         CHECK:    }
//         CHECK:    gpu.barrier
//         CHECK:    arith.minui
//         CHECK:    memref.load
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxnumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxnumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxnumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxnumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxnumf
//         CHECK:    arith.maxnumf
//         CHECK:    vector.broadcast %{{.*}} : f32 to vector<4xf32>
//         CHECK:    scf.for {{.*}} -> (vector<4xf32>) {
//         CHECK:      vector.transfer_read
//         CHECK:      arith.subf
//         CHECK:      math.exp
//         CHECK:      arith.addf
//         CHECK:      scf.yield
//         CHECK:    vector.reduction <add>, %{{.*}} : vector<4xf32> into f32
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    scf.if
//         CHECK:      memref.store {{.*}} : memref<32xf32, #gpu.address_space<workgroup>>
//         CHECK:    }
//         CHECK:    gpu.barrier
//         CHECK:    memref.load
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    arith.addf
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
  hal.executable.export public @small_reduction ordinal(0) layout(#pipeline_layout) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @small_reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1024x13xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 13], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x13xf32>> -> tensor<1024x13xf32>
      %3 = tensor.empty() : tensor<1024xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1024xf32>) -> tensor<1024xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<1024x13xf32>) outs(%4 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<1024xf32>
      flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [1024], strides = [1] : tensor<1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024xf32>>
      return
    }
  }
}
}

// Small reduction computes the whole reduction on a single thread.
//   CHECK-LABEL: func.func @small_reduction
//         CHECK: scf.for %{{.*}} = %c0 to %c13 step %c4
//         CHECK:   linalg.generic
//         CHECK:     arith.addf

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @group_reduction {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @group_reduction ordinal(0) layout(#pipeline_layout) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x64xf32>) outs(%4 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<8xf32>
      flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_reduction
//         CHECK:   %[[RD:.+]] = vector.transfer_read {{.*}} memref<8x64xf32, #hal.descriptor_type<storage_buffer>>, vector<2xf32>
//         CHECK:   %[[ADD:.+]] = arith.addf %[[RD]]
//         CHECK:   vector.reduction <add>, %[[ADD]]
// CHECK-COUNT-5:   gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf
//         CHECK:   scf.if
//         CHECK:     vector.transfer_write {{.*}} memref<8xf32, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @group_elementwise_reduction_elementwise {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @group_elementwise_reduction_elementwise ordinal(0) layout(#pipeline_layout) {
  ^bb0(%arg0: !hal.device, %arg1: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_elementwise_reduction_elementwise() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
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
      flow.dispatch.tensor.store %6, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_elementwise_reduction_elementwise
//         CHECK:   vector.transfer_read {{.*}} vector<2xf32>
//         CHECK:   arith.addf{{.*}} : vector<2xf32>
//         CHECK:   arith.addf{{.*}} : vector<2xf32>
//         CHECK:   arith.addf{{.*}} : vector<2xf32>
// CHECK-COUNT-5:   gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf
//         CHECK:   %[[SQRT_VEC:.+]] = math.sqrt
//         CHECK:   scf.if
//         CHECK:     vector.transfer_write %[[SQRT_VEC]], {{.*}} : vector<1xf32>, memref<8xf32, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @group_reduction_larger {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @group_reduction_larger ordinal(0) layout(#pipeline_layout) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction_larger() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<33x1024xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<33xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [33, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<33x1024xf32>> -> tensor<33x1024xf32>
      %3 = tensor.empty() : tensor<33xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<33xf32>) -> tensor<33xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<33x1024xf32>) outs(%4 : tensor<33xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<33xf32>
      flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [33], strides = [1] : tensor<33xf32> -> !flow.dispatch.tensor<writeonly:tensor<33xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_reduction_larger
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf
//         CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8xf32, #gpu.address_space<workgroup>>
//         CHECK: scf.if
//         CHECK:   memref.store %{{.*}}, %[[ALLOC]][%{{.*}}] : memref<8xf32, #gpu.address_space<workgroup>>
//         CHECK: }
//         CHECK: arith.minui
//         CHECK: memref.load
// CHECK-COUNT-3: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf
//         CHECK: %[[RES:.*]], %{{.*}} = gpu.shuffle  idx
//         CHECK: %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : f32 to vector<1xf32>
//         CHECK: scf.if
//         CHECK:   vector.transfer_write %[[RES_VEC]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @group_reduction_1d {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @group_reduction_1d ordinal(0) layout(#pipeline_layout) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction_1d() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<f32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [64], strides = [1] : !flow.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
      %3 = tensor.empty() : tensor<f32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%2 : tensor<64xf32>) outs(%4 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<f32>
      flow.dispatch.tensor.store %5, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_reduction_1d
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @group_elementwise_reduction_elementwise_4d {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) {
  hal.executable.export public @group_elementwise_reduction_elementwise_4d ordinal(0) layout(#pipeline_layout) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_elementwise_reduction_elementwise_4d() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x4x8x64xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x4x8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 4, 8, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x8x64xf32>> -> tensor<2x4x8x64xf32>
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
      flow.dispatch.tensor.store %6, %1, offsets = [0, 0, 0], sizes = [2, 4, 8], strides = [1, 1, 1] : tensor<2x4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x4x8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_elementwise_reduction_elementwise_4d
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

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
    hal.executable.export public @i4_dequant_matvec ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @i4_dequant_matvec() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32x128xi4>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x128xf16>>
        %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4096xf16>>
        %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4096, 32, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32x128xi4>> -> tensor<4096x32x128xi4>
        %6 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32xf16>> -> tensor<4096x32xf16>
        %7 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4096, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32xf16>> -> tensor<4096x32xf16>
        %8 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128xf16>> -> tensor<32x128xf16>
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
        flow.dispatch.tensor.store %13, %4, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<4096xf16>>
        return
      }
    }
  }
}

//   CHECK-LABEL: func.func @i4_dequant_matvec()
//         CHECK:   %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x8xf16>
//         CHECK:   %[[FOR:.+]] = scf.for %{{.+}} = %c0 to %c32 step %c4 iter_args(%[[ARG:.+]] = %[[CST]]) -> (vector<1x8xf16>)
//         CHECK:     %[[READ0:.+]] = vector.transfer_read {{.+}} : memref<4096x32x128xi4, #hal.descriptor_type<storage_buffer>>, vector<1x8xi4>
//         CHECK:     %[[READ1:.+]] = vector.transfer_read {{.+}} : memref<4096x32xf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
//         CHECK:     %[[READ2:.+]] = vector.transfer_read {{.+}} : memref<4096x32xf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
//         CHECK:     %[[READ3:.+]] = vector.transfer_read {{.+}} : memref<32x128xf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
//         CHECK:     %[[EXTEND:.+]] = arith.extui %[[READ0]] : vector<1x8xi4> to vector<1x8xi32>
//         CHECK:     %[[CVT:.+]] = arith.uitofp %[[EXTEND]] : vector<1x8xi32> to vector<1x8xf16>
//         CHECK:     %[[SUB:.+]] = arith.subf %[[CVT]], %[[READ1]] : vector<1x8xf16>
//         CHECK:     %[[MUL0:.+]] = arith.mulf %[[SUB]], %[[READ2]] : vector<1x8xf16>
//         CHECK:     %[[MUL1:.+]] = arith.mulf %[[READ3]], %[[MUL0]] : vector<1x8xf16>
//         CHECK:     %[[ADD:.+]] = arith.addf %[[MUL1]], %[[ARG]] : vector<1x8xf16>

//         CHECK:   %[[SCAST:.+]] = vector.shape_cast %[[FOR]] : vector<1x8xf16> to vector<8xf16>
//         CHECK:   vector.reduction <add>, %[[SCAST]] : vector<8xf16> into f16
// CHECK-COUNT-6:   gpu.shuffle  xor
//         CHECK:   scf.if
//         CHECK:     vector.transfer_write
