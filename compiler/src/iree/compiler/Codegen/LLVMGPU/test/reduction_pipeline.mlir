// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))" \
// RUN:   --iree-codegen-llvmgpu-enable-transform-dialect-jit %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @warp_reduction_dispatch {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
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

#map = affine_map<()[s0] -> (s0 * 4)>
//         CHECK:  #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 4)>
//   CHECK-LABEL:  func.func @warp_reduction_dispatch
//     CHECK-DAG:    %[[C0I:.+]] = arith.constant 0 : i32
//     CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//     CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : i32
//     CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : i32
//     CHECK-DAG:    %[[C4:.+]] = arith.constant 4 : i32
//     CHECK-DAG:    %[[C8:.+]] = arith.constant 8 : i32
//     CHECK-DAG:    %[[C16:.+]] = arith.constant 16 : i32
//     CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : i32
//     CHECK-DAG:    %[[C32I:.+]] = arith.constant 32 : index
//     CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
//     CHECK-DAG:    %[[C10240:.+]] = arith.constant 10240 : index
//     CHECK-DAG:    %[[IDENTITY:.+]] = arith.constant 0.000000e+00 : f32
//     CHECK-DAG:    %[[CF:.+]] = arith.constant 1.000000e+00 : f32
//     CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1xf32>
//     CHECK-DAG:    %[[TID:.+]] = gpu.thread_id  x
//         CHECK:    %[[TID4:.+]] = affine.apply #[[$MAP]]()[%[[TID]]]
//         CHECK:    %[[R0:.+]] = scf.for %{{.*}} = %[[TID4]] to %[[C10240]] step %[[C1024]] iter_args(%[[A0:.+]] = %[[CST]]) -> (vector<1xf32>) {
//         CHECK:      %[[V:.+]] = vector.transfer_read {{.*}} {in_bounds = [true]} : memref<512x10240xf32>, vector<4xf32>
//         CHECK:      %[[E:.+]] = vector.extract %[[A0]][0] : vector<1xf32>
//         CHECK:      %[[RL:.+]] = vector.reduction <add>, %[[V]], %[[E]] : vector<4xf32> into f32
//         CHECK:      %[[B:.+]] = vector.broadcast %[[RL:.*]] : f32 to vector<1xf32>
//         CHECK:      scf.yield %[[B]] : vector<1xf32>
//         CHECK:    }
//         CHECK:    %[[R1:.+]] = vector.extract %[[R0]][0] : vector<1xf32>
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
//         CHECK:    %[[ALLOC:.+]] = memref.alloc() : memref<8xf32, #gpu.address_space<workgroup>>
//         CHECK:    %[[WID:.+]] = arith.divui %{{.*}}, %{{.*}} : index
//         CHECK:    %[[LANE_ID:.*]] = arith.remui %[[TID]], %[[C32I]] : index
//         CHECK:    %[[LANE0:.*]] = arith.cmpi eq, %[[LANE_ID]], %[[C0]] : index
//         CHECK:    scf.if %[[LANE0]] {
//         CHECK:      memref.store %[[R6]], %[[ALLOC]][%[[WID]]] : memref<8xf32, #gpu.address_space<workgroup>>
//         CHECK:    }
//         CHECK:    gpu.barrier
//         CHECK:    %[[LANE_ID_IN_BOUNDS:.*]] = arith.minui %[[LANE_ID]]
//         CHECK:    %[[LOAD_VAL:.+]] = memref.load %[[ALLOC]][%[[LANE_ID_IN_BOUNDS]]] : memref<8xf32, #gpu.address_space<workgroup>>
//         CHECK:    %[[S5:.+]], %{{.*}} = gpu.shuffle  xor %[[LOAD_VAL]], %[[C1]], %[[C32]] : f32
//         CHECK:    %[[R7:.+]] = arith.addf %[[LOAD_VAL]], %[[S5]] : f32
//         CHECK:    %[[S6:.+]], %{{.*}} = gpu.shuffle  xor %[[R7]], %[[C2]], %[[C32]] : f32
//         CHECK:    %[[R8:.+]] = arith.addf %[[R7]], %[[S6]] : f32
//         CHECK:    %[[S7:.+]], %{{.*}} = gpu.shuffle  xor %[[R8]], %[[C4]], %[[C32]] : f32
//         CHECK:    %[[R9:.+]] = arith.addf %[[R8]], %[[S7]] : f32
//         CHECK:    %[[S9:.+]], %{{.*}} = gpu.shuffle  idx %[[R9]], %[[C0I]], %[[C32]] : f32
//         CHECK:    %[[R12:.+]] = arith.addf %[[S9]], %[[CF]] : f32
//         CHECK:    %[[R13:.+]] = vector.broadcast %[[R12]] : f32 to vector<1xf32>
//         CHECK:    %[[TID0:.+]] = arith.cmpi eq, %[[TID]], %[[C0]] : index
//         CHECK:    scf.if %[[TID0]] {
//         CHECK:      vector.transfer_write %[[R13]], %{{.*}}[%{{.*}}] {in_bounds = [true]} : vector<1xf32>, memref<512xf32>
//         CHECK:    }

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @warp_reduction_broadcast_dispatch {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
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
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x10240xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<512x10240xf32>>
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

//   CHECK-LABEL:  func.func @warp_reduction_broadcast_dispatch
//         CHECK:    scf.for {{.*}} -> (vector<1xf32>) {
//         CHECK:      vector.transfer_read {{.*}} : memref<512x10240xf32>, vector<4xf32>
//         CHECK:      vector.reduction <add>, {{.*}} : vector<4xf32> into f32
//         CHECK:      scf.yield
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
//         CHECK:    arith.remui
//         CHECK:    scf.if
//         CHECK:      memref.store {{.*}} : memref<16xf32, #gpu.address_space<workgroup>>
//         CHECK:    }
//         CHECK:    gpu.barrier
//         CHECK:    arith.minui
//         CHECK:    memref.load
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.addf
//         CHECK:    arith.addf
//         CHECK:    vector.broadcast %{{.*}} : f32 to vector<1xf32>
//         CHECK:    scf.for
//         CHECK:      vector.transfer_read
//         CHECK:      arith.divf {{.*}} : vector<4x1xf32>
//         CHECK:      vector.transfer_write {{.*}} : vector<4xf32>, memref<512x10240xf32>
//         CHECK:    }
//         CHECK:    return

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @softmax {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
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
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<12x128x40960xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<12x128x40960xf32>> -> tensor<12x128x40960xf32>
      %3 = tensor.empty() : tensor<12x128xf32>
      %4 = tensor.empty() : tensor<12x128x40960xf32>
      %5 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 4096]]>} ins(%cst : f32) outs(%3 : tensor<12x128xf32>) -> tensor<12x128xf32>
      %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 4096]]>} ins(%cst_0 : f32) outs(%3 : tensor<12x128xf32>) -> tensor<12x128xf32>
      %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<12x128x40960xf32>) outs(%5 : tensor<12x128xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 4096]]>} {
      ^bb0(%in: f32, %out: f32):
        %11 = arith.maxf %in, %out : f32
        linalg.yield %11 : f32
      } -> tensor<12x128xf32>
      %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %7 : tensor<12x128x40960xf32>, tensor<12x128xf32>) outs(%4 : tensor<12x128x40960xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 4096]]>} {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %11 = arith.subf %in, %in_2 : f32
        %12 = math.exp %11 : f32
        linalg.yield %12 : f32
      } -> tensor<12x128x40960xf32>
      %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%8 : tensor<12x128x40960xf32>) outs(%6 : tensor<12x128xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 4096]]>} {
      ^bb0(%in: f32, %out: f32):
        %11 = arith.addf %in, %out : f32
        linalg.yield %11 : f32
      } -> tensor<12x128xf32>
      %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %9 : tensor<12x128x40960xf32>, tensor<12x128xf32>) outs(%4 : tensor<12x128x40960xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1], [0, 0, 4096]]>} {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %11 = arith.divf %cst_1, %in_2 : f32
        %12 = arith.mulf %in, %11 : f32
        linalg.yield %12 : f32
      } -> tensor<12x128x40960xf32>
      flow.dispatch.tensor.store %10, %1, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : tensor<12x128x40960xf32> -> !flow.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL:  func.func @softmax
//         CHECK:    scf.for {{.*}} -> (vector<4xf32>) {
//         CHECK:      vector.transfer_read {{.*}} : memref<12x128x40960xf32>, vector<4xf32>
//         CHECK:      arith.maxf {{.*}} : vector<4xf32>
//         CHECK:      scf.yield
//         CHECK:    vector.reduction <maxf>, %{{.*}} : vector<4xf32> into f32
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxf
//         CHECK:    arith.remui
//         CHECK:    scf.if
//         CHECK:      memref.store {{.*}} : memref<32xf32, #gpu.address_space<workgroup>>
//         CHECK:    }
//         CHECK:    gpu.barrier
//         CHECK:    arith.minui
//         CHECK:    memref.load
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maxf
//         CHECK:    arith.maxf
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
//         CHECK:    arith.divf
//         CHECK:    scf.for
//         CHECK:      vector.transfer_read
//         CHECK:      arith.subf
//         CHECK:      math.exp
//         CHECK:      arith.mulf
//         CHECK:      vector.transfer_write
//         CHECK:    }
//         CHECK:    return
