// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-linalg-ext-decompose-softmax)), iree-spirv-select-lowering-strategy-pass, iree-spirv-lower-executable-target-pass)))' \
// RUN:   %s | FileCheck %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3,
    [Shader, GroupNonUniform, GroupNonUniformShuffle], [SPV_KHR_storage_buffer_storage_class]>, Unknown:Unknown,
    #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 32, cooperative_matrix_properties_khr = []>>}>

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>

hal.executable @warp_reduction_dispatch {
  hal.executable.variant public @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
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
//     CHECK-DAG:    %[[C0I:.+]] = arith.constant 0 : i32
//     CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//     CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : i32
//     CHECK-DAG:    %[[C2:.+]] = arith.constant 2 : i32
//     CHECK-DAG:    %[[C4:.+]] = arith.constant 4 : i32
//     CHECK-DAG:    %[[C8:.+]] = arith.constant 8 : i32
//     CHECK-DAG:    %[[C16:.+]] = arith.constant 16 : i32
//     CHECK-DAG:    %[[C32:.+]] = arith.constant 32 : i32
//     CHECK-DAG:    %[[C32I:.+]] = arith.constant 32 : index
//     CHECK-DAG:    %[[C512:.+]] = arith.constant 512 : index
//     CHECK-DAG:    %[[C10240:.+]] = arith.constant 10240 : index
//     CHECK-DAG:    %[[IDENTITY:.+]] = arith.constant 0.000000e+00 : f32
//     CHECK-DAG:    %[[CF:.+]] = arith.constant 1.000000e+00 : f32
//     CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
//     CHECK-DAG:    %[[TID:.+]] = gpu.thread_id  x
//         CHECK:    %[[R0:.+]] = scf.for %{{.*}} = %[[C0]] to %[[C10240]] step %[[C512]] iter_args(%[[A0:.+]] = %[[CST]]) -> (vector<4xf32>) {
//         CHECK:      %[[V:.+]] = vector.transfer_read {{.*}} {in_bounds = [true]} : memref<512x10240xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
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
//         CHECK:    %[[ALLOC:.+]] = memref.alloc() : memref<4xf32, #gpu.address_space<workgroup>>
//         CHECK:    %[[WID:.+]] = arith.divui %{{.*}}, %{{.*}} : index
//         CHECK:    %[[LANE_ID:.*]] = arith.remui %[[TID]], %[[C32I]] : index
//         CHECK:    %[[LANE0:.*]] = arith.cmpi eq, %[[LANE_ID]], %[[C0]] : index
//         CHECK:    scf.if %[[LANE0]] {
//         CHECK:      memref.store %[[R6]], %[[ALLOC]][%[[WID]]] : memref<4xf32, #gpu.address_space<workgroup>>
//         CHECK:    }
//         CHECK:    gpu.barrier
//         CHECK:    %[[LANE_ID_IN_BOUNDS:.*]] = arith.minui %[[LANE_ID]]
//         CHECK:    %[[LOAD_VAL:.+]] = memref.load %[[ALLOC]][%[[LANE_ID_IN_BOUNDS]]] : memref<4xf32, #gpu.address_space<workgroup>>
//         CHECK:    %[[S5:.+]], %{{.*}} = gpu.shuffle  xor %[[LOAD_VAL]], %[[C1]], %[[C32]] : f32
//         CHECK:    %[[R7:.+]] = arith.addf %[[LOAD_VAL]], %[[S5]] : f32
//         CHECK:    %[[S6:.+]], %{{.*}} = gpu.shuffle  xor %[[R7]], %[[C2]], %[[C32]] : f32
//         CHECK:    %[[R8:.+]] = arith.addf %[[R7]], %[[S6]] : f32
//         CHECK:    %[[S7:.+]], %{{.*}} = gpu.shuffle  idx %[[R8]], %[[C0I]], %[[C32]] : f32
//         CHECK:    %[[R12:.+]] = arith.addf %[[S7]], %[[CF]] : f32
//         CHECK:    %[[R13:.+]] = vector.splat %[[R12]] : vector<1xf32>
//         CHECK:    %[[TID0:.+]] = arith.cmpi eq, %[[TID]], %[[C0]] : index
//         CHECK:    scf.if %[[TID0]] {
//         CHECK:      vector.transfer_write %[[R13]], %{{.*}}[%{{.*}}] {in_bounds = [true]} : vector<1xf32>, memref<512xf32, #hal.descriptor_type<storage_buffer>>
//         CHECK:    }

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3,
    [Shader, GroupNonUniform, GroupNonUniformShuffle], [SPV_KHR_storage_buffer_storage_class]>, Unknown:Unknown,
    #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 32>>}>

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>

hal.executable @warp_reduction_dispatch {
  hal.executable.variant public @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
    hal.executable.export public @warp_reduction_dispatch ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @warp_reduction_dispatch() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x9216x9216xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<10x9216x9216xf16>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [10, 9216, 9216], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x9216x9216xf16>> -> tensor<10x9216x9216xf16>
        %3 = tensor.empty() : tensor<10x9216x9216xf16>
        %4 = tensor.empty() : tensor<10x9216xf16>
        %5 = linalg.fill ins(%cst : f16) outs(%4 : tensor<10x9216xf16>) -> tensor<10x9216xf16>
        %6 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%2 : tensor<10x9216x9216xf16>) outs(%5 : tensor<10x9216xf16>) {
        ^bb0(%in: f16, %out: f16):
          %8 = arith.addf %in, %out : f16
          linalg.yield %8 : f16
        } -> tensor<10x9216xf16>
        %7 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%2, %6 : tensor<10x9216x9216xf16>, tensor<10x9216xf16>) outs(%3 : tensor<10x9216x9216xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.divf %in, %in_0 : f16
          linalg.yield %8 : f16
        } -> tensor<10x9216x9216xf16>
        flow.dispatch.tensor.store %7, %1, offsets = [0, 0, 0], sizes = [10, 9216, 9216], strides = [1, 1, 1] : tensor<10x9216x9216xf16> -> !flow.dispatch.tensor<writeonly:tensor<10x9216x9216xf16>>
        return
      }
    }
  }
}

// Check fused elementwise ops

//         CHECK:  #[[$MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0 * 8)>

//   CHECK-LABEL:  func.func @warp_reduction_dispatch

//     CHECK-DAG:    %[[I0:.+]] = arith.constant 0 : i32
//     CHECK-DAG:    %[[I1:.+]] = arith.constant 1 : i32
//     CHECK-DAG:    %[[I2:.+]] = arith.constant 2 : i32
//     CHECK-DAG:    %[[I32:.+]] = arith.constant 32 : i32

//     CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//     CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
//     CHECK-DAG:    %[[C9216:.+]] = arith.constant 9216 : index

//     CHECK-DAG:    %[[F0:.+]] = arith.constant 0.000000e+00 : f16

//     CHECK-DAG:    %[[WGIDX:.+]] = hal.interface.workgroup.id[0] : index
//     CHECK-DAG:    %[[WGIDY:.+]] = hal.interface.workgroup.id[1] : index
//     CHECK-DAG:    %[[TIDX:.+]] = gpu.thread_id  x

//     CHECK-DAG:    %[[SPAN0:.+]] = hal.interface.binding.subspan set(0) binding(0)
//     CHECK-DAG:    %[[SPAN1:.+]] = hal.interface.binding.subspan set(0) binding(1)

//         CHECK:    gpu.barrier
//         CHECK:    %{{.+}}, %{{.+}} = gpu.shuffle  xor %{{.+}}, %[[I1]], %[[I32]] : i32
//         CHECK:    %{{.+}}, %{{.+}} = gpu.shuffle  xor %{{.+}}, %[[I2]], %[[I32]] : i32
//         CHECK:    %{{.+}}, %{{.+}} = gpu.shuffle  idx %{{.+}}, %[[I0]], %[[I32]] : i32
//         CHECK:    %[[ADD:.+]] = vector.reduction <add>, %{{.+}} : vector<2xf16> into f16
//         CHECK:    %[[ADD1:.+]] = arith.addf %[[ADD]], %[[F0]] : f16
//         CHECK:    %[[SPLAT:.+]] = vector.splat %[[ADD1]] : vector<4xf16>
//         CHECK:    scf.for %[[IV:.+]] = %[[C0]] to %[[C9216]] step %[[C1024]] {
//         CHECK:      %[[OFFSET:.+]] = affine.apply #[[$MAP]](%[[IV]])[%[[TIDX]]]
//         CHECK:      %[[READ:.+]] = vector.transfer_read %[[SPAN0]][%[[WGIDY]], %[[WGIDX]], %[[OFFSET]]], %[[F0]] {in_bounds = [true]} : memref<10x9216x9216xf16{{.*}}>, vector<8xf16>
//         CHECK:      %[[SLICE0:.+]] = vector.extract_strided_slice %[[READ]] {offsets = [0], sizes = [4], strides = [1]}
//         CHECK:      %[[DIV0:.+]] = arith.divf %[[SLICE0]], %[[SPLAT]] : vector<4xf16>
//         CHECK:      %[[SLICE1:.+]] = vector.insert_strided_slice %[[DIV0]], %cst {offsets = [0], strides = [1]}
//         CHECK:      %[[SLICE2:.+]] = vector.extract_strided_slice %[[READ]] {offsets = [4], sizes = [4], strides = [1]}
//         CHECK:      %[[DIV1:.+]] = arith.divf %[[SLICE2]], %[[SPLAT]] : vector<4xf16>
//         CHECK:      %[[SLICE3:.+]] = vector.insert_strided_slice %[[DIV1]], %[[SLICE1]] {offsets = [4], strides = [1]}
//         CHECK:      vector.transfer_write %[[SLICE3]], %[[SPAN1]][%[[WGIDY]], %[[WGIDX]], %{{.*}}] {in_bounds = [true]} : vector<8xf16>, memref<10x9216x9216xf16{{.*}}>
//         CHECK:    }

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3,
    [Shader, GroupNonUniform, GroupNonUniformShuffle], []>, Unknown:Unknown, #spirv.resource_limits<
      max_compute_shared_memory_size = 49152,
      max_compute_workgroup_invocations = 1024,
      max_compute_workgroup_size = [1024, 1024, 64],
      subgroup_size = 32>>}>

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>

hal.executable @softmax {
hal.executable.variant public @vulkan_spirv_fb target(#executable_target_vulkan_spirv_fb) {
  hal.executable.export public @softmax ordinal(0) layout(#pipeline_layout) {
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
      %3 = tensor.empty() : tensor<12x128x40960xf32>
      %4 = linalg.softmax dimension(2) ins(%2 : tensor<12x128x40960xf32>) outs(%3 : tensor<12x128x40960xf32>) -> tensor<12x128x40960xf32>
      flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [12, 128, 40960], strides = [1, 1, 1] : tensor<12x128x40960xf32> -> !flow.dispatch.tensor<writeonly:tensor<12x128x40960xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL:  func.func @softmax
//         CHECK:    scf.for {{.*}} -> (vector<4xf32>) {
//         CHECK:      vector.transfer_read {{.*}} : memref<12x128x40960xf32{{.+}}>, vector<4xf32>
//         CHECK:      arith.maximumf {{.*}} : vector<4xf32>
//         CHECK:      scf.yield
//         CHECK:    vector.reduction <maximumf>, %{{.*}} : vector<4xf32> into f32
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maximumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maximumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maximumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maximumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maximumf
//         CHECK:    arith.remui
//         CHECK:    scf.if
//         CHECK:      memref.store {{.*}} : memref<32xf32, #gpu.address_space<workgroup>>
//         CHECK:    }
//         CHECK:    gpu.barrier
//         CHECK:    arith.minui
//         CHECK:    memref.load
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maximumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maximumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maximumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maximumf
//         CHECK:    gpu.shuffle  xor
//         CHECK:    arith.maximumf
//         CHECK:    arith.maximumf
//         CHECK:    vector.splat %{{.*}} : vector<4xf32>
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
//         CHECK:    vector.splat
//         CHECK:    vector.splat
//         CHECK:    scf.for
//         CHECK:      vector.transfer_read
//         CHECK:      arith.subf
//         CHECK:      math.exp
//         CHECK:      arith.divf
//         CHECK:      vector.transfer_write
//         CHECK:    }
//         CHECK:    return

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

// CHECK-LABEL: func.func @dynamic_softmax
// CHECK-DAG:     %[[ADD_PAD:.+]] = arith.constant dense<0.000000e+00> : vector<1xf16>
// CHECK-DAG:     %[[MIN_F16:.+]] = arith.constant dense<0xFC00> : vector<1xf16>
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C0_F16:.+]] = arith.constant 0.000000e+00 : f16

// CHECK:         %[[DIM_LBITS:.+]] = hal.interface.constant.load[0] : i32
// CHECK:         %[[DIM_UBITS:.+]] = hal.interface.constant.load[1] : i32
// CHECK:         %[[EXTL:.+]] = arith.extui %[[DIM_LBITS]] : i32 to i64
// CHECK:         %[[EXTU:.+]] = arith.extui %[[DIM_UBITS]] : i32 to i64
// CHECK:         %[[SHIFTU:.+]] = arith.shli %[[EXTU]], %{{.*}} : i64
// CHECK:         %[[COMBINE:.+]] = arith.ori %[[EXTL]], %[[SHIFTU]] : i64
// CHECK:         %[[DYNAMIC_SIZE:.+]] = arith.index_castui %[[COMBINE]] : i64 to index

// Do the first local reduction.
// CHECK:         vector.transfer_write %[[MIN_F16]], %{{.*}} : vector<1xf16>, memref<1x64xf16, #gpu.address_space<workgroup>>
// CHECK:         scf.for {{.*}} %[[C0]] to %[[DYNAMIC_SIZE]] step %[[C64]]
// CHECK:           %[[MASK:.+]] = vector.create_mask %{{.*}} : vector<1xi1>
// CHECK-DAG:       %[[ACC:.+]] = vector.transfer_read %{{.*}}, %cst_2, %[[MASK]] {{.*}} : memref<1x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
// CHECK-DAG:       %[[NEW:.+]] = vector.transfer_read %{{.*}}, %cst_2, %[[MASK]] {{.*}} : memref<32x?xf16, #hal.descriptor_type<storage_buffer>>, vector<1xf16>
// CHECK:           %[[MAX:.+]] = arith.maximumf %[[NEW]], %[[ACC]] : vector<1xf16>
// CHECK:           vector.transfer_write %[[MAX]], %{{.*}}, %[[MASK]] {{.*}} : vector<1xf16>, memref<1x64xf16, #gpu.address_space<workgroup>>
// CHECK:           gpu.barrier

// Finish the first reduction.
// CHECK:         vector.transfer_read {{.*}} : memref<1x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
// CHECK-COUNT-6: gpu.shuffle  xor {{.*}} : i32

// Do the elementwise scaling and second local reduction.
// CHECK:         vector.transfer_write %[[ADD_PAD]], %{{.*}} : vector<1xf16>, memref<1x64xf16, #gpu.address_space<workgroup>>
// CHECK:         scf.for {{.*}} %[[C0]] to %[[DYNAMIC_SIZE]] step %[[C64]]
// CHECK:           %[[MASK2:.+]] = vector.create_mask %{{.*}} : vector<1xi1>
// CHECK:           vector.transfer_read %{{.*}}, %[[MASK]] {{.*}} : memref<32x?xf16, #hal.descriptor_type<storage_buffer>>, vector<1xf16>
// CHECK:           vector.transfer_read %{{.*}}, %[[MASK]] {{.*}} : memref<1x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
// CHECK:           arith.subf
// CHECK:           math.exp
// CHECK:           arith.addf
// CHECK:           vector.transfer_write %{{.*}}, %[[MASK2]] {{.*}} : vector<1xf16>, memref<1x64xf16, #gpu.address_space<workgroup>>
// CHECK:           gpu.barrier

// Finish the second reduction.
// CHECK:         gpu.subgroup_reduce add {{.*}} : (f16) -> f16

// Store the result back to global memory in a loop, recomputing the
// elementwise part.
// CHECK:         scf.for {{.*}} %[[C0]] to %[[DYNAMIC_SIZE]] step %[[C64]]
// CHECK:           %[[MASK3:.+]] = vector.create_mask %{{.*}} : vector<1xi1>
// CHECK:           vector.transfer_read {{.*}} %[[MASK3]] {{.*}} : memref<32x?xf16, #hal.descriptor_type<storage_buffer>>, vector<1xf16>
// CHECK:           arith.subf
// CHECK:           math.exp
// CHECK:           arith.divf
// CHECK:           vector.transfer_write {{.*}} %[[MASK3]] {{.*}} : vector<1xf16>, memref<32x?xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:         }
