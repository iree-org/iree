// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-select-lowering-strategy, iree-llvmgpu-lower-executable-target)))" %s | FileCheck %s

hal.executable @group_reduction_1d {
hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {target_arch = "gfx1100"}>) {
  hal.executable.export public @group_reduction_1d ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction_1d() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<f32>>
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

// CHECK:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction>
// CHECK-LABEL: hal.executable.export public @group_redution_1d
// CHECK-SAME:    subgroup_size = 32
// CHECK-SAME:    translation_info = #[[$TRANSLATION]]
// CHECK-SAME:    workgroup_size = [32 : index, 1 : index, 1 : index]
//CHECK-LABEL:      func.func @group_reduction_1d
// CHECK-COUNT-5:     gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

// -----

hal.executable @group_reduction_1d {
hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {target_arch = "gfx940"}>) {
  hal.executable.export public @group_reduction_1d ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction_1d() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<f32>>
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

// On CDNA, we prefer wave64 with subgroup size of 64.

// CHECK:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction>
// CHECK-LABEL: hal.executable.export public @group_redution_1d
// CHECK-SAME:    subgroup_size = 64
// CHECK-SAME:    translation_info = #[[$TRANSLATION]]
// CHECK-SAME:    workgroup_size = [64 : index, 1 : index, 1 : index]
//CHECK-LABEL:      func.func @group_reduction_1d
// CHECK-COUNT-5:     gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

// -----

hal.executable private @i4_dequant_matvec {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {target_arch = "gfx1100"}>) {
    hal.executable.export public @i4_dequant_matvec ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @i4_dequant_matvec() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32x128xi4>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x128xf16>>
        %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4096xf16>>
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

// CHECK:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction>
// CHECK-LABEL: hal.executable.export public @i4_dequent_matvec
// CHECK-SAME:    subgroup_size = 32
// CHECK-SAME:    translation_info = #[[$TRANSLATION]]
// CHECK-SAME:    workgroup_size = [64 : index, 1 : index, 1 : index]
//
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
//         CHECK:   %[[EXTRACT:.+]] = vector.extract_strided_slice %[[SCAST]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
//         CHECK:   vector.reduction <add>, %[[EXTRACT]] : vector<4xf16> into f16
// CHECK-COUNT-6:   gpu.shuffle  xor
//         CHECK:   scf.if
//         CHECK:     vector.transfer_write

// -----

hal.executable private @i4_dequant_matvec {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {target_arch = "gfx940"}>) {
    hal.executable.export public @i4_dequant_matvec ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @i4_dequant_matvec() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32x128xi4>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x128xf16>>
        %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4096xf16>>
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

// CHECK:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction>
// CHECK-LABEL: hal.executable.export public @i4_dequent_matvec
// CHECK-SAME:    subgroup_size = 64
// CHECK-SAME:    translation_info = #[[$TRANSLATION]]
// CHECK-SAME:    workgroup_size = [64 : index, 1 : index, 1 : index]
// CHECK-LABEL:     func.func @i4_dequant_matvec()
