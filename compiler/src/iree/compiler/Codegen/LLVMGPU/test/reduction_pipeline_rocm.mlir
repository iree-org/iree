// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:  --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:  %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 \
// RUN:  --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:  %s | FileCheck %s --check-prefix=CDNA3

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @group_reduction_1d {
hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
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

//         CDNA3: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [32, 1, 1] subgroup_size = 32>
//         CDNA3: func.func @group_reduction_1d()
//    CDNA3-SAME:    translation_info = #[[$TRANSLATION]]
// CDNA3-COUNT-5:     gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @group_reduction_1d {
hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
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

// On CDNA, we prefer wave64 with subgroup size of 64.

//        CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//        CHECK: func.func @group_reduction_1d
// CHECK-COUNT-5:     gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @i4_dequant_matvec {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
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

//        CDNA3: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 32>
//        CDNA3: func.func @i4_dequant_matvec()
//   CDNA3-SAME:    translation_info = #[[$TRANSLATION]]
//         CDNA3:   %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x8xf16>
//         CDNA3:   %[[FOR:.+]] = scf.for %{{.+}} = %c0 to %c32 step %c4 iter_args(%[[ARG:.+]] = %[[CST]]) -> (vector<1x8xf16>)
//         CDNA3:     %[[READ0:.+]] = vector.transfer_read {{.+}} : memref<4096x32x128xi4, #hal.descriptor_type<storage_buffer>>, vector<1x8xi4>
//         CDNA3:     %[[READ1:.+]] = vector.transfer_read {{.+}} : memref<4096x32xf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
//         CDNA3:     %[[READ2:.+]] = vector.transfer_read {{.+}} : memref<4096x32xf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
//         CDNA3:     %[[READ3:.+]] = vector.transfer_read {{.+}} : memref<32x128xf16, #hal.descriptor_type<storage_buffer>>, vector<1x8xf16>
//         CDNA3:     %[[EXTEND:.+]] = arith.extui %[[READ0]] : vector<1x8xi4> to vector<1x8xi32>
//         CDNA3:     %[[CVT:.+]] = arith.uitofp %[[EXTEND]] : vector<1x8xi32> to vector<1x8xf16>
//         CDNA3:     %[[SUB:.+]] = arith.subf %[[CVT]], %[[READ1]] : vector<1x8xf16>
//         CDNA3:     %[[MUL0:.+]] = arith.mulf %[[SUB]], %[[READ2]] : vector<1x8xf16>
//         CDNA3:     %[[MUL1:.+]] = arith.mulf %[[READ3]], %[[MUL0]] : vector<1x8xf16>
//         CDNA3:     %[[ADD:.+]] = arith.addf %[[MUL1]], %[[ARG]] : vector<1x8xf16>

//         CDNA3:   %[[SCAST:.+]] = vector.shape_cast %[[FOR]] : vector<1x8xf16> to vector<8xf16>
//         CDNA3:   vector.reduction <add>, %[[SCAST]] : vector<8xf16> into f16
// CDNA3-COUNT-6:   gpu.shuffle  xor
//         CDNA3:   scf.if
//         CDNA3:     vector.transfer_write

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @i4_dequant_matvec {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
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

//      CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//      CHECK: func.func @i4_dequant_matvec()
// CHECK-SAME:     translation_info = #[[$TRANSLATION]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @matvec_fp16 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matvec_fp16 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_fp16() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
        %5 = tensor.empty() : tensor<1x32000xf16>
        %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 8], [0, 0, 512]]>} ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 8], [0, 0, 512]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<1x32000xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        return
      }
    }
  }
}

// This matvec is expected to be reduced multiple rows at a time by a single workgroup.
// Check that we distribute it across subgroup threads properly. Thread 0 is expected to
// write 8 results at the end.
// TODO(kuhar): We should reduce the number of `gpu.shuffles` performed.

//          CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//          CHECK: func.func @matvec_fp16()
//     CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//      CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK-DAG:   %[[C512:.+]] = arith.constant 512 : index
//      CHECK-DAG:   %[[C4096:.+]] = arith.constant 4096 : index
//      CHECK-DAG:   %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<8x8xf16>
//          CHECK:   scf.for %{{.+}} = %[[C0]] to %[[C4096]] step %[[C512]] iter_args(%[[ARG:.+]] = %[[CST]]) -> (vector<8x8xf16>)
//      CHECK-DAG:     %[[MAT:.+]] = vector.transfer_read {{.+}} : memref<32000x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<8x8xf16>
//      CHECK-DAG:     %[[VEC:.+]] = vector.transfer_read {{.+}} : memref<1x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<8x8xf16>
//          CHECK:     %[[MUL:.+]] = arith.mulf %[[VEC]], %[[MAT]] : vector<8x8xf16>
//          CHECK:     %[[ADD:.+]] = arith.addf %[[ARG]], %[[MUL]] : vector<8x8xf16>

//          CHECK:   vector.reduction <add>, %{{.+}} : vector<8xf16> into f16
// CHECK-COUNT-24:   gpu.shuffle xor
//          CHECK:   scf.if
//          CHECK:     vector.transfer_write {{.+}} : vector<8xf16>, memref<1x32000xf16, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @matvec_fp16 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matvec_fp16 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_fp16() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
        %5 = tensor.empty() : tensor<1x32000xf16>
        %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 4], [0, 0, 512]]>} ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 4], [0, 0, 512]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<1x32000xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        return
      }
    }
  }
}

// Multi-row matvec with wave32.
// TODO(kuhar): We should reduce the number of `gpu.shuffles` performed.

//          CDNA3: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 32>
//          CDNA3: func.func @matvec_fp16()
//     CDNA3-SAME:     translation_info = #[[$TRANSLATION]]
//      CDNA3-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CDNA3-DAG:   %[[C512:.+]] = arith.constant 512 : index
//      CDNA3-DAG:   %[[C4096:.+]] = arith.constant 4096 : index
//      CDNA3-DAG:   %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<8x8xf16>
//          CDNA3:   scf.for %{{.+}} = %[[C0]] to %[[C4096]] step %[[C512]] iter_args(%[[ARG:.+]] = %[[CST]]) -> (vector<8x8xf16>)
//      CDNA3-DAG:     %[[MAT:.+]] = vector.transfer_read {{.+}} : memref<32000x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<8x8xf16>
//      CDNA3-DAG:     %[[VEC:.+]] = vector.transfer_read {{.+}} : memref<1x4096xf16, #hal.descriptor_type<storage_buffer>>, vector<8x8xf16>
//          CDNA3:     %[[MUL:.+]] = arith.mulf %[[VEC]], %[[MAT]] : vector<8x8xf16>
//          CDNA3:     %[[ADD:.+]] = arith.addf %[[ARG]], %[[MUL]] : vector<8x8xf16>

//          CDNA3:   vector.reduction <add>, %{{.+}} : vector<8xf16> into f16
// CDNA3-COUNT-24:   gpu.shuffle xor
//          CDNA3:   scf.if
//          CDNA3:     vector.transfer_write {{.+}} : vector<8xf16>, memref<1x32000xf16, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @multi_reduction {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @multi_reduction ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @multi_reduction() {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 2.304000e+05 : f32
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %c85483008 = arith.constant 85483008 : index
        %c165416448 = arith.constant 165416448 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c85483008) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<2x32x60x3840xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c165416448) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<2x32x60x3840xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 60, 3840], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x60x3840xf16>> -> tensor<2x32x60x3840xf16>
        %3 = tensor.empty() : tensor<2x32x60x3840xf32>
        %4 = tensor.empty() : tensor<2x32xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<2x32x60x3840xf16>) outs(%3 : tensor<2x32x60x3840xf32>) {
        ^bb0(%in: f16, %out: f32):
          %11 = arith.extf %in : f16 to f32
          linalg.yield %11 : f32
        } -> tensor<2x32x60x3840xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%4 : tensor<2x32xf32>) -> tensor<2x32xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%5 : tensor<2x32x60x3840xf32>) outs(%6 : tensor<2x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          %11 = arith.addf %in, %out : f32
          linalg.yield %11 : f32
        } -> tensor<2x32xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<2x32xf32>) outs(%4 : tensor<2x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          %11 = arith.divf %in, %cst_0 : f32
          linalg.yield %11 : f32
        } -> tensor<2x32xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%5, %8 : tensor<2x32x60x3840xf32>, tensor<2x32xf32>) outs(%6 : tensor<2x32xf32>) {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %11 = arith.subf %in, %in_2 : f32
          %12 = arith.mulf %11, %11 : f32
          %13 = arith.addf %12, %out : f32
          linalg.yield %13 : f32
        } -> tensor<2x32xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %8, %9 : tensor<2x32x60x3840xf16>, tensor<2x32xf32>, tensor<2x32xf32>) outs(%3 : tensor<2x32x60x3840xf32>) {
        ^bb0(%in: f16, %in_2: f32, %in_3: f32, %out: f32):
          %11 = arith.divf %in_3, %cst_0 : f32
          %12 = arith.addf %11, %cst_1 : f32
          %13 = math.rsqrt %12 : f32
          %14 = arith.extf %in : f16 to f32
          %15 = arith.subf %14, %in_2 : f32
          %16 = arith.mulf %15, %13 : f32
          linalg.yield %16 : f32
        } -> tensor<2x32x60x3840xf32>
        flow.dispatch.tensor.store %10, %1, offsets = [0, 0, 0, 0], sizes = [2, 32, 60, 3840], strides = [1, 1, 1, 1] : tensor<2x32x60x3840xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x60x3840xf32>>
        return
      }
    }
  }
}

// Check that all loops are singly nested.
//
//          CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//          CHECK: func.func @multi_reduction()
//     CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//      CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK-DAG:   %[[C225:.+]] = arith.constant 225 : index
//          CHECK:   %[[RES0:.+]] = scf.for %[[ARG0:[a-zA-Z0-9]+]] = %[[C0]] to %[[C225]] step %[[C1]]
//     CHECK-NEXT:     %[[DELIN:.+]]:2 = affine.delinearize_index %[[ARG0]] into (15, 15) : index, index
//          CHECK:   %[[RES1:.+]] = scf.for %[[ARG0:[a-zA-Z0-9]+]] = %[[C0]] to %[[C225]] step %[[C1]]
//     CHECK-NEXT:     %[[DELIN:.+]]:2 = affine.delinearize_index %[[ARG0]] into (15, 15) : index, index
//          CHECK:   scf.for %[[ARG0:[a-zA-Z0-9]+]] = %[[C0]] to %[[C225]] step %[[C1]]
//     CHECK-NEXT:     %[[DELIN:.+]]:2 = affine.delinearize_index %[[ARG0]] into (15, 15) : index, index
