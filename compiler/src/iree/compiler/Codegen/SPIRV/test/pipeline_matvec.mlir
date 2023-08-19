// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-linalg-to-spirv-pipeline)))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable @i4_dequant_matvec_f16 {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, Float16, StorageBuffer16BitAccess, GroupNonUniform, GroupNonUniformShuffle], [SPV_KHR_16bit_storage]>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @i4_dequant_matvec_f16 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @i4_dequant_matvec_f16() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x1xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x1xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1x86x128xf16>>
        %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x1x4096xf16>>
        %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4096, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>> -> tensor<4096x86x128xi4>
        %6 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4096, 86, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x1xf16>> -> tensor<4096x86x1xf16>
        %7 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [4096, 86, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x1xf16>> -> tensor<4096x86x1xf16>
        %8 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [1, 1, 86, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x86x128xf16>> -> tensor<1x1x86x128xf16>
        %9 = tensor.empty() : tensor<1x1x4096xf16>
        %10 = tensor.empty() : tensor<4096x86x128xf16>
        %11 = linalg.fill ins(%cst : f16) outs(%9 : tensor<1x1x4096xf16>) -> tensor<1x1x4096xf16>
        %12 = linalg.generic {
            indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%5, %6, %7 : tensor<4096x86x128xi4>, tensor<4096x86x1xf16>, tensor<4096x86x1xf16>) outs(%10 : tensor<4096x86x128xf16>) {
        ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
          %14 = arith.extui %in : i4 to i32
          %15 = arith.uitofp %14 : i32 to f16
          %16 = arith.subf %15, %in_1 : f16
          %17 = arith.mulf %16, %in_0 : f16
          linalg.yield %17 : f16
        } -> tensor<4096x86x128xf16>
        %13 = linalg.generic {
            indexing_maps = [
                affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
                affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
                affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
        ins(%8, %12 : tensor<1x1x86x128xf16>, tensor<4096x86x128xf16>) outs(%11 : tensor<1x1x4096xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %14 = arith.mulf %in, %in_0 : f16
          %15 = arith.addf %14, %out : f16
          linalg.yield %15 : f16
        } -> tensor<1x1x4096xf16>
        flow.dispatch.tensor.store %13, %4, offsets = [0, 0, 0], sizes = [1, 1, 4096], strides = [1, 1, 1] : tensor<1x1x4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<1x1x4096xf16>>
        return
      }
    }
  }
}

//   CHECK-LABEL: spirv.func @i4_dequant_matvec_f16()

//         CHECK: %[[C15:.+]] = spirv.Constant 15 : i32

//         CHECK: spirv.mlir.loop

// Load the quantized weight and get 8xi4 out of it.
//         CHECK:   spirv.Load "StorageBuffer" %{{.+}} : i32
// CHECK-COUNT-8:   spirv.BitwiseAnd %{{.+}}, %[[C15]] : i32

// CHECK-COUNT-2:   spirv.ConvertUToF %{{.+}} : vector<4xi32> to vector<4xf16>
// CHECK-COUNT-2:   spirv.FSub %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-4:   spirv.FMul %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:   spirv.FAdd %{{.+}}, %{{.+}} : vector<4xf16>
// CHECK-COUNT-2:   spirv.Bitcast %{{.+}} : vector<4xf16> to vector<2xf32>
// CHECK-COUNT-2:   spirv.VectorShuffle {{.+}} : vector<2xf32> -> vector<4xf32>

//         CHECK:   spirv.mlir.merge

//         CHECK: %[[LD:.+]] = spirv.Load "Function" %4 : vector<4xf32>
//         CHECK: %[[VS0:.+]] = spirv.VectorShuffle [0 : i32, 1 : i32] %[[LD]]
//         CHECK: spirv.Bitcast %[[VS0]] : vector<2xf32> to vector<4xf16>
//         CHECK: %[[VS1:.+]] = spirv.VectorShuffle [2 : i32, 3 : i32] %[[LD]]
//         CHECK: spirv.Bitcast %[[VS1]] : vector<2xf32> to vector<4xf16>

// CHECK-COUNT-5: spirv.GroupNonUniformShuffleXor <Subgroup>

//         CHECK: spirv.mlir.selection
