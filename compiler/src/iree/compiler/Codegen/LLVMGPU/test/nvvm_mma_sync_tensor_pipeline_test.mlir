// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))" %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
#translation = #iree_codegen.translation_info<LLVMGPUMatmulTensorCoreMmaSyncOnTensors pipeline_depth = 4>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation, workgroup_size = [2, 2, 1]>
#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#executable_target_cuda_nvptx_fb], legacy_sync}>
module attributes {hal.device.targets = [#device_target_cuda]} {
  hal.executable private @foo_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
      hal.executable.export public @matmul_f16xf16xf32_to_f16 ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @matmul_f16xf16xf32_to_f16() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3456x2048xf16>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1024xf16>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<3456x1024xf16>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [3456, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<3456x2048xf16>> -> tensor<3456x2048xf16>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1024xf16>> -> tensor<2048x1024xf16>
          %5 = tensor.empty() : tensor<3456x1024xf16>
          %6 = tensor.empty() : tensor<3456x1024xf32>
          %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<3456x1024xf32>) -> tensor<3456x1024xf32>
          %8 = linalg.matmul {compilation_info = #compilation} ins(%3, %4 : tensor<3456x2048xf16>, tensor<2048x1024xf16>) outs(%7 : tensor<3456x1024xf32>) -> tensor<3456x1024xf32>
          %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<3456x1024xf32>) outs(%5 : tensor<3456x1024xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          } -> tensor<3456x1024xf16>
          flow.dispatch.tensor.store %9, %2, offsets = [0, 0], sizes = [3456, 1024], strides = [1, 1] : tensor<3456x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<3456x1024xf16>>
          return
        }
      }
    }
  }
}
// CHECK: func.func @matmul_f16xf16xf32_to_f16
// CHECK-NOT:         memref.alloc
//
// After unrolling to mma shapes, there are (64 / 16) * (64 / 8) = 32 results in
// total.
//
// CHECK:             %[[SCF:.+]]:32 = scf.for
// CHECK-COUNT-31:    %{{.+}} = vector.insert_strided_slice %[[SCF]]#
// CHECK:             %[[MATMUL_RES:.+]] = vector.insert_strided_slice %[[SCF]]#
// CHECK:             %[[TRUNCF:.+]] = arith.truncf %[[MATMUL_RES]] : vector<64x64xf32> to vector<64x64xf16>
