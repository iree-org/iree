// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1201 --iree-codegen-llvmgpu-configuration-pipeline --iree-codegen-llvmgpu-rocdl-lowering-pipeline %s | FileCheck %s --check-prefix=OCP
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 --iree-codegen-llvmgpu-configuration-pipeline --iree-codegen-llvmgpu-rocdl-lowering-pipeline %s | FileCheck %s --check-prefix=OCP

// RUN: not iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-codegen-llvmgpu-configuration-pipeline --iree-codegen-llvmgpu-rocdl-lowering-pipeline -o /dev/null 2>&1 %s | FileCheck %s --check-prefix=ERRORS
// RUN: not iree-opt --split-input-file --iree-gpu-test-target=gfx908 --iree-codegen-llvmgpu-configuration-pipeline --iree-codegen-llvmgpu-rocdl-lowering-pipeline -o /dev/null 2>&1 %s | FileCheck %s --check-prefix=ERRORS

// With --iree-llvmgpu-enable-small-float-emulation, unsupported chips use software emulation.
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-llvmgpu-enable-small-float-emulation --iree-codegen-llvmgpu-configuration-pipeline --iree-codegen-llvmgpu-rocdl-lowering-pipeline %s | FileCheck %s --check-prefix=EMULATED
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx908 --iree-llvmgpu-enable-small-float-emulation --iree-codegen-llvmgpu-configuration-pipeline --iree-codegen-llvmgpu-rocdl-lowering-pipeline %s | FileCheck %s --check-prefix=EMULATED

#map = affine_map<(d0) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @ext_fp8_dispatch() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf8E4M3FN>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf8E5M2>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf8E4M3FN>> -> tensor<4096xf8E4M3FN>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf8E5M2>> -> tensor<4096xf8E5M2>
  %5 = tensor.empty() : tensor<4096xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel"]}
                       ins(%3, %4 : tensor<4096xf8E4M3FN>, tensor<4096xf8E5M2>)
                       outs(%5 : tensor<4096xf32>) {
  ^bb0(%in0: f8E4M3FN, %in1: f8E5M2, %out: f32):
    %7 = arith.extf %in0 : f8E4M3FN to f32
    %8 = arith.extf %in1 : f8E5M2 to f32
    %9 = arith.addf %7, %8 : f32
    linalg.yield %9 : f32
  } -> tensor<4096xf32>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf32>>
  return
}

// ERRORS: F8E5M2 and F8E4M3FN types are not supported on gfx942 (MI-300) or older chipsets; try F8E5M2FNUZ or F8E4M3FNUZ instead, or use --iree-llvmgpu-enable-small-float-emulation

//   OCP-LABEL: llvm.func @ext_fp8_dispatch
// OCP-COUNT-8:   rocdl.cvt.pk.f32.fp8 %{{.*}} : vector<2xf32>
// OCP-COUNT-8:   rocdl.cvt.pk.f32.bf8 %{{.*}} : vector<2xf32>
//         OCP:   %[[ADD:.+]] = llvm.fadd %{{.*}}, %{{.*}} : vector<16xf32>
//         OCP:   llvm.store %[[ADD]], %{{.*}} : vector<16xf32>, !llvm.ptr<7>

// EMULATED-LABEL: llvm.func @ext_fp8_dispatch
//       EMULATED:   llvm.fadd %{{.*}}, %{{.*}} : vector<16xf32>
//       EMULATED:   llvm.store %{{.*}}, %{{.*}} : vector<16xf32>, !llvm.ptr<7>
