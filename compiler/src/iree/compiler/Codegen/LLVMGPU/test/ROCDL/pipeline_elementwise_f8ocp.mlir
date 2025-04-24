// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1201 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline)))" %s | FileCheck %s --check-prefix=RDNA4
// RUN: not iree-opt --split-input-file --iree-gpu-test-target=gfx942 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline)))" -o /dev/null 2>&1 %s | FileCheck %s --check-prefix=ERRORS
// RUN: not iree-opt --split-input-file --iree-gpu-test-target=gfx908 --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-rocdl-pipeline)))" -o /dev/null 2>&1 %s | FileCheck %s --check-prefix=ERRORS

#map = affine_map<(d0) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @ext_fp8_dispatch {
  hal.executable.variant @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @ext_fp8_dispatch layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
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
    }
  }
}

// ERRORS: F8E5M2 and F8E4M3FN types are not supported on gfx942 (MI-300) or older chipsets; try F8E5M2FNUZ or F8E4M3FNUZ instead.

//   RDNA4-LABEL: hal.executable public @ext_fp8_dispatch
//         RDNA4:   hal.executable.variant public @rocm
// RDNA4-COUNT-8:     rocdl.cvt.pk.f32.fp8 %{{.*}} : vector<2xf32>
// RDNA4-COUNT-8:     rocdl.cvt.pk.f32.bf8 %{{.*}} : vector<2xf32>
//         RDNA4:     %[[ADD:.+]] = llvm.fadd %{{.*}}, %{{.*}} : vector<16xf32>
//         RDNA4:     llvm.store %[[ADD]], %{{.*}} : vector<16xf32>, !llvm.ptr<7>

// -----
