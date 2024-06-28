// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetBackends=rocm},iree-hal-transformation-pipeline{serialize-executables=false})' --iree-rocm-target-chip=mi300x %s | FileCheck %s --check-prefix=GFX942
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetBackends=rocm},iree-hal-transformation-pipeline{serialize-executables=false})' --iree-rocm-target-chip=gfx940 %s | FileCheck %s --check-prefix=GFX940
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetBackends=rocm},iree-hal-transformation-pipeline{serialize-executables=false})' --iree-rocm-target-chip=rx7900xtx %s | FileCheck %s --check-prefix=GFX1100
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetBackends=rocm},iree-hal-transformation-pipeline{serialize-executables=false})' --iree-rocm-target-chip=gfx941 --iree-rocm-target-features=+sramecc,-xnack %s | FileCheck %s --check-prefix=GFX941

// GFX942: target = #iree_gpu.target<arch = "gfx942",
// GFX942-SAME: wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8,
// GFX942-SAME:         subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32,
// GFX942-SAME:         mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>, <MFMA_I8_16x16x32_I32>, <MFMA_I8_32x32x16_I32>],
// GFX942-SAME:         subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
// GFX942-SAME:         max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>,
// GFX942-SAME: chip = <wgp_count = 304>>

// GFX940: target = #iree_gpu.target<arch = "gfx940",
// GFX940-SAME:         mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>, <MFMA_I8_16x16x32_I32>, <MFMA_I8_32x32x16_I32>],

// GFX1100: target = #iree_gpu.target<arch = "gfx1100",
// GFX1100-SAME:        mma = [<WMMA_F16_16x16x16_F32>, <WMMA_F16_16x16x16_F16>]
// GFX1100-SAME:        subgroup_size_choices = [32, 64]

// GFX941: target = #iree_gpu.target<arch = "gfx941",
// GFX941-SAME:         features = "+sramecc,-xnack"


stream.executable public @reduce_dispatch {
  stream.executable.export @reduce_dispatch workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @reduce_dispatch(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<16xf32>>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<f32>>
      %0 = tensor.empty() : tensor<f32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[0], sizes=[16], strides=[1] : !flow.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%1 : tensor<16xf32>) outs(%0 : tensor<f32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %4 = arith.addf %arg2, %arg3 : f32
        linalg.yield %4 : f32
      } -> tensor<f32>
      flow.dispatch.tensor.store %3, %arg1, offsets=[], sizes=[], strides=[] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
      return
    }
  }
}
