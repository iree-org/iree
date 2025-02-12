// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=mi300x %s | FileCheck %s --check-prefixes=GFX942,MI300X
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=mi300a %s | FileCheck %s --check-prefixes=GFX942,MI300A
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=mi308x %s | FileCheck %s --check-prefixes=GFX942,MI308X
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=mi325x %s | FileCheck %s --check-prefixes=GFX942,MI325X
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=gfx941 --iree-hip-target-features=+sramecc,-xnack %s | FileCheck %s --check-prefix=GFX941
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=gfx942 %s | FileCheck %s --check-prefix=GFX940
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=rx7900xtx %s | FileCheck %s --check-prefix=GFX1100
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=w7900 %s | FileCheck %s --check-prefix=GFX1100
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=v710 %s | FileCheck %s --check-prefix=GFX1101
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=w7700 %s | FileCheck %s --check-prefix=GFX1101

// GFX942: target = #iree_gpu.target<arch = "gfx942",
// GFX942-SAME: wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8,
// GFX942-SAME:         subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32,
// GFX942-SAME:         mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>],
// GFX942-SAME:         subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
// GFX942-SAME:         max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
// GFX942-SAME:         max_workgroup_counts = [2147483647, 2147483647, 2147483647],
// MI300X: chip = <wgp_count = 304, sku = "mi300x">>
// MI300A: chip = <wgp_count = 228, sku = "mi300a">>
// MI308X: chip = <wgp_count = 80, sku = "mi308x">>
// MI325X: chip = <wgp_count = 304, sku = "mi325x">>

// GFX941: target = #iree_gpu.target<arch = "gfx941",
// GFX941-SAME:         features = "+sramecc,-xnack"

// GFX940: target = #iree_gpu.target<arch = "gfx942",
// GFX940-SAME:         mma =  [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>],

// GFX1100: target = #iree_gpu.target<arch = "gfx1100",
// GFX1100-SAME:        mma = [<WMMA_F32_16x16x16_F16>, <WMMA_F16_16x16x16_F16>, <WMMA_I32_16x16x16_I8>, <WMMA_I32_16x16x16_I8>, <WMMA_I32_16x16x16_I8>]
// GFX1100-SAME:        subgroup_size_choices = [32, 64]

// GFX1101: target = #iree_gpu.target<arch = "gfx1101",
// GFX1101-SAME:        mma = [<WMMA_F32_16x16x16_F16>, <WMMA_F16_16x16x16_F16>, <WMMA_I32_16x16x16_I8>, <WMMA_I32_16x16x16_I8>, <WMMA_I32_16x16x16_I8>]
// GFX1101-SAME:        subgroup_size_choices = [32, 64]

stream.executable public @reduce_dispatch {
  stream.executable.export @reduce_dispatch workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @reduce_dispatch() {
      return
    }
  }
}
