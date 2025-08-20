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
// RUN:   --iree-hip-target=gfx950 %s | FileCheck %s --check-prefixes=GFX950
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=mi350x %s | FileCheck %s --check-prefixes=GFX950,MI350X
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=mi355x %s | FileCheck %s --check-prefixes=GFX950,MI355X
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
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=gfx1200 %s | FileCheck %s --check-prefix=GFX1200
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=rx9060xt %s | FileCheck %s --check-prefixes=GFX1200,RX9060XT
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=gfx1201 %s | FileCheck %s --check-prefix=GFX1201
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=rx9070xt %s | FileCheck %s --check-prefixes=GFX1201,RX9070XT
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=rx9070 %s | FileCheck %s --check-prefixes=GFX1201,RX9070
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=r9700 %s | FileCheck %s --check-prefixes=GFX1201,R9700

// GFX942: target_info = #iree_gpu.target<arch = "gfx942",
// GFX942-SAME: wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8,
// GFX942-SAME:         subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32,
// GFX942-SAME:         mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>],
// GFX942-SAME:         subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
// GFX942-SAME:         max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
// GFX942-SAME:         max_workgroup_counts = [2147483647, 2147483647, 2147483647],
// MI300X: chip = <wgp_count = 304, sku = "mi300x", memory_bandwidth_tbps = 5.300000e+00 : f32, perf_tflops = {fp16 = 1.307400e+03 : f32, fp32 = 1.634000e+02 : f32, fp8 = 2.614900e+03 : f32, int8 = 2.614900e+03 : f32}>>
// MI300A: chip = <wgp_count = 228, sku = "mi300a", memory_bandwidth_tbps = 5.300000e+00 : f32, perf_tflops = {fp16 = 980.599975 : f32, fp32 = 1.226000e+02 : f32, fp8 = 1.961200e+03 : f32, int8 = 1.961200e+03 : f32}>>
// MI308X: chip = <wgp_count = 80, sku = "mi308x", memory_bandwidth_tbps = 5.300000e+00 : f32, perf_tflops = {fp16 = 1.884000e+02 : f32, fp32 = 2.900000e+01 : f32, fp8 = 1.768000e+02 : f32, int8 = 1.768000e+02 : f32}>>
// MI325X: chip = <wgp_count = 304, sku = "mi325x", memory_bandwidth_tbps = 5.300000e+00 : f32, perf_tflops = {fp16 = 1.307400e+03 : f32, fp32 = 1.634000e+02 : f32, fp8 = 2.614900e+03 : f32, int8 = 2.614900e+03 : f32}>>

// GFX950: target_info = #iree_gpu.target<arch = "gfx950",
// GFX950-SAME:         mma = [<MFMA_F32_16x16x32_F16>, <MFMA_F32_32x32x16_F16>, <MFMA_F32_16x16x32_BF16>, <MFMA_F32_32x32x16_BF16>, <MFMA_F32_16x16x128_F8E5M2>, <MFMA_F32_16x16x128_F8E5M2_F8E4M3FN>, <MFMA_F32_16x16x128_F8E4M3FN>, <MFMA_F32_16x16x128_F8E4M3FN_F8E5M2>, <MFMA_F32_32x32x64_F8E5M2>, <MFMA_F32_32x32x64_F8E5M2_F8E4M3FN>, <MFMA_F32_32x32x64_F8E4M3FN>, <MFMA_F32_32x32x64_F8E4M3FN_F8E5M2>, <MFMA_I32_16x16x64_I8>, <MFMA_I32_32x32x32_I8>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2>, <MFMA_F32_16x16x32_F8E5M2_F8E4M3FN>, <MFMA_F32_16x16x32_F8E4M3FN>, <MFMA_F32_16x16x32_F8E4M3FN_F8E5M2>, <MFMA_F32_32x32x16_F8E5M2>, <MFMA_F32_32x32x16_F8E5M2_F8E4M3FN>, <MFMA_F32_32x32x16_F8E4M3FN>, <MFMA_F32_32x32x16_F8E4M3FN_F8E5M2>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>],
// GFX950-SAME:         scaled_mma = [<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E8M0FNU, rhs_elem_type = f8E8M0FNU, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E5M2, rhs_elem_type = f8E5M2, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E5M2FNUZ, rhs_elem_type = f8E5M2FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FNUZ, rhs_elem_type = f8E4M3FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E8M0FNU, rhs_elem_type = f8E8M0FNU, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E5M2, rhs_elem_type = f8E5M2, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E5M2FNUZ, rhs_elem_type = f8E5M2FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E4M3FNUZ, rhs_elem_type = f8E4M3FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>],
// GFX950-SAME:         subgroup_size_choices = [64],
// GFX950-SAME:         max_workgroup_memory_bytes = 163840,
// MI350X: chip = <wgp_count = 256, sku = "mi350x", memory_bandwidth_tbps = 8.000000e+00 : f32, perf_tflops = {fp16 = 2.300000e+03 : f32, fp32 = 1.442000e+02 : f32, fp4 = 9.200000e+03 : f32, fp6 = 9.200000e+03 : f32, fp8 = 4.600000e+03 : f32, int8 = 4.600000e+03 : f32}>>
// MI355X: chip = <wgp_count = 256, sku = "mi355x", memory_bandwidth_tbps = 8.000000e+00 : f32, perf_tflops = {fp16 = 2.500000e+03 : f32, fp32 = 1.573000e+02 : f32, fp4 = 1.000000e+04 : f32, fp6 = 1.000000e+04 : f32, fp8 = 5.000000e+03 : f32, int8 = 5.000000e+03 : f32}>>

// GFX1100: target_info = #iree_gpu.target<arch = "gfx1100",
// GFX1100-SAME:        mma = [<WMMAR3_F32_16x16x16_F16>, <WMMAR3_F16_16x16x16_F16>, <WMMAR3_F32_16x16x16_BF16>, <WMMAR3_BF16_16x16x16_BF16>, <WMMAR3_I32_16x16x16_I8>]
// GFX1100-SAME:        subgroup_size_choices = [32, 64]

// GFX1101: target_info = #iree_gpu.target<arch = "gfx1101",
// GFX1101-SAME:        mma = [<WMMAR3_F32_16x16x16_F16>, <WMMAR3_F16_16x16x16_F16>, <WMMAR3_F32_16x16x16_BF16>, <WMMAR3_BF16_16x16x16_BF16>, <WMMAR3_I32_16x16x16_I8>]
// GFX1101-SAME:        subgroup_size_choices = [32, 64]

// GFX1200: target_info = #iree_gpu.target<arch = "gfx1200",
// GFX1200-SAME:        mma = [<WMMAR4_F32_16x16x16_F16>, <WMMAR4_F16_16x16x16_F16>, <WMMAR4_F32_16x16x16_BF16>, <WMMAR4_BF16_16x16x16_BF16>, <WMMAR4_F32_16x16x16_F8E5M2>, <WMMAR4_F32_16x16x16_F8E5M2_F8E4M3FN>, <WMMAR4_F32_16x16x16_F8E4M3FN>, <WMMAR4_F32_16x16x16_F8E4M3FN_F8E5M2>,  <WMMAR4_I32_16x16x16_I8>]
// GFX1200-SAME:        subgroup_size_choices = [32, 64]
//
// RX9060XT: chip = <wgp_count = 16, sku = "rx9060xt", memory_bandwidth_tbps = 3.200000e-01 : f32, perf_tflops = {fp16 = 1.030000e+02 : f32, fp32 = 2.560000e+01 : f32, fp8 = 2.050000e+02 : f32, int8 = 2.050000e+02 : f32}>>

// GFX1201: target_info = #iree_gpu.target<arch = "gfx1201",
// GFX1201-SAME:        mma = [<WMMAR4_F32_16x16x16_F16>, <WMMAR4_F16_16x16x16_F16>, <WMMAR4_F32_16x16x16_BF16>, <WMMAR4_BF16_16x16x16_BF16>, <WMMAR4_F32_16x16x16_F8E5M2>, <WMMAR4_F32_16x16x16_F8E5M2_F8E4M3FN>, <WMMAR4_F32_16x16x16_F8E4M3FN>, <WMMAR4_F32_16x16x16_F8E4M3FN_F8E5M2>,  <WMMAR4_I32_16x16x16_I8>]
// GFX1201-SAME:        subgroup_size_choices = [32, 64]
//
// RX9070XT: chip = <wgp_count = 32, sku = "rx9070xt", memory_bandwidth_tbps = 6.400000e-01 : f32, perf_tflops = {fp16 = 1.950000e+02 : f32, fp32 = 4.870000e+01 : f32, fp8 = 3.890000e+02 : f32, int8 = 3.890000e+02 : f32}>>
// RX9070:   chip = <wgp_count = 28, sku = "rx9070", memory_bandwidth_tbps = 6.400000e-01 : f32, perf_tflops = {fp16 = 1.450000e+02 : f32, fp32 = 3.610000e+01 : f32, fp8 = 2.890000e+02 : f32, int8 = 2.890000e+02 : f32}>>
// R9700:    chip = <wgp_count = 32, sku = "r9700", memory_bandwidth_tbps = 6.400000e-01 : f32, perf_tflops = {fp16 = 1.910000e+02 : f32, fp32 = 4.780000e+01 : f32, fp8 = 3.830000e+02 : f32, int8 = 3.830000e+02 : f32}>>

stream.executable public @reduce_dispatch {
  stream.executable.export @reduce_dispatch workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root(%arg0)
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @reduce_dispatch() {
      return
    }
  }
}
