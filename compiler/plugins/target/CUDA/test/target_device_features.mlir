// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=cuda},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-cuda-target=sm_89 %s | FileCheck %s --check-prefix=SM89
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=cuda},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-cuda-target=ada %s | FileCheck %s --check-prefix=SM89
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=cuda},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-cuda-target=rtx4090 %s | FileCheck %s --check-prefixes=SM89,RTX4090
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=cuda},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-cuda-target=sm_89 --iree-cuda-target-features=+ptx80 %s | FileCheck %s --check-prefix=PTX80
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=cuda},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-cuda-target=sm_120 --iree-cuda-target-features=+ptx87 %s | FileCheck %s --check-prefix=SM120
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=cuda},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-cuda-target=sm_121 --iree-cuda-target-features=+ptx88 %s | FileCheck %s --check-prefix=SM121

// SM89: target_info = #iree_gpu.target<arch = "sm_89", features = "+ptx78",
// SM89-SAME: wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8,
// SM89-SAME:         subgroup = shuffle|arithmetic, dot = dp4xi8toi32,
// SM89-SAME:         mma = [<NV_MMA_SYNC_F32_16x8x16_F16>, <NV_MMA_SYNC_F16_16x8x16_F16>, <NV_MMA_SYNC_F32_16x8x16_BF16>, <NV_WMMA_F32_16x16x16_F16>, <NV_WMMA_F16_16x16x16_F16>],
// SM89-SAME:         subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024],
// SM89-SAME:         max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 101376,
// SM89-SAME:         max_workgroup_counts = [2147483647, 65535, 65535]>
// RTX4090: chip = <wgp_count = 128>

// PTX80: target_info = #iree_gpu.target<arch = "sm_89", features = "+ptx80",
// SM120: target_info = #iree_gpu.target<arch = "sm_120", features = "+ptx87",
// SM120-SAME: wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8,
// SM120-SAME:         subgroup = shuffle|arithmetic, dot = dp4xi8toi32,
// SM120-SAME:         mma = [<NV_MMA_SYNC_F32_16x8x16_F16>, <NV_MMA_SYNC_F16_16x8x16_F16>, <NV_MMA_SYNC_F32_16x8x16_BF16>, <NV_WMMA_F32_16x16x16_F16>, <NV_WMMA_F16_16x16x16_F16>],
// SM120-SAME:         subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 64],
// SM120-SAME:         max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 101376,
// SM120-SAME:         max_workgroup_counts = [2147483647, 65535, 65535]>
// SM121: target_info = #iree_gpu.target<arch = "sm_121", features = "+ptx88",

stream.executable public @target_device_features {
  stream.executable.export @target_device_features workgroups()
      -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @target_device_features() {
      return
    }
  }
}
