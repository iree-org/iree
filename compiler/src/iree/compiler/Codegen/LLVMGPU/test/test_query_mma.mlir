// RUN: iree-opt --split-input-file --iree-test-llvmgpu-query-mma %s | FileCheck %s

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
{abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "",
wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8,
subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32,
mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>,
<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E4M3FNUZ>,
<MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>],
subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128,
simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none", waves_per_eu = 2 : i64}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
module {
  hal.executable private @main {
    hal.executable.variant public @main target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @entry_point layout(#pipeline_layout)
      builtin.module {
        func.func @fn() {
          return
        }
      }
    }
  }
}

// CHECK: Printing MMA Collection Before querying supported mma instrinsic instructions, size: 0
// CHECK: Printing MMA Collection After querying supported mma instrinsic instructions, size: 9
// CHECK: MFMA_F32_16x16x4_F32
// CHECK-SAME: MFMA_F32_16x16x16_F16
// CHECK-SAME: MFMA_F32_32x32x8_F16
// CHECK-SAME: MFMA_F32_16x16x16_BF16
// CHECK-SAME: MFMA_F32_32x32x8_BF16
// CHECK-SAME: MFMA_F32_16x16x32_F8E4M3FNUZ
// CHECK-SAME: MFMA_F32_16x16x32_F8E5M2FNUZ
// CHECK-SAME: MFMA_I32_16x16x32_I8
// CHECK-SAME: MFMA_I32_32x32x16_I8
// CHECK-LABEL: func.func @fn

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
module {
  hal.executable private @main {
    hal.executable.variant public @main target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @entry_point layout(#pipeline_layout)
      builtin.module {
        func.func @fn_empty() {
          return
        }
      }
    }
  }
}

// CHECK: Printing MMA Collection Before querying supported mma instrinsic instructions, size: 0
// CHECK: Printing MMA Collection After querying supported mma instrinsic instructions, size: 0
// CHECK-LABEL: func.func @fn_empty
