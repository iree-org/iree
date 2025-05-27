// RUN: iree-opt --split-input-file --iree-hal-serialize-all-executables --iree-rocm-container-type=hsaco %s | FileCheck %s

// This smoketest verifies that serializing with the --iree-rocm-container-type=
// flag set to raw HSACO ELF files produces an embedded ELF file. This cannot be
// used with the IREE runtime but can be useful when using
// --compile-mode=hal-executable and wanting an HSACO to pass to other tooling
// without needing to unwrap the Flatbuffer. To avoid test churn we just check
// that the `.ELF` magic bytes are present at the start and ignore the contents.

//      CHECK: hal.executable public @executable
//      CHECK: hal.executable.binary public @rocm_hsaco_fb attributes {
// CHECK-SAME:   data = dense<"0x7F454C46
// CHECK-SAME:   format = "rocm-hsaco-fb"

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  abi = "hip",
  iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>
}>
hal.executable public @executable {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target) {
    hal.executable.export public @export ordinal(0) layout(#pipeline_layout)
    builtin.module {
      llvm.func @export() {
        llvm.return
      }
    }
  }
}
