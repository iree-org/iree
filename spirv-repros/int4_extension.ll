; Issue: SPIR-V backend emits SPV_INTEL_int4 for i4 integer types
;
; AMD HIP SPIR-V JIT rejects this extension:
;   "Invalid SPIR-V module: input SPIR-V module uses unknown extension
;    'SPV_INTEL_int4'"
;
; This is arguably correct behavior from the SPIR-V backend — i4 requires
; an extension in SPIR-V. The fix should be on the producer side (IREE):
; widen i4 to i8 before emitting SPIR-V, or on the HIP JIT side to
; support i4 natively.
;
; Reproduce:
;   llc -mtriple=spirv64-amd-amdhsa -filetype=obj int4_extension.ll -o int4.spv
;   strings int4.spv | grep SPV_INTEL_int4

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64-amd-amdhsa"

define spir_kernel void @test(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %val = load i4, ptr addrspace(1) %in, align 1
  %ext = zext i4 %val to i32
  store i32 %ext, ptr addrspace(1) %out, align 4
  ret void
}
