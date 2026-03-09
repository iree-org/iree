; RUN: llc -mtriple=spirv64-amd-amdhsa -filetype=obj %s -o %t.spv
; RUN: not grep -q 'spirv.llvm_fma' %t.spv
;
; Bug 8: AMD vendor mode passes standard math intrinsics through as external
; function calls instead of lowering them to OpenCL.std extended instructions.
;
; In SPIRVPrepareFunctions::substituteIntrinsicCalls(), the default case
; converts ALL unhandled intrinsics to external function calls when
; Triple::Vendor == AMD. This includes standard LLVM math intrinsics
; (llvm.fma, llvm.sin, llvm.cos, etc.) that the SPIR-V backend can lower
; natively via GlobalISel to OpenCL.std extended instructions.
;
; Without the fix, the resulting SPIR-V contains:
;   OpFunctionCall %... %spirv_llvm_fma_v16f32 ...
; instead of:
;   OpExtInst %... %opencl OpenCL.std fma ...
;
; At JIT time, these unresolved symbols cause the kernel to be silently
; dropped by comgr (no error, all-zero output).
;
; Fix: Restrict AMD pass-through to only amdgcn-specific intrinsics
; (llvm.amdgcn.*, llvm.r600.*). Standard math intrinsics go through the
; SPIR-V backend's normal GlobalISel lowering.
;
; Verification:
;   # With unfixed LLVM:
;   strings out.spv | grep spirv.llvm_fma  # should find external function
;   # With fixed LLVM:
;   strings out.spv | grep OpenCL.std      # should find OpenCL.std (fma via ExtInst)

target triple = "spirv64-amd-amdhsa"

declare <16 x float> @llvm.fma.v16f32(<16 x float>, <16 x float>, <16 x float>)

define spir_kernel void @fma_kernel(ptr addrspace(1) %a, ptr addrspace(1) %b,
                                    ptr addrspace(1) %c, ptr addrspace(1) %out) {
entry:
  %va = load <16 x float>, ptr addrspace(1) %a, align 64
  %vb = load <16 x float>, ptr addrspace(1) %b, align 64
  %vc = load <16 x float>, ptr addrspace(1) %c, align 64
  %result = call <16 x float> @llvm.fma.v16f32(<16 x float> %va, <16 x float> %vb, <16 x float> %vc)
  store <16 x float> %result, ptr addrspace(1) %out, align 64
  ret void
}
