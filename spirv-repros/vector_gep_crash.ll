; Bug: SPIRVEmitIntrinsics creates llvm.spv.gep with incorrect return type
; for vector-indexed GEPs
;
; When a getelementptr uses a vector index (e.g., <1 x i64>), the result type
; is a vector of pointers (<1 x ptr addrspace(1)>). SPIRVEmitIntrinsics rewrites
; this to an llvm.spv.gep intrinsic but assigns a scalar pointer return type
; (ptr addrspace(4)) instead of the expected vector type, causing the module
; verifier to fail.
;
; Reproduce:
;   llc -mtriple=spirv64-amd-amdhsa vector_gep_crash.ll -o /dev/null
;
; Expected: No crash. Vector GEPs should be lowered correctly.
; Actual:
;   Intrinsic has incorrect return type!
;   ptr addrspace(4) @llvm.spv.gep.v1p1.p1
;   LLVM ERROR: Broken function found, compilation aborted!
;
; Note: This repro requires llvm.assume to be stripped first (separate bug).
;       If your build still has the llvm.assume bug, you'll see that crash
;       instead. Use: sed -i '/call void @llvm.assume/d' on any file with
;       llvm.assume calls before testing this bug.

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define spir_kernel void @test(ptr addrspace(1) %p) {
  %gep = getelementptr i8, ptr addrspace(1) %p, <1 x i64> zeroinitializer
  ret void
}
