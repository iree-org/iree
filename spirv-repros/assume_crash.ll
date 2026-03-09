; Bug: llvm.assume with operand bundles crashes SPIRVEmitIntrinsics
;
; The SPIR-V backend's SPIRVEmitIntrinsics pass iterates operand bundle
; arguments as if they were regular call arguments, causing an out-of-bounds
; assertion when calling paramHasAttr().
;
; Reproduce:
;   llc -mtriple=spirv64-amd-amdhsa assume_crash.ll -o /dev/null
;
; Expected: No crash. llvm.assume should be ignored or stripped.
; Actual:
;   Assertion `ArgNo < arg_size() && "Param index out of bounds!"' failed.
;   in llvm/lib/IR/Instructions.cpp:414 (CallBase::paramHasAttr)
;   Running pass 'SPIRV emit intrinsics'

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64-amd-amdhsa"

define spir_kernel void @test(ptr addrspace(1) %p) {
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %p, i64 64) ]
  ret void
}

declare void @llvm.assume(i1 noundef)
