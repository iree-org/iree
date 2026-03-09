; RUN: not llc -mtriple=spirv64-amd-amdhsa -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: NumElems >= 2 && "SPIR-V OpTypeVector requires at least 2 components"
;
; Bug 6: <1 x T> nested inside aggregate types crashes getOpTypeVector
;
; SPIR-V OpTypeVector requires NumElements >= 2. Upstream PR #180735 added a
; scalarization guard in getOrCreateSPIRVType(), but the fix does not cover
; the createSPIRVType() path reached through findSPIRVType() when processing
; nested types (e.g., arrays of <1 x T> vectors).
;
; The crash path is:
;   getOrCreateSPIRVType([8 x <1 x float>])
;     -> getOrCreateSPIRVPointerType  (for alloca)
;     -> createSPIRVType              (for array element)
;     -> findSPIRVType(<1 x float>)
;     -> restOfCreateSPIRVType
;     -> createSPIRVType              (line 1177: isVectorTy branch)
;     -> getOpTypeVector(1, ...)      (assertion fires)
;
; Top-level <1 x T> is correctly scalarized by getOrCreateSPIRVType (the fix),
; but <1 x T> encountered during recursive type creation is not.
;
; This is what IREE's WMMA codegen produces for gfx1201 (RDNA4). The WMMA
; intrinsic result type is [4 x [4 x [8 x <1 x float>]]], introducing
; <1 x float> as a nested element type throughout the IR.
;
; Impact: Blocks ALL WMMA matmul types (f16, bf16, i8, f8) on the SPIR-V path.

target triple = "spirv64-amd-amdhsa"

define spir_kernel void @vec1_nested(ptr addrspace(1) %out) {
entry:
  %v = alloca [8 x <1 x float>], align 4, addrspace(0)
  %p = getelementptr [8 x <1 x float>], ptr addrspace(0) %v, i32 0, i32 0
  store <1 x float> <float 1.0>, ptr addrspace(0) %p, align 4
  %r = load <1 x float>, ptr addrspace(0) %p, align 4
  %s = extractelement <1 x float> %r, i32 0
  store float %s, ptr addrspace(1) %out, align 4
  ret void
}
