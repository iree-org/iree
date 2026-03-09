; RUN: not llc -mtriple=spirv64-amd-amdhsa -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: PHI node operands are not the same type as the result!
;
; Bug 7: [1 x <N x T>] array types cause PHI type mismatch in SPIR-V output
;
; When LLVM IR contains `phi [1 x <16 x float>]`, the SPIR-V backend's
; SPIRVEmitIntrinsics pass rewrites the insertvalue/extractvalue + PHI chain
; in a way that produces inconsistent types: the PHI result type gets collapsed
; but the zeroinitializer operand keeps the original array type.
;
; With the Bug 5 fix (isa<PHINode> in replaceMemInstrUses), the verifier
; catches this at compile time:
;
;   PHI node operands are not the same type as the result!
;     %acc = phi [1 x <16 x float>] [ %1, %entry ], [ %4, %loop ]
;
; Without the Bug 5 fix, this file hits UNREACHABLE "illegal aggregate
; intrinsic user" instead (Bug 5 masks Bug 7).
;
; This pattern is common in IREE's f32 matmul codegen (tile accumulator loops).
; IREE workaround: unwrapSingleElementArrayTypes() in ROCMTarget.cpp rewrites
; phi [1 x T] -> phi T before SPIR-V emission.

target triple = "spirv64-amd-amdhsa"

define spir_kernel void @array1_phi(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi [1 x <16 x float>] [ zeroinitializer, %entry ], [ %acc.next, %loop ]

  ; Load a vector and accumulate into the [1 x <16 x float>] wrapper
  %ptr = getelementptr float, ptr addrspace(1) %in, i32 %i
  %val = load float, ptr addrspace(1) %ptr, align 4
  %splat = insertelement <16 x float> poison, float %val, i32 0
  %bcast = shufflevector <16 x float> %splat, <16 x float> poison,
                         <16 x i32> zeroinitializer
  %old = extractvalue [1 x <16 x float>] %acc, 0
  %sum = fadd <16 x float> %old, %bcast
  %acc.next = insertvalue [1 x <16 x float>] undef, <16 x float> %sum, 0

  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  %result = extractvalue [1 x <16 x float>] %acc, 0
  store <16 x float> %result, ptr addrspace(1) %out, align 64
  ret void
}
