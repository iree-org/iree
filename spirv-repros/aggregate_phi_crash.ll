; Reproducer: SPIRVEmitIntrinsics crashes on aggregate insertvalue with PHI user.
;
; The pass rewrites `insertvalue` on aggregate types to `@llvm.spv.insertv`
; and calls `replaceMemInstrUses()` to update users. That function handles
; AssignType intrinsics, memory instructions, ReturnInst, and CallInst — but
; not PHINode. When the insertvalue result feeds back into a loop PHI, the
; UNREACHABLE at SPIRVEmitIntrinsics.cpp:1489 fires.
;
; RUN: llc -mtriple=spirv64 %s -o /dev/null
;
; LLVM ERROR:
;   illegal aggregate intrinsic user
;   UNREACHABLE executed at SPIRVEmitIntrinsics.cpp:1489
;
; Affects: any loop that accumulates into an aggregate type ([1 x float], etc.)
; via insertvalue/extractvalue with a PHI back-edge. Common in IREE's codegen
; for reductions (softmax, attention, dot_general, etc.) — 8 IREE e2e tests.

define void @aggregate_phi_crash(ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i32 [ %next, %loop ], [ 0, %entry ]
  %agg = phi [1 x float] [ %agg.new, %loop ], [ zeroinitializer, %entry ]
  %prev = extractvalue [1 x float] %agg, 0
  %sum = fadd float %prev, 1.0
  %agg.new = insertvalue [1 x float] poison, float %sum, 0
  %next = add i32 %i, 1
  %cond = icmp slt i32 %next, 64
  br i1 %cond, label %loop, label %exit

exit:
  %final = extractvalue [1 x float] %agg, 0
  store float %final, ptr addrspace(1) %out, align 4
  ret void
}
