; Bug: InstructionSelect crashes with "incompatible result and operand types
; in a bitcast" when lowering llvm.masked.load of <8 x i8>
;
; The SPIR-V backend's instruction selector internally generates a bitcast
; during masked load lowering that is incompatible in SPIR-V's type system.
;
; Reproduce:
;   llc -mtriple=spirv64-amd-amdhsa bitcast_crash.ll -o /dev/null
;
; Expected: No crash. Masked loads should be lowered correctly.
; Actual:
;   LLVM ERROR: incompatible result and operand types in a bitcast
;   Running pass 'InstructionSelect' on function '@test'

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define spir_kernel void @test(<8 x i1> %mask) {
  %v = call <8 x i8> @llvm.masked.load.v8i8.p1(ptr addrspace(1) null, <8 x i1> %mask, <8 x i8> zeroinitializer)
  store <8 x i8> %v, ptr addrspace(3) null, align 1
  ret void
}

declare <8 x i8> @llvm.masked.load.v8i8.p1(ptr addrspace(1), <8 x i1>, <8 x i8>)
