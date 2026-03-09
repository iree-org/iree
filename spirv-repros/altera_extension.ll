; Bug: SPIR-V backend spuriously emits SPV_ALTERA_arbitrary_precision_integers
; for code that only uses standard integer types (i32, i64)
;
; The extension is triggered by urem in a loop. The backend likely introduces
; non-standard integer widths during strength reduction of urem-by-constant.
; AMD HIP SPIR-V JIT rejects this extension:
;   "Invalid SPIR-V module: input SPIR-V module uses unknown extension
;    'SPV_ALTERA_arbitrary_precision_integers'"
;
; Reproduce:
;   llc -mtriple=spirv64-amd-amdhsa -filetype=obj altera_extension.ll -o altera.spv
;   strings altera.spv | grep SPV_ALTERA
;
; Expected: No SPV_ALTERA extension for standard integer widths.
; Actual: SPIR-V output contains OpExtension "SPV_ALTERA_arbitrary_precision_integers"

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define spir_kernel void @test(i32 %0) {
  br label %2

2:
  %3 = phi i32 [ %8, %4 ], [ %0, %1 ]
  br i1 false, label %4, label %9

4:
  %5 = urem i32 %3, 8
  %6 = zext i32 %5 to i64
  %7 = or i64 0, %6
  %8 = add i32 %3, 1
  br label %2

9:
  ret void
}
