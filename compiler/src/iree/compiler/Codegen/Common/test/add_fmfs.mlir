// RUN: iree-opt -iree-codegen-add-fast-math-flags --split-input-file %s | FileCheck %s

// LABEL: llvm.func @fmfs
llvm.func @fmfs() -> f32 {
  %c3 = llvm.mlir.constant(3.000000e+00 : f32) : f32
  %c6 = llvm.mlir.constant(6.000000e+00 : f32) : f32
  %mul = llvm.fmul %c3, %c3 : f32
  %add = llvm.fadd %c3, %c6 : f32
  llvm.return %add : f32
}

// CHECK: llvm.fmul %{{.*}}, %{{.*}}  {fastmathFlags = #llvm.fastmath<contract>} : f32
// CHECK: llvm.fadd %{{.*}}, %{{.*}}  {fastmathFlags = #llvm.fastmath<contract>} : f32
