// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-unfuse-fma-pass))" %s | FileCheck %s

func.func @fma_unfused(%a : f32, %b: f32, %c: f32) -> f32 {
    %0 = "llvm.intr.fma"(%a, %b, %c) : (f32, f32, f32) -> f32
    return %0 : f32
}

// CHECK: func.func @fma_unfused(%[[A:.+]]: f32, %[[B:.+]]: f32, %[[C:.+]]: f32)
// CHECK: %[[MUL:.+]] = llvm.fmul %[[A]], %[[B]]
// CHECK: %[[RES:.+]] = llvm.fadd %[[MUL]], %[[C]]
// CHECK: return %[[RES]]
