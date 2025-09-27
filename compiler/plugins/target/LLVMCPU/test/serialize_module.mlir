// RUN: mkdir -p %t-dump
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(iree-hal-serialize-all-executables{dump-intermediates-path="%t-dump"}))' %s -o %t.mlir
// RUN: cat %t-dump/*.codegen.ll | FileCheck %s

hal.executable private @simple_dispatch_0 {
  hal.executable.variant public @embedded_elf_riscv_64 target(<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "sifive-x390", cpu_features = "+m,+d,+zvl1024b,+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 256 : i64, target_triple = "riscv64-unknown-unknown-eabi-elf"}>) {
    hal.executable.export public @simple_dispatch_0_elementwise_2_f32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}
    builtin.module {
      llvm.func @simple_dispatch_0_elementwise_2_f32(%arg0: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg2: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}) -> i32 {
        %0 = llvm.mlir.constant(0 : i32) : i32
        llvm.return %0 : i32
      }
    }
  }
}

// CHECK: attributes
// CHECK-SAME: "target-cpu"="sifive-x390"
// CHECK-SAME: "target-features"="{{.*}}zvl1024b
