// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-target-test-propagate-target-attrs)))' %s | FileCheck %s

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

// CHECK: llvm.func @simple_dispatch_0_elementwise_2_f32
// CHECK-SAME: target_cpu = "sifive-x390"
// CHECK-SAME: target_features = #llvm.target_features<[{{.*}}"+zvl1024b"{{.*}}]>

// -----

hal.executable private @simple_dispatch_0_cpu_only {
  hal.executable.variant public @embedded_elf_riscv_64 target(<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "sifive-x390", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 256 : i64, target_triple = "riscv64-unknown-unknown-eabi-elf"}>) {
    hal.executable.export public @simple_dispatch_0_elementwise_2_f32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}
    builtin.module {
      llvm.func @simple_dispatch_0_elementwise_2_f32(%arg0: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg2: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}) -> i32 {
        %0 = llvm.mlir.constant(0 : i32) : i32
        llvm.return %0 : i32
      }
    }
  }
}

// CHECK: llvm.func @simple_dispatch_0_elementwise_2_f32
// CHECK-SAME: target_cpu = "sifive-x390"
// CHECK-SAME: target_features = #llvm.target_features<[{{.*}}"+zvl1024b"{{.*}}]>

// -----

hal.executable private @simple_dispatch_0_features_only {
  hal.executable.variant public @embedded_elf_riscv_64 target(<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+d,+zvl1024b,+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 256 : i64, target_triple = "riscv64-unknown-unknown-eabi-elf"}>) {
    hal.executable.export public @simple_dispatch_0_elementwise_2_f32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}
    builtin.module {
      llvm.func @simple_dispatch_0_elementwise_2_f32(%arg0: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg2: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}) -> i32 {
        %0 = llvm.mlir.constant(0 : i32) : i32
        llvm.return %0 : i32
      }
    }
  }
}

// CHECK: llvm.func @simple_dispatch_0_elementwise_2_f32
// CHECK-NOT: target_cpu = "sifive-x390"
// CHECK-SAME: target_features = #llvm.target_features<[{{.*}}"+zvl1024b"{{.*}}]>
