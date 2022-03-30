// RUN: iree-opt %s  -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))' --iree-codegen-use-linalg-transform-interp --linalg-transform-file-name=%p/linalg_transform_spec.mlir -allow-unregistered-dialect | FileCheck %s

#device_target_cpu = #hal.device.target<"cpu", {executable_targets = [#hal.executable.target<"llvm", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>]}>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>, #hal.descriptor_set.binding<1, storage_buffer>, #hal.descriptor_set.binding<2, storage_buffer>]>]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>

hal.executable private @pad_matmul_static_dispatch_0 {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.entry_point public @pad_matmul_static_dispatch_0 ordinal(0) layout(#executable_layout)
    builtin.module {
      func.func @pad_matmul_static_dispatch_0() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:250x500xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:500x1020xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:250x1020xf32>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [250, 500], strides = [1, 1] : !flow.dispatch.tensor<readonly:250x500xf32> -> tensor<250x500xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [500, 1020], strides = [1, 1] : !flow.dispatch.tensor<readonly:500x1020xf32> -> tensor<500x1020xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [250, 1020], strides = [1, 1] : !flow.dispatch.tensor<readwrite:250x1020xf32> -> tensor<250x1020xf32>
        
        //      CHECK: "iree_linalg_ext.hal.executable.entry_point"()
        //  CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
        //  CHECK-DAG:   %[[C125:.*]] = arith.constant 125 : index
        //      CHECK:   iree_linalg_ext.hal.return %[[C125]], %[[C1]], %[[C1]] : index, index, index
        //      CHECK: iree_linalg_ext.hal.interface.workgroup.id[0] : index
        //      CHECK: iree_linalg_ext.hal.interface.workgroup.count[0] : index
        //      CHECK: subview
        //      CHECK: subview
        //      CHECK: matmul{{.*}}ins{{.*}}outs
        //      CHECK: return
        %6 = linalg.matmul ins(%3, %4 : tensor<250x500xf32>, tensor<500x1020xf32>) outs(%5 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [250, 1020], strides = [1, 1] : tensor<250x1020xf32> -> !flow.dispatch.tensor<readwrite:250x1020xf32>
        return
      }
    }
  }
}
