// Both paths through inparallel should produce similar output; i.e. running either:
//   * X -> inparallel -> bufferize -> inparallelToHAL (i.e. distribution on buffers)
//   * X -> inparallel -> inparallelToHAL (i.e. distribution on tensors) -> bufferize (i.e. distribution on buffers)
// should be consistent and provide a "no abstraction gap" lowering path.
// To this end, we use the same --check-prefix=INPARALLEL for both cases.
// 
// RUN: iree-opt %s  -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))' --iree-codegen-use-linalg-transform-interp --linalg-transform-file-name=%p/linalg_transform_inparallel_buffers_spec.mlir -split-input-file | FileCheck %s --check-prefix=INPARALLEL
// RUN: iree-opt %s  -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))' --iree-codegen-use-linalg-transform-interp --linalg-transform-file-name=%p/linalg_transform_inparallel_tensors_spec.mlir -split-input-file | FileCheck %s --check-prefix=INPARALLEL
// TODO: Remove spurious alloca + copy due to dispatch region creation, bufferization, codegen ordering.
// RUN: iree-opt %s  -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))' --iree-codegen-use-linalg-transform-interp --linalg-transform-file-name=%p/linalg_transform_inparallel_tensors_spec.mlir -split-input-file | FileCheck %s --check-prefix=INPARALLEL-TENSORS

#device_target_cpu = #hal.device.target<"cpu", {executable_targets = [#hal.executable.target<"llvm", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>]}>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>, #hal.descriptor_set.binding<1, storage_buffer>, #hal.descriptor_set.binding<2, storage_buffer>]>]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>

hal.executable private @matmul_tensors {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.entry_point public @matmul_tensors ordinal(0) layout(#executable_layout)
    builtin.module {
      func.func @matmul_tensors() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:250x500xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:500x1020xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:250x1020xf32>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [250, 500], strides = [1, 1] : !flow.dispatch.tensor<readonly:250x500xf32> -> tensor<250x500xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [500, 1020], strides = [1, 1] : !flow.dispatch.tensor<readonly:500x1020xf32> -> tensor<500x1020xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [250, 1020], strides = [1, 1] : !flow.dispatch.tensor<readwrite:250x1020xf32> -> tensor<250x1020xf32>

        //     INPARALLEL: hal.executable.entry_point public @matmul_tensors ordinal(0) layout(#executable_layout) {
        //     INPARALLEL:   %[[C1:.*]] = arith.constant 1 : index
        // INPARALLEL-DAG: %[[C125:.*]] = arith.constant 125 : index
        // INPARALLEL-DAG: %[[C255:.*]] = arith.constant 255 : index
        //     INPARALLEL: hal.return %[[C125]], %[[C255]], %[[C1]] : index, index, index
        
        //      CHECK: memref.assume_alignment %{{.*}}, 64 : memref<250x1020xf32>
        // CHECK-NEXT: linalg.matmul{{.*}}ins(%{{.*}} : memref<250x500xf32>, memref<500x1020xf32>) outs(%{{.*}} : memref<250x1020xf32>)
        // CHECK-NEXT: return

        //     INPARALLEL: %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
        //     INPARALLEL: %[[offx:.*]] = affine.apply {{.*}}()[%[[workgroup_id_x]]]
        //     INPARALLEL:  %[[szx:.*]] = affine.min {{.*}}()[%[[workgroup_id_x]]]
        //     INPARALLEL: memref.subview %{{.*}}[%[[offx]], 0] [%[[szx]], 1020] [1, 1] : memref<250x1020xf32> to memref<?x1020xf32, {{.*}}>
        //     INPARALLEL: %[[workgroup_id_y:.*]] = hal.interface.workgroup.id[1] : index
        //     INPARALLEL: %[[offy:.*]] = affine.apply {{.*}}()[%[[workgroup_id_y]]]
        //     INPARALLEL:  %[[szy:.*]] = affine.min {{.*}}()[%[[workgroup_id_y]]]
        //     INPARALLEL: memref.subview %{{.*}}[0, %[[offy]]] [%[[szx]], %[[szy]]] [1, 1] : memref<?x1020xf32, {{.*}}> to memref<?x?xf32, {{.*}}>
        //     INPARALLEL: memref.subview %{{.*}}[%[[offx]], 0] [%[[szx]], 500] [1, 1] : memref<250x500xf32> to memref<?x500xf32, {{.*}}>
        //     INPARALLEL: memref.subview %{{.*}}[0, %[[offy]]] [500, %[[szy]]] [1, 1] : memref<500x1020xf32> to memref<500x?xf32, {{.*}}>
        //     INPARALLEL: linalg.matmul{{.*}}ins({{.*}} : memref<?x500xf32, {{.*}}>, memref<500x?xf32, {{.*}}>) outs(%{{.*}} : memref<?x?xf32, {{.*}}>)
        %6 = linalg.matmul
          ins(%3, %4 : tensor<250x500xf32>, tensor<500x1020xf32>) outs(%5 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [250, 1020], strides = [1, 1] : tensor<250x1020xf32> -> !flow.dispatch.tensor<readwrite:250x1020xf32>
        return
      }
    }
  }
}

// -----

#device_target_cpu = #hal.device.target<"cpu", {executable_targets = [#hal.executable.target<"llvm", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>]}>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>, #hal.descriptor_set.binding<1, storage_buffer>, #hal.descriptor_set.binding<2, storage_buffer>]>]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>

hal.executable private @matmul_tensors {
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
    hal.executable.entry_point public @matmul_tensors ordinal(0) layout(#executable_layout)
    builtin.module {
      func.func @matmul_tensors() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:250x500xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:500x1020xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:250x1020xf32>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [250, 500], strides = [1, 1] : !flow.dispatch.tensor<readonly:250x500xf32> -> tensor<250x500xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [500, 1020], strides = [1, 1] : !flow.dispatch.tensor<readonly:500x1020xf32> -> tensor<500x1020xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [250, 1020], strides = [1, 1] : !flow.dispatch.tensor<readwrite:250x1020xf32> -> tensor<250x1020xf32>

        //     INPARALLEL: hal.executable.entry_point public @matmul_tensors ordinal(0) layout(#executable_layout) {
        //     INPARALLEL:   %[[C1:.*]] = arith.constant 1 : index
        // INPARALLEL-DAG: %[[C125:.*]] = arith.constant 125 : index
        // INPARALLEL-DAG: %[[C255:.*]] = arith.constant 255 : index
        //     INPARALLEL: hal.return %[[C125]], %[[C255]], %[[C1]] : index, index, index
        
        //      CHECK: memref.assume_alignment %{{.*}}, 64 : memref<250x1020xf32>
        // CHECK-NEXT: linalg.fill{{.*}}ins(%{{.*}} : f32) outs(%{{.*}} : memref<250x1020xf32>)
        // CHECK-NEXT: linalg.matmul{{.*}}ins(%{{.*}} : memref<250x500xf32>, memref<500x1020xf32>) outs(%{{.*}} : memref<250x1020xf32>)
        // CHECK-NEXT: return
        %6 = linalg.init_tensor [250, 1020] : tensor<250x1020xf32>
        %cst = arith.constant 0.0 : f32
        %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<250x1020xf32>) -> tensor<250x1020xf32>

        // TODO: Remove spurious alloca due to dispatch region creation, bufferization, codegen ordering.
        // This is also empirically related to interface.binding.subspan having a MemAlloc effect.
        //     INPARALLEL-TENSORS: alloca
        
        // Fill is not fused here, but it was already in the dispatch region, so we get a race.
        //     INPARALLEL: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<250x1020xf32>)
        //     INPARALLEL: %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
        //     INPARALLEL: %[[offx:.*]] = affine.apply {{.*}}()[%[[workgroup_id_x]]]
        //     INPARALLEL:  %[[szx:.*]] = affine.min {{.*}}()[%[[workgroup_id_x]]]
        //     INPARALLEL: memref.subview %{{.*}}[%[[offx]], 0] [%[[szx]], 1020] [1, 1] : memref<250x1020xf32> to memref<?x1020xf32, {{.*}}>
        //     INPARALLEL: %[[workgroup_id_y:.*]] = hal.interface.workgroup.id[1] : index
        //     INPARALLEL: %[[offy:.*]] = affine.apply {{.*}}()[%[[workgroup_id_y]]]
        //     INPARALLEL:  %[[szy:.*]] = affine.min {{.*}}()[%[[workgroup_id_y]]]
        //     INPARALLEL: memref.subview %{{.*}}[0, %[[offy]]] [%[[szx]], %[[szy]]] [1, 1] : memref<?x1020xf32, {{.*}}> to memref<?x?xf32, {{.*}}>
        //     INPARALLEL: memref.subview %{{.*}}[%[[offx]], 0] [%[[szx]], 500] [1, 1] : memref<250x500xf32> to memref<?x500xf32, {{.*}}>
        //     INPARALLEL: memref.subview %{{.*}}[0, %[[offy]]] [500, %[[szy]]] [1, 1] : memref<500x1020xf32> to memref<500x?xf32, {{.*}}>
        //     INPARALLEL: linalg.matmul{{.*}}ins({{.*}} : memref<?x500xf32, {{.*}}>, memref<500x?xf32, {{.*}}>) outs(%{{.*}} : memref<?x?xf32, {{.*}}>)

        // TODO: Remove spurious copy due to dispatch region creation, bufferization, codegen ordering.
        // This is also empirically related to interface.binding.subspan having a MemAlloc effect.
        //     INPARALLEL-TENSORS: subview
        //     INPARALLEL-TENSORS: linalg.generic
        %8 = linalg.matmul
          ins(%3, %4 : tensor<250x500xf32>, tensor<500x1020xf32>) outs(%7 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
        flow.dispatch.tensor.store %8, %2, offsets = [0, 0], sizes = [250, 1020], strides = [1, 1] : tensor<250x1020xf32> -> !flow.dispatch.tensor<readwrite:250x1020xf32>
        return
      }
    }
  }
}
