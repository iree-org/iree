// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' --split-input-file %s | FileCheck %s

module attributes {hal.device.targets = [#hal.device.target<"llvm-cpu", {executable_targets = [#hal.executable.target<"llvm-cpu", "system-elf-riscv_64", {cpu = "generic-rv64", cpu_features = "+m,+a,+f,+d,+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 64 : index, target_triple = "riscv64"}>]}>]} {
  hal.executable private @main_dispatch_77 {
    hal.executable.variant public @system_elf_riscv_64 target(<"llvm-cpu", "system-elf-riscv_64", {cpu = "generic-rv64", cpu_features = "+m,+a,+f,+d,+v", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 64 : index, target_triple = "riscv64"}>) {
      hal.executable.export public @main_dispatch_77_generic_1x257x257x21 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @main_dispatch_77_generic_1x257x257x21() {
          %c1115136 = arith.constant 1115136 : index
          %c0 = arith.constant 0 : index
          %cst = arith.constant 2.000000e+00 : f32
          %cst_0 = arith.constant 0.000000e+00 : f32
          %cst_1 = arith.constant 1.600000e+01 : f32
          %c1_i32 = arith.constant 1 : i32
          %c32_i32 = arith.constant 32 : i32
          %cst_2 = arith.constant 1.000000e+00 : f32
          %c0_i32 = arith.constant 0 : i32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c1115136) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x33x33x21xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x257x257x21xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 33, 33, 21], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x33x33x21xf32>> -> tensor<1x33x33x21xf32>
          %3 = tensor.empty() : tensor<1x257x257x21xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3 : tensor<1x257x257x21xf32>) {
          ^bb0(%out: f32):
            %5 = linalg.index 1 : index
            %6 = linalg.index 0 : index
            %7 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 257)>(%5, %6)
            %8 = linalg.index 2 : index
            %9 = linalg.index 3 : index
            %10 = arith.index_cast %7 : index to i32
            %11 = arith.index_cast %8 : index to i32
            %12 = arith.uitofp %10 : i32 to f32
            %13 = arith.mulf %12, %cst : f32
            %14 = arith.addf %13, %cst_0 : f32
            %15 = arith.divf %14, %cst_1 : f32
            %16 = math.floor %15 : f32
            %17 = arith.subf %15, %16 : f32
            %18 = arith.fptosi %16 : f32 to i32
            %19 = arith.uitofp %11 : i32 to f32
            %20 = arith.mulf %19, %cst : f32
            %21 = arith.addf %20, %cst_0 : f32
            %22 = arith.divf %21, %cst_1 : f32
            %23 = math.floor %22 : f32
            %24 = arith.subf %22, %23 : f32
            %25 = arith.fptosi %23 : f32 to i32
            %26 = arith.addi %18, %c1_i32 : i32
            %27 = arith.cmpi slt, %18, %c0_i32 : i32
            %28 = arith.select %27, %c0_i32, %18 : i32
            %29 = arith.cmpi sgt, %18, %c32_i32 : i32
            %30 = arith.select %29, %c32_i32, %28 : i32
            %31 = arith.cmpi slt, %26, %c0_i32 : i32
            %32 = arith.select %31, %c0_i32, %26 : i32
            %33 = arith.cmpi sgt, %26, %c32_i32 : i32
            %34 = arith.select %33, %c32_i32, %32 : i32
            %35 = arith.index_cast %30 : i32 to index
            %36 = arith.index_cast %34 : i32 to index
            %37 = arith.addi %25, %c1_i32 : i32
            %38 = arith.cmpi slt, %25, %c0_i32 : i32
            %39 = arith.select %38, %c0_i32, %25 : i32
            %40 = arith.cmpi sgt, %25, %c32_i32 : i32
            %41 = arith.select %40, %c32_i32, %39 : i32
            %42 = arith.cmpi slt, %37, %c0_i32 : i32
            %43 = arith.select %42, %c0_i32, %37 : i32
            %44 = arith.cmpi sgt, %37, %c32_i32 : i32
            %45 = arith.select %44, %c32_i32, %43 : i32
            %46 = arith.index_cast %41 : i32 to index
            %47 = arith.index_cast %45 : i32 to index
            %extracted = tensor.extract %2[%c0, %35, %46, %9] : tensor<1x33x33x21xf32>
            %extracted_3 = tensor.extract %2[%c0, %35, %47, %9] : tensor<1x33x33x21xf32>
            %extracted_4 = tensor.extract %2[%c0, %36, %46, %9] : tensor<1x33x33x21xf32>
            %extracted_5 = tensor.extract %2[%c0, %36, %47, %9] : tensor<1x33x33x21xf32>
            %48 = arith.subf %cst_2, %24 : f32
            %49 = arith.mulf %extracted, %48 : f32
            %50 = arith.mulf %extracted_3, %24 : f32
            %51 = arith.addf %49, %50 : f32
            %52 = arith.mulf %extracted_4, %48 : f32
            %53 = arith.mulf %extracted_5, %24 : f32
            %54 = arith.addf %52, %53 : f32
            %55 = arith.subf %cst_2, %17 : f32
            %56 = arith.mulf %51, %55 : f32
            %57 = arith.mulf %54, %17 : f32
            %58 = arith.addf %56, %57 : f32
            linalg.yield %58 : f32
          } -> tensor<1x257x257x21xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [1, 257, 257, 21], strides = [1, 1, 1, 1] : tensor<1x257x257x21xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x257x257x21xf32>>
          return
        }
      }
    }
  }
}

// CHECK-LABEL: func.func @main_dispatch_77_generic_1x257x257x21
//     CHECK-8: vector.load
