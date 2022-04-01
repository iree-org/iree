// RUN: iree-opt %s  -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))' -iree-set-num-workgroups-from-linalg-ext | FileCheck %s

hal.executable @_matmul_static_dispatch_0 {
hal.executable.variant public @embedded_elf_x86_64, target = <"llvm", "embedded-elf-x86_64", {cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}> {
  // CHECK: hal.executable.entry_point public @_matmul_static_dispatch_0 
  // CHECK:   %[[C1:.*]] = arith.constant 1 : index
  // CHECK:   %[[C3:.*]] = arith.constant 3 : index
  // CHECK:   hal.return %[[C3]], %[[C1]], %[[C1]] : index, index, index
  // CHECK: }
  hal.executable.entry_point public @_matmul_static_dispatch_0 ordinal(0) layout(#hal.executable.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>]>) {translation_info = #iree_codegen.translation_info<LinalgTransformInterpCodegen>}
  builtin.module {
    // CHECK: func @_matmul_static_dispatch_0
    func @_matmul_static_dispatch_0() {
      %cst = arith.constant dense<[[1.500000e+01, 1.400000e+01, 1.300000e+01], [1.200000e+01, 1.100000e+01, 1.000000e+01], [9.000000e+00, 8.000000e+00, 7.000000e+00], [6.000000e+00, 5.000000e+00, 4.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<5x3xf32>
      %cst_0 = arith.constant dense<[[1.500000e+01, 1.400000e+01, 1.300000e+01, 1.200000e+01, 1.100000e+01], [1.000000e+01, 9.000000e+00, 8.000000e+00, 7.000000e+00, 6.000000e+00], [5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<3x5xf32>
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:5x5xf32>
      %1 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [5, 5], strides = [1, 1] : !flow.dispatch.tensor<readwrite:5x5xf32> -> tensor<5x5xf32>
      
      // The body of this op is folded into the top-level hal.executable.entry_point body.
      // CHECK-NOT: iree_linalg_ext.hal.executable.entry_point
      "iree_linalg_ext.hal.executable.entry_point"() ({
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        // CHECK-NOT: iree_linalg_ext.hal.return
        iree_linalg_ext.hal.return %c3, %c1, %c1 : index, index, index
      }) : () -> ()

      // CHECK: = hal.interface.workgroup.id[0] : index
      %2 = iree_linalg_ext.hal.interface.workgroup.id[0] : index
      // Unused, just goes away.
      // CHECK-NOT: = hal.interface.workgroup.id[0] : index
      %3 = iree_linalg_ext.hal.interface.workgroup.count[0] : index
      %4 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%2]
      %5 = affine.min affine_map<()[s0] -> (s0 * -2 + 5, 2)>()[%2]
      %6 = tensor.extract_slice %1[%4, 0] [%5, 5] [1, 1] : tensor<5x5xf32> to tensor<?x5xf32>
      %7 = tensor.extract_slice %cst[%4, 0] [%5, 3] [1, 1] : tensor<5x3xf32> to tensor<?x3xf32>
      %8 = linalg.matmul {iree_linalg_transform.matched} ins(%7, %cst_0 : tensor<?x3xf32>, tensor<3x5xf32>) outs(%6 : tensor<?x5xf32>) -> tensor<?x5xf32>
      %9 = tensor.insert_slice %8 into %1[%4, 0] [%5, 5] [1, 1] : tensor<?x5xf32> into tensor<5x5xf32>
      flow.dispatch.tensor.store %9, %0, offsets = [0, 0], sizes = [5, 5], strides = [1, 1] : tensor<5x5xf32> -> !flow.dispatch.tensor<readwrite:5x5xf32>
      return
    }
  }
}
}
