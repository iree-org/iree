// RUN: iree-opt %s \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-specialize-exports, cse)))" \
// RUN: --split-input-file | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
hal.executable private @single_specialization_executable {
  hal.executable.variant public @variant target(#hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">) {
    hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %6 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %23 = arith.index_castui %6 : i32 to index
        %24 = util.assume.int
            %23<umin = 256, umax = 1048320, udiv = 256>
          : index
        %25 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>>
        %26 = iree_tensor_ext.dispatch.workload.ordinal %24#0, 0 : index
        %27 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%26}
        %28 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024xf32>>{%26}
        %30 = iree_tensor_ext.dispatch.tensor.load %27, offsets = [0, 0], sizes = [%26, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%26} -> tensor<?x4096xf16>
        %31 = iree_tensor_ext.dispatch.tensor.load %25, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>> -> tensor<1024x4096xf16>
        %33 = tensor.empty(%26) : tensor<?x1024xf32>
        %34 = linalg.fill ins(%cst : f32) outs(%33 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
        %35 = linalg.matmul_transpose_b ins(%30, %31 : tensor<?x4096xf16>, tensor<1024x4096xf16>) outs(%34 : tensor<?x1024xf32>)
          {iree_codegen.specialization_ranges = #util<int.assumption.multi_array[
            [<umin = 128, umax = 4096, udiv = 128>, <umin = 128, umax = 4096, udiv = 128>, <umin = 64, udiv = 64>],
            [<umin = 4096, udiv = 256>, <umin = 4096, udiv = 256>, <udiv = 64>]
          ]>} -> tensor<?x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %35, %28, offsets = [0, 0], sizes = [%26, 1024], strides = [1, 1] : tensor<?x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024xf32>>{%26}
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable private @single_specialization_executable

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32 ordinal(0)
//  CHECK-SAME:     condition(%{{.*}}: !hal.device, %[[W:.+]]: index) -> i1
//   CHECK-DAG:       %[[TRUE:.+]] = arith.constant true
//       CHECK:       %[[UMIN:.+]] = arith.cmpi ule, %c128, %[[W]]
//       CHECK:       %[[CMIN:.+]] = arith.andi %[[UMIN]], %[[TRUE]]
//       CHECK:       %[[UMAX:.+]] = arith.cmpi uge, %c4096, %[[W]]
//       CHECK:       %[[CMAX:.+]] = arith.andi %[[UMAX]], %[[CMIN]]
//       CHECK:       %[[UREM:.+]] = arith.remui %[[W]], %c128
//       CHECK:       %[[UDIV:.+]] = arith.cmpi eq, %[[UREM]], %c0
//       CHECK:       %[[CDIV:.+]] = arith.andi %[[UDIV]], %[[CMAX]]
//       CHECK:       hal.return %[[CDIV]]
//       CHECK:     fallback(@matmul_transpose_b_Dx1024x4096_f16xf16xf32_0)
//  CHECK-SAME:     count(%arg0: !hal.device, %arg1: index)
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0 ordinal(1)
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   builtin.module
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32
//       CHECK:       util.assume.int %{{.*}}<umin = 128, umax = 4096, udiv = 128>
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0
//       CHECK:       util.assume.int %{{.*}}<umin = 256, umax = 1048320, udiv = 256>

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
hal.executable private @multiple_specialization_executable {
  hal.executable.variant public @variant target(#hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">) {
    hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %6 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %23 = arith.index_castui %6 : i32 to index
        %24 = util.assume.int
            %23<umin = 256, umax = 1048320, udiv = 256>
          : index
        %25 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>>
        %26 = iree_tensor_ext.dispatch.workload.ordinal %24#0, 0 : index
        %27 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%26}
        %28 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024xf32>>{%26}
        %30 = iree_tensor_ext.dispatch.tensor.load %27, offsets = [0, 0], sizes = [%26, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%26} -> tensor<?x4096xf16>
        %31 = iree_tensor_ext.dispatch.tensor.load %25, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>> -> tensor<1024x4096xf16>
        %33 = tensor.empty(%26) : tensor<?x1024xf32>
        %34 = linalg.fill ins(%cst : f32) outs(%33 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
        %35 = linalg.matmul_transpose_b ins(%30, %31 : tensor<?x4096xf16>, tensor<1024x4096xf16>) outs(%34 : tensor<?x1024xf32>)
          {iree_codegen.specialization_ranges = #util<int.assumption.multi_array[
            [<umin = 128, umax = 4096, udiv = 128>, <umin = 128, umax = 4096, udiv = 128>, <umin = 64, udiv = 64>],
            [<umin = 0, udiv = 512>, <umin = 0, udiv = 512>, <udiv = 64>],
            [<udiv = 16>, <udiv = 16>, <udiv = 64>], // Always applies, skipped.
            [<umin = 0, udiv = 512>, <umin = 0, udiv = 512>, <udiv = 64>] // Previous always applies, skipped.
          ]>} -> tensor<?x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %35, %28, offsets = [0, 0], sizes = [%26, 1024], strides = [1, 1] : tensor<?x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024xf32>>{%26}
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable private @multiple_specialization_executable

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32 ordinal(0)
//  CHECK-SAME:     condition(%{{.*}}: !hal.device, %[[W:.+]]: index) -> i1
//       CHECK:     fallback(@matmul_transpose_b_Dx1024x4096_f16xf16xf32_0)

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0 ordinal(1)
//  CHECK-SAME:     condition(%{{.*}}: !hal.device, %[[W:.+]]: index) -> i1
//       CHECK:     fallback(@matmul_transpose_b_Dx1024x4096_f16xf16xf32_0_1)

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0_1 ordinal(2)
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   builtin.module
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32
//       CHECK:       util.assume.int %{{.*}}<umin = 128, umax = 4096, udiv = 128>
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0
//       CHECK:       util.assume.int %{{.*}}<umin = 0, udiv = 512>
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0_1
//       CHECK:       util.assume.int %{{.*}}<umin = 256, umax = 1048320, udiv = 256>
