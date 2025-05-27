// RUN: iree-opt %s \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-specialize-exports, cse)))" \
// RUN:   --split-input-file | FileCheck %s

#executable_target_embedded_elf_aarch64 = #hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
hal.executable private @single_specialization_executable {
  hal.executable.variant public @variant target(#executable_target_embedded_elf_aarch64) {
    hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %1 = arith.index_castui %0 : i32 to index
        %2 = util.assume.int %1<umin = 256, umax = 1048320, udiv = 256> : index
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>>
        %4 = iree_tensor_ext.dispatch.workload.ordinal %2, 0 : index
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%4}
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024xf32>>{%4}
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%4, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%4} -> tensor<?x4096xf16>
        %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>> -> tensor<1024x4096xf16>
        %9 = tensor.empty(%4) : tensor<?x1024xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
        %11 = linalg.matmul_transpose_b {
          iree_codegen.specialization_ranges = #util<int.assumption.multi_array[
            [<umin = 128, umax = 4096, udiv = 128>, <umin = 128, umax = 4096, udiv = 128>, <umin = 64, udiv = 64>],
            [<umin = 4096, udiv = 256>, <umin = 4096, udiv = 256>, <udiv = 64>]]>}
          ins(%7, %8 : tensor<?x4096xf16>, tensor<1024x4096xf16>) outs(%10 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%4, 1024], strides = [1, 1] : tensor<?x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024xf32>>{%4}
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
//  CHECK-SAME:     count(%{{[A-Za-z0-9]*}}: !hal.device
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0 ordinal(1)
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   builtin.module
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32
//       CHECK:       util.assume.int %{{.*}}<umin = 128, umax = 4096, udiv = 128>
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0
//       CHECK:       util.assume.int %{{.*}}<umin = 256, umax = 1048320, udiv = 256>

// -----

#executable_target_embedded_elf_aarch64 = #hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
hal.executable private @multiple_specialization_executable {
  hal.executable.variant public @variant target(#executable_target_embedded_elf_aarch64) {
    hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %1 = arith.index_castui %0 : i32 to index
        %2 = util.assume.int %1<umin = 256, umax = 1048320, udiv = 256> : index
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>>
        %4 = iree_tensor_ext.dispatch.workload.ordinal %2, 0 : index
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%4}
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024xf32>>{%4}
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%4, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%4} -> tensor<?x4096xf16>
        %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>> -> tensor<1024x4096xf16>
        %9 = tensor.empty(%4) : tensor<?x1024xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
        %11 = linalg.matmul_transpose_b {
          iree_codegen.specialization_ranges = #util<int.assumption.multi_array[
            [<umin = 128, umax = 4096, udiv = 128>, <umin = 128, umax = 4096, udiv = 128>, <umin = 64, udiv = 64>],
            [<umin = 0, udiv = 512>, <umin = 0, udiv = 512>, <udiv = 64>], [<udiv = 16>, <udiv = 16>, <udiv = 64>],
            [<umin = 0, udiv = 512>, <umin = 0, udiv = 512>, <udiv = 64>]]>}
          ins(%7, %8 : tensor<?x4096xf16>, tensor<1024x4096xf16>) outs(%10 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%4, 1024], strides = [1, 1] : tensor<?x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024xf32>>{%4}
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

// -----

#executable_target_embedded_elf_aarch64 = #hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
hal.executable private @multiple_dimension_assume {
  hal.executable.variant public @variant target(#executable_target_embedded_elf_aarch64) {
    hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4:2 = util.assume.int
            %2[<umin = 256, umax = 8192, udiv = 256>, <udiv = 128>],
            %3<udiv = 512>
          : index, index
        %5 = iree_tensor_ext.dispatch.workload.ordinal %4#1, 1 : index
        %6 = iree_tensor_ext.dispatch.workload.ordinal %4#0, 0 : index
        %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%6}
        %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%5}
        %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %5}
        %10 = iree_tensor_ext.dispatch.tensor.load %7, offsets = [0, 0], sizes = [%6, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%6} -> tensor<?x4096xf16>
        %11 = iree_tensor_ext.dispatch.tensor.load %8, offsets = [0, 0], sizes = [%5, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%5} -> tensor<?x4096xf16>
        %12 = tensor.empty(%6, %5) : tensor<?x?xf32>
        %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %14 = linalg.matmul_transpose_b {
          iree_codegen.specialization_ranges = #util<int.assumption.multi_array[
            [<umin = 128, umax = 4096, udiv = 128>, <umin = 128, umax = 4096, udiv = 128>, <umin = 64, udiv = 64>],
            [<umin = 4096, udiv = 256>, <umin = 4096, udiv = 256>, <udiv = 64>]]>}
          ins(%10, %11 : tensor<?x4096xf16>, tensor<?x4096xf16>) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
        iree_tensor_ext.dispatch.tensor.store %14, %9, offsets = [0, 0], sizes = [%6, %5], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %5}
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable private @multiple_dimension_assume

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32 ordinal(0)
//  CHECK-SAME:     condition(%{{.*}}: !hal.device, %[[W0:[A-Za-z0-9]+]]: index, %[[W1:[A-Za-z0-9]+]]: index) -> i1
//       CHECK:       %[[TRUE:.+]] = arith.constant true
//       CHECK:       %[[UMIN:.+]] = arith.cmpi ule, %c128, %[[W0]]
//       CHECK:       %[[CMIN:.+]] = arith.andi %[[UMIN]], %[[TRUE]]
//       CHECK:       %[[UMAX:.+]] = arith.cmpi uge, %c4096, %[[W0]]
//       CHECK:       %[[CMAX:.+]] = arith.andi %[[UMAX]], %[[CMIN]]
//       CHECK:       %[[UREM:.+]] = arith.remui %[[W0]], %c128
//       CHECK:       %[[UDIV:.+]] = arith.cmpi eq, %[[UREM]], %c0
//       CHECK:       %[[CDIV:.+]] = arith.andi %[[UDIV]], %[[CMAX]]
//       CHECK:       %[[UMIN1:.+]] = arith.cmpi ule, %c128, %[[W1]]
//       CHECK:       %[[CMIN1:.+]] = arith.andi %[[UMIN1]], %[[CDIV]]
//       CHECK:       %[[UMAX1:.+]] = arith.cmpi uge, %c4096, %[[W1]]
//       CHECK:       %[[CMAX1:.+]] = arith.andi %[[UMAX1]], %[[CMIN1]]
//       CHECK:       %[[UREM1:.+]] = arith.remui %[[W1]], %c128
//       CHECK:       %[[UDIV1:.+]] = arith.cmpi eq, %[[UREM1]], %c0
//       CHECK:       %[[CDIV1:.+]] = arith.andi %[[UDIV1]], %[[CMAX1]]
//       CHECK:       hal.return %[[CDIV1]]
//       CHECK:     fallback(@matmul_transpose_b_Dx1024x4096_f16xf16xf32_0)
//  CHECK-SAME:     count(%{{[A-Za-z0-9]*}}: !hal.device
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0 ordinal(1)
//  CHECK-SAME:     condition(%{{.*}}: !hal.device, %[[W0:[A-Za-z0-9]+]]: index, %[[W1:[A-Za-z0-9]+]]: index) -> i1
//       CHECK:       %[[TRUE:.+]] = arith.constant true
//       CHECK:       %[[UMIN:.+]] = arith.cmpi ule, %c4096, %[[W0]]
//       CHECK:       %[[CMIN:.+]] = arith.andi %[[UMIN]], %[[TRUE]]
//       CHECK:       %[[UREM:.+]] = arith.remui %[[W0]], %c256
//       CHECK:       %[[UDIV:.+]] = arith.cmpi eq, %[[UREM]], %c0
//       CHECK:       %[[CDIV:.+]] = arith.andi %[[UDIV]], %[[CMIN]]
//       CHECK:       %[[UMIN1:.+]] = arith.cmpi ule, %c4096, %[[W1]]
//       CHECK:       %[[CMIN1:.+]] = arith.andi %[[UMIN1]], %[[CDIV]]
//       CHECK:       %[[UREM1:.+]] = arith.remui %[[W1]], %c256
//       CHECK:       %[[UDIV1:.+]] = arith.cmpi eq, %[[UREM1]], %c0
//       CHECK:       %[[CDIV1:.+]] = arith.andi %[[UDIV1]], %[[CMIN1]]
//       CHECK:       hal.return %[[CDIV1]]
//       CHECK:     fallback(@matmul_transpose_b_Dx1024x4096_f16xf16xf32_0_1)
//  CHECK-SAME:     count(%{{[A-Za-z0-9]*}}: !hal.device
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0_1 ordinal(2)
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   builtin.module
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32
//       CHECK:       util.assume.int
//  CHECK-NEXT:         <umin = 128, umax = 4096, udiv = 128>
//  CHECK-NEXT:         <umin = 128, umax = 4096, udiv = 128>
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0
//       CHECK:       util.assume.int
//  CHECK-NEXT:         <umin = 4096, udiv = 256>
//  CHECK-NEXT:         <umin = 4096, udiv = 256>
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0_1
//       CHECK:       util.assume.int
//  CHECK-NEXT:         [<umin = 256, umax = 8192, udiv = 256>, <udiv = 128>]
//  CHECK-NEXT:         <udiv = 512>

// -----

#executable_target_embedded_elf_aarch64 = #hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
hal.executable private @unrelated_int_assume {
  hal.executable.variant public @variant target(#executable_target_embedded_elf_aarch64) {
    hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %12 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %1 = arith.index_castui %0 : i32 to index
        %13 = arith.index_castui %12 : i32 to index
        %2:2 = util.assume.int
            %1[<umin = 256, umax = 1048320, udiv = 256>, <udiv = 128>],
            %13[<umin = 0, umax = 1000, udiv = 2>, <umin = 1, umax = 2000, udiv = 3>]
          : index, index
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%2#1) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>>
        %4 = iree_tensor_ext.dispatch.workload.ordinal %2#0, 0 : index
        %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%4}
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024xf32>>{%4}
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%4, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%4} -> tensor<?x4096xf16>
        %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>> -> tensor<1024x4096xf16>
        %9 = tensor.empty(%4) : tensor<?x1024xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
        %11 = linalg.matmul_transpose_b {
          iree_codegen.specialization_ranges = #util<int.assumption.multi_array[
            [<umin = 4096, udiv = 256>, <udiv = 256>, <udiv = 64>]]>}
          ins(%7, %8 : tensor<?x4096xf16>, tensor<1024x4096xf16>) outs(%10 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%4, 1024], strides = [1, 1] : tensor<?x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x1024xf32>>{%4}
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable private @unrelated_int_assume

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32 ordinal(0)
//  CHECK-SAME:     condition(%{{.*}}: !hal.device, %[[W:.+]]: index) -> i1
//       CHECK:     fallback(@matmul_transpose_b_Dx1024x4096_f16xf16xf32_0)
//  CHECK-SAME:     count(%{{[A-Za-z0-9]*}}: !hal.device
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0 ordinal(1)
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   builtin.module
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32
//       CHECK:       util.assume.int
//  CHECK-NEXT:         <umin = 4096, udiv = 256>
//  CHECK-NEXT:         [<umin = 0, umax = 1000, udiv = 2>, <umin = 1, umax = 2000, udiv = 3>]
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0
//       CHECK:       util.assume.int
//  CHECK-NEXT:         [<umin = 256, umax = 1048320, udiv = 256>, <udiv = 128>]
//  CHECK-NEXT:         [<umin = 0, umax = 1000, udiv = 2>, <umin = 1, umax = 2000, udiv = 3>]

// -----

#executable_target_embedded_elf_aarch64 = #hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
hal.executable private @no_assume {
  hal.executable.variant public @variant target(#executable_target_embedded_elf_aarch64) {
    hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %5 = iree_tensor_ext.dispatch.workload.ordinal %2, 1 : index
        %6 = iree_tensor_ext.dispatch.workload.ordinal %3, 0 : index
        %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%6}
        %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%5}
        %9 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %5}
        %10 = iree_tensor_ext.dispatch.tensor.load %7, offsets = [0, 0], sizes = [%6, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%6} -> tensor<?x4096xf16>
        %11 = iree_tensor_ext.dispatch.tensor.load %8, offsets = [0, 0], sizes = [%5, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%5} -> tensor<?x4096xf16>
        %12 = tensor.empty(%6, %5) : tensor<?x?xf32>
        %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %14 = linalg.matmul_transpose_b {
          iree_codegen.specialization_ranges = #util<int.assumption.multi_array[
            [<umin = 128, umax = 4096, udiv = 128>, <umin = 128, umax = 4096, udiv = 128>, <umin = 64, udiv = 64>],
            [<umin = 4096, udiv = 256>, <umin = 4096, udiv = 256>, <udiv = 64>]]>}
          ins(%10, %11 : tensor<?x4096xf16>, tensor<?x4096xf16>) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
        iree_tensor_ext.dispatch.tensor.store %14, %9, offsets = [0, 0], sizes = [%6, %5], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %5}
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable private @no_assume

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32 ordinal(0)
//  CHECK-SAME:     condition(%{{.*}}: !hal.device
//       CHECK:     fallback(@matmul_transpose_b_Dx1024x4096_f16xf16xf32_0)
//  CHECK-SAME:     count(%{{[A-Za-z0-9]*}}: !hal.device
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0 ordinal(1)
//  CHECK-SAME:     condition(%{{.*}}: !hal.device
//       CHECK:     fallback(@matmul_transpose_b_Dx1024x4096_f16xf16xf32_0_1)
//  CHECK-SAME:     count(%{{[A-Za-z0-9]*}}: !hal.device
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   hal.executable.export public @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0_1 ordinal(2)
//  CHECK-NEXT:       iree_tensor_ext.dispatch.workgroup_count_from_slice

//       CHECK:   builtin.module
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32
//   CHECK-DAG:       util.assume.int %{{.*}}<umin = 128, umax = 4096, udiv = 128>
//   CHECK-DAG:       util.assume.int %{{.*}}<umin = 128, umax = 4096, udiv = 128>
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0
//   CHECK-DAG:       util.assume.int %{{.*}}<umin = 4096, udiv = 256>
//   CHECK-DAG:       util.assume.int %{{.*}}<umin = 4096, udiv = 256>
//       CHECK:     func.func @matmul_transpose_b_Dx1024x4096_f16xf16xf32_0_1
//   CHECK-NOT:       util.assume.int
