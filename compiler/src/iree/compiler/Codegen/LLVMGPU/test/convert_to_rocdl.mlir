// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx908 --iree-convert-to-rocdl %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx908 --iree-convert-to-rocdl --iree-hip-index-bits=32 %s | FileCheck %s --check-prefix=INDEX32

// Test that that standard and GPU ops are converted to LLVM and NVVM.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
builtin.module {
  func.func @abs_ex_dispatch_0() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) flags(ReadOnly) : memref<16xf32>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<16xf32>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<16xf32>
    %3 = gpu.block_id x
    %4 = gpu.block_dim x
    %5 = gpu.thread_id x
    %6 = arith.muli %3, %4 : index
    %7 = arith.addi %6, %5 : index
    %9 = memref.load %1[%7] : memref<16xf32>
    %10 = memref.load %2[%7] : memref<16xf32>
    %11 = arith.addf %9, %10 : f32
    memref.store %11, %0[%7] : memref<16xf32>
    return
  }
}
//   CHECK-LABEL: llvm.func @abs_ex_dispatch_0
// INDEX32-LABEL: llvm.func @abs_ex_dispatch_0
//    CHECK-SAME: (%{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef, llvm.readonly},
//    CHECK-SAME:  %{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef},
//    CHECK-SAME:  %{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef},
//    CHECK-SAME:  %{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef, llvm.readnone})
//         CHECK:    rocdl.workgroup.dim.x
//         CHECK:    llvm.getelementptr inbounds|nuw %{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
//       INDEX32:    llvm.getelementptr inbounds|nuw %{{.*}} : (!llvm.ptr, i32) -> !llvm.ptr, f32
//         CHECK:    llvm.fadd


// -----
// Test that maximum and minum are converted to max and min on rocm
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
builtin.module {
  func.func @reduction_maximum() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) :
        memref<32x64x64xf32, strided<[4096, 64, 1], offset: ?>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<32x64x64xf32,
        strided<[4096, 64, 1], offset: ?>>
  %2 = vector.load %0[%c0, %c0, %c0] : memref<32x64x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<2xf32>
  %3 = vector.reduction <maximumf>, %2 : vector<2xf32> into f32
  %4 = vector.splat %3 : vector<2xf32>
  vector.store %4, %1[%c0, %c0, %c0] : memref<32x64x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<2xf32>
  return
  }
}
// CHECK-LABEL: llvm.func @reduction_maximum
// CHECK:  llvm.intr.vector.reduce.fmax({{.*}})  : (vector<2xf32>) -> f32

// -----
// Test that gpu barriers be lowered to `s_waitcnt lgkmcnt(0)\0As_barrier` on rocm
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
builtin.module {
  func.func @simple_barrier() {
    gpu.barrier
    return
  }
}
// CHECK-LABEL: llvm.func @simple_barrier
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att ";;;WARNING: BREAKS DEBUG WATCHES\0As_waitcnt lgkmcnt(0)\0As_barrier", ""  : () -> ()

// -----
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
builtin.module {
  func.func @masked_load_store() {
    %c0 = arith.constant 0 : index
    %idx = gpu.thread_id x
    %pass_thru = arith.constant dense<0.000000e+00> : vector<1xf32>
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<64xf32, #gpu.address_space<global>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<64xf32, #gpu.address_space<global>>
    %mask = vector.create_mask %idx : vector<1xi1>
    %ld = vector.maskedload %0[%idx], %mask, %pass_thru : memref<64xf32, #gpu.address_space<global>>, vector<1xi1>, vector<1xf32> into vector<1xf32>
    vector.maskedstore %1[%idx], %mask, %ld : memref<64xf32, #gpu.address_space<global>>, vector<1xi1>, vector<1xf32>
    return
  }
}
// CHECK-LABEL: llvm.func @masked_load_store
//       CHECK:   %[[MASK_BIT:.+]] = llvm.icmp "sgt" {{.*}} : vector<1xi64>
//       CHECK:   llvm.intr.masked.load %{{.*}}, %[[MASK_BIT]]
//       CHECK:   llvm.intr.masked.store %{{.*}}, %[[MASK_BIT]]

// -----
// Test workgroup size lowering
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
builtin.module attributes {} {
  func.func @interface_wg_size() {
    %c0 = arith.constant 0.0 : f32
    %workgroup_size_x = hal.interface.workgroup.size[0] : index
    %workgroup_size_y = hal.interface.workgroup.size[1] : index
    %subspan = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<64x64xf32>
    memref.store %c0, %subspan[%workgroup_size_x, %workgroup_size_y] : memref<64x64xf32>
    return
  }
}
// CHECK-LABEL: llvm.func @interface_wg_size
//       CHECK:   %[[WGDIMX:.+]] = rocdl.workgroup.dim.x
//       CHECK:   %[[WGDIMY:.+]] = rocdl.workgroup.dim.y

// -----

// Check that the operations generated by emulate bit widths are lowered correctly

module {
  func.func @emulation_lowering() {
    %cst = arith.constant dense<4> : vector<2x4xi8>
    %cst_0 = arith.constant dense<15> : vector<2x4xi8>
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant dense<0> : vector<2x8xi4>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<8x1x2xf16>
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %thread_id_x = gpu.thread_id  x upper_bound 64
    %0 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<131072x192xf16, #gpu.address_space<global>>
    %1 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<131072x192xf16, #gpu.address_space<global>>
    %2 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : memref<402653184xi8, #gpu.address_space<global>>
    %3 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(4) alignment(64) offset(%c0) flags(Indirect) : memref<131072x192x32xf16, #gpu.address_space<global>>
    %4 = arith.divui %thread_id_x, %c4 : index
    %5 = arith.remui %thread_id_x, %c4 : index
    %6 = arith.muli %5, %c8 : index
    %workgroup_id_x = hal.interface.workgroup.id[0] upper_bound 6 : index
    %workgroup_id_y = hal.interface.workgroup.id[1] upper_bound 131072 : index
    %7 = arith.muli %4, %c2 : index
    %8 = arith.muli %workgroup_id_x, %c32 : index
    %9 = arith.addi %7, %8 : index
    %10 = vector.load %0[%workgroup_id_y, %9] : memref<131072x192xf16, #gpu.address_space<global>>, vector<2xf16>
    %11 = vector.broadcast %10 : vector<2xf16> to vector<1x2xf16>
    %12 = vector.insert %11, %cst_2 [0] : vector<1x2xf16> into vector<8x1x2xf16>
    %13 = vector.insert %11, %12 [1] : vector<1x2xf16> into vector<8x1x2xf16>
    %14 = vector.insert %11, %13 [2] : vector<1x2xf16> into vector<8x1x2xf16>
    %15 = vector.insert %11, %14 [3] : vector<1x2xf16> into vector<8x1x2xf16>
    %16 = vector.insert %11, %15 [4] : vector<1x2xf16> into vector<8x1x2xf16>
    %17 = vector.insert %11, %16 [5] : vector<1x2xf16> into vector<8x1x2xf16>
    %18 = vector.insert %11, %17 [6] : vector<1x2xf16> into vector<8x1x2xf16>
    %19 = vector.insert %11, %18 [7] : vector<1x2xf16> into vector<8x1x2xf16>
    %20 = vector.transpose %19, [1, 2, 0] : vector<8x1x2xf16> to vector<1x2x8xf16>
    %21 = vector.load %1[%workgroup_id_y, %9] : memref<131072x192xf16, #gpu.address_space<global>>, vector<2xf16>
    %22 = vector.broadcast %21 : vector<2xf16> to vector<1x2xf16>
    %23 = vector.insert %22, %cst_2 [0] : vector<1x2xf16> into vector<8x1x2xf16>
    %24 = vector.insert %22, %23 [1] : vector<1x2xf16> into vector<8x1x2xf16>
    %25 = vector.insert %22, %24 [2] : vector<1x2xf16> into vector<8x1x2xf16>
    %26 = vector.insert %22, %25 [3] : vector<1x2xf16> into vector<8x1x2xf16>
    %27 = vector.insert %22, %26 [4] : vector<1x2xf16> into vector<8x1x2xf16>
    %28 = vector.insert %22, %27 [5] : vector<1x2xf16> into vector<8x1x2xf16>
    %29 = vector.insert %22, %28 [6] : vector<1x2xf16> into vector<8x1x2xf16>
    %30 = vector.insert %22, %29 [7] : vector<1x2xf16> into vector<8x1x2xf16>
    %31 = vector.transpose %30, [1, 2, 0] : vector<8x1x2xf16> to vector<1x2x8xf16>
    %c3072 = arith.constant 3072 : index
    %32 = arith.muli %workgroup_id_y, %c3072 : index
    %c16 = arith.constant 16 : index
    %33 = arith.muli %9, %c16 : index
    %34 = arith.addi %32, %33 : index
    %c2_3 = arith.constant 2 : index
    %c0_4 = arith.constant 0 : index
    %c-1 = arith.constant -1 : index
    %35 = arith.cmpi slt, %6, %c0_4 : index
    %36 = arith.subi %c-1, %6 : index
    %37 = arith.select %35, %36, %6 : index
    %38 = arith.divsi %37, %c2_3 : index
    %39 = arith.subi %c-1, %38 : index
    %40 = arith.select %35, %39, %38 : index
    %41 = arith.addi %34, %40 : index
    %42 = vector.load %2[%41] : memref<402653184xi8, #gpu.address_space<global>>, vector<4xi8>
    %43 = vector.bitcast %42 : vector<4xi8> to vector<8xi4>
    %44 = vector.insert %43, %cst_1 [0] : vector<8xi4> into vector<2x8xi4>
    %45 = arith.addi %9, %c1 : index
    %c3072_5 = arith.constant 3072 : index
    %46 = arith.muli %workgroup_id_y, %c3072_5 : index
    %c16_6 = arith.constant 16 : index
    %47 = arith.muli %45, %c16_6 : index
    %48 = arith.addi %46, %47 : index
    %c2_7 = arith.constant 2 : index
    %c0_8 = arith.constant 0 : index
    %c-1_9 = arith.constant -1 : index
    %49 = arith.cmpi slt, %6, %c0_8 : index
    %50 = arith.subi %c-1_9, %6 : index
    %51 = arith.select %49, %50, %6 : index
    %52 = arith.divsi %51, %c2_7 : index
    %53 = arith.subi %c-1_9, %52 : index
    %54 = arith.select %49, %53, %52 : index
    %55 = arith.addi %48, %54 : index
    %56 = vector.load %2[%55] : memref<402653184xi8, #gpu.address_space<global>>, vector<4xi8>
    %57 = vector.bitcast %56 : vector<4xi8> to vector<8xi4>
    %58 = vector.insert %57, %44 [1] : vector<8xi4> into vector<2x8xi4>
    %59 = vector.bitcast %58 : vector<2x8xi4> to vector<2x4xi8>
    %60 = arith.andi %59, %cst_0 : vector<2x4xi8>
    %61 = arith.shrui %59, %cst : vector<2x4xi8>
    %62 = vector.interleave %60, %61 : vector<2x4xi8> -> vector<2x8xi8>
    %63 = arith.extui %62 : vector<2x8xi8> to vector<2x8xi32>
    %64 = arith.uitofp %63 : vector<2x8xi32> to vector<2x8xf16>
    %65 = vector.extract %20[0] : vector<2x8xf16> from vector<1x2x8xf16>
    %66 = arith.mulf %64, %65 : vector<2x8xf16>
    %67 = vector.extract %31[0] : vector<2x8xf16> from vector<1x2x8xf16>
    %68 = arith.addf %66, %67 : vector<2x8xf16>
    %69 = vector.extract %68[0] : vector<8xf16> from vector<2x8xf16>
    vector.store %69, %3[%workgroup_id_y, %9, %6] : memref<131072x192x32xf16, #gpu.address_space<global>>, vector<8xf16>
    %70 = vector.extract %68[1] : vector<8xf16> from vector<2x8xf16>
    vector.store %70, %3[%workgroup_id_y, %45, %6] : memref<131072x192x32xf16, #gpu.address_space<global>>, vector<8xf16>
    return
  }
}
// CHECK-LABEL: llvm.func @emulation_lowering(
//   CHECK-NOT:   builtin.unrealized_conversion_cast

// -----
// Test that an unused binding still appears in the kernargs
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
builtin.module {
  func.func @missing_ptr_dispatch_copy_idx_0() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
    %1 = arith.index_castui %0 : i32 to index
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) offset(%1) flags(ReadOnly) : memref<16xf32, strided<[1], offset : ?>, #gpu.address_space<global>>
    %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<16xf32, #gpu.address_space<global>>
    %4 = memref.load %2[%c0] : memref<16xf32, strided<[1], offset : ?>, #gpu.address_space<global>>
    memref.store %4, %3[%c0] : memref<16xf32, #gpu.address_space<global>>
    return
  }
}
//   CHECK-LABEL: llvm.func @missing_ptr_dispatch_copy_idx_0
//    CHECK-SAME: (%[[arg0:.+]]: !llvm.ptr<1> {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef, llvm.readonly},
//    CHECK-SAME:  %[[arg1:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef, llvm.readnone},
//    CHECK-SAME:  %[[arg2:.+]]: !llvm.ptr<1> {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef},
//    CHECK-SAME:  %[[arg3:.+]]: i32 {llvm.noundef})
//         CHECK:   llvm.zext %[[arg3]] : i32 to i64
//         CHECK:   llvm.insertvalue %[[arg0]]
//         CHECK:   llvm.insertvalue %[[arg2]]

// -----
// Test lowering of iree_codegen.null_pointer.
module {
  func.func private @foo(!iree_codegen.null_pointer)
  func.func @null_pointer() {
    %0 = iree_codegen.null_pointer
    call @foo(%0) : (!iree_codegen.null_pointer) -> ()
    return
  }
}
//   CHECK-LABEL: llvm.func @null_pointer
//   CHECK:       llvm.mlir.zero : !llvm.ptr

// -----

module {
  func.func private @foo(vector<4xf32>)
  func.func @swap_mfma() {
    %in = arith.constant 1.0 : f32
    %out = arith.constant dense<0.0> : vector<4xf32>
    rocdl.s.setprio 1 { iree_gpu.swap_mfma = -10 }
    rocdl.s.setprio 2 { iree_gpu.swap_mfma = 1 }
    rocdl.s.setprio 3 { iree_gpu.swap_mfma = 2 }
    rocdl.s.setprio 4 { iree_gpu.swap_mfma = 5 }
    %0 = amdgpu.mfma %in * %in + %out {
      abid = 0 : i32, cbsz = 0 : i32, k = 1 : i32, m = 4 : i32, n = 4 : i32, blocks = 16 : i32
    }  blgp = none : f32, f32, vector<4xf32>
    %1 = amdgpu.mfma %in * %in + %0 {
      abid = 0 : i32, cbsz = 0 : i32, k = 1 : i32, m = 4 : i32, n = 4 : i32, blocks = 16 : i32
    }  blgp = none : f32, f32, vector<4xf32>
    %2 = amdgpu.mfma %in * %in + %1 {
      abid = 0 : i32, cbsz = 0 : i32, k = 1 : i32, m = 4 : i32, n = 4 : i32, blocks = 16 : i32
    }  blgp = none : f32, f32, vector<4xf32>
    call @foo(%2) : (vector<4xf32>) -> ()
    return
  }
}
//   CHECK-LABEL: llvm.func @swap_mfma
//   CHECK:         rocdl.s.setprio 1
//   CHECK:         rocdl.mfma
//   CHECK-NEXT:    rocdl.s.setprio 2
//   CHECK:         rocdl.mfma
//   CHECK-NEXT:    rocdl.s.setprio 3
//   CHECK:         rocdl.mfma
//   CHECK-NEXT:    rocdl.s.setprio 4
