// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx908 --iree-convert-to-rocdl %s | FileCheck %s

// Test that that standard and GPU ops are converted to LLVM and NVVM.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
builtin.module {
  func.func @abs_ex_dispatch_0() {
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
//    CHECK-SAME: (%{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef, llvm.readonly},
//    CHECK-SAME:  %{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef},
//    CHECK-SAME:  %{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef},
//    CHECK-SAME:  %{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef, llvm.readnone})
//         CHECK:    llvm.call @__ockl_get_local_size({{.*}}) : (i32) -> (i64
//         CHECK:    llvm.getelementptr inbounds|nuw %{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
//         CHECK:    llvm.fadd


// -----
// Test that maximum and minimum are converted to max and min on rocm
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
  %4 = vector.broadcast %3 : f32 to vector<2xf32>
  vector.store %4, %1[%c0, %c0, %c0] : memref<32x64x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<2xf32>
  return
  }
}
// CHECK-LABEL: llvm.func @reduction_maximum
// CHECK:  llvm.intr.vector.reduce.fmax({{.*}})  : (vector<2xf32>) -> f32

// -----
// Test that gpu barriers be lowered to mmra fences
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
builtin.module {
  func.func @simple_barrier() {
    gpu.barrier memfence [#gpu.address_space<workgroup>]
    return
  }
}
// CHECK: #[[$MMRA:.+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">
// CHECK-LABEL: llvm.func @simple_barrier
// CHECK: llvm.fence syncscope("workgroup") release {llvm.mmra = #[[$MMRA]]}
// CHECK: rocdl.s.barrier
// CHECK: llvm.fence syncscope("workgroup") acquire {llvm.mmra = #[[$MMRA]]}

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
//       CHECK:   llvm.call @__ockl_get_local_size({{.*}}) : (i32) -> (i64
//       CHECK:   llvm.call @__ockl_get_local_size({{.*}}) : (i32) -> (i64

// -----

// Check that the operations generated by emulate bit widths are lowered correctly

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module {
  func.func @emulation_lowering() {
    %cst = arith.constant dense<4> : vector<4xi8>
    %cst_0 = arith.constant dense<4> : vector<4xi8>
    %cst_1 = arith.constant dense<15> : vector<4xi8>
    %cst_2 = arith.constant dense<15> : vector<4xi8>
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %thread_id_x = gpu.thread_id x upper_bound 64
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<131072x192xf16, #gpu.address_space<global>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<131072x192xf16, #gpu.address_space<global>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : memref<402653184xi8, #gpu.address_space<global>>
    %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) flags(Indirect) : memref<131072x192x32xf16, #gpu.address_space<global>>
    %4 = arith.divui %thread_id_x, %c4 : index
    %5 = arith.remui %thread_id_x, %c4 : index
    %6 = arith.muli %5, %c8 : index
    %workgroup_id_x = hal.interface.workgroup.id[0] upper_bound 6 : index
    %workgroup_id_y = hal.interface.workgroup.id[1] upper_bound 131072 : index
    %7 = arith.muli %4, %c2 : index
    %8 = arith.muli %workgroup_id_x, %c32 : index
    %9 = arith.addi %7, %8 : index
    %10 = vector.load %0[%workgroup_id_y, %9] : memref<131072x192xf16, #gpu.address_space<global>>, vector<2xf16>
    %11 = vector.broadcast %10 : vector<2xf16> to vector<2xf16>
    %12:2 = vector.to_elements %11 : vector<2xf16>
    %13:2 = vector.to_elements %11 : vector<2xf16>
    %14:2 = vector.to_elements %11 : vector<2xf16>
    %15:2 = vector.to_elements %11 : vector<2xf16>
    %16:2 = vector.to_elements %11 : vector<2xf16>
    %17:2 = vector.to_elements %11 : vector<2xf16>
    %18:2 = vector.to_elements %11 : vector<2xf16>
    %19:2 = vector.to_elements %11 : vector<2xf16>
    %20 = vector.from_elements %12#0, %13#0, %14#0, %15#0, %16#0, %17#0, %18#0, %19#0 : vector<8xf16>
    %21 = vector.from_elements %12#1, %13#1, %14#1, %15#1, %16#1, %17#1, %18#1, %19#1 : vector<8xf16>
    %22 = vector.load %1[%workgroup_id_y, %9] : memref<131072x192xf16, #gpu.address_space<global>>, vector<2xf16>
    %23 = vector.broadcast %22 : vector<2xf16> to vector<2xf16>
    %24:2 = vector.to_elements %23 : vector<2xf16>
    %25:2 = vector.to_elements %23 : vector<2xf16>
    %26:2 = vector.to_elements %23 : vector<2xf16>
    %27:2 = vector.to_elements %23 : vector<2xf16>
    %28:2 = vector.to_elements %23 : vector<2xf16>
    %29:2 = vector.to_elements %23 : vector<2xf16>
    %30:2 = vector.to_elements %23 : vector<2xf16>
    %31:2 = vector.to_elements %23 : vector<2xf16>
    %32 = vector.from_elements %24#0, %25#0, %26#0, %27#0, %28#0, %29#0, %30#0, %31#0 : vector<8xf16>
    %33 = vector.from_elements %24#1, %25#1, %26#1, %27#1, %28#1, %29#1, %30#1, %31#1 : vector<8xf16>
    %c3072 = arith.constant 3072 : index
    %34 = arith.muli %workgroup_id_y, %c3072 : index
    %c16 = arith.constant 16 : index
    %35 = arith.muli %9, %c16 : index
    %36 = arith.addi %34, %35 : index
    %c2_13 = arith.constant 2 : index
    %c0_14 = arith.constant 0 : index
    %c-1 = arith.constant -1 : index
    %37 = arith.cmpi slt, %6, %c0_14 : index
    %38 = arith.subi %c-1, %6 : index
    %39 = arith.select %37, %38, %6 : index
    %40 = arith.divsi %39, %c2_13 : index
    %41 = arith.subi %c-1, %40 : index
    %42 = arith.select %37, %41, %40 : index
    %43 = arith.addi %36, %42 : index
    %44 = vector.load %2[%43] : memref<402653184xi8, #gpu.address_space<global>>, vector<4xi8>
    %45 = vector.bitcast %44 : vector<4xi8> to vector<8xi4>
    %46 = arith.addi %9, %c1 : index
    %c3072_15 = arith.constant 3072 : index
    %47 = arith.muli %workgroup_id_y, %c3072_15 : index
    %c16_16 = arith.constant 16 : index
    %48 = arith.muli %46, %c16_16 : index
    %49 = arith.addi %47, %48 : index
    %c2_17 = arith.constant 2 : index
    %c0_18 = arith.constant 0 : index
    %c-1_19 = arith.constant -1 : index
    %50 = arith.cmpi slt, %6, %c0_18 : index
    %51 = arith.subi %c-1_19, %6 : index
    %52 = arith.select %50, %51, %6 : index
    %53 = arith.divsi %52, %c2_17 : index
    %54 = arith.subi %c-1_19, %53 : index
    %55 = arith.select %50, %54, %53 : index
    %56 = arith.addi %49, %55 : index
    %57 = vector.load %2[%56] : memref<402653184xi8, #gpu.address_space<global>>, vector<4xi8>
    %58 = vector.bitcast %57 : vector<4xi8> to vector<8xi4>
    %59 = vector.bitcast %45 : vector<8xi4> to vector<4xi8>
    %60 = vector.bitcast %58 : vector<8xi4> to vector<4xi8>
    %61 = arith.andi %59, %cst_1 : vector<4xi8>
    %62 = arith.andi %60, %cst_2 : vector<4xi8>
    %63 = arith.shrui %59, %cst : vector<4xi8>
    %64 = arith.shrui %60, %cst_0 : vector<4xi8>
    %cst_andi = arith.constant dense<0> : vector<2x4xi8>
    %65 = vector.insert %61, %cst_andi [0] : vector<4xi8> into vector<2x4xi8>
    %66 = vector.insert %62, %65 [1] : vector<4xi8> into vector<2x4xi8>
    %cst_shrui = arith.constant dense<0> : vector<2x4xi8>
    %67 = vector.insert %63, %cst_shrui [0] : vector<4xi8> into vector<2x4xi8>
    %68 = vector.insert %64, %67 [1] : vector<4xi8> into vector<2x4xi8>
    %69 = vector.interleave %66, %68 : vector<2x4xi8> -> vector<2x8xi8>
    %70 = vector.extract %69[0] : vector<8xi8> from vector<2x8xi8>
    %71 = vector.extract %69[1] : vector<8xi8> from vector<2x8xi8>
    %72 = arith.extui %70 : vector<8xi8> to vector<8xi32>
    %73 = arith.extui %71 : vector<8xi8> to vector<8xi32>
    %74 = arith.uitofp %72 : vector<8xi32> to vector<8xf16>
    %75 = arith.uitofp %73 : vector<8xi32> to vector<8xf16>
    %76 = arith.mulf %74, %20 : vector<8xf16>
    %77 = arith.mulf %75, %21 : vector<8xf16>
    %78 = arith.addf %76, %32 : vector<8xf16>
    %79 = arith.addf %77, %33 : vector<8xf16>
    vector.store %78, %3[%workgroup_id_y, %9, %6] : memref<131072x192x32xf16, #gpu.address_space<global>>, vector<8xf16>
    vector.store %79, %3[%workgroup_id_y, %46, %6] : memref<131072x192x32xf16, #gpu.address_space<global>>, vector<8xf16>
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
    %0 = amdgpu.mfma 4x4x1 %in * %in + %out {
      abid = 0 : i32, cbsz = 0 : i32, blocks = 16 : i32
    }  blgp = none : f32, f32, vector<4xf32>
    %1 = amdgpu.mfma 4x4x1 %in * %in + %0 {
      abid = 0 : i32, cbsz = 0 : i32, blocks = 16 : i32
    }  blgp = none : f32, f32, vector<4xf32>
    %2 = amdgpu.mfma 4x4x1 %in * %in + %1 {
      abid = 0 : i32, cbsz = 0 : i32, blocks = 16 : i32
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

// -----

builtin.module {
  func.func @shared_memory_lowering() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
    %0 = memref.alloc() : memref<1x16x32xf32, #gpu.address_space<workgroup>>
    %1 = memref.alloc() : memref<1x32x16xf32, #gpu.address_space<workgroup>>
    %2 = memref.alloc() : memref<1x8x16xf32, #gpu.address_space<workgroup>>
    vector.store %cst, %1[%c0, %c0, %c0] : memref<1x32x16xf32, #gpu.address_space<workgroup>>, vector<4xf32>
    vector.store %cst, %2[%c0, %c0, %c0] : memref<1x8x16xf32, #gpu.address_space<workgroup>>, vector<4xf32>
    vector.store %cst, %0[%c0, %c0, %c0] : memref<1x16x32xf32, #gpu.address_space<workgroup>>, vector<4xf32>
    return
  }
}

// CHECK-DAG: llvm.mlir.global private @__shared_memory___1() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<1 x array<16 x array<32 x f32>>>
// CHECK-DAG: llvm.mlir.global private @__shared_memory___0() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<1 x array<32 x array<16 x f32>>>
// CHECK-DAG: llvm.mlir.global private @__shared_memory__() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<1 x array<8 x array<16 x f32>>>
// CHECK-LABEL: llvm.func @shared_memory_lowering() {
//   CHECK-DAG: %[[A1:.+]] = llvm.mlir.addressof @__shared_memory___1
//   CHECK-DAG: llvm.getelementptr %[[A1]][0, 0, 0, 0]
//   CHECK-DAG: %[[A0:.+]] = llvm.mlir.addressof @__shared_memory___0
//   CHECK-DAG: llvm.getelementptr %[[A0]][0, 0, 0, 0]
//   CHECK-DAG: %[[A:.+]] = llvm.mlir.addressof @__shared_memory__
//   CHECK-DAG: llvm.getelementptr %[[A]][0, 0, 0, 0]

// -----

builtin.module {
  func.func @global_subgroup_barrier() {
    iree_gpu.global_subgroup_barrier
    return
  }
}

// CHECK-LABEL: llvm.func @global_subgroup_barrier
//       CHECK:   llvm.inline_asm has_side_effects asm_dialect = att ";;;WARNING: BREAKS DEBUG WATCHES{{.*}}s_barrier"
