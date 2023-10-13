// RUN: iree-opt --iree-codegen-hoist-vector-extract-insert-slice %s | FileCheck %s

func.func @hoist_unrolled_vector_for_mma() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xf32>
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<3456x2048xf16>
  memref.assume_alignment %0, 64 : memref<3456x2048xf16>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<2048x1024xf16>
  memref.assume_alignment %1, 64 : memref<2048x1024xf16>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<3456x1024xf32>
  memref.assume_alignment %2, 64 : memref<3456x1024xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %3 = gpu.thread_id  x
  %4 = gpu.thread_id  y
  %5 = affine.apply affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 8) * 128)>()[%workgroup_id_x, %4]
  %6 = affine.apply affine_map<()[s0, s1] -> (s0 * 128 + s1 * 32 - (s0 floordiv 8) * 1024)>()[%workgroup_id_x, %3]
  %7 = scf.for %arg0 = %c0 to %c2048 step %c64 iter_args(%arg1 = %cst_0) -> (vector<32x32xf32>) {
    %26 = vector.transfer_read %0[%5, %arg0], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %27 = affine.apply affine_map<(d0) -> (d0 + 16)>(%arg0)
    %28 = vector.transfer_read %0[%5, %27], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %29 = affine.apply affine_map<(d0) -> (d0 + 32)>(%arg0)
    %30 = vector.transfer_read %0[%5, %29], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %31 = affine.apply affine_map<(d0) -> (d0 + 48)>(%arg0)
    %32 = vector.transfer_read %0[%5, %31], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %33 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
    %34 = vector.transfer_read %0[%33, %arg0], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %35 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
    %36 = affine.apply affine_map<(d0) -> (d0 + 16)>(%arg0)
    %37 = vector.transfer_read %0[%35, %36], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %38 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
    %39 = affine.apply affine_map<(d0) -> (d0 + 32)>(%arg0)
    %40 = vector.transfer_read %0[%38, %39], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %41 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
    %42 = affine.apply affine_map<(d0) -> (d0 + 48)>(%arg0)
    %43 = vector.transfer_read %0[%41, %42], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %44 = vector.transfer_read %1[%arg0, %6], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %45 = affine.apply affine_map<(d0) -> (d0 + 16)>(%arg0)
    %46 = vector.transfer_read %1[%45, %6], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %47 = affine.apply affine_map<(d0) -> (d0 + 32)>(%arg0)
    %48 = vector.transfer_read %1[%47, %6], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %49 = affine.apply affine_map<(d0) -> (d0 + 48)>(%arg0)
    %50 = vector.transfer_read %1[%49, %6], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %51 = affine.apply affine_map<(d0) -> (d0 + 16)>(%6)
    %52 = vector.transfer_read %1[%arg0, %51], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %53 = affine.apply affine_map<(d0) -> (d0 + 16)>(%6)
    %54 = affine.apply affine_map<(d0) -> (d0 + 16)>(%arg0)
    %55 = vector.transfer_read %1[%54, %53], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %56 = affine.apply affine_map<(d0) -> (d0 + 16)>(%6)
    %57 = affine.apply affine_map<(d0) -> (d0 + 32)>(%arg0)
    %58 = vector.transfer_read %1[%57, %56], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %59 = affine.apply affine_map<(d0) -> (d0 + 16)>(%6)
    %60 = affine.apply affine_map<(d0) -> (d0 + 48)>(%arg0)
    %61 = vector.transfer_read %1[%60, %59], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %62 = vector.extract_strided_slice %44 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %63 = vector.extract_strided_slice %arg1 {offsets = [0, 0], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %64 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %26, %62, %63 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %65 = vector.extract_strided_slice %44 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %66 = vector.extract_strided_slice %arg1 {offsets = [0, 8], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %67 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %26, %65, %66 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %68 = vector.extract_strided_slice %52 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %69 = vector.extract_strided_slice %arg1 {offsets = [0, 16], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %70 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %26, %68, %69 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %71 = vector.extract_strided_slice %52 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %72 = vector.extract_strided_slice %arg1 {offsets = [0, 24], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %73 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %26, %71, %72 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %74 = vector.extract_strided_slice %44 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %75 = vector.extract_strided_slice %arg1 {offsets = [16, 0], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %76 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %34, %74, %75 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %77 = vector.extract_strided_slice %44 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %78 = vector.extract_strided_slice %arg1 {offsets = [16, 8], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %79 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %34, %77, %78 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %80 = vector.extract_strided_slice %52 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %81 = vector.extract_strided_slice %arg1 {offsets = [16, 16], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %82 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %34, %80, %81 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %83 = vector.extract_strided_slice %52 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %84 = vector.extract_strided_slice %arg1 {offsets = [16, 24], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %85 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %34, %83, %84 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %86 = vector.extract_strided_slice %46 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %87 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %28, %86, %64 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %88 = vector.extract_strided_slice %46 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %89 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %28, %88, %67 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %90 = vector.extract_strided_slice %55 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %91 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %28, %90, %70 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %92 = vector.extract_strided_slice %55 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %93 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %28, %92, %73 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %94 = vector.extract_strided_slice %46 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %95 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %37, %94, %76 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %96 = vector.extract_strided_slice %46 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %97 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %37, %96, %79 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %98 = vector.extract_strided_slice %55 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %99 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %37, %98, %82 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %100 = vector.extract_strided_slice %55 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %101 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %37, %100, %85 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %102 = vector.extract_strided_slice %48 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %103 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %30, %102, %87 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %104 = vector.extract_strided_slice %48 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %105 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %30, %104, %89 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %106 = vector.extract_strided_slice %58 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %107 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %30, %106, %91 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %108 = vector.extract_strided_slice %58 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %109 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %30, %108, %93 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %110 = vector.extract_strided_slice %48 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %111 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %110, %95 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %112 = vector.extract_strided_slice %48 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %113 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %112, %97 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %114 = vector.extract_strided_slice %58 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %115 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %114, %99 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %116 = vector.extract_strided_slice %58 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %117 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %116, %101 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %118 = vector.extract_strided_slice %50 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %119 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %32, %118, %103 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %120 = vector.extract_strided_slice %50 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %121 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %32, %120, %105 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %122 = vector.extract_strided_slice %61 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %123 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %32, %122, %107 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %124 = vector.extract_strided_slice %61 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %125 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %32, %124, %109 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %126 = vector.extract_strided_slice %50 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %127 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %126, %111 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %128 = vector.extract_strided_slice %50 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %129 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %128, %113 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %130 = vector.extract_strided_slice %61 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %131 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %130, %115 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %132 = vector.extract_strided_slice %61 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %133 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %132, %117 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %134 = vector.insert_strided_slice %119, %cst_0 {offsets = [0, 0], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %135 = vector.insert_strided_slice %121, %134 {offsets = [0, 8], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %136 = vector.insert_strided_slice %123, %135 {offsets = [0, 16], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %137 = vector.insert_strided_slice %125, %136 {offsets = [0, 24], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %138 = vector.insert_strided_slice %127, %137 {offsets = [16, 0], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %139 = vector.insert_strided_slice %129, %138 {offsets = [16, 8], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %140 = vector.insert_strided_slice %131, %139 {offsets = [16, 16], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %141 = vector.insert_strided_slice %133, %140 {offsets = [16, 24], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    scf.yield %141 : vector<32x32xf32>
  }
  %8 = vector.extract_strided_slice %7 {offsets = [0, 0], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  vector.transfer_write %8, %2[%5, %6] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %9 = vector.extract_strided_slice %7 {offsets = [0, 8], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %10 = affine.apply affine_map<(d0) -> (d0 + 8)>(%6)
  vector.transfer_write %9, %2[%5, %10] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %11 = vector.extract_strided_slice %7 {offsets = [0, 16], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %12 = affine.apply affine_map<(d0) -> (d0 + 16)>(%6)
  vector.transfer_write %11, %2[%5, %12] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %13 = vector.extract_strided_slice %7 {offsets = [0, 24], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %14 = affine.apply affine_map<(d0) -> (d0 + 24)>(%6)
  vector.transfer_write %13, %2[%5, %14] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %15 = vector.extract_strided_slice %7 {offsets = [16, 0], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %16 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
  vector.transfer_write %15, %2[%16, %6] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %17 = vector.extract_strided_slice %7 {offsets = [16, 8], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %18 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
  %19 = affine.apply affine_map<(d0) -> (d0 + 8)>(%6)
  vector.transfer_write %17, %2[%18, %19] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %20 = vector.extract_strided_slice %7 {offsets = [16, 16], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %21 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
  %22 = affine.apply affine_map<(d0) -> (d0 + 16)>(%6)
  vector.transfer_write %20, %2[%21, %22] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %23 = vector.extract_strided_slice %7 {offsets = [16, 24], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %24 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
  %25 = affine.apply affine_map<(d0) -> (d0 + 24)>(%6)
  vector.transfer_write %23, %2[%24, %25] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  return
}
// CHECK-LABEL: func.func @hoist_unrolled_vector_for_mma
// CHECK:         %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<16x8xf32>
// CHECK:         %[[RES:.+]]:8 = scf.for {{.+}} iter_args(%[[ARG0:.+]] = %[[INIT]]
// CHECK-NOT:       vector.extract_strided_slice %[[ARG0]]
// vector.insert_strided_slice ops are folded to their consumers.
// CHECK-NOT:     vector.insert_strided_slice
// CHECK:        vector.transfer_write %[[RES]]#0
// CHECK:        vector.transfer_write %[[RES]]#1
// CHECK:        vector.transfer_write %[[RES]]#2
// CHECK:        vector.transfer_write %[[RES]]#3
// CHECK:        vector.transfer_write %[[RES]]#4
// CHECK:        vector.transfer_write %[[RES]]#5
// CHECK:        vector.transfer_write %[[RES]]#6
// CHECK:        vector.transfer_write %[[RES]]#7
