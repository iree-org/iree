// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-lowering,canonicalize,cse))" --split-input-file %s | FileCheck %s

func.func @main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32() {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c131072 = arith.constant 131072 : index
    %c0 = arith.constant 0 : index
    %c671872 = arith.constant 671872 : index
    %c17449088 = arith.constant 17449088 : index
    %alloc = memref.alloc() : memref<512x20xf16, #gpu.address_space<workgroup>>
    %alloc_0 = memref.alloc() : memref<1x32x20xf16, #gpu.address_space<workgroup>>
    %thread_id_x = gpu.thread_id  x upper_bound 256
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c131072) flags("ReadOnly|Indirect") : memref<130x130x16xf16, strided<[2080, 16, 1], offset: ?>, #gpu.address_space<global>>
    %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<130x130x16xf16, strided<[2080, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
    memref.assume_alignment %1, 64 : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<512x144xf16, #gpu.address_space<global>>
    %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<512x144xf16, #gpu.address_space<global>> to memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>
    memref.assume_alignment %3, 64 : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>
    %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x16xf16, #gpu.address_space<global>>
    %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<32x16xf16, #gpu.address_space<global>> to memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>
    memref.assume_alignment %5, 64 : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>
    %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c671872) flags(Indirect) : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1], offset: ?>, #gpu.address_space<global>>
    %7 = amdgpu.fat_raw_buffer_cast %6 resetOffset : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
    memref.assume_alignment %7, 64 : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
    %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c17449088) flags(Indirect) : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1], offset: ?>, #gpu.address_space<global>>
    %9 = amdgpu.fat_raw_buffer_cast %8 resetOffset : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
    memref.assume_alignment %9, 64 : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
    %subview = memref.subview %alloc_0[0, 0, 0] [1, 32, 16] [1, 1, 1] : memref<1x32x20xf16, #gpu.address_space<workgroup>> to memref<1x32x16xf16, strided<[640, 20, 1]>, #gpu.address_space<workgroup>>
    %subview_1 = memref.subview %alloc[0, 0] [512, 16] [1, 1] : memref<512x20xf16, #gpu.address_space<workgroup>> to memref<512x16xf16, strided<[20, 1]>, #gpu.address_space<workgroup>>
    %10 = arith.floordivsi %thread_id_x, %c64 : index
    %11 = arith.muli %10, %c8 overflow<nsw> : index
    %12 = gpu.lane_id upper_bound 64
    %13 = arith.remsi %12, %c64 : index
    %14 = arith.cmpi slt, %13, %c0 : index
    %15 = arith.addi %13, %c64 : index
    %16 = arith.select %14, %15, %13 : index
    %17 = arith.divsi %16, %c16 : index
    %18 = arith.remsi %12, %c16 : index
    %19 = arith.cmpi slt, %18, %c0 : index
    %20 = arith.addi %18, %c16 : index
    %21 = arith.select %19, %20, %18 : index
    %22 = arith.muli %17, %c4 : index
    %expand_shape = memref.expand_shape %subview_1 [[0, 1], [2, 3]] output_shape [32, 16, 1, 16] : memref<512x16xf16, strided<[20, 1]>, #gpu.address_space<workgroup>> into memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>
    %expand_shape_2 = memref.expand_shape %subview [[0], [1, 2], [3, 4]] output_shape [1, 2, 16, 1, 16] : memref<1x32x16xf16, strided<[640, 20, 1]>, #gpu.address_space<workgroup>> into memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>
    gpu.barrier
    scf.for %arg0 = %c0 to %c8 step %c1 {
      %23 = vector.transfer_read %expand_shape_2[%c0, %c0, %21, %c0, %22], %cst {in_bounds = [true, true, true, true]} : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<2x1x1x4xf16>
      %24 = vector.transfer_read %expand_shape[%11, %21, %c0, %22], %cst {in_bounds = [true, true, true, true]} : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<8x1x1x4xf16>
      %25 = "foo.bla"(%23, %24) : (vector<2x1x1x4xf16>, vector<8x1x1x4xf16>) -> vector<1x2x8x4x1xf32>
    }
    return
}
