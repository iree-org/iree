

==================================
State just before the start of the lowering of the transfer ops
==================================
func.func @main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32() {
  %cst = arith.constant dense<0.000000e+00> : vector<8x1x1x2x4xf16>
  %0 = ub.poison : vector<1x2x4x8x1xf16>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<2x4x8x1xf16>
  %1 = ub.poison : vector<2x4x8x1xf32>
  %2 = ub.poison : vector<4x8x1xf32>
  %cst_1 = arith.constant dense<0.000000e+00> : vector<1x2x4x8x1xf32>
  %3 = ub.poison : vector<1x2x8x4x1xf32>
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c48 = arith.constant 48 : index
  %c768 = arith.constant 768 : index
  %c512 = arith.constant 512 : index
  %c256 = arith.constant 256 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %cst_2 = arith.constant dense<0.000000e+00> : vector<2x8x4x1xf32>
  %c8 = arith.constant 8 : index
  %cst_3 = arith.constant dense<0.000000e+00> : vector<1x2x8x4x1xf32>
  %cst_4 = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %c671872 = arith.constant 671872 : index
  %c17449088 = arith.constant 17449088 : index
  %alloc = memref.alloc() : memref<512x20xf16, #gpu.address_space<workgroup>>
  %alloc_5 = memref.alloc() : memref<1x32x20xf16, #gpu.address_space<workgroup>>
  %alloca = memref.alloca() : memref<1x2x4x8x1xf32, #gpu.address_space<private>>
  %thread_id_x = gpu.thread_id  x upper_bound 256
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c131072) flags("ReadOnly|Indirect") : memref<130x130x16xf16, strided<[2080, 16, 1], offset: ?>, #gpu.address_space<global>>
  %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<130x130x16xf16, strided<[2080, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %5, 64 : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  %6 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<512x144xf16, #gpu.address_space<global>>
  %7 = amdgpu.fat_raw_buffer_cast %6 resetOffset : memref<512x144xf16, #gpu.address_space<global>> to memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %7, 64 : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>
  %8 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x16xf16, #gpu.address_space<global>>
  %9 = amdgpu.fat_raw_buffer_cast %8 resetOffset : memref<32x16xf16, #gpu.address_space<global>> to memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %9, 64 : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>
  %10 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c671872) flags(Indirect) : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1], offset: ?>, #gpu.address_space<global>>
  %11 = amdgpu.fat_raw_buffer_cast %10 resetOffset : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %11, 64 : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  %12 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(4) alignment(64) offset(%c17449088) flags(Indirect) : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1], offset: ?>, #gpu.address_space<global>>
  %13 = amdgpu.fat_raw_buffer_cast %12 resetOffset : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %13, 64 : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  %subview = memref.subview %alloc_5[0, 0, 0] [1, 32, 16] [1, 1, 1] : memref<1x32x20xf16, #gpu.address_space<workgroup>> to memref<1x32x16xf16, strided<[640, 20, 1]>, #gpu.address_space<workgroup>>
  %subview_6 = memref.subview %alloc[0, 0] [512, 16] [1, 1] : memref<512x20xf16, #gpu.address_space<workgroup>> to memref<512x16xf16, strided<[20, 1]>, #gpu.address_space<workgroup>>
  %14 = arith.floordivsi %thread_id_x, %c64 : index
  %15 = arith.muli %14, %c8 overflow<nsw> : index
  %16 = gpu.lane_id upper_bound 64
  %17 = arith.remsi %16, %c64 : index
  %18 = arith.cmpi slt, %17, %c0 : index
  %19 = arith.addi %17, %c64 : index
  %20 = arith.select %18, %19, %17 : index
  %21 = arith.divsi %20, %c16 : index
  %22 = arith.remsi %16, %c16 : index
  %23 = arith.cmpi slt, %22, %c0 : index
  %24 = arith.addi %22, %c16 : index
  %25 = arith.select %23, %24, %22 : index
  %26 = arith.muli %21, %c4 : index
  %27 = arith.muli %14, %c64 overflow<nsw> : index
  %28 = arith.addi %16, %27 : index
  %29 = arith.floordivsi %28, %c8 : index
  %30 = arith.remsi %28, %c8 : index
  %31 = arith.cmpi slt, %30, %c0 : index
  %32 = arith.addi %30, %c8 : index
  %33 = arith.select %31, %32, %30 : index
  %34 = arith.floordivsi %28, %c2 : index
  %35 = arith.remsi %28, %c2 : index
  %36 = arith.cmpi slt, %35, %c0 : index
  %37 = arith.addi %35, %c2 : index
  %38 = arith.select %36, %37, %35 : index
  %39 = arith.addi %28, %c256 : index
  %40 = arith.floordivsi %39, %c2 : index
  %41 = arith.remsi %39, %c2 : index
  %42 = arith.cmpi slt, %41, %c0 : index
  %43 = arith.addi %41, %c2 : index
  %44 = arith.select %42, %43, %41 : index
  %45 = arith.addi %28, %c512 : index
  %46 = arith.floordivsi %45, %c2 : index
  %47 = arith.remsi %45, %c2 : index
  %48 = arith.cmpi slt, %47, %c0 : index
  %49 = arith.addi %47, %c2 : index
  %50 = arith.select %48, %49, %47 : index
  %51 = arith.addi %28, %c768 : index
  %52 = arith.floordivsi %51, %c2 : index
  %53 = arith.remsi %51, %c2 : index
  %54 = arith.cmpi slt, %53, %c0 : index
  %55 = arith.addi %53, %c2 : index
  %56 = arith.select %54, %55, %53 : index
  %expand_shape = memref.expand_shape %subview_6 [[0, 1], [2, 3]] output_shape [32, 16, 1, 16] : memref<512x16xf16, strided<[20, 1]>, #gpu.address_space<workgroup>> into memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>
  %expand_shape_7 = memref.expand_shape %subview [[0], [1, 2], [3, 4]] output_shape [1, 2, 16, 1, 16] : memref<1x32x16xf16, strided<[640, 20, 1]>, #gpu.address_space<workgroup>> into memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>
  %57 = arith.muli %33, %c2 overflow<nsw> : index
  %58 = arith.floordivsi %57, %c48 : index
  %59 = arith.remsi %57, %c48 : index
  %60 = arith.cmpi slt, %59, %c0 : index
  %61 = arith.addi %59, %c48 : index
  %62 = arith.select %60, %61, %59 : index
  %63 = arith.divsi %62, %c16 : index
  %64 = arith.remsi %57, %c16 : index
  %65 = arith.cmpi slt, %64, %c0 : index
  %66 = arith.addi %64, %c16 : index
  %67 = arith.select %65, %66, %64 : index
  %68 = arith.muli %38, %c8 overflow<nsw> : index
  %69 = arith.muli %44, %c8 overflow<nsw> : index
  %70 = arith.muli %50, %c8 overflow<nsw> : index
  %71 = arith.muli %56, %c8 overflow<nsw> : index
  %subview_8 = memref.subview %alloca[0, 0, 0, 0, 0] [1, 2, 4, 8, 1] [1, 1, 1, 1, 1] : memref<1x2x4x8x1xf32, #gpu.address_space<private>> to memref<1x2x4x8xf32, strided<[64, 32, 8, 1]>, #gpu.address_space<private>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  gpu.barrier
  %72 = arith.muli %workgroup_id_y, %c128 overflow<nsw> : index
  %73 = arith.addi %29, %72 : index
  %74 = arith.muli %workgroup_id_x, %c32 overflow<nsw> : index
  %75 = arith.addi %73, %74 : index
  %76 = arith.floordivsi %75, %c128 : index
  %77 = arith.remsi %75, %c128 : index
  %78 = arith.cmpi slt, %77, %c0 : index
  %79 = arith.addi %77, %c128 : index
  %80 = arith.select %78, %79, %77 : index
  %81 = arith.addi %58, %76 : index
  %82 = arith.addi %63, %80 : index
  %83 = vector.transfer_read %5[%81, %82, %67], %cst_4 {in_bounds = [true]} : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf16>
  %84 = vector.transfer_read %7[%34, %68], %cst_4 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
  %85 = vector.transfer_read %7[%40, %69], %cst_4 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
  %86 = vector.transfer_read %7[%46, %70], %cst_4 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
  %87 = vector.transfer_read %7[%52, %71], %cst_4 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
  vector.transfer_write %83, %alloc_5[%c0, %29, %57] {in_bounds = [true]} : vector<2xf16>, memref<1x32x20xf16, #gpu.address_space<workgroup>>
  vector.transfer_write %84, %alloc[%34, %68] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
  vector.transfer_write %85, %alloc[%40, %69] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
  vector.transfer_write %86, %alloc[%46, %70] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
  vector.transfer_write %87, %alloc[%52, %71] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
  %88 = scf.for %arg0 = %c0 to %c8 step %c1 iter_args(%arg1 = %cst_3) -> (vector<1x2x8x4x1xf32>) {
    %500 = arith.addi %arg0, %c1 : index
    %501 = arith.muli %500, %c16 overflow<nsw> : index
    %502 = arith.addi %501, %57 : index
    %503 = arith.floordivsi %502, %c48 : index
    %504 = arith.remsi %502, %c48 : index
    %505 = arith.cmpi slt, %504, %c0 : index
    %506 = arith.addi %504, %c48 : index
    %507 = arith.select %505, %506, %504 : index
    %508 = arith.divsi %507, %c16 : index
    %509 = arith.remsi %502, %c16 : index
    %510 = arith.cmpi slt, %509, %c0 : index
    %511 = arith.addi %509, %c16 : index
    %512 = arith.select %510, %511, %509 : index
    %513 = arith.addi %503, %76 : index
    %514 = arith.addi %508, %80 : index
    %515 = vector.transfer_read %5[%513, %514, %512], %cst_4 {in_bounds = [true]} : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf16>
    %516 = arith.addi %501, %68 : index
    %517 = vector.transfer_read %7[%34, %516], %cst_4 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    %518 = arith.addi %501, %69 : index
    %519 = vector.transfer_read %7[%40, %518], %cst_4 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    %520 = arith.addi %501, %70 : index
    %521 = vector.transfer_read %7[%46, %520], %cst_4 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    %522 = arith.addi %501, %71 : index
    %523 = vector.transfer_read %7[%52, %522], %cst_4 {in_bounds = [true]} : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    gpu.barrier
    %524 = vector.transfer_read %expand_shape_7[%c0, %c0, %25, %c0, %26], %cst_4 {in_bounds = [true, true, true, true]} : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<2x1x1x4xf16>
    %525 = vector.transfer_read %expand_shape[%15, %25, %c0, %26], %cst_4 {in_bounds = [true, true, true, true]} : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<8x1x1x4xf16>
    %526 = vector.extract_strided_slice %525 {offsets = [0, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
    %527 = vector.transpose %526, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %528 = vector.extract_strided_slice %525 {offsets = [1, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
    %529 = vector.transpose %528, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %530 = vector.extract_strided_slice %525 {offsets = [2, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
    %531 = vector.transpose %530, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %532 = vector.extract_strided_slice %525 {offsets = [3, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
    %533 = vector.transpose %532, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %534 = vector.extract_strided_slice %525 {offsets = [4, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
    %535 = vector.transpose %534, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %536 = vector.extract_strided_slice %525 {offsets = [5, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
    %537 = vector.transpose %536, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %538 = vector.extract_strided_slice %525 {offsets = [6, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
    %539 = vector.transpose %538, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %540 = vector.extract_strided_slice %525 {offsets = [7, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
    %541 = vector.transpose %540, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %542 = vector.extract %524[0, 0] : vector<1x4xf16> from vector<2x1x1x4xf16>
    %543 = vector.extract %527[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %544 = vector.extract %arg1[0, 0, 0] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %545 = vector.shape_cast %542 : vector<1x4xf16> to vector<4xf16>
    %546 = vector.shape_cast %543 : vector<1x4xf16> to vector<4xf16>
    %547 = vector.shape_cast %544 : vector<4x1xf32> to vector<4xf32>
    %548 = amdgpu.mfma %545 * %546 + %547 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %549 = vector.shape_cast %548 : vector<4xf32> to vector<4x1xf32>
    %550 = vector.extract %529[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %551 = vector.extract %arg1[0, 0, 1] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %552 = vector.shape_cast %550 : vector<1x4xf16> to vector<4xf16>
    %553 = vector.shape_cast %551 : vector<4x1xf32> to vector<4xf32>
    %554 = amdgpu.mfma %545 * %552 + %553 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %555 = vector.shape_cast %554 : vector<4xf32> to vector<4x1xf32>
    %556 = vector.extract %531[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %557 = vector.extract %arg1[0, 0, 2] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %558 = vector.shape_cast %556 : vector<1x4xf16> to vector<4xf16>
    %559 = vector.shape_cast %557 : vector<4x1xf32> to vector<4xf32>
    %560 = amdgpu.mfma %545 * %558 + %559 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %561 = vector.shape_cast %560 : vector<4xf32> to vector<4x1xf32>
    %562 = vector.extract %533[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %563 = vector.extract %arg1[0, 0, 3] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %564 = vector.shape_cast %562 : vector<1x4xf16> to vector<4xf16>
    %565 = vector.shape_cast %563 : vector<4x1xf32> to vector<4xf32>
    %566 = amdgpu.mfma %545 * %564 + %565 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %567 = vector.shape_cast %566 : vector<4xf32> to vector<4x1xf32>
    %568 = vector.extract %535[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %569 = vector.extract %arg1[0, 0, 4] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %570 = vector.shape_cast %568 : vector<1x4xf16> to vector<4xf16>
    %571 = vector.shape_cast %569 : vector<4x1xf32> to vector<4xf32>
    %572 = amdgpu.mfma %545 * %570 + %571 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %573 = vector.shape_cast %572 : vector<4xf32> to vector<4x1xf32>
    %574 = vector.extract %537[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %575 = vector.extract %arg1[0, 0, 5] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %576 = vector.shape_cast %574 : vector<1x4xf16> to vector<4xf16>
    %577 = vector.shape_cast %575 : vector<4x1xf32> to vector<4xf32>
    %578 = amdgpu.mfma %545 * %576 + %577 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %579 = vector.shape_cast %578 : vector<4xf32> to vector<4x1xf32>
    %580 = vector.extract %539[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %581 = vector.extract %arg1[0, 0, 6] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %582 = vector.shape_cast %580 : vector<1x4xf16> to vector<4xf16>
    %583 = vector.shape_cast %581 : vector<4x1xf32> to vector<4xf32>
    %584 = amdgpu.mfma %545 * %582 + %583 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %585 = vector.shape_cast %584 : vector<4xf32> to vector<4x1xf32>
    %586 = vector.extract %541[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %587 = vector.extract %arg1[0, 0, 7] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %588 = vector.shape_cast %586 : vector<1x4xf16> to vector<4xf16>
    %589 = vector.shape_cast %587 : vector<4x1xf32> to vector<4xf32>
    %590 = amdgpu.mfma %545 * %588 + %589 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %591 = vector.shape_cast %590 : vector<4xf32> to vector<4x1xf32>
    %592 = vector.extract %524[1, 0] : vector<1x4xf16> from vector<2x1x1x4xf16>
    %593 = vector.extract %arg1[0, 1, 0] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %594 = vector.shape_cast %592 : vector<1x4xf16> to vector<4xf16>
    %595 = vector.shape_cast %593 : vector<4x1xf32> to vector<4xf32>
    %596 = amdgpu.mfma %594 * %546 + %595 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %597 = vector.shape_cast %596 : vector<4xf32> to vector<4x1xf32>
    %598 = vector.extract %arg1[0, 1, 1] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %599 = vector.shape_cast %598 : vector<4x1xf32> to vector<4xf32>
    %600 = amdgpu.mfma %594 * %552 + %599 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %601 = vector.shape_cast %600 : vector<4xf32> to vector<4x1xf32>
    %602 = vector.extract %arg1[0, 1, 2] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %603 = vector.shape_cast %602 : vector<4x1xf32> to vector<4xf32>
    %604 = amdgpu.mfma %594 * %558 + %603 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %605 = vector.shape_cast %604 : vector<4xf32> to vector<4x1xf32>
    %606 = vector.extract %arg1[0, 1, 3] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %607 = vector.shape_cast %606 : vector<4x1xf32> to vector<4xf32>
    %608 = amdgpu.mfma %594 * %564 + %607 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %609 = vector.shape_cast %608 : vector<4xf32> to vector<4x1xf32>
    %610 = vector.extract %arg1[0, 1, 4] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %611 = vector.shape_cast %610 : vector<4x1xf32> to vector<4xf32>
    %612 = amdgpu.mfma %594 * %570 + %611 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %613 = vector.shape_cast %612 : vector<4xf32> to vector<4x1xf32>
    %614 = vector.extract %arg1[0, 1, 5] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %615 = vector.shape_cast %614 : vector<4x1xf32> to vector<4xf32>
    %616 = amdgpu.mfma %594 * %576 + %615 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %617 = vector.shape_cast %616 : vector<4xf32> to vector<4x1xf32>
    %618 = vector.extract %arg1[0, 1, 6] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %619 = vector.shape_cast %618 : vector<4x1xf32> to vector<4xf32>
    %620 = amdgpu.mfma %594 * %582 + %619 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %621 = vector.shape_cast %620 : vector<4xf32> to vector<4x1xf32>
    %622 = vector.extract %arg1[0, 1, 7] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %623 = vector.shape_cast %622 : vector<4x1xf32> to vector<4xf32>
    %624 = amdgpu.mfma %594 * %588 + %623 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %625 = vector.shape_cast %624 : vector<4xf32> to vector<4x1xf32>
    %626 = vector.insert_strided_slice %549, %cst_2 {offsets = [0, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %627 = vector.insert_strided_slice %555, %626 {offsets = [0, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %628 = vector.insert_strided_slice %561, %627 {offsets = [0, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %629 = vector.insert_strided_slice %567, %628 {offsets = [0, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %630 = vector.insert_strided_slice %573, %629 {offsets = [0, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %631 = vector.insert_strided_slice %579, %630 {offsets = [0, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %632 = vector.insert_strided_slice %585, %631 {offsets = [0, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %633 = vector.insert_strided_slice %591, %632 {offsets = [0, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %634 = vector.insert_strided_slice %597, %633 {offsets = [1, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %635 = vector.insert_strided_slice %601, %634 {offsets = [1, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %636 = vector.insert_strided_slice %605, %635 {offsets = [1, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %637 = vector.insert_strided_slice %609, %636 {offsets = [1, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %638 = vector.insert_strided_slice %613, %637 {offsets = [1, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %639 = vector.insert_strided_slice %617, %638 {offsets = [1, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %640 = vector.insert_strided_slice %621, %639 {offsets = [1, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %641 = vector.insert_strided_slice %625, %640 {offsets = [1, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %642 = vector.insert %641, %3 [0] : vector<2x8x4x1xf32> into vector<1x2x8x4x1xf32>
    gpu.barrier
    vector.transfer_write %515, %alloc_5[%c0, %29, %57] {in_bounds = [true]} : vector<2xf16>, memref<1x32x20xf16, #gpu.address_space<workgroup>>
    vector.transfer_write %517, %alloc[%34, %68] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
    vector.transfer_write %519, %alloc[%40, %69] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
    vector.transfer_write %521, %alloc[%46, %70] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
    vector.transfer_write %523, %alloc[%52, %71] {in_bounds = [true]} : vector<8xf16>, memref<512x20xf16, #gpu.address_space<workgroup>>
    scf.yield %642 : vector<1x2x8x4x1xf32>
  }
  gpu.barrier
  %89 = vector.transfer_read %expand_shape_7[%c0, %c0, %25, %c0, %26], %cst_4 {in_bounds = [true, true, true, true]} : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<2x1x1x4xf16>
  %90 = vector.transfer_read %expand_shape[%15, %25, %c0, %26], %cst_4 {in_bounds = [true, true, true, true]} : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<8x1x1x4xf16>
  %91 = vector.extract_strided_slice %90 {offsets = [0, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
  %92 = vector.transpose %91, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %93 = vector.extract_strided_slice %90 {offsets = [1, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
  %94 = vector.transpose %93, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %95 = vector.extract_strided_slice %90 {offsets = [2, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
  %96 = vector.transpose %95, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %97 = vector.extract_strided_slice %90 {offsets = [3, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
  %98 = vector.transpose %97, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %99 = vector.extract_strided_slice %90 {offsets = [4, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
  %100 = vector.transpose %99, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %101 = vector.extract_strided_slice %90 {offsets = [5, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
  %102 = vector.transpose %101, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %103 = vector.extract_strided_slice %90 {offsets = [6, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
  %104 = vector.transpose %103, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %105 = vector.extract_strided_slice %90 {offsets = [7, 0, 0, 0], sizes = [1, 1, 1, 4], strides = [1, 1, 1, 1]} : vector<8x1x1x4xf16> to vector<1x1x1x4xf16>
  %106 = vector.transpose %105, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %107 = vector.extract %89[0, 0] : vector<1x4xf16> from vector<2x1x1x4xf16>
  %108 = vector.extract %92[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %109 = vector.extract %88[0, 0, 0] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %110 = vector.shape_cast %107 : vector<1x4xf16> to vector<4xf16>
  %111 = vector.shape_cast %108 : vector<1x4xf16> to vector<4xf16>
  %112 = vector.shape_cast %109 : vector<4x1xf32> to vector<4xf32>
  %113 = amdgpu.mfma %110 * %111 + %112 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %114 = vector.shape_cast %113 : vector<4xf32> to vector<4x1xf32>
  %115 = vector.extract %94[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %116 = vector.extract %88[0, 0, 1] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %117 = vector.shape_cast %115 : vector<1x4xf16> to vector<4xf16>
  %118 = vector.shape_cast %116 : vector<4x1xf32> to vector<4xf32>
  %119 = amdgpu.mfma %110 * %117 + %118 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %120 = vector.shape_cast %119 : vector<4xf32> to vector<4x1xf32>
  %121 = vector.extract %96[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %122 = vector.extract %88[0, 0, 2] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %123 = vector.shape_cast %121 : vector<1x4xf16> to vector<4xf16>
  %124 = vector.shape_cast %122 : vector<4x1xf32> to vector<4xf32>
  %125 = amdgpu.mfma %110 * %123 + %124 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %126 = vector.shape_cast %125 : vector<4xf32> to vector<4x1xf32>
  %127 = vector.extract %98[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %128 = vector.extract %88[0, 0, 3] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %129 = vector.shape_cast %127 : vector<1x4xf16> to vector<4xf16>
  %130 = vector.shape_cast %128 : vector<4x1xf32> to vector<4xf32>
  %131 = amdgpu.mfma %110 * %129 + %130 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %132 = vector.shape_cast %131 : vector<4xf32> to vector<4x1xf32>
  %133 = vector.extract %100[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %134 = vector.extract %88[0, 0, 4] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %135 = vector.shape_cast %133 : vector<1x4xf16> to vector<4xf16>
  %136 = vector.shape_cast %134 : vector<4x1xf32> to vector<4xf32>
  %137 = amdgpu.mfma %110 * %135 + %136 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %138 = vector.shape_cast %137 : vector<4xf32> to vector<4x1xf32>
  %139 = vector.extract %102[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %140 = vector.extract %88[0, 0, 5] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %141 = vector.shape_cast %139 : vector<1x4xf16> to vector<4xf16>
  %142 = vector.shape_cast %140 : vector<4x1xf32> to vector<4xf32>
  %143 = amdgpu.mfma %110 * %141 + %142 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %144 = vector.shape_cast %143 : vector<4xf32> to vector<4x1xf32>
  %145 = vector.extract %104[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %146 = vector.extract %88[0, 0, 6] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %147 = vector.shape_cast %145 : vector<1x4xf16> to vector<4xf16>
  %148 = vector.shape_cast %146 : vector<4x1xf32> to vector<4xf32>
  %149 = amdgpu.mfma %110 * %147 + %148 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %150 = vector.shape_cast %149 : vector<4xf32> to vector<4x1xf32>
  %151 = vector.extract %106[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %152 = vector.extract %88[0, 0, 7] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %153 = vector.shape_cast %151 : vector<1x4xf16> to vector<4xf16>
  %154 = vector.shape_cast %152 : vector<4x1xf32> to vector<4xf32>
  %155 = amdgpu.mfma %110 * %153 + %154 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %156 = vector.shape_cast %155 : vector<4xf32> to vector<4x1xf32>
  %157 = vector.extract %89[1, 0] : vector<1x4xf16> from vector<2x1x1x4xf16>
  %158 = vector.extract %88[0, 1, 0] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %159 = vector.shape_cast %157 : vector<1x4xf16> to vector<4xf16>
  %160 = vector.shape_cast %158 : vector<4x1xf32> to vector<4xf32>
  %161 = amdgpu.mfma %159 * %111 + %160 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %162 = vector.shape_cast %161 : vector<4xf32> to vector<4x1xf32>
  %163 = vector.extract %88[0, 1, 1] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %164 = vector.shape_cast %163 : vector<4x1xf32> to vector<4xf32>
  %165 = amdgpu.mfma %159 * %117 + %164 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %166 = vector.shape_cast %165 : vector<4xf32> to vector<4x1xf32>
  %167 = vector.extract %88[0, 1, 2] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %168 = vector.shape_cast %167 : vector<4x1xf32> to vector<4xf32>
  %169 = amdgpu.mfma %159 * %123 + %168 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %170 = vector.shape_cast %169 : vector<4xf32> to vector<4x1xf32>
  %171 = vector.extract %88[0, 1, 3] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %172 = vector.shape_cast %171 : vector<4x1xf32> to vector<4xf32>
  %173 = amdgpu.mfma %159 * %129 + %172 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %174 = vector.shape_cast %173 : vector<4xf32> to vector<4x1xf32>
  %175 = vector.extract %88[0, 1, 4] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %176 = vector.shape_cast %175 : vector<4x1xf32> to vector<4xf32>
  %177 = amdgpu.mfma %159 * %135 + %176 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %178 = vector.shape_cast %177 : vector<4xf32> to vector<4x1xf32>
  %179 = vector.extract %88[0, 1, 5] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %180 = vector.shape_cast %179 : vector<4x1xf32> to vector<4xf32>
  %181 = amdgpu.mfma %159 * %141 + %180 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %182 = vector.shape_cast %181 : vector<4xf32> to vector<4x1xf32>
  %183 = vector.extract %88[0, 1, 6] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %184 = vector.shape_cast %183 : vector<4x1xf32> to vector<4xf32>
  %185 = amdgpu.mfma %159 * %147 + %184 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %186 = vector.shape_cast %185 : vector<4xf32> to vector<4x1xf32>
  %187 = vector.extract %88[0, 1, 7] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %188 = vector.shape_cast %187 : vector<4x1xf32> to vector<4xf32>
  %189 = amdgpu.mfma %159 * %153 + %188 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %190 = vector.shape_cast %189 : vector<4xf32> to vector<4x1xf32>
  %191 = vector.insert_strided_slice %114, %cst_2 {offsets = [0, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %192 = vector.insert_strided_slice %120, %191 {offsets = [0, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %193 = vector.insert_strided_slice %126, %192 {offsets = [0, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %194 = vector.insert_strided_slice %132, %193 {offsets = [0, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %195 = vector.insert_strided_slice %138, %194 {offsets = [0, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %196 = vector.insert_strided_slice %144, %195 {offsets = [0, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %197 = vector.insert_strided_slice %150, %196 {offsets = [0, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %198 = vector.insert_strided_slice %156, %197 {offsets = [0, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %199 = vector.insert_strided_slice %162, %198 {offsets = [1, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %200 = vector.insert_strided_slice %166, %199 {offsets = [1, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %201 = vector.insert_strided_slice %170, %200 {offsets = [1, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %202 = vector.insert_strided_slice %174, %201 {offsets = [1, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %203 = vector.insert_strided_slice %178, %202 {offsets = [1, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %204 = vector.insert_strided_slice %182, %203 {offsets = [1, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %205 = vector.insert_strided_slice %186, %204 {offsets = [1, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %206 = vector.insert_strided_slice %190, %205 {offsets = [1, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
  %207 = vector.insert %206, %3 [0] : vector<2x8x4x1xf32> into vector<1x2x8x4x1xf32>
  %208 = vector.extract_strided_slice %207 {offsets = [0, 0, 0, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %209 = vector.transpose %208, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %210 = vector.insert_strided_slice %209, %cst_1 {offsets = [0, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %211 = vector.extract_strided_slice %207 {offsets = [0, 0, 1, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %212 = vector.transpose %211, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %213 = vector.insert_strided_slice %212, %210 {offsets = [0, 0, 0, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %214 = vector.extract_strided_slice %207 {offsets = [0, 0, 2, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %215 = vector.transpose %214, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %216 = vector.insert_strided_slice %215, %213 {offsets = [0, 0, 0, 2, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %217 = vector.extract_strided_slice %207 {offsets = [0, 0, 3, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %218 = vector.transpose %217, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %219 = vector.insert_strided_slice %218, %216 {offsets = [0, 0, 0, 3, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %220 = vector.extract_strided_slice %207 {offsets = [0, 0, 4, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %221 = vector.transpose %220, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %222 = vector.insert_strided_slice %221, %219 {offsets = [0, 0, 0, 4, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %223 = vector.extract_strided_slice %207 {offsets = [0, 0, 5, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %224 = vector.transpose %223, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %225 = vector.insert_strided_slice %224, %222 {offsets = [0, 0, 0, 5, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %226 = vector.extract_strided_slice %207 {offsets = [0, 0, 6, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %227 = vector.transpose %226, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %228 = vector.insert_strided_slice %227, %225 {offsets = [0, 0, 0, 6, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %229 = vector.extract_strided_slice %207 {offsets = [0, 0, 7, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %230 = vector.transpose %229, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %231 = vector.insert_strided_slice %230, %228 {offsets = [0, 0, 0, 7, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %232 = vector.extract_strided_slice %207 {offsets = [0, 0, 0, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %233 = vector.transpose %232, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %234 = vector.insert_strided_slice %233, %231 {offsets = [0, 0, 1, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %235 = vector.extract_strided_slice %207 {offsets = [0, 0, 1, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %236 = vector.transpose %235, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %237 = vector.insert_strided_slice %236, %234 {offsets = [0, 0, 1, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %238 = vector.extract_strided_slice %207 {offsets = [0, 0, 2, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %239 = vector.transpose %238, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %240 = vector.insert_strided_slice %239, %237 {offsets = [0, 0, 1, 2, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %241 = vector.extract_strided_slice %207 {offsets = [0, 0, 3, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %242 = vector.transpose %241, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %243 = vector.insert_strided_slice %242, %240 {offsets = [0, 0, 1, 3, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %244 = vector.extract_strided_slice %207 {offsets = [0, 0, 4, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %245 = vector.transpose %244, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %246 = vector.insert_strided_slice %245, %243 {offsets = [0, 0, 1, 4, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %247 = vector.extract_strided_slice %207 {offsets = [0, 0, 5, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %248 = vector.transpose %247, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %249 = vector.insert_strided_slice %248, %246 {offsets = [0, 0, 1, 5, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %250 = vector.extract_strided_slice %207 {offsets = [0, 0, 6, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %251 = vector.transpose %250, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %252 = vector.insert_strided_slice %251, %249 {offsets = [0, 0, 1, 6, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %253 = vector.extract_strided_slice %207 {offsets = [0, 0, 7, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %254 = vector.transpose %253, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %255 = vector.insert_strided_slice %254, %252 {offsets = [0, 0, 1, 7, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %256 = vector.extract_strided_slice %207 {offsets = [0, 0, 0, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %257 = vector.transpose %256, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %258 = vector.insert_strided_slice %257, %255 {offsets = [0, 0, 2, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %259 = vector.extract_strided_slice %207 {offsets = [0, 0, 1, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %260 = vector.transpose %259, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %261 = vector.insert_strided_slice %260, %258 {offsets = [0, 0, 2, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %262 = vector.extract_strided_slice %207 {offsets = [0, 0, 2, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %263 = vector.transpose %262, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %264 = vector.insert_strided_slice %263, %261 {offsets = [0, 0, 2, 2, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %265 = vector.extract_strided_slice %207 {offsets = [0, 0, 3, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %266 = vector.transpose %265, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %267 = vector.insert_strided_slice %266, %264 {offsets = [0, 0, 2, 3, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %268 = vector.extract_strided_slice %207 {offsets = [0, 0, 4, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %269 = vector.transpose %268, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %270 = vector.insert_strided_slice %269, %267 {offsets = [0, 0, 2, 4, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %271 = vector.extract_strided_slice %207 {offsets = [0, 0, 5, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %272 = vector.transpose %271, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %273 = vector.insert_strided_slice %272, %270 {offsets = [0, 0, 2, 5, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %274 = vector.extract_strided_slice %207 {offsets = [0, 0, 6, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %275 = vector.transpose %274, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %276 = vector.insert_strided_slice %275, %273 {offsets = [0, 0, 2, 6, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %277 = vector.extract_strided_slice %207 {offsets = [0, 0, 7, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %278 = vector.transpose %277, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %279 = vector.insert_strided_slice %278, %276 {offsets = [0, 0, 2, 7, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %280 = vector.extract_strided_slice %207 {offsets = [0, 0, 0, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %281 = vector.transpose %280, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %282 = vector.insert_strided_slice %281, %279 {offsets = [0, 0, 3, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %283 = vector.extract_strided_slice %207 {offsets = [0, 0, 1, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %284 = vector.transpose %283, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %285 = vector.insert_strided_slice %284, %282 {offsets = [0, 0, 3, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %286 = vector.extract_strided_slice %207 {offsets = [0, 0, 2, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %287 = vector.transpose %286, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %288 = vector.insert_strided_slice %287, %285 {offsets = [0, 0, 3, 2, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %289 = vector.extract_strided_slice %207 {offsets = [0, 0, 3, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %290 = vector.transpose %289, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %291 = vector.insert_strided_slice %290, %288 {offsets = [0, 0, 3, 3, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %292 = vector.extract_strided_slice %207 {offsets = [0, 0, 4, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %293 = vector.transpose %292, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %294 = vector.insert_strided_slice %293, %291 {offsets = [0, 0, 3, 4, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %295 = vector.extract_strided_slice %207 {offsets = [0, 0, 5, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %296 = vector.transpose %295, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %297 = vector.insert_strided_slice %296, %294 {offsets = [0, 0, 3, 5, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %298 = vector.extract_strided_slice %207 {offsets = [0, 0, 6, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %299 = vector.transpose %298, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %300 = vector.insert_strided_slice %299, %297 {offsets = [0, 0, 3, 6, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %301 = vector.extract_strided_slice %207 {offsets = [0, 0, 7, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %302 = vector.transpose %301, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %303 = vector.insert_strided_slice %302, %300 {offsets = [0, 0, 3, 7, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %304 = vector.extract_strided_slice %207 {offsets = [0, 1, 0, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %305 = vector.transpose %304, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %306 = vector.insert_strided_slice %305, %303 {offsets = [0, 1, 0, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %307 = vector.extract_strided_slice %207 {offsets = [0, 1, 1, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %308 = vector.transpose %307, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %309 = vector.insert_strided_slice %308, %306 {offsets = [0, 1, 0, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %310 = vector.extract_strided_slice %207 {offsets = [0, 1, 2, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %311 = vector.transpose %310, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %312 = vector.insert_strided_slice %311, %309 {offsets = [0, 1, 0, 2, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %313 = vector.extract_strided_slice %207 {offsets = [0, 1, 3, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %314 = vector.transpose %313, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %315 = vector.insert_strided_slice %314, %312 {offsets = [0, 1, 0, 3, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %316 = vector.extract_strided_slice %207 {offsets = [0, 1, 4, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %317 = vector.transpose %316, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %318 = vector.insert_strided_slice %317, %315 {offsets = [0, 1, 0, 4, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %319 = vector.extract_strided_slice %207 {offsets = [0, 1, 5, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %320 = vector.transpose %319, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %321 = vector.insert_strided_slice %320, %318 {offsets = [0, 1, 0, 5, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %322 = vector.extract_strided_slice %207 {offsets = [0, 1, 6, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %323 = vector.transpose %322, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %324 = vector.insert_strided_slice %323, %321 {offsets = [0, 1, 0, 6, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %325 = vector.extract_strided_slice %207 {offsets = [0, 1, 7, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %326 = vector.transpose %325, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %327 = vector.insert_strided_slice %326, %324 {offsets = [0, 1, 0, 7, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %328 = vector.extract_strided_slice %207 {offsets = [0, 1, 0, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %329 = vector.transpose %328, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %330 = vector.insert_strided_slice %329, %327 {offsets = [0, 1, 1, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %331 = vector.extract_strided_slice %207 {offsets = [0, 1, 1, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %332 = vector.transpose %331, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %333 = vector.insert_strided_slice %332, %330 {offsets = [0, 1, 1, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %334 = vector.extract_strided_slice %207 {offsets = [0, 1, 2, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %335 = vector.transpose %334, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %336 = vector.insert_strided_slice %335, %333 {offsets = [0, 1, 1, 2, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %337 = vector.extract_strided_slice %207 {offsets = [0, 1, 3, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %338 = vector.transpose %337, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %339 = vector.insert_strided_slice %338, %336 {offsets = [0, 1, 1, 3, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %340 = vector.extract_strided_slice %207 {offsets = [0, 1, 4, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %341 = vector.transpose %340, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %342 = vector.insert_strided_slice %341, %339 {offsets = [0, 1, 1, 4, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %343 = vector.extract_strided_slice %207 {offsets = [0, 1, 5, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %344 = vector.transpose %343, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %345 = vector.insert_strided_slice %344, %342 {offsets = [0, 1, 1, 5, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %346 = vector.extract_strided_slice %207 {offsets = [0, 1, 6, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %347 = vector.transpose %346, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %348 = vector.insert_strided_slice %347, %345 {offsets = [0, 1, 1, 6, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %349 = vector.extract_strided_slice %207 {offsets = [0, 1, 7, 1, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %350 = vector.transpose %349, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %351 = vector.insert_strided_slice %350, %348 {offsets = [0, 1, 1, 7, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %352 = vector.extract_strided_slice %207 {offsets = [0, 1, 0, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %353 = vector.transpose %352, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %354 = vector.insert_strided_slice %353, %351 {offsets = [0, 1, 2, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %355 = vector.extract_strided_slice %207 {offsets = [0, 1, 1, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %356 = vector.transpose %355, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %357 = vector.insert_strided_slice %356, %354 {offsets = [0, 1, 2, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %358 = vector.extract_strided_slice %207 {offsets = [0, 1, 2, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %359 = vector.transpose %358, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %360 = vector.insert_strided_slice %359, %357 {offsets = [0, 1, 2, 2, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %361 = vector.extract_strided_slice %207 {offsets = [0, 1, 3, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %362 = vector.transpose %361, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %363 = vector.insert_strided_slice %362, %360 {offsets = [0, 1, 2, 3, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %364 = vector.extract_strided_slice %207 {offsets = [0, 1, 4, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %365 = vector.transpose %364, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %366 = vector.insert_strided_slice %365, %363 {offsets = [0, 1, 2, 4, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %367 = vector.extract_strided_slice %207 {offsets = [0, 1, 5, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %368 = vector.transpose %367, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %369 = vector.insert_strided_slice %368, %366 {offsets = [0, 1, 2, 5, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %370 = vector.extract_strided_slice %207 {offsets = [0, 1, 6, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %371 = vector.transpose %370, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %372 = vector.insert_strided_slice %371, %369 {offsets = [0, 1, 2, 6, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %373 = vector.extract_strided_slice %207 {offsets = [0, 1, 7, 2, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %374 = vector.transpose %373, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %375 = vector.insert_strided_slice %374, %372 {offsets = [0, 1, 2, 7, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %376 = vector.extract_strided_slice %207 {offsets = [0, 1, 0, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %377 = vector.transpose %376, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %378 = vector.insert_strided_slice %377, %375 {offsets = [0, 1, 3, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %379 = vector.extract_strided_slice %207 {offsets = [0, 1, 1, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %380 = vector.transpose %379, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %381 = vector.insert_strided_slice %380, %378 {offsets = [0, 1, 3, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %382 = vector.extract_strided_slice %207 {offsets = [0, 1, 2, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %383 = vector.transpose %382, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %384 = vector.insert_strided_slice %383, %381 {offsets = [0, 1, 3, 2, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %385 = vector.extract_strided_slice %207 {offsets = [0, 1, 3, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %386 = vector.transpose %385, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %387 = vector.insert_strided_slice %386, %384 {offsets = [0, 1, 3, 3, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %388 = vector.extract_strided_slice %207 {offsets = [0, 1, 4, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %389 = vector.transpose %388, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %390 = vector.insert_strided_slice %389, %387 {offsets = [0, 1, 3, 4, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %391 = vector.extract_strided_slice %207 {offsets = [0, 1, 5, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %392 = vector.transpose %391, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %393 = vector.insert_strided_slice %392, %390 {offsets = [0, 1, 3, 5, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %394 = vector.extract_strided_slice %207 {offsets = [0, 1, 6, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %395 = vector.transpose %394, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %396 = vector.insert_strided_slice %395, %393 {offsets = [0, 1, 3, 6, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %397 = vector.extract_strided_slice %207 {offsets = [0, 1, 7, 3, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x8x4x1xf32> to vector<1x1x1x1x1xf32>
  %398 = vector.transpose %397, [0, 1, 3, 2, 4] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x1xf32>
  %399 = vector.insert_strided_slice %398, %396 {offsets = [0, 1, 3, 7, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x1xf32> into vector<1x2x4x8x1xf32>
  %400 = vector.extract %399[0] : vector<2x4x8x1xf32> from vector<1x2x4x8x1xf32>
  %401 = vector.shape_cast %400 : vector<2x4x8x1xf32> to vector<2x4x8xf32>
  vector.transfer_write %401, %subview_8[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<2x4x8xf32>, memref<1x2x4x8xf32, strided<[64, 32, 8, 1]>, #gpu.address_space<private>>
  %402 = vector.transfer_read %9[%15, %25], %cst_4 {in_bounds = [true, true]} : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8x1xf16>
  %403 = arith.extf %402 : vector<8x1xf16> to vector<8x1xf32>
  %404 = vector.insert %403, %2 [0] : vector<8x1xf32> into vector<4x8x1xf32>
  %405 = vector.insert %403, %404 [1] : vector<8x1xf32> into vector<4x8x1xf32>
  %406 = vector.insert %403, %405 [2] : vector<8x1xf32> into vector<4x8x1xf32>
  %407 = vector.insert %403, %406 [3] : vector<8x1xf32> into vector<4x8x1xf32>
  %408 = vector.insert %407, %1 [0] : vector<4x8x1xf32> into vector<2x4x8x1xf32>
  %409 = vector.insert %407, %408 [1] : vector<4x8x1xf32> into vector<2x4x8x1xf32>
  %410 = vector.extract_strided_slice %400 {offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %411 = vector.extract_strided_slice %409 {offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %412 = arith.addf %410, %411 : vector<1x1x8x1xf32>
  %413 = vector.extract_strided_slice %400 {offsets = [0, 1, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %414 = vector.extract_strided_slice %409 {offsets = [0, 1, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %415 = arith.addf %413, %414 : vector<1x1x8x1xf32>
  %416 = vector.extract_strided_slice %400 {offsets = [0, 2, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %417 = vector.extract_strided_slice %409 {offsets = [0, 2, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %418 = arith.addf %416, %417 : vector<1x1x8x1xf32>
  %419 = vector.extract_strided_slice %400 {offsets = [0, 3, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %420 = vector.extract_strided_slice %409 {offsets = [0, 3, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %421 = arith.addf %419, %420 : vector<1x1x8x1xf32>
  %422 = vector.extract_strided_slice %400 {offsets = [1, 0, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %423 = vector.extract_strided_slice %409 {offsets = [1, 0, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %424 = arith.addf %422, %423 : vector<1x1x8x1xf32>
  %425 = vector.extract_strided_slice %400 {offsets = [1, 1, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %426 = vector.extract_strided_slice %409 {offsets = [1, 1, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %427 = arith.addf %425, %426 : vector<1x1x8x1xf32>
  %428 = vector.extract_strided_slice %400 {offsets = [1, 2, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %429 = vector.extract_strided_slice %409 {offsets = [1, 2, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %430 = arith.addf %428, %429 : vector<1x1x8x1xf32>
  %431 = vector.extract_strided_slice %400 {offsets = [1, 3, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %432 = vector.extract_strided_slice %409 {offsets = [1, 3, 0, 0], sizes = [1, 1, 8, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf32> to vector<1x1x8x1xf32>
  %433 = arith.addf %431, %432 : vector<1x1x8x1xf32>
  %434 = arith.truncf %412 : vector<1x1x8x1xf32> to vector<1x1x8x1xf16>
  %435 = vector.insert_strided_slice %434, %cst_0 {offsets = [0, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x8x1xf16> into vector<2x4x8x1xf16>
  %436 = arith.truncf %415 : vector<1x1x8x1xf32> to vector<1x1x8x1xf16>
  %437 = vector.insert_strided_slice %436, %435 {offsets = [0, 1, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x8x1xf16> into vector<2x4x8x1xf16>
  %438 = arith.truncf %418 : vector<1x1x8x1xf32> to vector<1x1x8x1xf16>
  %439 = vector.insert_strided_slice %438, %437 {offsets = [0, 2, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x8x1xf16> into vector<2x4x8x1xf16>
  %440 = arith.truncf %421 : vector<1x1x8x1xf32> to vector<1x1x8x1xf16>
  %441 = vector.insert_strided_slice %440, %439 {offsets = [0, 3, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x8x1xf16> into vector<2x4x8x1xf16>
  %442 = arith.truncf %424 : vector<1x1x8x1xf32> to vector<1x1x8x1xf16>
  %443 = vector.insert_strided_slice %442, %441 {offsets = [1, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x8x1xf16> into vector<2x4x8x1xf16>
  %444 = arith.truncf %427 : vector<1x1x8x1xf32> to vector<1x1x8x1xf16>
  %445 = vector.insert_strided_slice %444, %443 {offsets = [1, 1, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x8x1xf16> into vector<2x4x8x1xf16>
  %446 = arith.truncf %430 : vector<1x1x8x1xf32> to vector<1x1x8x1xf16>
  %447 = vector.insert_strided_slice %446, %445 {offsets = [1, 2, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x8x1xf16> into vector<2x4x8x1xf16>
  %448 = arith.truncf %433 : vector<1x1x8x1xf32> to vector<1x1x8x1xf16>
  %449 = vector.insert_strided_slice %448, %447 {offsets = [1, 3, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x8x1xf16> into vector<2x4x8x1xf16>
  %450 = vector.insert %449, %0 [0] : vector<2x4x8x1xf16> into vector<1x2x4x8x1xf16>
  %451 = arith.muli %workgroup_id_x, %c2 overflow<nsw> : index
  vector.transfer_write %449, %11[%workgroup_id_y, %451, %26, %15, %25] {in_bounds = [true, true, true, true]} : vector<2x4x8x1xf16>, memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  %452 = vector.extract_strided_slice %450 {offsets = [0, 0, 0, 0, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %453 = vector.transpose %452, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %454 = vector.insert_strided_slice %453, %cst {offsets = [0, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %455 = vector.extract_strided_slice %450 {offsets = [0, 1, 0, 0, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %456 = vector.transpose %455, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %457 = vector.insert_strided_slice %456, %454 {offsets = [0, 0, 0, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %458 = vector.extract_strided_slice %450 {offsets = [0, 0, 0, 1, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %459 = vector.transpose %458, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %460 = vector.insert_strided_slice %459, %457 {offsets = [1, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %461 = vector.extract_strided_slice %450 {offsets = [0, 1, 0, 1, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %462 = vector.transpose %461, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %463 = vector.insert_strided_slice %462, %460 {offsets = [1, 0, 0, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %464 = vector.extract_strided_slice %450 {offsets = [0, 0, 0, 2, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %465 = vector.transpose %464, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %466 = vector.insert_strided_slice %465, %463 {offsets = [2, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %467 = vector.extract_strided_slice %450 {offsets = [0, 1, 0, 2, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %468 = vector.transpose %467, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %469 = vector.insert_strided_slice %468, %466 {offsets = [2, 0, 0, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %470 = vector.extract_strided_slice %450 {offsets = [0, 0, 0, 3, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %471 = vector.transpose %470, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %472 = vector.insert_strided_slice %471, %469 {offsets = [3, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %473 = vector.extract_strided_slice %450 {offsets = [0, 1, 0, 3, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %474 = vector.transpose %473, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %475 = vector.insert_strided_slice %474, %472 {offsets = [3, 0, 0, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %476 = vector.extract_strided_slice %450 {offsets = [0, 0, 0, 4, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %477 = vector.transpose %476, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %478 = vector.insert_strided_slice %477, %475 {offsets = [4, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %479 = vector.extract_strided_slice %450 {offsets = [0, 1, 0, 4, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %480 = vector.transpose %479, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %481 = vector.insert_strided_slice %480, %478 {offsets = [4, 0, 0, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %482 = vector.extract_strided_slice %450 {offsets = [0, 0, 0, 5, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %483 = vector.transpose %482, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %484 = vector.insert_strided_slice %483, %481 {offsets = [5, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %485 = vector.extract_strided_slice %450 {offsets = [0, 1, 0, 5, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %486 = vector.transpose %485, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %487 = vector.insert_strided_slice %486, %484 {offsets = [5, 0, 0, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %488 = vector.extract_strided_slice %450 {offsets = [0, 0, 0, 6, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %489 = vector.transpose %488, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %490 = vector.insert_strided_slice %489, %487 {offsets = [6, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %491 = vector.extract_strided_slice %450 {offsets = [0, 1, 0, 6, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %492 = vector.transpose %491, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %493 = vector.insert_strided_slice %492, %490 {offsets = [6, 0, 0, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %494 = vector.extract_strided_slice %450 {offsets = [0, 0, 0, 7, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %495 = vector.transpose %494, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %496 = vector.insert_strided_slice %495, %493 {offsets = [7, 0, 0, 0, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  %497 = vector.extract_strided_slice %450 {offsets = [0, 1, 0, 7, 0], sizes = [1, 1, 4, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<1x2x4x8x1xf16> to vector<1x1x4x1x1xf16>
  %498 = vector.transpose %497, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %499 = vector.insert_strided_slice %498, %496 {offsets = [7, 0, 0, 1, 0], strides = [1, 1, 1, 1, 1]} : vector<1x1x1x1x4xf16> into vector<8x1x1x2x4xf16>
  vector.transfer_write %499, %13[%15, %25, %workgroup_id_y, %451, %26] {in_bounds = [true, true, true, true, true]} : vector<8x1x1x2x4xf16>, memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  gpu.barrier
  memref.dealloc %alloc_5 : memref<1x32x20xf16, #gpu.address_space<workgroup>>
  memref.dealloc %alloc : memref<512x20xf16, #gpu.address_space<workgroup>>
  return
}


==================================
State just before the start of flattening process
==================================
func.func @main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32() {
  %cst = arith.constant dense<0.000000e+00> : vector<2x4x8x1xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<8x1xf16>
  %c3 = arith.constant 3 : index
  %cst_1 = arith.constant dense<0.000000e+00> : vector<2x4x8x1xf16>
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c48 = arith.constant 48 : index
  %c768 = arith.constant 768 : index
  %c512 = arith.constant 512 : index
  %c256 = arith.constant 256 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %cst_2 = arith.constant dense<0.000000e+00> : vector<2x8x4x1xf32>
  %c8 = arith.constant 8 : index
  %cst_3 = arith.constant dense<0.000000e+00> : vector<1x2x8x4x1xf32>
  %c1 = arith.constant 1 : index
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %c671872 = arith.constant 671872 : index
  %c17449088 = arith.constant 17449088 : index
  %alloc = memref.alloc() : memref<512x20xf16, #gpu.address_space<workgroup>>
  %alloc_4 = memref.alloc() : memref<1x32x20xf16, #gpu.address_space<workgroup>>
  %alloca = memref.alloca() : memref<1x2x4x8x1xf32, #gpu.address_space<private>>
  %thread_id_x = gpu.thread_id  x upper_bound 256
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c131072) flags("ReadOnly|Indirect") : memref<130x130x16xf16, strided<[2080, 16, 1], offset: ?>, #gpu.address_space<global>>
  %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<130x130x16xf16, strided<[2080, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %1, 64 : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<512x144xf16, #gpu.address_space<global>>
  %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<512x144xf16, #gpu.address_space<global>> to memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %3, 64 : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x16xf16, #gpu.address_space<global>>
  %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<32x16xf16, #gpu.address_space<global>> to memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %5, 64 : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>
  %6 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c671872) flags(Indirect) : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1], offset: ?>, #gpu.address_space<global>>
  %7 = amdgpu.fat_raw_buffer_cast %6 resetOffset : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %7, 64 : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  %8 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(4) alignment(64) offset(%c17449088) flags(Indirect) : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1], offset: ?>, #gpu.address_space<global>>
  %9 = amdgpu.fat_raw_buffer_cast %8 resetOffset : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  memref.assume_alignment %9, 64 : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
  %subview = memref.subview %alloc_4[0, 0, 0] [1, 32, 16] [1, 1, 1] : memref<1x32x20xf16, #gpu.address_space<workgroup>> to memref<1x32x16xf16, strided<[640, 20, 1]>, #gpu.address_space<workgroup>>
  %subview_5 = memref.subview %alloc[0, 0] [512, 16] [1, 1] : memref<512x20xf16, #gpu.address_space<workgroup>> to memref<512x16xf16, strided<[20, 1]>, #gpu.address_space<workgroup>>
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
  %23 = arith.muli %10, %c64 overflow<nsw> : index
  %24 = arith.addi %12, %23 : index
  %25 = arith.floordivsi %24, %c8 : index
  %26 = arith.remsi %24, %c8 : index
  %27 = arith.cmpi slt, %26, %c0 : index
  %28 = arith.addi %26, %c8 : index
  %29 = arith.select %27, %28, %26 : index
  %30 = arith.floordivsi %24, %c2 : index
  %31 = arith.remsi %24, %c2 : index
  %32 = arith.cmpi slt, %31, %c0 : index
  %33 = arith.addi %31, %c2 : index
  %34 = arith.select %32, %33, %31 : index
  %35 = arith.addi %24, %c256 : index
  %36 = arith.floordivsi %35, %c2 : index
  %37 = arith.remsi %35, %c2 : index
  %38 = arith.cmpi slt, %37, %c0 : index
  %39 = arith.addi %37, %c2 : index
  %40 = arith.select %38, %39, %37 : index
  %41 = arith.addi %24, %c512 : index
  %42 = arith.floordivsi %41, %c2 : index
  %43 = arith.remsi %41, %c2 : index
  %44 = arith.cmpi slt, %43, %c0 : index
  %45 = arith.addi %43, %c2 : index
  %46 = arith.select %44, %45, %43 : index
  %47 = arith.addi %24, %c768 : index
  %48 = arith.floordivsi %47, %c2 : index
  %49 = arith.remsi %47, %c2 : index
  %50 = arith.cmpi slt, %49, %c0 : index
  %51 = arith.addi %49, %c2 : index
  %52 = arith.select %50, %51, %49 : index
  %expand_shape = memref.expand_shape %subview_5 [[0, 1], [2, 3]] output_shape [32, 16, 1, 16] : memref<512x16xf16, strided<[20, 1]>, #gpu.address_space<workgroup>> into memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>
  %expand_shape_6 = memref.expand_shape %subview [[0], [1, 2], [3, 4]] output_shape [1, 2, 16, 1, 16] : memref<1x32x16xf16, strided<[640, 20, 1]>, #gpu.address_space<workgroup>> into memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>
  %53 = arith.muli %29, %c2 overflow<nsw> : index
  %54 = arith.floordivsi %53, %c48 : index
  %55 = arith.remsi %53, %c48 : index
  %56 = arith.cmpi slt, %55, %c0 : index
  %57 = arith.addi %55, %c48 : index
  %58 = arith.select %56, %57, %55 : index
  %59 = arith.divsi %58, %c16 : index
  %60 = arith.remsi %53, %c16 : index
  %61 = arith.cmpi slt, %60, %c0 : index
  %62 = arith.addi %60, %c16 : index
  %63 = arith.select %61, %62, %60 : index
  %64 = arith.muli %34, %c8 overflow<nsw> : index
  %65 = arith.muli %40, %c8 overflow<nsw> : index
  %66 = arith.muli %46, %c8 overflow<nsw> : index
  %67 = arith.muli %52, %c8 overflow<nsw> : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  gpu.barrier
  %68 = arith.muli %workgroup_id_y, %c128 overflow<nsw> : index
  %69 = arith.addi %25, %68 : index
  %70 = arith.muli %workgroup_id_x, %c32 overflow<nsw> : index
  %71 = arith.addi %69, %70 : index
  %72 = arith.floordivsi %71, %c128 : index
  %73 = arith.remsi %71, %c128 : index
  %74 = arith.cmpi slt, %73, %c0 : index
  %75 = arith.addi %73, %c128 : index
  %76 = arith.select %74, %75, %73 : index
  %77 = arith.addi %54, %72 : index
  %78 = arith.addi %59, %76 : index
  %79 = vector.load %1[%77, %78, %63] : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf16>
  %80 = vector.load %3[%30, %64] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
  %81 = vector.load %3[%36, %65] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
  %82 = vector.load %3[%42, %66] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
  %83 = vector.load %3[%48, %67] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
  vector.store %79, %alloc_4[%c0, %25, %53] : memref<1x32x20xf16, #gpu.address_space<workgroup>>, vector<2xf16>
  vector.store %80, %alloc[%30, %64] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  vector.store %81, %alloc[%36, %65] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  vector.store %82, %alloc[%42, %66] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  vector.store %83, %alloc[%48, %67] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
  %84 = scf.for %arg0 = %c0 to %c8 step %c1 iter_args(%arg1 = %cst_3) -> (vector<1x2x8x4x1xf32>) {
    %598 = arith.addi %arg0, %c1 : index
    %599 = arith.muli %598, %c16 overflow<nsw> : index
    %600 = arith.addi %599, %53 : index
    %601 = arith.floordivsi %600, %c48 : index
    %602 = arith.remsi %600, %c48 : index
    %603 = arith.cmpi slt, %602, %c0 : index
    %604 = arith.addi %602, %c48 : index
    %605 = arith.select %603, %604, %602 : index
    %606 = arith.divsi %605, %c16 : index
    %607 = arith.remsi %600, %c16 : index
    %608 = arith.cmpi slt, %607, %c0 : index
    %609 = arith.addi %607, %c16 : index
    %610 = arith.select %608, %609, %607 : index
    %611 = arith.addi %601, %72 : index
    %612 = arith.addi %606, %76 : index
    %613 = vector.load %1[%611, %612, %610] : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf16>
    %614 = arith.addi %599, %64 : index
    %615 = vector.load %3[%30, %614] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    %616 = arith.addi %599, %65 : index
    %617 = vector.load %3[%36, %616] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    %618 = arith.addi %599, %66 : index
    %619 = vector.load %3[%42, %618] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    %620 = arith.addi %599, %67 : index
    %621 = vector.load %3[%48, %620] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    gpu.barrier
    %622 = vector.load %expand_shape_6[%c0, %c0, %21, %c0, %22] : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %623 = vector.load %expand_shape_6[%c0, %c1, %21, %c0, %22] : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %624 = vector.load %expand_shape[%11, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %625 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
    %626 = vector.load %expand_shape[%625, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %627 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%11]
    %628 = vector.load %expand_shape[%627, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %629 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%11]
    %630 = vector.load %expand_shape[%629, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %631 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%11]
    %632 = vector.load %expand_shape[%631, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %633 = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%11]
    %634 = vector.load %expand_shape[%633, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %635 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%11]
    %636 = vector.load %expand_shape[%635, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %637 = affine.apply affine_map<()[s0] -> (s0 + 7)>()[%11]
    %638 = vector.load %expand_shape[%637, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %639 = vector.shape_cast %624 : vector<4xf16> to vector<1x1x1x4xf16>
    %640 = vector.transpose %639, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %641 = vector.shape_cast %626 : vector<4xf16> to vector<1x1x1x4xf16>
    %642 = vector.transpose %641, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %643 = vector.shape_cast %628 : vector<4xf16> to vector<1x1x1x4xf16>
    %644 = vector.transpose %643, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %645 = vector.shape_cast %630 : vector<4xf16> to vector<1x1x1x4xf16>
    %646 = vector.transpose %645, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %647 = vector.shape_cast %632 : vector<4xf16> to vector<1x1x1x4xf16>
    %648 = vector.transpose %647, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %649 = vector.shape_cast %634 : vector<4xf16> to vector<1x1x1x4xf16>
    %650 = vector.transpose %649, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %651 = vector.shape_cast %636 : vector<4xf16> to vector<1x1x1x4xf16>
    %652 = vector.transpose %651, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %653 = vector.shape_cast %638 : vector<4xf16> to vector<1x1x1x4xf16>
    %654 = vector.transpose %653, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
    %655 = vector.extract %640[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %656 = vector.extract %arg1[0, 0, 0] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %657 = vector.shape_cast %655 : vector<1x4xf16> to vector<4xf16>
    %658 = vector.shape_cast %656 : vector<4x1xf32> to vector<4xf32>
    %659 = amdgpu.mfma %622 * %657 + %658 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %660 = vector.shape_cast %659 : vector<4xf32> to vector<4x1xf32>
    %661 = vector.extract %642[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %662 = vector.extract %arg1[0, 0, 1] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %663 = vector.shape_cast %661 : vector<1x4xf16> to vector<4xf16>
    %664 = vector.shape_cast %662 : vector<4x1xf32> to vector<4xf32>
    %665 = amdgpu.mfma %622 * %663 + %664 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %666 = vector.shape_cast %665 : vector<4xf32> to vector<4x1xf32>
    %667 = vector.extract %644[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %668 = vector.extract %arg1[0, 0, 2] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %669 = vector.shape_cast %667 : vector<1x4xf16> to vector<4xf16>
    %670 = vector.shape_cast %668 : vector<4x1xf32> to vector<4xf32>
    %671 = amdgpu.mfma %622 * %669 + %670 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %672 = vector.shape_cast %671 : vector<4xf32> to vector<4x1xf32>
    %673 = vector.extract %646[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %674 = vector.extract %arg1[0, 0, 3] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %675 = vector.shape_cast %673 : vector<1x4xf16> to vector<4xf16>
    %676 = vector.shape_cast %674 : vector<4x1xf32> to vector<4xf32>
    %677 = amdgpu.mfma %622 * %675 + %676 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %678 = vector.shape_cast %677 : vector<4xf32> to vector<4x1xf32>
    %679 = vector.extract %648[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %680 = vector.extract %arg1[0, 0, 4] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %681 = vector.shape_cast %679 : vector<1x4xf16> to vector<4xf16>
    %682 = vector.shape_cast %680 : vector<4x1xf32> to vector<4xf32>
    %683 = amdgpu.mfma %622 * %681 + %682 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %684 = vector.shape_cast %683 : vector<4xf32> to vector<4x1xf32>
    %685 = vector.extract %650[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %686 = vector.extract %arg1[0, 0, 5] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %687 = vector.shape_cast %685 : vector<1x4xf16> to vector<4xf16>
    %688 = vector.shape_cast %686 : vector<4x1xf32> to vector<4xf32>
    %689 = amdgpu.mfma %622 * %687 + %688 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %690 = vector.shape_cast %689 : vector<4xf32> to vector<4x1xf32>
    %691 = vector.extract %652[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %692 = vector.extract %arg1[0, 0, 6] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %693 = vector.shape_cast %691 : vector<1x4xf16> to vector<4xf16>
    %694 = vector.shape_cast %692 : vector<4x1xf32> to vector<4xf32>
    %695 = amdgpu.mfma %622 * %693 + %694 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %696 = vector.shape_cast %695 : vector<4xf32> to vector<4x1xf32>
    %697 = vector.extract %654[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
    %698 = vector.extract %arg1[0, 0, 7] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %699 = vector.shape_cast %697 : vector<1x4xf16> to vector<4xf16>
    %700 = vector.shape_cast %698 : vector<4x1xf32> to vector<4xf32>
    %701 = amdgpu.mfma %622 * %699 + %700 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %702 = vector.shape_cast %701 : vector<4xf32> to vector<4x1xf32>
    %703 = vector.extract %arg1[0, 1, 0] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %704 = vector.shape_cast %703 : vector<4x1xf32> to vector<4xf32>
    %705 = amdgpu.mfma %623 * %657 + %704 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %706 = vector.shape_cast %705 : vector<4xf32> to vector<4x1xf32>
    %707 = vector.extract %arg1[0, 1, 1] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %708 = vector.shape_cast %707 : vector<4x1xf32> to vector<4xf32>
    %709 = amdgpu.mfma %623 * %663 + %708 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %710 = vector.shape_cast %709 : vector<4xf32> to vector<4x1xf32>
    %711 = vector.extract %arg1[0, 1, 2] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %712 = vector.shape_cast %711 : vector<4x1xf32> to vector<4xf32>
    %713 = amdgpu.mfma %623 * %669 + %712 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %714 = vector.shape_cast %713 : vector<4xf32> to vector<4x1xf32>
    %715 = vector.extract %arg1[0, 1, 3] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %716 = vector.shape_cast %715 : vector<4x1xf32> to vector<4xf32>
    %717 = amdgpu.mfma %623 * %675 + %716 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %718 = vector.shape_cast %717 : vector<4xf32> to vector<4x1xf32>
    %719 = vector.extract %arg1[0, 1, 4] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %720 = vector.shape_cast %719 : vector<4x1xf32> to vector<4xf32>
    %721 = amdgpu.mfma %623 * %681 + %720 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %722 = vector.shape_cast %721 : vector<4xf32> to vector<4x1xf32>
    %723 = vector.extract %arg1[0, 1, 5] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %724 = vector.shape_cast %723 : vector<4x1xf32> to vector<4xf32>
    %725 = amdgpu.mfma %623 * %687 + %724 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %726 = vector.shape_cast %725 : vector<4xf32> to vector<4x1xf32>
    %727 = vector.extract %arg1[0, 1, 6] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %728 = vector.shape_cast %727 : vector<4x1xf32> to vector<4xf32>
    %729 = amdgpu.mfma %623 * %693 + %728 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %730 = vector.shape_cast %729 : vector<4xf32> to vector<4x1xf32>
    %731 = vector.extract %arg1[0, 1, 7] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
    %732 = vector.shape_cast %731 : vector<4x1xf32> to vector<4xf32>
    %733 = amdgpu.mfma %623 * %699 + %732 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %734 = vector.shape_cast %733 : vector<4xf32> to vector<4x1xf32>
    %735 = vector.insert_strided_slice %660, %cst_2 {offsets = [0, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %736 = vector.insert_strided_slice %666, %735 {offsets = [0, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %737 = vector.insert_strided_slice %672, %736 {offsets = [0, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %738 = vector.insert_strided_slice %678, %737 {offsets = [0, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %739 = vector.insert_strided_slice %684, %738 {offsets = [0, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %740 = vector.insert_strided_slice %690, %739 {offsets = [0, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %741 = vector.insert_strided_slice %696, %740 {offsets = [0, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %742 = vector.insert_strided_slice %702, %741 {offsets = [0, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %743 = vector.insert_strided_slice %706, %742 {offsets = [1, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %744 = vector.insert_strided_slice %710, %743 {offsets = [1, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %745 = vector.insert_strided_slice %714, %744 {offsets = [1, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %746 = vector.insert_strided_slice %718, %745 {offsets = [1, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %747 = vector.insert_strided_slice %722, %746 {offsets = [1, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %748 = vector.insert_strided_slice %726, %747 {offsets = [1, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %749 = vector.insert_strided_slice %730, %748 {offsets = [1, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %750 = vector.insert_strided_slice %734, %749 {offsets = [1, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %751 = vector.broadcast %750 : vector<2x8x4x1xf32> to vector<1x2x8x4x1xf32>
    gpu.barrier
    vector.store %613, %alloc_4[%c0, %25, %53] : memref<1x32x20xf16, #gpu.address_space<workgroup>>, vector<2xf16>
    vector.store %615, %alloc[%30, %64] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    vector.store %617, %alloc[%36, %65] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    vector.store %619, %alloc[%42, %66] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    vector.store %621, %alloc[%48, %67] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    scf.yield %751 : vector<1x2x8x4x1xf32>
  }
  gpu.barrier
  %85 = vector.load %expand_shape_6[%c0, %c0, %21, %c0, %22] : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
  %86 = vector.load %expand_shape_6[%c0, %c1, %21, %c0, %22] : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
  %87 = vector.load %expand_shape[%11, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
  %88 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
  %89 = vector.load %expand_shape[%88, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
  %90 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%11]
  %91 = vector.load %expand_shape[%90, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
  %92 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%11]
  %93 = vector.load %expand_shape[%92, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
  %94 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%11]
  %95 = vector.load %expand_shape[%94, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
  %96 = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%11]
  %97 = vector.load %expand_shape[%96, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
  %98 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%11]
  %99 = vector.load %expand_shape[%98, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
  %100 = affine.apply affine_map<()[s0] -> (s0 + 7)>()[%11]
  %101 = vector.load %expand_shape[%100, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
  %102 = vector.shape_cast %87 : vector<4xf16> to vector<1x1x1x4xf16>
  %103 = vector.transpose %102, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %104 = vector.shape_cast %89 : vector<4xf16> to vector<1x1x1x4xf16>
  %105 = vector.transpose %104, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %106 = vector.shape_cast %91 : vector<4xf16> to vector<1x1x1x4xf16>
  %107 = vector.transpose %106, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %108 = vector.shape_cast %93 : vector<4xf16> to vector<1x1x1x4xf16>
  %109 = vector.transpose %108, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %110 = vector.shape_cast %95 : vector<4xf16> to vector<1x1x1x4xf16>
  %111 = vector.transpose %110, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %112 = vector.shape_cast %97 : vector<4xf16> to vector<1x1x1x4xf16>
  %113 = vector.transpose %112, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %114 = vector.shape_cast %99 : vector<4xf16> to vector<1x1x1x4xf16>
  %115 = vector.transpose %114, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %116 = vector.shape_cast %101 : vector<4xf16> to vector<1x1x1x4xf16>
  %117 = vector.transpose %116, [0, 2, 1, 3] : vector<1x1x1x4xf16> to vector<1x1x1x4xf16>
  %118 = vector.extract %103[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %119 = vector.extract %84[0, 0, 0] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %120 = vector.shape_cast %118 : vector<1x4xf16> to vector<4xf16>
  %121 = vector.shape_cast %119 : vector<4x1xf32> to vector<4xf32>
  %122 = amdgpu.mfma %85 * %120 + %121 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %123 = vector.shape_cast %122 : vector<4xf32> to vector<4x1xf32>
  %124 = vector.extract %105[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %125 = vector.extract %84[0, 0, 1] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %126 = vector.shape_cast %124 : vector<1x4xf16> to vector<4xf16>
  %127 = vector.shape_cast %125 : vector<4x1xf32> to vector<4xf32>
  %128 = amdgpu.mfma %85 * %126 + %127 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %129 = vector.shape_cast %128 : vector<4xf32> to vector<4x1xf32>
  %130 = vector.extract %107[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %131 = vector.extract %84[0, 0, 2] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %132 = vector.shape_cast %130 : vector<1x4xf16> to vector<4xf16>
  %133 = vector.shape_cast %131 : vector<4x1xf32> to vector<4xf32>
  %134 = amdgpu.mfma %85 * %132 + %133 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %135 = vector.shape_cast %134 : vector<4xf32> to vector<4x1xf32>
  %136 = vector.extract %109[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %137 = vector.extract %84[0, 0, 3] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %138 = vector.shape_cast %136 : vector<1x4xf16> to vector<4xf16>
  %139 = vector.shape_cast %137 : vector<4x1xf32> to vector<4xf32>
  %140 = amdgpu.mfma %85 * %138 + %139 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %141 = vector.shape_cast %140 : vector<4xf32> to vector<4x1xf32>
  %142 = vector.extract %111[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %143 = vector.extract %84[0, 0, 4] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %144 = vector.shape_cast %142 : vector<1x4xf16> to vector<4xf16>
  %145 = vector.shape_cast %143 : vector<4x1xf32> to vector<4xf32>
  %146 = amdgpu.mfma %85 * %144 + %145 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %147 = vector.shape_cast %146 : vector<4xf32> to vector<4x1xf32>
  %148 = vector.extract %113[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %149 = vector.extract %84[0, 0, 5] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %150 = vector.shape_cast %148 : vector<1x4xf16> to vector<4xf16>
  %151 = vector.shape_cast %149 : vector<4x1xf32> to vector<4xf32>
  %152 = amdgpu.mfma %85 * %150 + %151 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %153 = vector.shape_cast %152 : vector<4xf32> to vector<4x1xf32>
  %154 = vector.extract %115[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %155 = vector.extract %84[0, 0, 6] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %156 = vector.shape_cast %154 : vector<1x4xf16> to vector<4xf16>
  %157 = vector.shape_cast %155 : vector<4x1xf32> to vector<4xf32>
  %158 = amdgpu.mfma %85 * %156 + %157 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %159 = vector.shape_cast %158 : vector<4xf32> to vector<4x1xf32>
  %160 = vector.extract %117[0, 0] : vector<1x4xf16> from vector<1x1x1x4xf16>
  %161 = vector.extract %84[0, 0, 7] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %162 = vector.shape_cast %160 : vector<1x4xf16> to vector<4xf16>
  %163 = vector.shape_cast %161 : vector<4x1xf32> to vector<4xf32>
  %164 = amdgpu.mfma %85 * %162 + %163 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %165 = vector.shape_cast %164 : vector<4xf32> to vector<4x1xf32>
  %166 = vector.extract %84[0, 1, 0] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %167 = vector.shape_cast %166 : vector<4x1xf32> to vector<4xf32>
  %168 = amdgpu.mfma %86 * %120 + %167 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %169 = vector.shape_cast %168 : vector<4xf32> to vector<4x1xf32>
  %170 = vector.extract %84[0, 1, 1] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %171 = vector.shape_cast %170 : vector<4x1xf32> to vector<4xf32>
  %172 = amdgpu.mfma %86 * %126 + %171 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %173 = vector.shape_cast %172 : vector<4xf32> to vector<4x1xf32>
  %174 = vector.extract %84[0, 1, 2] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %175 = vector.shape_cast %174 : vector<4x1xf32> to vector<4xf32>
  %176 = amdgpu.mfma %86 * %132 + %175 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %177 = vector.shape_cast %176 : vector<4xf32> to vector<4x1xf32>
  %178 = vector.extract %84[0, 1, 3] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %179 = vector.shape_cast %178 : vector<4x1xf32> to vector<4xf32>
  %180 = amdgpu.mfma %86 * %138 + %179 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %181 = vector.shape_cast %180 : vector<4xf32> to vector<4x1xf32>
  %182 = vector.extract %84[0, 1, 4] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %183 = vector.shape_cast %182 : vector<4x1xf32> to vector<4xf32>
  %184 = amdgpu.mfma %86 * %144 + %183 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %185 = vector.shape_cast %184 : vector<4xf32> to vector<4x1xf32>
  %186 = vector.extract %84[0, 1, 5] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %187 = vector.shape_cast %186 : vector<4x1xf32> to vector<4xf32>
  %188 = amdgpu.mfma %86 * %150 + %187 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %189 = vector.shape_cast %188 : vector<4xf32> to vector<4x1xf32>
  %190 = vector.extract %84[0, 1, 6] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %191 = vector.shape_cast %190 : vector<4x1xf32> to vector<4xf32>
  %192 = amdgpu.mfma %86 * %156 + %191 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %193 = vector.shape_cast %192 : vector<4xf32> to vector<4x1xf32>
  %194 = vector.extract %84[0, 1, 7] : vector<4x1xf32> from vector<1x2x8x4x1xf32>
  %195 = vector.shape_cast %194 : vector<4x1xf32> to vector<4xf32>
  %196 = amdgpu.mfma %86 * %162 + %195 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  %197 = vector.shape_cast %196 : vector<4xf32> to vector<4x1xf32>
  %198 = vector.extract %123[0] : vector<1xf32> from vector<4x1xf32>
  %199 = vector.insert_strided_slice %198, %cst {offsets = [0, 0, 0, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %200 = vector.extract %129[0] : vector<1xf32> from vector<4x1xf32>
  %201 = vector.insert_strided_slice %200, %199 {offsets = [0, 0, 1, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %202 = vector.extract %135[0] : vector<1xf32> from vector<4x1xf32>
  %203 = vector.insert_strided_slice %202, %201 {offsets = [0, 0, 2, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %204 = vector.extract %141[0] : vector<1xf32> from vector<4x1xf32>
  %205 = vector.insert_strided_slice %204, %203 {offsets = [0, 0, 3, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %206 = vector.extract %147[0] : vector<1xf32> from vector<4x1xf32>
  %207 = vector.insert_strided_slice %206, %205 {offsets = [0, 0, 4, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %208 = vector.extract %153[0] : vector<1xf32> from vector<4x1xf32>
  %209 = vector.insert_strided_slice %208, %207 {offsets = [0, 0, 5, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %210 = vector.extract %159[0] : vector<1xf32> from vector<4x1xf32>
  %211 = vector.insert_strided_slice %210, %209 {offsets = [0, 0, 6, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %212 = vector.extract %165[0] : vector<1xf32> from vector<4x1xf32>
  %213 = vector.insert_strided_slice %212, %211 {offsets = [0, 0, 7, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %214 = vector.extract %123[1] : vector<1xf32> from vector<4x1xf32>
  %215 = vector.insert_strided_slice %214, %213 {offsets = [0, 1, 0, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %216 = vector.extract %129[1] : vector<1xf32> from vector<4x1xf32>
  %217 = vector.insert_strided_slice %216, %215 {offsets = [0, 1, 1, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %218 = vector.extract %135[1] : vector<1xf32> from vector<4x1xf32>
  %219 = vector.insert_strided_slice %218, %217 {offsets = [0, 1, 2, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %220 = vector.extract %141[1] : vector<1xf32> from vector<4x1xf32>
  %221 = vector.insert_strided_slice %220, %219 {offsets = [0, 1, 3, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %222 = vector.extract %147[1] : vector<1xf32> from vector<4x1xf32>
  %223 = vector.insert_strided_slice %222, %221 {offsets = [0, 1, 4, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %224 = vector.extract %153[1] : vector<1xf32> from vector<4x1xf32>
  %225 = vector.insert_strided_slice %224, %223 {offsets = [0, 1, 5, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %226 = vector.extract %159[1] : vector<1xf32> from vector<4x1xf32>
  %227 = vector.insert_strided_slice %226, %225 {offsets = [0, 1, 6, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %228 = vector.extract %165[1] : vector<1xf32> from vector<4x1xf32>
  %229 = vector.insert_strided_slice %228, %227 {offsets = [0, 1, 7, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %230 = vector.extract %123[2] : vector<1xf32> from vector<4x1xf32>
  %231 = vector.insert_strided_slice %230, %229 {offsets = [0, 2, 0, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %232 = vector.extract %129[2] : vector<1xf32> from vector<4x1xf32>
  %233 = vector.insert_strided_slice %232, %231 {offsets = [0, 2, 1, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %234 = vector.extract %135[2] : vector<1xf32> from vector<4x1xf32>
  %235 = vector.insert_strided_slice %234, %233 {offsets = [0, 2, 2, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %236 = vector.extract %141[2] : vector<1xf32> from vector<4x1xf32>
  %237 = vector.insert_strided_slice %236, %235 {offsets = [0, 2, 3, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %238 = vector.extract %147[2] : vector<1xf32> from vector<4x1xf32>
  %239 = vector.insert_strided_slice %238, %237 {offsets = [0, 2, 4, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %240 = vector.extract %153[2] : vector<1xf32> from vector<4x1xf32>
  %241 = vector.insert_strided_slice %240, %239 {offsets = [0, 2, 5, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %242 = vector.extract %159[2] : vector<1xf32> from vector<4x1xf32>
  %243 = vector.insert_strided_slice %242, %241 {offsets = [0, 2, 6, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %244 = vector.extract %165[2] : vector<1xf32> from vector<4x1xf32>
  %245 = vector.insert_strided_slice %244, %243 {offsets = [0, 2, 7, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %246 = vector.extract %123[3] : vector<1xf32> from vector<4x1xf32>
  %247 = vector.insert_strided_slice %246, %245 {offsets = [0, 3, 0, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %248 = vector.extract %129[3] : vector<1xf32> from vector<4x1xf32>
  %249 = vector.insert_strided_slice %248, %247 {offsets = [0, 3, 1, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %250 = vector.extract %135[3] : vector<1xf32> from vector<4x1xf32>
  %251 = vector.insert_strided_slice %250, %249 {offsets = [0, 3, 2, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %252 = vector.extract %141[3] : vector<1xf32> from vector<4x1xf32>
  %253 = vector.insert_strided_slice %252, %251 {offsets = [0, 3, 3, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %254 = vector.extract %147[3] : vector<1xf32> from vector<4x1xf32>
  %255 = vector.insert_strided_slice %254, %253 {offsets = [0, 3, 4, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %256 = vector.extract %153[3] : vector<1xf32> from vector<4x1xf32>
  %257 = vector.insert_strided_slice %256, %255 {offsets = [0, 3, 5, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %258 = vector.extract %159[3] : vector<1xf32> from vector<4x1xf32>
  %259 = vector.insert_strided_slice %258, %257 {offsets = [0, 3, 6, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %260 = vector.extract %165[3] : vector<1xf32> from vector<4x1xf32>
  %261 = vector.insert_strided_slice %260, %259 {offsets = [0, 3, 7, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %262 = vector.extract %169[0] : vector<1xf32> from vector<4x1xf32>
  %263 = vector.insert_strided_slice %262, %261 {offsets = [1, 0, 0, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %264 = vector.extract %173[0] : vector<1xf32> from vector<4x1xf32>
  %265 = vector.insert_strided_slice %264, %263 {offsets = [1, 0, 1, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %266 = vector.extract %177[0] : vector<1xf32> from vector<4x1xf32>
  %267 = vector.insert_strided_slice %266, %265 {offsets = [1, 0, 2, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %268 = vector.extract %181[0] : vector<1xf32> from vector<4x1xf32>
  %269 = vector.insert_strided_slice %268, %267 {offsets = [1, 0, 3, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %270 = vector.extract %185[0] : vector<1xf32> from vector<4x1xf32>
  %271 = vector.insert_strided_slice %270, %269 {offsets = [1, 0, 4, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %272 = vector.extract %189[0] : vector<1xf32> from vector<4x1xf32>
  %273 = vector.insert_strided_slice %272, %271 {offsets = [1, 0, 5, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %274 = vector.extract %193[0] : vector<1xf32> from vector<4x1xf32>
  %275 = vector.insert_strided_slice %274, %273 {offsets = [1, 0, 6, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %276 = vector.extract %197[0] : vector<1xf32> from vector<4x1xf32>
  %277 = vector.insert_strided_slice %276, %275 {offsets = [1, 0, 7, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %278 = vector.extract %169[1] : vector<1xf32> from vector<4x1xf32>
  %279 = vector.insert_strided_slice %278, %277 {offsets = [1, 1, 0, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %280 = vector.extract %173[1] : vector<1xf32> from vector<4x1xf32>
  %281 = vector.insert_strided_slice %280, %279 {offsets = [1, 1, 1, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %282 = vector.extract %177[1] : vector<1xf32> from vector<4x1xf32>
  %283 = vector.insert_strided_slice %282, %281 {offsets = [1, 1, 2, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %284 = vector.extract %181[1] : vector<1xf32> from vector<4x1xf32>
  %285 = vector.insert_strided_slice %284, %283 {offsets = [1, 1, 3, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %286 = vector.extract %185[1] : vector<1xf32> from vector<4x1xf32>
  %287 = vector.insert_strided_slice %286, %285 {offsets = [1, 1, 4, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %288 = vector.extract %189[1] : vector<1xf32> from vector<4x1xf32>
  %289 = vector.insert_strided_slice %288, %287 {offsets = [1, 1, 5, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %290 = vector.extract %193[1] : vector<1xf32> from vector<4x1xf32>
  %291 = vector.insert_strided_slice %290, %289 {offsets = [1, 1, 6, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %292 = vector.extract %197[1] : vector<1xf32> from vector<4x1xf32>
  %293 = vector.insert_strided_slice %292, %291 {offsets = [1, 1, 7, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %294 = vector.extract %169[2] : vector<1xf32> from vector<4x1xf32>
  %295 = vector.insert_strided_slice %294, %293 {offsets = [1, 2, 0, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %296 = vector.extract %173[2] : vector<1xf32> from vector<4x1xf32>
  %297 = vector.insert_strided_slice %296, %295 {offsets = [1, 2, 1, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %298 = vector.extract %177[2] : vector<1xf32> from vector<4x1xf32>
  %299 = vector.insert_strided_slice %298, %297 {offsets = [1, 2, 2, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %300 = vector.extract %181[2] : vector<1xf32> from vector<4x1xf32>
  %301 = vector.insert_strided_slice %300, %299 {offsets = [1, 2, 3, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %302 = vector.extract %185[2] : vector<1xf32> from vector<4x1xf32>
  %303 = vector.insert_strided_slice %302, %301 {offsets = [1, 2, 4, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %304 = vector.extract %189[2] : vector<1xf32> from vector<4x1xf32>
  %305 = vector.insert_strided_slice %304, %303 {offsets = [1, 2, 5, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %306 = vector.extract %193[2] : vector<1xf32> from vector<4x1xf32>
  %307 = vector.insert_strided_slice %306, %305 {offsets = [1, 2, 6, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %308 = vector.extract %197[2] : vector<1xf32> from vector<4x1xf32>
  %309 = vector.insert_strided_slice %308, %307 {offsets = [1, 2, 7, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %310 = vector.extract %169[3] : vector<1xf32> from vector<4x1xf32>
  %311 = vector.insert_strided_slice %310, %309 {offsets = [1, 3, 0, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %312 = vector.extract %173[3] : vector<1xf32> from vector<4x1xf32>
  %313 = vector.insert_strided_slice %312, %311 {offsets = [1, 3, 1, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %314 = vector.extract %177[3] : vector<1xf32> from vector<4x1xf32>
  %315 = vector.insert_strided_slice %314, %313 {offsets = [1, 3, 2, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %316 = vector.extract %181[3] : vector<1xf32> from vector<4x1xf32>
  %317 = vector.insert_strided_slice %316, %315 {offsets = [1, 3, 3, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %318 = vector.extract %185[3] : vector<1xf32> from vector<4x1xf32>
  %319 = vector.insert_strided_slice %318, %317 {offsets = [1, 3, 4, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %320 = vector.extract %189[3] : vector<1xf32> from vector<4x1xf32>
  %321 = vector.insert_strided_slice %320, %319 {offsets = [1, 3, 5, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %322 = vector.extract %193[3] : vector<1xf32> from vector<4x1xf32>
  %323 = vector.insert_strided_slice %322, %321 {offsets = [1, 3, 6, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %324 = vector.extract %197[3] : vector<1xf32> from vector<4x1xf32>
  %325 = vector.insert_strided_slice %324, %323 {offsets = [1, 3, 7, 0], strides = [1]} : vector<1xf32> into vector<2x4x8x1xf32>
  %326 = vector.shape_cast %325 : vector<2x4x8x1xf32> to vector<2x4x8xf32>
  %327 = vector.extract %326[0, 0] : vector<8xf32> from vector<2x4x8xf32>
  scf.for %arg0 = %c0 to %c8 step %c1 {
    %598 = vector.extractelement %327[%arg0 : index] : vector<8xf32>
    memref.store %598, %alloca[%c0, %c0, %c0, %arg0, %c0] : memref<1x2x4x8x1xf32, #gpu.address_space<private>>
  }
  %328 = vector.extract %326[0, 1] : vector<8xf32> from vector<2x4x8xf32>
  scf.for %arg0 = %c0 to %c8 step %c1 {
    %598 = vector.extractelement %328[%arg0 : index] : vector<8xf32>
    memref.store %598, %alloca[%c0, %c0, %c1, %arg0, %c0] : memref<1x2x4x8x1xf32, #gpu.address_space<private>>
  }
  %329 = vector.extract %326[0, 2] : vector<8xf32> from vector<2x4x8xf32>
  scf.for %arg0 = %c0 to %c8 step %c1 {
    %598 = vector.extractelement %329[%arg0 : index] : vector<8xf32>
    memref.store %598, %alloca[%c0, %c0, %c2, %arg0, %c0] : memref<1x2x4x8x1xf32, #gpu.address_space<private>>
  }
  %330 = vector.extract %326[0, 3] : vector<8xf32> from vector<2x4x8xf32>
  scf.for %arg0 = %c0 to %c8 step %c1 {
    %598 = vector.extractelement %330[%arg0 : index] : vector<8xf32>
    memref.store %598, %alloca[%c0, %c0, %c3, %arg0, %c0] : memref<1x2x4x8x1xf32, #gpu.address_space<private>>
  }
  %331 = vector.extract %326[1, 0] : vector<8xf32> from vector<2x4x8xf32>
  scf.for %arg0 = %c0 to %c8 step %c1 {
    %598 = vector.extractelement %331[%arg0 : index] : vector<8xf32>
    memref.store %598, %alloca[%c0, %c1, %c0, %arg0, %c0] : memref<1x2x4x8x1xf32, #gpu.address_space<private>>
  }
  %332 = vector.extract %326[1, 1] : vector<8xf32> from vector<2x4x8xf32>
  scf.for %arg0 = %c0 to %c8 step %c1 {
    %598 = vector.extractelement %332[%arg0 : index] : vector<8xf32>
    memref.store %598, %alloca[%c0, %c1, %c1, %arg0, %c0] : memref<1x2x4x8x1xf32, #gpu.address_space<private>>
  }
  %333 = vector.extract %326[1, 2] : vector<8xf32> from vector<2x4x8xf32>
  scf.for %arg0 = %c0 to %c8 step %c1 {
    %598 = vector.extractelement %333[%arg0 : index] : vector<8xf32>
    memref.store %598, %alloca[%c0, %c1, %c2, %arg0, %c0] : memref<1x2x4x8x1xf32, #gpu.address_space<private>>
  }
  %334 = vector.extract %326[1, 3] : vector<8xf32> from vector<2x4x8xf32>
  scf.for %arg0 = %c0 to %c8 step %c1 {
    %598 = vector.extractelement %334[%arg0 : index] : vector<8xf32>
    memref.store %598, %alloca[%c0, %c1, %c3, %arg0, %c0] : memref<1x2x4x8x1xf32, #gpu.address_space<private>>
  }
  %335 = vector.load %5[%11, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %336 = vector.insert %335, %cst_0 [0] : vector<1xf16> into vector<8x1xf16>
  %337 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
  %338 = vector.load %5[%337, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %339 = vector.insert %338, %336 [1] : vector<1xf16> into vector<8x1xf16>
  %340 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%11]
  %341 = vector.load %5[%340, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %342 = vector.insert %341, %339 [2] : vector<1xf16> into vector<8x1xf16>
  %343 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%11]
  %344 = vector.load %5[%343, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %345 = vector.insert %344, %342 [3] : vector<1xf16> into vector<8x1xf16>
  %346 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%11]
  %347 = vector.load %5[%346, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %348 = vector.insert %347, %345 [4] : vector<1xf16> into vector<8x1xf16>
  %349 = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%11]
  %350 = vector.load %5[%349, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %351 = vector.insert %350, %348 [5] : vector<1xf16> into vector<8x1xf16>
  %352 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%11]
  %353 = vector.load %5[%352, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %354 = vector.insert %353, %351 [6] : vector<1xf16> into vector<8x1xf16>
  %355 = affine.apply affine_map<()[s0] -> (s0 + 7)>()[%11]
  %356 = vector.load %5[%355, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %357 = vector.insert %356, %354 [7] : vector<1xf16> into vector<8x1xf16>
  %358 = arith.extf %357 : vector<8x1xf16> to vector<8x1xf32>
  %359 = vector.extract %325[0, 0] : vector<8x1xf32> from vector<2x4x8x1xf32>
  %360 = arith.addf %359, %358 : vector<8x1xf32>
  %361 = vector.extract %325[0, 1] : vector<8x1xf32> from vector<2x4x8x1xf32>
  %362 = arith.addf %361, %358 : vector<8x1xf32>
  %363 = vector.extract %325[0, 2] : vector<8x1xf32> from vector<2x4x8x1xf32>
  %364 = arith.addf %363, %358 : vector<8x1xf32>
  %365 = vector.extract %325[0, 3] : vector<8x1xf32> from vector<2x4x8x1xf32>
  %366 = arith.addf %365, %358 : vector<8x1xf32>
  %367 = vector.extract %325[1, 0] : vector<8x1xf32> from vector<2x4x8x1xf32>
  %368 = arith.addf %367, %358 : vector<8x1xf32>
  %369 = vector.extract %325[1, 1] : vector<8x1xf32> from vector<2x4x8x1xf32>
  %370 = arith.addf %369, %358 : vector<8x1xf32>
  %371 = vector.extract %325[1, 2] : vector<8x1xf32> from vector<2x4x8x1xf32>
  %372 = arith.addf %371, %358 : vector<8x1xf32>
  %373 = vector.extract %325[1, 3] : vector<8x1xf32> from vector<2x4x8x1xf32>
  %374 = arith.addf %373, %358 : vector<8x1xf32>
  %375 = arith.truncf %360 : vector<8x1xf32> to vector<8x1xf16>
  %376 = vector.insert_strided_slice %375, %cst_1 {offsets = [0, 0, 0, 0], strides = [1, 1]} : vector<8x1xf16> into vector<2x4x8x1xf16>
  %377 = arith.truncf %362 : vector<8x1xf32> to vector<8x1xf16>
  %378 = vector.insert_strided_slice %377, %376 {offsets = [0, 1, 0, 0], strides = [1, 1]} : vector<8x1xf16> into vector<2x4x8x1xf16>
  %379 = arith.truncf %364 : vector<8x1xf32> to vector<8x1xf16>
  %380 = vector.insert_strided_slice %379, %378 {offsets = [0, 2, 0, 0], strides = [1, 1]} : vector<8x1xf16> into vector<2x4x8x1xf16>
  %381 = arith.truncf %366 : vector<8x1xf32> to vector<8x1xf16>
  %382 = vector.insert_strided_slice %381, %380 {offsets = [0, 3, 0, 0], strides = [1, 1]} : vector<8x1xf16> into vector<2x4x8x1xf16>
  %383 = arith.truncf %368 : vector<8x1xf32> to vector<8x1xf16>
  %384 = vector.insert_strided_slice %383, %382 {offsets = [1, 0, 0, 0], strides = [1, 1]} : vector<8x1xf16> into vector<2x4x8x1xf16>
  %385 = arith.truncf %370 : vector<8x1xf32> to vector<8x1xf16>
  %386 = vector.insert_strided_slice %385, %384 {offsets = [1, 1, 0, 0], strides = [1, 1]} : vector<8x1xf16> into vector<2x4x8x1xf16>
  %387 = arith.truncf %372 : vector<8x1xf32> to vector<8x1xf16>
  %388 = vector.insert_strided_slice %387, %386 {offsets = [1, 2, 0, 0], strides = [1, 1]} : vector<8x1xf16> into vector<2x4x8x1xf16>
  %389 = arith.truncf %374 : vector<8x1xf32> to vector<8x1xf16>
  %390 = vector.insert_strided_slice %389, %388 {offsets = [1, 3, 0, 0], strides = [1, 1]} : vector<8x1xf16> into vector<2x4x8x1xf16>
  %391 = arith.muli %workgroup_id_x, %c2 overflow<nsw> : index
  %392 = vector.extract %375[0] : vector<1xf16> from vector<8x1xf16>
  vector.store %392, %7[%workgroup_id_y, %391, %22, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %393 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
  %394 = vector.extract %375[1] : vector<1xf16> from vector<8x1xf16>
  vector.store %394, %7[%workgroup_id_y, %391, %22, %393, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %395 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%11]
  %396 = vector.extract %375[2] : vector<1xf16> from vector<8x1xf16>
  vector.store %396, %7[%workgroup_id_y, %391, %22, %395, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %397 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%11]
  %398 = vector.extract %375[3] : vector<1xf16> from vector<8x1xf16>
  vector.store %398, %7[%workgroup_id_y, %391, %22, %397, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %399 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%11]
  %400 = vector.extract %375[4] : vector<1xf16> from vector<8x1xf16>
  vector.store %400, %7[%workgroup_id_y, %391, %22, %399, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %401 = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%11]
  %402 = vector.extract %375[5] : vector<1xf16> from vector<8x1xf16>
  vector.store %402, %7[%workgroup_id_y, %391, %22, %401, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %403 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%11]
  %404 = vector.extract %375[6] : vector<1xf16> from vector<8x1xf16>
  vector.store %404, %7[%workgroup_id_y, %391, %22, %403, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %405 = affine.apply affine_map<()[s0] -> (s0 + 7)>()[%11]
  %406 = vector.extract %375[7] : vector<1xf16> from vector<8x1xf16>
  vector.store %406, %7[%workgroup_id_y, %391, %22, %405, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %407 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%22]
  %408 = vector.extract %377[0] : vector<1xf16> from vector<8x1xf16>
  vector.store %408, %7[%workgroup_id_y, %391, %407, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %409 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
  %410 = vector.extract %377[1] : vector<1xf16> from vector<8x1xf16>
  vector.store %410, %7[%workgroup_id_y, %391, %407, %409, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %411 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%11]
  %412 = vector.extract %377[2] : vector<1xf16> from vector<8x1xf16>
  vector.store %412, %7[%workgroup_id_y, %391, %407, %411, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %413 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%11]
  %414 = vector.extract %377[3] : vector<1xf16> from vector<8x1xf16>
  vector.store %414, %7[%workgroup_id_y, %391, %407, %413, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %415 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%11]
  %416 = vector.extract %377[4] : vector<1xf16> from vector<8x1xf16>
  vector.store %416, %7[%workgroup_id_y, %391, %407, %415, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %417 = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%11]
  %418 = vector.extract %377[5] : vector<1xf16> from vector<8x1xf16>
  vector.store %418, %7[%workgroup_id_y, %391, %407, %417, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %419 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%11]
  %420 = vector.extract %377[6] : vector<1xf16> from vector<8x1xf16>
  vector.store %420, %7[%workgroup_id_y, %391, %407, %419, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %421 = affine.apply affine_map<()[s0] -> (s0 + 7)>()[%11]
  %422 = vector.extract %377[7] : vector<1xf16> from vector<8x1xf16>
  vector.store %422, %7[%workgroup_id_y, %391, %407, %421, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %423 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%22]
  %424 = vector.extract %379[0] : vector<1xf16> from vector<8x1xf16>
  vector.store %424, %7[%workgroup_id_y, %391, %423, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %425 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
  %426 = vector.extract %379[1] : vector<1xf16> from vector<8x1xf16>
  vector.store %426, %7[%workgroup_id_y, %391, %423, %425, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %427 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%11]
  %428 = vector.extract %379[2] : vector<1xf16> from vector<8x1xf16>
  vector.store %428, %7[%workgroup_id_y, %391, %423, %427, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %429 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%11]
  %430 = vector.extract %379[3] : vector<1xf16> from vector<8x1xf16>
  vector.store %430, %7[%workgroup_id_y, %391, %423, %429, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %431 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%11]
  %432 = vector.extract %379[4] : vector<1xf16> from vector<8x1xf16>
  vector.store %432, %7[%workgroup_id_y, %391, %423, %431, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %433 = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%11]
  %434 = vector.extract %379[5] : vector<1xf16> from vector<8x1xf16>
  vector.store %434, %7[%workgroup_id_y, %391, %423, %433, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %435 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%11]
  %436 = vector.extract %379[6] : vector<1xf16> from vector<8x1xf16>
  vector.store %436, %7[%workgroup_id_y, %391, %423, %435, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %437 = affine.apply affine_map<()[s0] -> (s0 + 7)>()[%11]
  %438 = vector.extract %379[7] : vector<1xf16> from vector<8x1xf16>
  vector.store %438, %7[%workgroup_id_y, %391, %423, %437, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %439 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%22]
  %440 = vector.extract %381[0] : vector<1xf16> from vector<8x1xf16>
  vector.store %440, %7[%workgroup_id_y, %391, %439, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %441 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
  %442 = vector.extract %381[1] : vector<1xf16> from vector<8x1xf16>
  vector.store %442, %7[%workgroup_id_y, %391, %439, %441, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %443 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%11]
  %444 = vector.extract %381[2] : vector<1xf16> from vector<8x1xf16>
  vector.store %444, %7[%workgroup_id_y, %391, %439, %443, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %445 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%11]
  %446 = vector.extract %381[3] : vector<1xf16> from vector<8x1xf16>
  vector.store %446, %7[%workgroup_id_y, %391, %439, %445, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %447 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%11]
  %448 = vector.extract %381[4] : vector<1xf16> from vector<8x1xf16>
  vector.store %448, %7[%workgroup_id_y, %391, %439, %447, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %449 = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%11]
  %450 = vector.extract %381[5] : vector<1xf16> from vector<8x1xf16>
  vector.store %450, %7[%workgroup_id_y, %391, %439, %449, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %451 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%11]
  %452 = vector.extract %381[6] : vector<1xf16> from vector<8x1xf16>
  vector.store %452, %7[%workgroup_id_y, %391, %439, %451, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %453 = affine.apply affine_map<()[s0] -> (s0 + 7)>()[%11]
  %454 = vector.extract %381[7] : vector<1xf16> from vector<8x1xf16>
  vector.store %454, %7[%workgroup_id_y, %391, %439, %453, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %455 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%391]
  %456 = vector.extract %383[0] : vector<1xf16> from vector<8x1xf16>
  vector.store %456, %7[%workgroup_id_y, %455, %22, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %457 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
  %458 = vector.extract %383[1] : vector<1xf16> from vector<8x1xf16>
  vector.store %458, %7[%workgroup_id_y, %455, %22, %457, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %459 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%11]
  %460 = vector.extract %383[2] : vector<1xf16> from vector<8x1xf16>
  vector.store %460, %7[%workgroup_id_y, %455, %22, %459, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %461 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%11]
  %462 = vector.extract %383[3] : vector<1xf16> from vector<8x1xf16>
  vector.store %462, %7[%workgroup_id_y, %455, %22, %461, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %463 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%11]
  %464 = vector.extract %383[4] : vector<1xf16> from vector<8x1xf16>
  vector.store %464, %7[%workgroup_id_y, %455, %22, %463, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %465 = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%11]
  %466 = vector.extract %383[5] : vector<1xf16> from vector<8x1xf16>
  vector.store %466, %7[%workgroup_id_y, %455, %22, %465, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %467 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%11]
  %468 = vector.extract %383[6] : vector<1xf16> from vector<8x1xf16>
  vector.store %468, %7[%workgroup_id_y, %455, %22, %467, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %469 = affine.apply affine_map<()[s0] -> (s0 + 7)>()[%11]
  %470 = vector.extract %383[7] : vector<1xf16> from vector<8x1xf16>
  vector.store %470, %7[%workgroup_id_y, %455, %22, %469, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %471 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%22]
  %472 = vector.extract %385[0] : vector<1xf16> from vector<8x1xf16>
  vector.store %472, %7[%workgroup_id_y, %455, %471, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %473 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
  %474 = vector.extract %385[1] : vector<1xf16> from vector<8x1xf16>
  vector.store %474, %7[%workgroup_id_y, %455, %471, %473, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %475 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%11]
  %476 = vector.extract %385[2] : vector<1xf16> from vector<8x1xf16>
  vector.store %476, %7[%workgroup_id_y, %455, %471, %475, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %477 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%11]
  %478 = vector.extract %385[3] : vector<1xf16> from vector<8x1xf16>
  vector.store %478, %7[%workgroup_id_y, %455, %471, %477, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %479 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%11]
  %480 = vector.extract %385[4] : vector<1xf16> from vector<8x1xf16>
  vector.store %480, %7[%workgroup_id_y, %455, %471, %479, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %481 = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%11]
  %482 = vector.extract %385[5] : vector<1xf16> from vector<8x1xf16>
  vector.store %482, %7[%workgroup_id_y, %455, %471, %481, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %483 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%11]
  %484 = vector.extract %385[6] : vector<1xf16> from vector<8x1xf16>
  vector.store %484, %7[%workgroup_id_y, %455, %471, %483, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %485 = affine.apply affine_map<()[s0] -> (s0 + 7)>()[%11]
  %486 = vector.extract %385[7] : vector<1xf16> from vector<8x1xf16>
  vector.store %486, %7[%workgroup_id_y, %455, %471, %485, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %487 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%22]
  %488 = vector.extract %387[0] : vector<1xf16> from vector<8x1xf16>
  vector.store %488, %7[%workgroup_id_y, %455, %487, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %489 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
  %490 = vector.extract %387[1] : vector<1xf16> from vector<8x1xf16>
  vector.store %490, %7[%workgroup_id_y, %455, %487, %489, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %491 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%11]
  %492 = vector.extract %387[2] : vector<1xf16> from vector<8x1xf16>
  vector.store %492, %7[%workgroup_id_y, %455, %487, %491, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %493 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%11]
  %494 = vector.extract %387[3] : vector<1xf16> from vector<8x1xf16>
  vector.store %494, %7[%workgroup_id_y, %455, %487, %493, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %495 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%11]
  %496 = vector.extract %387[4] : vector<1xf16> from vector<8x1xf16>
  vector.store %496, %7[%workgroup_id_y, %455, %487, %495, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %497 = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%11]
  %498 = vector.extract %387[5] : vector<1xf16> from vector<8x1xf16>
  vector.store %498, %7[%workgroup_id_y, %455, %487, %497, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %499 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%11]
  %500 = vector.extract %387[6] : vector<1xf16> from vector<8x1xf16>
  vector.store %500, %7[%workgroup_id_y, %455, %487, %499, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %501 = affine.apply affine_map<()[s0] -> (s0 + 7)>()[%11]
  %502 = vector.extract %387[7] : vector<1xf16> from vector<8x1xf16>
  vector.store %502, %7[%workgroup_id_y, %455, %487, %501, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %503 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%22]
  %504 = vector.extract %389[0] : vector<1xf16> from vector<8x1xf16>
  vector.store %504, %7[%workgroup_id_y, %455, %503, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %505 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
  %506 = vector.extract %389[1] : vector<1xf16> from vector<8x1xf16>
  vector.store %506, %7[%workgroup_id_y, %455, %503, %505, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %507 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%11]
  %508 = vector.extract %389[2] : vector<1xf16> from vector<8x1xf16>
  vector.store %508, %7[%workgroup_id_y, %455, %503, %507, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %509 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%11]
  %510 = vector.extract %389[3] : vector<1xf16> from vector<8x1xf16>
  vector.store %510, %7[%workgroup_id_y, %455, %503, %509, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %511 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%11]
  %512 = vector.extract %389[4] : vector<1xf16> from vector<8x1xf16>
  vector.store %512, %7[%workgroup_id_y, %455, %503, %511, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %513 = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%11]
  %514 = vector.extract %389[5] : vector<1xf16> from vector<8x1xf16>
  vector.store %514, %7[%workgroup_id_y, %455, %503, %513, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %515 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%11]
  %516 = vector.extract %389[6] : vector<1xf16> from vector<8x1xf16>
  vector.store %516, %7[%workgroup_id_y, %455, %503, %515, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %517 = affine.apply affine_map<()[s0] -> (s0 + 7)>()[%11]
  %518 = vector.extract %389[7] : vector<1xf16> from vector<8x1xf16>
  vector.store %518, %7[%workgroup_id_y, %455, %503, %517, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
  %519 = vector.extract_strided_slice %390 {offsets = [0, 0, 0, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %520 = vector.broadcast %519 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %521 = vector.transpose %520, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %522 = vector.extract_strided_slice %390 {offsets = [1, 0, 0, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %523 = vector.broadcast %522 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %524 = vector.transpose %523, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %525 = vector.extract_strided_slice %390 {offsets = [0, 0, 1, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %526 = vector.broadcast %525 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %527 = vector.transpose %526, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %528 = vector.extract_strided_slice %390 {offsets = [1, 0, 1, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %529 = vector.broadcast %528 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %530 = vector.transpose %529, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %531 = vector.extract_strided_slice %390 {offsets = [0, 0, 2, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %532 = vector.broadcast %531 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %533 = vector.transpose %532, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %534 = vector.extract_strided_slice %390 {offsets = [1, 0, 2, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %535 = vector.broadcast %534 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %536 = vector.transpose %535, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %537 = vector.extract_strided_slice %390 {offsets = [0, 0, 3, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %538 = vector.broadcast %537 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %539 = vector.transpose %538, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %540 = vector.extract_strided_slice %390 {offsets = [1, 0, 3, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %541 = vector.broadcast %540 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %542 = vector.transpose %541, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %543 = vector.extract_strided_slice %390 {offsets = [0, 0, 4, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %544 = vector.broadcast %543 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %545 = vector.transpose %544, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %546 = vector.extract_strided_slice %390 {offsets = [1, 0, 4, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %547 = vector.broadcast %546 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %548 = vector.transpose %547, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %549 = vector.extract_strided_slice %390 {offsets = [0, 0, 5, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %550 = vector.broadcast %549 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %551 = vector.transpose %550, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %552 = vector.extract_strided_slice %390 {offsets = [1, 0, 5, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %553 = vector.broadcast %552 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %554 = vector.transpose %553, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %555 = vector.extract_strided_slice %390 {offsets = [0, 0, 6, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %556 = vector.broadcast %555 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %557 = vector.transpose %556, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %558 = vector.extract_strided_slice %390 {offsets = [1, 0, 6, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %559 = vector.broadcast %558 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %560 = vector.transpose %559, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %561 = vector.extract_strided_slice %390 {offsets = [0, 0, 7, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %562 = vector.broadcast %561 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %563 = vector.transpose %562, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %564 = vector.extract_strided_slice %390 {offsets = [1, 0, 7, 0], sizes = [1, 4, 1, 1], strides = [1, 1, 1, 1]} : vector<2x4x8x1xf16> to vector<1x4x1x1xf16>
  %565 = vector.broadcast %564 : vector<1x4x1x1xf16> to vector<1x1x4x1x1xf16>
  %566 = vector.transpose %565, [3, 4, 0, 1, 2] : vector<1x1x4x1x1xf16> to vector<1x1x1x1x4xf16>
  %567 = vector.extract %521[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %567, %9[%11, %21, %workgroup_id_y, %391, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %568 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%391]
  %569 = vector.extract %524[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %569, %9[%11, %21, %workgroup_id_y, %568, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %570 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
  %571 = vector.extract %527[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %571, %9[%570, %21, %workgroup_id_y, %391, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %572 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%391]
  %573 = vector.extract %530[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %573, %9[%570, %21, %workgroup_id_y, %572, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %574 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%11]
  %575 = vector.extract %533[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %575, %9[%574, %21, %workgroup_id_y, %391, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %576 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%391]
  %577 = vector.extract %536[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %577, %9[%574, %21, %workgroup_id_y, %576, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %578 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%11]
  %579 = vector.extract %539[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %579, %9[%578, %21, %workgroup_id_y, %391, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %580 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%391]
  %581 = vector.extract %542[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %581, %9[%578, %21, %workgroup_id_y, %580, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %582 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%11]
  %583 = vector.extract %545[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %583, %9[%582, %21, %workgroup_id_y, %391, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %584 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%391]
  %585 = vector.extract %548[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %585, %9[%582, %21, %workgroup_id_y, %584, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %586 = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%11]
  %587 = vector.extract %551[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %587, %9[%586, %21, %workgroup_id_y, %391, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %588 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%391]
  %589 = vector.extract %554[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %589, %9[%586, %21, %workgroup_id_y, %588, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %590 = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%11]
  %591 = vector.extract %557[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %591, %9[%590, %21, %workgroup_id_y, %391, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %592 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%391]
  %593 = vector.extract %560[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %593, %9[%590, %21, %workgroup_id_y, %592, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %594 = affine.apply affine_map<()[s0] -> (s0 + 7)>()[%11]
  %595 = vector.extract %563[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %595, %9[%594, %21, %workgroup_id_y, %391, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  %596 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%391]
  %597 = vector.extract %566[0, 0, 0, 0] : vector<4xf16> from vector<1x1x1x1x4xf16>
  vector.store %597, %9[%594, %21, %workgroup_id_y, %596, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
  gpu.barrier
  memref.dealloc %alloc_4 : memref<1x32x20xf16, #gpu.address_space<workgroup>>
  memref.dealloc %alloc : memref<512x20xf16, #gpu.address_space<workgroup>>
  return
}
#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 + 2)>
#map2 = affine_map<()[s0] -> (s0 + 3)>
#map3 = affine_map<()[s0] -> (s0 + 4)>
#map4 = affine_map<()[s0] -> (s0 + 5)>
#map5 = affine_map<()[s0] -> (s0 + 6)>
#map6 = affine_map<()[s0] -> (s0 + 7)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module {
  func.func @main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32() {
    %cst = arith.constant dense<0.000000e+00> : vector<64xf16>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<8xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c48 = arith.constant 48 : index
    %c768 = arith.constant 768 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c131072 = arith.constant 131072 : index
    %c0 = arith.constant 0 : index
    %c671872 = arith.constant 671872 : index
    %c17449088 = arith.constant 17449088 : index
    %alloc = memref.alloc() : memref<512x20xf16, #gpu.address_space<workgroup>>
    %alloc_2 = memref.alloc() : memref<1x32x20xf16, #gpu.address_space<workgroup>>
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
    %subview = memref.subview %alloc_2[0, 0, 0] [1, 32, 16] [1, 1, 1] : memref<1x32x20xf16, #gpu.address_space<workgroup>> to memref<1x32x16xf16, strided<[640, 20, 1]>, #gpu.address_space<workgroup>>
    %subview_3 = memref.subview %alloc[0, 0] [512, 16] [1, 1] : memref<512x20xf16, #gpu.address_space<workgroup>> to memref<512x16xf16, strided<[20, 1]>, #gpu.address_space<workgroup>>
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
    %23 = arith.muli %10, %c64 overflow<nsw> : index
    %24 = arith.addi %12, %23 : index
    %25 = arith.floordivsi %24, %c8 : index
    %26 = arith.remsi %24, %c8 : index
    %27 = arith.cmpi slt, %26, %c0 : index
    %28 = arith.addi %26, %c8 : index
    %29 = arith.select %27, %28, %26 : index
    %30 = arith.floordivsi %24, %c2 : index
    %31 = arith.remsi %24, %c2 : index
    %32 = arith.cmpi slt, %31, %c0 : index
    %33 = arith.addi %31, %c2 : index
    %34 = arith.select %32, %33, %31 : index
    %35 = arith.addi %24, %c256 : index
    %36 = arith.floordivsi %35, %c2 : index
    %37 = arith.remsi %35, %c2 : index
    %38 = arith.cmpi slt, %37, %c0 : index
    %39 = arith.addi %37, %c2 : index
    %40 = arith.select %38, %39, %37 : index
    %41 = arith.addi %24, %c512 : index
    %42 = arith.floordivsi %41, %c2 : index
    %43 = arith.remsi %41, %c2 : index
    %44 = arith.cmpi slt, %43, %c0 : index
    %45 = arith.addi %43, %c2 : index
    %46 = arith.select %44, %45, %43 : index
    %47 = arith.addi %24, %c768 : index
    %48 = arith.floordivsi %47, %c2 : index
    %49 = arith.remsi %47, %c2 : index
    %50 = arith.cmpi slt, %49, %c0 : index
    %51 = arith.addi %49, %c2 : index
    %52 = arith.select %50, %51, %49 : index
    %expand_shape = memref.expand_shape %subview_3 [[0, 1], [2, 3]] output_shape [32, 16, 1, 16] : memref<512x16xf16, strided<[20, 1]>, #gpu.address_space<workgroup>> into memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>
    %expand_shape_4 = memref.expand_shape %subview [[0], [1, 2], [3, 4]] output_shape [1, 2, 16, 1, 16] : memref<1x32x16xf16, strided<[640, 20, 1]>, #gpu.address_space<workgroup>> into memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>
    %53 = arith.muli %29, %c2 overflow<nsw> : index
    %54 = arith.floordivsi %53, %c48 : index
    %55 = arith.remsi %53, %c48 : index
    %56 = arith.cmpi slt, %55, %c0 : index
    %57 = arith.addi %55, %c48 : index
    %58 = arith.select %56, %57, %55 : index
    %59 = arith.divsi %58, %c16 : index
    %60 = arith.remsi %53, %c16 : index
    %61 = arith.cmpi slt, %60, %c0 : index
    %62 = arith.addi %60, %c16 : index
    %63 = arith.select %61, %62, %60 : index
    %64 = arith.muli %34, %c8 overflow<nsw> : index
    %65 = arith.muli %40, %c8 overflow<nsw> : index
    %66 = arith.muli %46, %c8 overflow<nsw> : index
    %67 = arith.muli %52, %c8 overflow<nsw> : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    gpu.barrier
    %68 = arith.muli %workgroup_id_y, %c128 overflow<nsw> : index
    %69 = arith.addi %25, %68 : index
    %70 = arith.muli %workgroup_id_x, %c32 overflow<nsw> : index
    %71 = arith.addi %69, %70 : index
    %72 = arith.floordivsi %71, %c128 : index
    %73 = arith.remsi %71, %c128 : index
    %74 = arith.cmpi slt, %73, %c0 : index
    %75 = arith.addi %73, %c128 : index
    %76 = arith.select %74, %75, %73 : index
    %77 = arith.addi %54, %72 : index
    %78 = arith.addi %59, %76 : index
    %79 = vector.load %1[%77, %78, %63] : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf16>
    %80 = vector.load %3[%30, %64] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    %81 = vector.load %3[%36, %65] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    %82 = vector.load %3[%42, %66] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    %83 = vector.load %3[%48, %67] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    vector.store %79, %alloc_2[%c0, %25, %53] : memref<1x32x20xf16, #gpu.address_space<workgroup>>, vector<2xf16>
    vector.store %80, %alloc[%30, %64] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    vector.store %81, %alloc[%36, %65] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    vector.store %82, %alloc[%42, %66] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    vector.store %83, %alloc[%48, %67] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %84 = scf.for %arg0 = %c0 to %c8 step %c1 iter_args(%arg1 = %cst_1) -> (vector<64xf32>) {
      %396 = arith.addi %arg0, %c1 : index
      %397 = arith.muli %396, %c16 overflow<nsw> : index
      %398 = arith.addi %397, %53 : index
      %399 = arith.floordivsi %398, %c48 : index
      %400 = arith.remsi %398, %c48 : index
      %401 = arith.cmpi slt, %400, %c0 : index
      %402 = arith.addi %400, %c48 : index
      %403 = arith.select %401, %402, %400 : index
      %404 = arith.divsi %403, %c16 : index
      %405 = arith.remsi %398, %c16 : index
      %406 = arith.cmpi slt, %405, %c0 : index
      %407 = arith.addi %405, %c16 : index
      %408 = arith.select %406, %407, %405 : index
      %409 = arith.addi %399, %72 : index
      %410 = arith.addi %404, %76 : index
      %411 = vector.load %1[%409, %410, %408] : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf16>
      %412 = arith.addi %397, %64 : index
      %413 = vector.load %3[%30, %412] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
      %414 = arith.addi %397, %65 : index
      %415 = vector.load %3[%36, %414] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
      %416 = arith.addi %397, %66 : index
      %417 = vector.load %3[%42, %416] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
      %418 = arith.addi %397, %67 : index
      %419 = vector.load %3[%48, %418] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
      gpu.barrier
      %420 = vector.load %expand_shape_4[%c0, %c0, %21, %c0, %22] : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
      %421 = vector.load %expand_shape_4[%c0, %c1, %21, %c0, %22] : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
      %422 = vector.load %expand_shape[%11, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
      %423 = affine.apply #map()[%11]
      %424 = vector.load %expand_shape[%423, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
      %425 = affine.apply #map1()[%11]
      %426 = vector.load %expand_shape[%425, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
      %427 = affine.apply #map2()[%11]
      %428 = vector.load %expand_shape[%427, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
      %429 = affine.apply #map3()[%11]
      %430 = vector.load %expand_shape[%429, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
      %431 = affine.apply #map4()[%11]
      %432 = vector.load %expand_shape[%431, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
      %433 = affine.apply #map5()[%11]
      %434 = vector.load %expand_shape[%433, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
      %435 = affine.apply #map6()[%11]
      %436 = vector.load %expand_shape[%435, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
      %437 = vector.extract_strided_slice %arg1 {offsets = [0], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %438 = amdgpu.mfma %420 * %422 + %437 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %439 = vector.extract_strided_slice %arg1 {offsets = [4], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %440 = amdgpu.mfma %420 * %424 + %439 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %441 = vector.extract_strided_slice %arg1 {offsets = [8], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %442 = amdgpu.mfma %420 * %426 + %441 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %443 = vector.extract_strided_slice %arg1 {offsets = [12], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %444 = amdgpu.mfma %420 * %428 + %443 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %445 = vector.extract_strided_slice %arg1 {offsets = [16], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %446 = amdgpu.mfma %420 * %430 + %445 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %447 = vector.extract_strided_slice %arg1 {offsets = [20], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %448 = amdgpu.mfma %420 * %432 + %447 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %449 = vector.extract_strided_slice %arg1 {offsets = [24], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %450 = amdgpu.mfma %420 * %434 + %449 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %451 = vector.extract_strided_slice %arg1 {offsets = [28], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %452 = amdgpu.mfma %420 * %436 + %451 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %453 = vector.extract_strided_slice %arg1 {offsets = [32], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %454 = amdgpu.mfma %421 * %422 + %453 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %455 = vector.extract_strided_slice %arg1 {offsets = [36], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %456 = amdgpu.mfma %421 * %424 + %455 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %457 = vector.extract_strided_slice %arg1 {offsets = [40], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %458 = amdgpu.mfma %421 * %426 + %457 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %459 = vector.extract_strided_slice %arg1 {offsets = [44], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %460 = amdgpu.mfma %421 * %428 + %459 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %461 = vector.extract_strided_slice %arg1 {offsets = [48], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %462 = amdgpu.mfma %421 * %430 + %461 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %463 = vector.extract_strided_slice %arg1 {offsets = [52], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %464 = amdgpu.mfma %421 * %432 + %463 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %465 = vector.extract_strided_slice %arg1 {offsets = [56], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %466 = amdgpu.mfma %421 * %434 + %465 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %467 = vector.extract_strided_slice %arg1 {offsets = [60], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
      %468 = amdgpu.mfma %421 * %436 + %467 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %469 = vector.insert_strided_slice %438, %cst_1 {offsets = [0], strides = [1]} : vector<4xf32> into vector<64xf32>
      %470 = vector.insert_strided_slice %440, %469 {offsets = [4], strides = [1]} : vector<4xf32> into vector<64xf32>
      %471 = vector.insert_strided_slice %442, %470 {offsets = [8], strides = [1]} : vector<4xf32> into vector<64xf32>
      %472 = vector.insert_strided_slice %444, %471 {offsets = [12], strides = [1]} : vector<4xf32> into vector<64xf32>
      %473 = vector.insert_strided_slice %446, %472 {offsets = [16], strides = [1]} : vector<4xf32> into vector<64xf32>
      %474 = vector.insert_strided_slice %448, %473 {offsets = [20], strides = [1]} : vector<4xf32> into vector<64xf32>
      %475 = vector.insert_strided_slice %450, %474 {offsets = [24], strides = [1]} : vector<4xf32> into vector<64xf32>
      %476 = vector.insert_strided_slice %452, %475 {offsets = [28], strides = [1]} : vector<4xf32> into vector<64xf32>
      %477 = vector.insert_strided_slice %454, %476 {offsets = [32], strides = [1]} : vector<4xf32> into vector<64xf32>
      %478 = vector.insert_strided_slice %456, %477 {offsets = [36], strides = [1]} : vector<4xf32> into vector<64xf32>
      %479 = vector.insert_strided_slice %458, %478 {offsets = [40], strides = [1]} : vector<4xf32> into vector<64xf32>
      %480 = vector.insert_strided_slice %460, %479 {offsets = [44], strides = [1]} : vector<4xf32> into vector<64xf32>
      %481 = vector.insert_strided_slice %462, %480 {offsets = [48], strides = [1]} : vector<4xf32> into vector<64xf32>
      %482 = vector.insert_strided_slice %464, %481 {offsets = [52], strides = [1]} : vector<4xf32> into vector<64xf32>
      %483 = vector.insert_strided_slice %466, %482 {offsets = [56], strides = [1]} : vector<4xf32> into vector<64xf32>
      %484 = vector.insert_strided_slice %468, %483 {offsets = [60], strides = [1]} : vector<4xf32> into vector<64xf32>
      gpu.barrier
      vector.store %411, %alloc_2[%c0, %25, %53] : memref<1x32x20xf16, #gpu.address_space<workgroup>>, vector<2xf16>
      vector.store %413, %alloc[%30, %64] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      vector.store %415, %alloc[%36, %65] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      vector.store %417, %alloc[%42, %66] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      vector.store %419, %alloc[%48, %67] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      scf.yield %484 : vector<64xf32>
    }
    gpu.barrier
    %85 = vector.load %expand_shape_4[%c0, %c0, %21, %c0, %22] : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %86 = vector.load %expand_shape_4[%c0, %c1, %21, %c0, %22] : memref<1x2x16x1x16xf16, strided<[640, 320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %87 = vector.load %expand_shape[%11, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %88 = affine.apply #map()[%11]
    %89 = vector.load %expand_shape[%88, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %90 = affine.apply #map1()[%11]
    %91 = vector.load %expand_shape[%90, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %92 = affine.apply #map2()[%11]
    %93 = vector.load %expand_shape[%92, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %94 = affine.apply #map3()[%11]
    %95 = vector.load %expand_shape[%94, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %96 = affine.apply #map4()[%11]
    %97 = vector.load %expand_shape[%96, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %98 = affine.apply #map5()[%11]
    %99 = vector.load %expand_shape[%98, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %100 = affine.apply #map6()[%11]
    %101 = vector.load %expand_shape[%100, %21, %c0, %22] : memref<32x16x1x16xf16, strided<[320, 20, 16, 1]>, #gpu.address_space<workgroup>>, vector<4xf16>
    %102 = vector.extract_strided_slice %84 {offsets = [0], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %103 = amdgpu.mfma %85 * %87 + %102 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %104 = vector.extract_strided_slice %84 {offsets = [4], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %105 = amdgpu.mfma %85 * %89 + %104 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %106 = vector.extract_strided_slice %84 {offsets = [8], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %107 = amdgpu.mfma %85 * %91 + %106 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %108 = vector.extract_strided_slice %84 {offsets = [12], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %109 = amdgpu.mfma %85 * %93 + %108 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %110 = vector.extract_strided_slice %84 {offsets = [16], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %111 = amdgpu.mfma %85 * %95 + %110 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %112 = vector.extract_strided_slice %84 {offsets = [20], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %113 = amdgpu.mfma %85 * %97 + %112 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %114 = vector.extract_strided_slice %84 {offsets = [24], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %115 = amdgpu.mfma %85 * %99 + %114 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %116 = vector.extract_strided_slice %84 {offsets = [28], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %117 = amdgpu.mfma %85 * %101 + %116 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %118 = vector.extract_strided_slice %84 {offsets = [32], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %119 = amdgpu.mfma %86 * %87 + %118 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %120 = vector.extract_strided_slice %84 {offsets = [36], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %121 = amdgpu.mfma %86 * %89 + %120 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %122 = vector.extract_strided_slice %84 {offsets = [40], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %123 = amdgpu.mfma %86 * %91 + %122 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %124 = vector.extract_strided_slice %84 {offsets = [44], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %125 = amdgpu.mfma %86 * %93 + %124 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %126 = vector.extract_strided_slice %84 {offsets = [48], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %127 = amdgpu.mfma %86 * %95 + %126 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %128 = vector.extract_strided_slice %84 {offsets = [52], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %129 = amdgpu.mfma %86 * %97 + %128 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %130 = vector.extract_strided_slice %84 {offsets = [56], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %131 = amdgpu.mfma %86 * %99 + %130 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %132 = vector.extract_strided_slice %84 {offsets = [60], sizes = [4], strides = [1]} : vector<64xf32> to vector<4xf32>
    %133 = amdgpu.mfma %86 * %101 + %132 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %134 = vector.extract_strided_slice %103 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %135 = vector.insert_strided_slice %134, %cst_1 {offsets = [0], strides = [1]} : vector<1xf32> into vector<64xf32>
    %136 = vector.extract_strided_slice %105 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %137 = vector.insert_strided_slice %136, %135 {offsets = [1], strides = [1]} : vector<1xf32> into vector<64xf32>
    %138 = vector.extract_strided_slice %107 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %139 = vector.insert_strided_slice %138, %137 {offsets = [2], strides = [1]} : vector<1xf32> into vector<64xf32>
    %140 = vector.extract_strided_slice %109 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %141 = vector.insert_strided_slice %140, %139 {offsets = [3], strides = [1]} : vector<1xf32> into vector<64xf32>
    %142 = vector.extract_strided_slice %111 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %143 = vector.insert_strided_slice %142, %141 {offsets = [4], strides = [1]} : vector<1xf32> into vector<64xf32>
    %144 = vector.extract_strided_slice %113 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %145 = vector.insert_strided_slice %144, %143 {offsets = [5], strides = [1]} : vector<1xf32> into vector<64xf32>
    %146 = vector.extract_strided_slice %115 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %147 = vector.insert_strided_slice %146, %145 {offsets = [6], strides = [1]} : vector<1xf32> into vector<64xf32>
    %148 = vector.extract_strided_slice %117 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %149 = vector.insert_strided_slice %148, %147 {offsets = [7], strides = [1]} : vector<1xf32> into vector<64xf32>
    %150 = vector.extract_strided_slice %103 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %151 = vector.insert_strided_slice %150, %149 {offsets = [8], strides = [1]} : vector<1xf32> into vector<64xf32>
    %152 = vector.extract_strided_slice %105 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %153 = vector.insert_strided_slice %152, %151 {offsets = [9], strides = [1]} : vector<1xf32> into vector<64xf32>
    %154 = vector.extract_strided_slice %107 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %155 = vector.insert_strided_slice %154, %153 {offsets = [10], strides = [1]} : vector<1xf32> into vector<64xf32>
    %156 = vector.extract_strided_slice %109 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %157 = vector.insert_strided_slice %156, %155 {offsets = [11], strides = [1]} : vector<1xf32> into vector<64xf32>
    %158 = vector.extract_strided_slice %111 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %159 = vector.insert_strided_slice %158, %157 {offsets = [12], strides = [1]} : vector<1xf32> into vector<64xf32>
    %160 = vector.extract_strided_slice %113 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %161 = vector.insert_strided_slice %160, %159 {offsets = [13], strides = [1]} : vector<1xf32> into vector<64xf32>
    %162 = vector.extract_strided_slice %115 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %163 = vector.insert_strided_slice %162, %161 {offsets = [14], strides = [1]} : vector<1xf32> into vector<64xf32>
    %164 = vector.extract_strided_slice %117 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %165 = vector.insert_strided_slice %164, %163 {offsets = [15], strides = [1]} : vector<1xf32> into vector<64xf32>
    %166 = vector.extract_strided_slice %103 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %167 = vector.insert_strided_slice %166, %165 {offsets = [16], strides = [1]} : vector<1xf32> into vector<64xf32>
    %168 = vector.extract_strided_slice %105 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %169 = vector.insert_strided_slice %168, %167 {offsets = [17], strides = [1]} : vector<1xf32> into vector<64xf32>
    %170 = vector.extract_strided_slice %107 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %171 = vector.insert_strided_slice %170, %169 {offsets = [18], strides = [1]} : vector<1xf32> into vector<64xf32>
    %172 = vector.extract_strided_slice %109 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %173 = vector.insert_strided_slice %172, %171 {offsets = [19], strides = [1]} : vector<1xf32> into vector<64xf32>
    %174 = vector.extract_strided_slice %111 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %175 = vector.insert_strided_slice %174, %173 {offsets = [20], strides = [1]} : vector<1xf32> into vector<64xf32>
    %176 = vector.extract_strided_slice %113 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %177 = vector.insert_strided_slice %176, %175 {offsets = [21], strides = [1]} : vector<1xf32> into vector<64xf32>
    %178 = vector.extract_strided_slice %115 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %179 = vector.insert_strided_slice %178, %177 {offsets = [22], strides = [1]} : vector<1xf32> into vector<64xf32>
    %180 = vector.extract_strided_slice %117 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %181 = vector.insert_strided_slice %180, %179 {offsets = [23], strides = [1]} : vector<1xf32> into vector<64xf32>
    %182 = vector.extract_strided_slice %103 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %183 = vector.insert_strided_slice %182, %181 {offsets = [24], strides = [1]} : vector<1xf32> into vector<64xf32>
    %184 = vector.extract_strided_slice %105 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %185 = vector.insert_strided_slice %184, %183 {offsets = [25], strides = [1]} : vector<1xf32> into vector<64xf32>
    %186 = vector.extract_strided_slice %107 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %187 = vector.insert_strided_slice %186, %185 {offsets = [26], strides = [1]} : vector<1xf32> into vector<64xf32>
    %188 = vector.extract_strided_slice %109 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %189 = vector.insert_strided_slice %188, %187 {offsets = [27], strides = [1]} : vector<1xf32> into vector<64xf32>
    %190 = vector.extract_strided_slice %111 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %191 = vector.insert_strided_slice %190, %189 {offsets = [28], strides = [1]} : vector<1xf32> into vector<64xf32>
    %192 = vector.extract_strided_slice %113 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %193 = vector.insert_strided_slice %192, %191 {offsets = [29], strides = [1]} : vector<1xf32> into vector<64xf32>
    %194 = vector.extract_strided_slice %115 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %195 = vector.insert_strided_slice %194, %193 {offsets = [30], strides = [1]} : vector<1xf32> into vector<64xf32>
    %196 = vector.extract_strided_slice %117 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %197 = vector.insert_strided_slice %196, %195 {offsets = [31], strides = [1]} : vector<1xf32> into vector<64xf32>
    %198 = vector.extract_strided_slice %119 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %199 = vector.insert_strided_slice %198, %197 {offsets = [32], strides = [1]} : vector<1xf32> into vector<64xf32>
    %200 = vector.extract_strided_slice %121 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %201 = vector.insert_strided_slice %200, %199 {offsets = [33], strides = [1]} : vector<1xf32> into vector<64xf32>
    %202 = vector.extract_strided_slice %123 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %203 = vector.insert_strided_slice %202, %201 {offsets = [34], strides = [1]} : vector<1xf32> into vector<64xf32>
    %204 = vector.extract_strided_slice %125 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %205 = vector.insert_strided_slice %204, %203 {offsets = [35], strides = [1]} : vector<1xf32> into vector<64xf32>
    %206 = vector.extract_strided_slice %127 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %207 = vector.insert_strided_slice %206, %205 {offsets = [36], strides = [1]} : vector<1xf32> into vector<64xf32>
    %208 = vector.extract_strided_slice %129 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %209 = vector.insert_strided_slice %208, %207 {offsets = [37], strides = [1]} : vector<1xf32> into vector<64xf32>
    %210 = vector.extract_strided_slice %131 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %211 = vector.insert_strided_slice %210, %209 {offsets = [38], strides = [1]} : vector<1xf32> into vector<64xf32>
    %212 = vector.extract_strided_slice %133 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %213 = vector.insert_strided_slice %212, %211 {offsets = [39], strides = [1]} : vector<1xf32> into vector<64xf32>
    %214 = vector.extract_strided_slice %119 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %215 = vector.insert_strided_slice %214, %213 {offsets = [40], strides = [1]} : vector<1xf32> into vector<64xf32>
    %216 = vector.extract_strided_slice %121 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %217 = vector.insert_strided_slice %216, %215 {offsets = [41], strides = [1]} : vector<1xf32> into vector<64xf32>
    %218 = vector.extract_strided_slice %123 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %219 = vector.insert_strided_slice %218, %217 {offsets = [42], strides = [1]} : vector<1xf32> into vector<64xf32>
    %220 = vector.extract_strided_slice %125 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %221 = vector.insert_strided_slice %220, %219 {offsets = [43], strides = [1]} : vector<1xf32> into vector<64xf32>
    %222 = vector.extract_strided_slice %127 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %223 = vector.insert_strided_slice %222, %221 {offsets = [44], strides = [1]} : vector<1xf32> into vector<64xf32>
    %224 = vector.extract_strided_slice %129 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %225 = vector.insert_strided_slice %224, %223 {offsets = [45], strides = [1]} : vector<1xf32> into vector<64xf32>
    %226 = vector.extract_strided_slice %131 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %227 = vector.insert_strided_slice %226, %225 {offsets = [46], strides = [1]} : vector<1xf32> into vector<64xf32>
    %228 = vector.extract_strided_slice %133 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %229 = vector.insert_strided_slice %228, %227 {offsets = [47], strides = [1]} : vector<1xf32> into vector<64xf32>
    %230 = vector.extract_strided_slice %119 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %231 = vector.insert_strided_slice %230, %229 {offsets = [48], strides = [1]} : vector<1xf32> into vector<64xf32>
    %232 = vector.extract_strided_slice %121 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %233 = vector.insert_strided_slice %232, %231 {offsets = [49], strides = [1]} : vector<1xf32> into vector<64xf32>
    %234 = vector.extract_strided_slice %123 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %235 = vector.insert_strided_slice %234, %233 {offsets = [50], strides = [1]} : vector<1xf32> into vector<64xf32>
    %236 = vector.extract_strided_slice %125 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %237 = vector.insert_strided_slice %236, %235 {offsets = [51], strides = [1]} : vector<1xf32> into vector<64xf32>
    %238 = vector.extract_strided_slice %127 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %239 = vector.insert_strided_slice %238, %237 {offsets = [52], strides = [1]} : vector<1xf32> into vector<64xf32>
    %240 = vector.extract_strided_slice %129 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %241 = vector.insert_strided_slice %240, %239 {offsets = [53], strides = [1]} : vector<1xf32> into vector<64xf32>
    %242 = vector.extract_strided_slice %131 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %243 = vector.insert_strided_slice %242, %241 {offsets = [54], strides = [1]} : vector<1xf32> into vector<64xf32>
    %244 = vector.extract_strided_slice %133 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %245 = vector.insert_strided_slice %244, %243 {offsets = [55], strides = [1]} : vector<1xf32> into vector<64xf32>
    %246 = vector.extract_strided_slice %119 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %247 = vector.insert_strided_slice %246, %245 {offsets = [56], strides = [1]} : vector<1xf32> into vector<64xf32>
    %248 = vector.extract_strided_slice %121 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %249 = vector.insert_strided_slice %248, %247 {offsets = [57], strides = [1]} : vector<1xf32> into vector<64xf32>
    %250 = vector.extract_strided_slice %123 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %251 = vector.insert_strided_slice %250, %249 {offsets = [58], strides = [1]} : vector<1xf32> into vector<64xf32>
    %252 = vector.extract_strided_slice %125 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %253 = vector.insert_strided_slice %252, %251 {offsets = [59], strides = [1]} : vector<1xf32> into vector<64xf32>
    %254 = vector.extract_strided_slice %127 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %255 = vector.insert_strided_slice %254, %253 {offsets = [60], strides = [1]} : vector<1xf32> into vector<64xf32>
    %256 = vector.extract_strided_slice %129 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %257 = vector.insert_strided_slice %256, %255 {offsets = [61], strides = [1]} : vector<1xf32> into vector<64xf32>
    %258 = vector.extract_strided_slice %131 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %259 = vector.insert_strided_slice %258, %257 {offsets = [62], strides = [1]} : vector<1xf32> into vector<64xf32>
    %260 = vector.extract_strided_slice %133 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %261 = vector.insert_strided_slice %260, %259 {offsets = [63], strides = [1]} : vector<1xf32> into vector<64xf32>
    %262 = vector.load %5[%11, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %263 = vector.insert_strided_slice %262, %cst_0 {offsets = [0], strides = [1]} : vector<1xf16> into vector<8xf16>
    %264 = vector.load %5[%88, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %265 = vector.insert_strided_slice %264, %263 {offsets = [1], strides = [1]} : vector<1xf16> into vector<8xf16>
    %266 = vector.load %5[%90, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %267 = vector.insert_strided_slice %266, %265 {offsets = [2], strides = [1]} : vector<1xf16> into vector<8xf16>
    %268 = vector.load %5[%92, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %269 = vector.insert_strided_slice %268, %267 {offsets = [3], strides = [1]} : vector<1xf16> into vector<8xf16>
    %270 = vector.load %5[%94, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %271 = vector.insert_strided_slice %270, %269 {offsets = [4], strides = [1]} : vector<1xf16> into vector<8xf16>
    %272 = vector.load %5[%96, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %273 = vector.insert_strided_slice %272, %271 {offsets = [5], strides = [1]} : vector<1xf16> into vector<8xf16>
    %274 = vector.load %5[%98, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %275 = vector.insert_strided_slice %274, %273 {offsets = [6], strides = [1]} : vector<1xf16> into vector<8xf16>
    %276 = vector.load %5[%100, %21] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %277 = vector.insert_strided_slice %276, %275 {offsets = [7], strides = [1]} : vector<1xf16> into vector<8xf16>
    %278 = arith.extf %277 : vector<8xf16> to vector<8xf32>
    %279 = vector.extract_strided_slice %261 {offsets = [0], sizes = [8], strides = [1]} : vector<64xf32> to vector<8xf32>
    %280 = arith.addf %279, %278 : vector<8xf32>
    %281 = vector.extract_strided_slice %261 {offsets = [8], sizes = [8], strides = [1]} : vector<64xf32> to vector<8xf32>
    %282 = arith.addf %281, %278 : vector<8xf32>
    %283 = vector.extract_strided_slice %261 {offsets = [16], sizes = [8], strides = [1]} : vector<64xf32> to vector<8xf32>
    %284 = arith.addf %283, %278 : vector<8xf32>
    %285 = vector.extract_strided_slice %261 {offsets = [24], sizes = [8], strides = [1]} : vector<64xf32> to vector<8xf32>
    %286 = arith.addf %285, %278 : vector<8xf32>
    %287 = vector.extract_strided_slice %261 {offsets = [32], sizes = [8], strides = [1]} : vector<64xf32> to vector<8xf32>
    %288 = arith.addf %287, %278 : vector<8xf32>
    %289 = vector.extract_strided_slice %261 {offsets = [40], sizes = [8], strides = [1]} : vector<64xf32> to vector<8xf32>
    %290 = arith.addf %289, %278 : vector<8xf32>
    %291 = vector.extract_strided_slice %261 {offsets = [48], sizes = [8], strides = [1]} : vector<64xf32> to vector<8xf32>
    %292 = arith.addf %291, %278 : vector<8xf32>
    %293 = vector.extract_strided_slice %261 {offsets = [56], sizes = [8], strides = [1]} : vector<64xf32> to vector<8xf32>
    %294 = arith.addf %293, %278 : vector<8xf32>
    %295 = arith.truncf %280 : vector<8xf32> to vector<8xf16>
    %296 = vector.insert_strided_slice %295, %cst {offsets = [0], strides = [1]} : vector<8xf16> into vector<64xf16>
    %297 = arith.truncf %282 : vector<8xf32> to vector<8xf16>
    %298 = vector.insert_strided_slice %297, %296 {offsets = [8], strides = [1]} : vector<8xf16> into vector<64xf16>
    %299 = arith.truncf %284 : vector<8xf32> to vector<8xf16>
    %300 = vector.insert_strided_slice %299, %298 {offsets = [16], strides = [1]} : vector<8xf16> into vector<64xf16>
    %301 = arith.truncf %286 : vector<8xf32> to vector<8xf16>
    %302 = vector.insert_strided_slice %301, %300 {offsets = [24], strides = [1]} : vector<8xf16> into vector<64xf16>
    %303 = arith.truncf %288 : vector<8xf32> to vector<8xf16>
    %304 = vector.insert_strided_slice %303, %302 {offsets = [32], strides = [1]} : vector<8xf16> into vector<64xf16>
    %305 = arith.truncf %290 : vector<8xf32> to vector<8xf16>
    %306 = vector.insert_strided_slice %305, %304 {offsets = [40], strides = [1]} : vector<8xf16> into vector<64xf16>
    %307 = arith.truncf %292 : vector<8xf32> to vector<8xf16>
    %308 = vector.insert_strided_slice %307, %306 {offsets = [48], strides = [1]} : vector<8xf16> into vector<64xf16>
    %309 = arith.truncf %294 : vector<8xf32> to vector<8xf16>
    %310 = vector.insert_strided_slice %309, %308 {offsets = [56], strides = [1]} : vector<8xf16> into vector<64xf16>
    %311 = arith.muli %workgroup_id_x, %c2 overflow<nsw> : index
    %312 = vector.extract_strided_slice %295 {offsets = [0], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %312, %7[%workgroup_id_y, %311, %22, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %313 = vector.extract_strided_slice %295 {offsets = [1], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %313, %7[%workgroup_id_y, %311, %22, %88, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %314 = vector.extract_strided_slice %295 {offsets = [2], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %314, %7[%workgroup_id_y, %311, %22, %90, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %315 = vector.extract_strided_slice %295 {offsets = [3], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %315, %7[%workgroup_id_y, %311, %22, %92, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %316 = vector.extract_strided_slice %295 {offsets = [4], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %316, %7[%workgroup_id_y, %311, %22, %94, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %317 = vector.extract_strided_slice %295 {offsets = [5], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %317, %7[%workgroup_id_y, %311, %22, %96, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %318 = vector.extract_strided_slice %295 {offsets = [6], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %318, %7[%workgroup_id_y, %311, %22, %98, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %319 = vector.extract_strided_slice %295 {offsets = [7], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %319, %7[%workgroup_id_y, %311, %22, %100, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %320 = affine.apply #map()[%22]
    %321 = vector.extract_strided_slice %297 {offsets = [0], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %321, %7[%workgroup_id_y, %311, %320, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %322 = vector.extract_strided_slice %297 {offsets = [1], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %322, %7[%workgroup_id_y, %311, %320, %88, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %323 = vector.extract_strided_slice %297 {offsets = [2], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %323, %7[%workgroup_id_y, %311, %320, %90, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %324 = vector.extract_strided_slice %297 {offsets = [3], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %324, %7[%workgroup_id_y, %311, %320, %92, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %325 = vector.extract_strided_slice %297 {offsets = [4], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %325, %7[%workgroup_id_y, %311, %320, %94, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %326 = vector.extract_strided_slice %297 {offsets = [5], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %326, %7[%workgroup_id_y, %311, %320, %96, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %327 = vector.extract_strided_slice %297 {offsets = [6], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %327, %7[%workgroup_id_y, %311, %320, %98, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %328 = vector.extract_strided_slice %297 {offsets = [7], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %328, %7[%workgroup_id_y, %311, %320, %100, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %329 = affine.apply #map1()[%22]
    %330 = vector.extract_strided_slice %299 {offsets = [0], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %330, %7[%workgroup_id_y, %311, %329, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %331 = vector.extract_strided_slice %299 {offsets = [1], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %331, %7[%workgroup_id_y, %311, %329, %88, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %332 = vector.extract_strided_slice %299 {offsets = [2], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %332, %7[%workgroup_id_y, %311, %329, %90, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %333 = vector.extract_strided_slice %299 {offsets = [3], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %333, %7[%workgroup_id_y, %311, %329, %92, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %334 = vector.extract_strided_slice %299 {offsets = [4], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %334, %7[%workgroup_id_y, %311, %329, %94, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %335 = vector.extract_strided_slice %299 {offsets = [5], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %335, %7[%workgroup_id_y, %311, %329, %96, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %336 = vector.extract_strided_slice %299 {offsets = [6], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %336, %7[%workgroup_id_y, %311, %329, %98, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %337 = vector.extract_strided_slice %299 {offsets = [7], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %337, %7[%workgroup_id_y, %311, %329, %100, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %338 = affine.apply #map2()[%22]
    %339 = vector.extract_strided_slice %301 {offsets = [0], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %339, %7[%workgroup_id_y, %311, %338, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %340 = vector.extract_strided_slice %301 {offsets = [1], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %340, %7[%workgroup_id_y, %311, %338, %88, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %341 = vector.extract_strided_slice %301 {offsets = [2], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %341, %7[%workgroup_id_y, %311, %338, %90, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %342 = vector.extract_strided_slice %301 {offsets = [3], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %342, %7[%workgroup_id_y, %311, %338, %92, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %343 = vector.extract_strided_slice %301 {offsets = [4], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %343, %7[%workgroup_id_y, %311, %338, %94, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %344 = vector.extract_strided_slice %301 {offsets = [5], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %344, %7[%workgroup_id_y, %311, %338, %96, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %345 = vector.extract_strided_slice %301 {offsets = [6], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %345, %7[%workgroup_id_y, %311, %338, %98, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %346 = vector.extract_strided_slice %301 {offsets = [7], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %346, %7[%workgroup_id_y, %311, %338, %100, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %347 = affine.apply #map()[%311]
    %348 = vector.extract_strided_slice %303 {offsets = [0], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %348, %7[%workgroup_id_y, %347, %22, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %349 = vector.extract_strided_slice %303 {offsets = [1], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %349, %7[%workgroup_id_y, %347, %22, %88, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %350 = vector.extract_strided_slice %303 {offsets = [2], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %350, %7[%workgroup_id_y, %347, %22, %90, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %351 = vector.extract_strided_slice %303 {offsets = [3], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %351, %7[%workgroup_id_y, %347, %22, %92, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %352 = vector.extract_strided_slice %303 {offsets = [4], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %352, %7[%workgroup_id_y, %347, %22, %94, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %353 = vector.extract_strided_slice %303 {offsets = [5], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %353, %7[%workgroup_id_y, %347, %22, %96, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %354 = vector.extract_strided_slice %303 {offsets = [6], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %354, %7[%workgroup_id_y, %347, %22, %98, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %355 = vector.extract_strided_slice %303 {offsets = [7], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %355, %7[%workgroup_id_y, %347, %22, %100, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %356 = vector.extract_strided_slice %305 {offsets = [0], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %356, %7[%workgroup_id_y, %347, %320, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %357 = vector.extract_strided_slice %305 {offsets = [1], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %357, %7[%workgroup_id_y, %347, %320, %88, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %358 = vector.extract_strided_slice %305 {offsets = [2], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %358, %7[%workgroup_id_y, %347, %320, %90, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %359 = vector.extract_strided_slice %305 {offsets = [3], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %359, %7[%workgroup_id_y, %347, %320, %92, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %360 = vector.extract_strided_slice %305 {offsets = [4], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %360, %7[%workgroup_id_y, %347, %320, %94, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %361 = vector.extract_strided_slice %305 {offsets = [5], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %361, %7[%workgroup_id_y, %347, %320, %96, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %362 = vector.extract_strided_slice %305 {offsets = [6], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %362, %7[%workgroup_id_y, %347, %320, %98, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %363 = vector.extract_strided_slice %305 {offsets = [7], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %363, %7[%workgroup_id_y, %347, %320, %100, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %364 = vector.extract_strided_slice %307 {offsets = [0], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %364, %7[%workgroup_id_y, %347, %329, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %365 = vector.extract_strided_slice %307 {offsets = [1], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %365, %7[%workgroup_id_y, %347, %329, %88, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %366 = vector.extract_strided_slice %307 {offsets = [2], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %366, %7[%workgroup_id_y, %347, %329, %90, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %367 = vector.extract_strided_slice %307 {offsets = [3], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %367, %7[%workgroup_id_y, %347, %329, %92, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %368 = vector.extract_strided_slice %307 {offsets = [4], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %368, %7[%workgroup_id_y, %347, %329, %94, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %369 = vector.extract_strided_slice %307 {offsets = [5], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %369, %7[%workgroup_id_y, %347, %329, %96, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %370 = vector.extract_strided_slice %307 {offsets = [6], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %370, %7[%workgroup_id_y, %347, %329, %98, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %371 = vector.extract_strided_slice %307 {offsets = [7], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %371, %7[%workgroup_id_y, %347, %329, %100, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %372 = vector.extract_strided_slice %309 {offsets = [0], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %372, %7[%workgroup_id_y, %347, %338, %11, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %373 = vector.extract_strided_slice %309 {offsets = [1], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %373, %7[%workgroup_id_y, %347, %338, %88, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %374 = vector.extract_strided_slice %309 {offsets = [2], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %374, %7[%workgroup_id_y, %347, %338, %90, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %375 = vector.extract_strided_slice %309 {offsets = [3], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %375, %7[%workgroup_id_y, %347, %338, %92, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %376 = vector.extract_strided_slice %309 {offsets = [4], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %376, %7[%workgroup_id_y, %347, %338, %94, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %377 = vector.extract_strided_slice %309 {offsets = [5], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %377, %7[%workgroup_id_y, %347, %338, %96, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %378 = vector.extract_strided_slice %309 {offsets = [6], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %378, %7[%workgroup_id_y, %347, %338, %98, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %379 = vector.extract_strided_slice %309 {offsets = [7], sizes = [1], strides = [1]} : vector<8xf16> to vector<1xf16>
    vector.store %379, %7[%workgroup_id_y, %347, %338, %100, %21] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %380 = vector.shuffle %310, %310 [0, 8, 16, 24] : vector<64xf16>, vector<64xf16>
    %381 = vector.shuffle %310, %310 [32, 40, 48, 56] : vector<64xf16>, vector<64xf16>
    %382 = vector.shuffle %310, %310 [1, 9, 17, 25] : vector<64xf16>, vector<64xf16>
    %383 = vector.shuffle %310, %310 [33, 41, 49, 57] : vector<64xf16>, vector<64xf16>
    %384 = vector.shuffle %310, %310 [2, 10, 18, 26] : vector<64xf16>, vector<64xf16>
    %385 = vector.shuffle %310, %310 [34, 42, 50, 58] : vector<64xf16>, vector<64xf16>
    %386 = vector.shuffle %310, %310 [3, 11, 19, 27] : vector<64xf16>, vector<64xf16>
    %387 = vector.shuffle %310, %310 [35, 43, 51, 59] : vector<64xf16>, vector<64xf16>
    %388 = vector.shuffle %310, %310 [4, 12, 20, 28] : vector<64xf16>, vector<64xf16>
    %389 = vector.shuffle %310, %310 [36, 44, 52, 60] : vector<64xf16>, vector<64xf16>
    %390 = vector.shuffle %310, %310 [5, 13, 21, 29] : vector<64xf16>, vector<64xf16>
    %391 = vector.shuffle %310, %310 [37, 45, 53, 61] : vector<64xf16>, vector<64xf16>
    %392 = vector.shuffle %310, %310 [6, 14, 22, 30] : vector<64xf16>, vector<64xf16>
    %393 = vector.shuffle %310, %310 [38, 46, 54, 62] : vector<64xf16>, vector<64xf16>
    %394 = vector.shuffle %310, %310 [7, 15, 23, 31] : vector<64xf16>, vector<64xf16>
    %395 = vector.shuffle %310, %310 [39, 47, 55, 63] : vector<64xf16>, vector<64xf16>
    vector.store %380, %9[%11, %21, %workgroup_id_y, %311, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %381, %9[%11, %21, %workgroup_id_y, %347, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %382, %9[%88, %21, %workgroup_id_y, %311, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %383, %9[%88, %21, %workgroup_id_y, %347, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %384, %9[%90, %21, %workgroup_id_y, %311, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %385, %9[%90, %21, %workgroup_id_y, %347, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %386, %9[%92, %21, %workgroup_id_y, %311, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %387, %9[%92, %21, %workgroup_id_y, %347, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %388, %9[%94, %21, %workgroup_id_y, %311, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %389, %9[%94, %21, %workgroup_id_y, %347, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %390, %9[%96, %21, %workgroup_id_y, %311, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %391, %9[%96, %21, %workgroup_id_y, %347, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %392, %9[%98, %21, %workgroup_id_y, %311, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %393, %9[%98, %21, %workgroup_id_y, %347, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %394, %9[%100, %21, %workgroup_id_y, %311, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    vector.store %395, %9[%100, %21, %workgroup_id_y, %347, %22] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    gpu.barrier
    memref.dealloc %alloc_2 : memref<1x32x20xf16, #gpu.address_space<workgroup>>
    memref.dealloc %alloc : memref<512x20xf16, #gpu.address_space<workgroup>>
    return
  }
}

