#map = affine_map<()[s0] -> (s0 + 16)>
#map1 = affine_map<()[s0, s1] -> (s0 * 16 + s1)>
#map2 = affine_map<()[s0, s1] -> (s0 * 16 + s1 + 16)>
#map3 = affine_map<()[s0, s1] -> (s0 * 16 + s1 + 32)>
#map4 = affine_map<()[s0, s1] -> (s0 * 16 + s1 + 48)>
#map5 = affine_map<()[s0, s1] -> (s0 * 16 + s1 + 64)>
#map6 = affine_map<()[s0, s1] -> (s0 * 16 + s1 + 80)>
#map7 = affine_map<()[s0, s1] -> (s0 * 16 + s1 + 96)>
#map8 = affine_map<()[s0, s1] -> (s0 * 16 + s1 + 112)>
#map9 = affine_map<()[s0] -> (s0 + 1)>
#map10 = affine_map<()[s0] -> (s0 + 2)>
#map11 = affine_map<()[s0] -> (s0 + 3)>
#map12 = affine_map<()[s0] -> (s0 + 4)>
#map13 = affine_map<()[s0] -> (s0 + 5)>
#map14 = affine_map<()[s0] -> (s0 + 6)>
#map15 = affine_map<()[s0] -> (s0 + 7)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module {
  func.func @main$async_dispatch_2_conv_128x128x512x3x3x16_f16xf16xf32() {
    %cst = arith.constant dense<0.000000e+00> : vector<8x1xf16>
    %0 = ub.poison : vector<2x4x8x1xf32>
    %1 = ub.poison : vector<4x8x1xf32>
    %2 = ub.poison : vector<4x1xf32>
    %3 = ub.poison : vector<4xf32>
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
    %cst_0 = arith.constant dense<0.000000e+00> : vector<2x8x4x1xf32>
    %c8 = arith.constant 8 : index
    %cst_1 = arith.constant dense<0.000000e+00> : vector<1x2x8x4x1xf32>
    %c1 = arith.constant 1 : index
    %c131072 = arith.constant 131072 : index
    %c0 = arith.constant 0 : index
    %c671872 = arith.constant 671872 : index
    %c17449088 = arith.constant 17449088 : index
    %alloc = memref.alloc() : memref<512x20xf16, #gpu.address_space<workgroup>>
    %alloc_2 = memref.alloc() : memref<1x32x20xf16, #gpu.address_space<workgroup>>
    %thread_id_x = gpu.thread_id  x upper_bound 256
    %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c131072) flags("ReadOnly|Indirect") : memref<130x130x16xf16, strided<[2080, 16, 1], offset: ?>, #gpu.address_space<global>>
    %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<130x130x16xf16, strided<[2080, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
    memref.assume_alignment %5, 64 : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
    %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<512x144xf16, #gpu.address_space<global>>
    %7 = amdgpu.fat_raw_buffer_cast %6 resetOffset : memref<512x144xf16, #gpu.address_space<global>> to memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>
    memref.assume_alignment %7, 64 : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>
    %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x16xf16, #gpu.address_space<global>>
    %9 = amdgpu.fat_raw_buffer_cast %8 resetOffset : memref<32x16xf16, #gpu.address_space<global>> to memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>
    memref.assume_alignment %9, 64 : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>
    %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c671872) flags(Indirect) : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1], offset: ?>, #gpu.address_space<global>>
    %11 = amdgpu.fat_raw_buffer_cast %10 resetOffset : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
    memref.assume_alignment %11, 64 : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
    %12 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c17449088) flags(Indirect) : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1], offset: ?>, #gpu.address_space<global>>
    %13 = amdgpu.fat_raw_buffer_cast %12 resetOffset : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1], offset: ?>, #gpu.address_space<global>> to memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
    memref.assume_alignment %13, 64 : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>
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
    %83 = vector.load %5[%81, %82, %67] : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf16>
    %84 = vector.load %7[%34, %68] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    %85 = vector.load %7[%40, %69] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    %86 = vector.load %7[%46, %70] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    %87 = vector.load %7[%52, %71] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
    vector.store %83, %alloc_2[%c0, %29, %57] : memref<1x32x20xf16, #gpu.address_space<workgroup>>, vector<2xf16>
    vector.store %84, %alloc[%34, %68] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    vector.store %85, %alloc[%40, %69] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    vector.store %86, %alloc[%46, %70] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    vector.store %87, %alloc[%52, %71] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %88 = scf.for %arg0 = %c0 to %c8 step %c1 iter_args(%arg1 = %cst_1) -> (vector<1x2x8x4x1xf32>) {
      %518 = arith.addi %arg0, %c1 : index
      %519 = arith.muli %518, %c16 overflow<nsw> : index
      %520 = arith.addi %519, %57 : index
      %521 = arith.floordivsi %520, %c48 : index
      %522 = arith.remsi %520, %c48 : index
      %523 = arith.cmpi slt, %522, %c0 : index
      %524 = arith.addi %522, %c48 : index
      %525 = arith.select %523, %524, %522 : index
      %526 = arith.divsi %525, %c16 : index
      %527 = arith.remsi %520, %c16 : index
      %528 = arith.cmpi slt, %527, %c0 : index
      %529 = arith.addi %527, %c16 : index
      %530 = arith.select %528, %529, %527 : index
      %531 = arith.addi %521, %76 : index
      %532 = arith.addi %526, %80 : index
      %533 = vector.load %5[%531, %532, %530] : memref<130x130x16xf16, strided<[2080, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<2xf16>
      %534 = arith.addi %519, %68 : index
      %535 = vector.load %7[%34, %534] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
      %536 = arith.addi %519, %69 : index
      %537 = vector.load %7[%40, %536] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
      %538 = arith.addi %519, %70 : index
      %539 = vector.load %7[%46, %538] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
      %540 = arith.addi %519, %71 : index
      %541 = vector.load %7[%52, %540] : memref<512x144xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<8xf16>
      gpu.barrier
      %542 = vector.load %alloc_2[%c0, %25, %26] : memref<1x32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
      %543 = affine.apply #map()[%25]
      %544 = vector.load %alloc_2[%c0, %543, %26] : memref<1x32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
      %545 = affine.apply #map1()[%15, %25]
      %546 = vector.load %alloc[%545, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
      %547 = affine.apply #map2()[%15, %25]
      %548 = vector.load %alloc[%547, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
      %549 = affine.apply #map3()[%15, %25]
      %550 = vector.load %alloc[%549, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
      %551 = affine.apply #map4()[%15, %25]
      %552 = vector.load %alloc[%551, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
      %553 = affine.apply #map5()[%15, %25]
      %554 = vector.load %alloc[%553, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
      %555 = affine.apply #map6()[%15, %25]
      %556 = vector.load %alloc[%555, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
      %557 = affine.apply #map7()[%15, %25]
      %558 = vector.load %alloc[%557, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
      %559 = affine.apply #map8()[%15, %25]
      %560 = vector.load %alloc[%559, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
      %561 = vector.extract %arg1[0, 0, 0, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %562 = vector.insert_strided_slice %561, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %563 = vector.extract %arg1[0, 0, 0, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %564 = vector.insert_strided_slice %563, %562 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %565 = vector.extract %arg1[0, 0, 0, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %566 = vector.insert_strided_slice %565, %564 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %567 = vector.extract %arg1[0, 0, 0, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %568 = vector.insert_strided_slice %567, %566 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %569 = amdgpu.mfma %542 * %546 + %568 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %570 = vector.extract_strided_slice %569 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %571 = vector.insert %570, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %572 = vector.extract_strided_slice %569 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %573 = vector.insert %572, %571 [1] : vector<1xf32> into vector<4x1xf32>
      %574 = vector.extract_strided_slice %569 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %575 = vector.insert %574, %573 [2] : vector<1xf32> into vector<4x1xf32>
      %576 = vector.extract_strided_slice %569 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %577 = vector.insert %576, %575 [3] : vector<1xf32> into vector<4x1xf32>
      %578 = vector.extract %arg1[0, 0, 1, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %579 = vector.insert_strided_slice %578, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %580 = vector.extract %arg1[0, 0, 1, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %581 = vector.insert_strided_slice %580, %579 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %582 = vector.extract %arg1[0, 0, 1, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %583 = vector.insert_strided_slice %582, %581 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %584 = vector.extract %arg1[0, 0, 1, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %585 = vector.insert_strided_slice %584, %583 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %586 = amdgpu.mfma %542 * %548 + %585 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %587 = vector.extract_strided_slice %586 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %588 = vector.insert %587, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %589 = vector.extract_strided_slice %586 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %590 = vector.insert %589, %588 [1] : vector<1xf32> into vector<4x1xf32>
      %591 = vector.extract_strided_slice %586 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %592 = vector.insert %591, %590 [2] : vector<1xf32> into vector<4x1xf32>
      %593 = vector.extract_strided_slice %586 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %594 = vector.insert %593, %592 [3] : vector<1xf32> into vector<4x1xf32>
      %595 = vector.extract %arg1[0, 0, 2, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %596 = vector.insert_strided_slice %595, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %597 = vector.extract %arg1[0, 0, 2, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %598 = vector.insert_strided_slice %597, %596 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %599 = vector.extract %arg1[0, 0, 2, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %600 = vector.insert_strided_slice %599, %598 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %601 = vector.extract %arg1[0, 0, 2, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %602 = vector.insert_strided_slice %601, %600 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %603 = amdgpu.mfma %542 * %550 + %602 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %604 = vector.extract_strided_slice %603 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %605 = vector.insert %604, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %606 = vector.extract_strided_slice %603 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %607 = vector.insert %606, %605 [1] : vector<1xf32> into vector<4x1xf32>
      %608 = vector.extract_strided_slice %603 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %609 = vector.insert %608, %607 [2] : vector<1xf32> into vector<4x1xf32>
      %610 = vector.extract_strided_slice %603 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %611 = vector.insert %610, %609 [3] : vector<1xf32> into vector<4x1xf32>
      %612 = vector.extract %arg1[0, 0, 3, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %613 = vector.insert_strided_slice %612, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %614 = vector.extract %arg1[0, 0, 3, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %615 = vector.insert_strided_slice %614, %613 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %616 = vector.extract %arg1[0, 0, 3, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %617 = vector.insert_strided_slice %616, %615 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %618 = vector.extract %arg1[0, 0, 3, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %619 = vector.insert_strided_slice %618, %617 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %620 = amdgpu.mfma %542 * %552 + %619 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %621 = vector.extract_strided_slice %620 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %622 = vector.insert %621, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %623 = vector.extract_strided_slice %620 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %624 = vector.insert %623, %622 [1] : vector<1xf32> into vector<4x1xf32>
      %625 = vector.extract_strided_slice %620 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %626 = vector.insert %625, %624 [2] : vector<1xf32> into vector<4x1xf32>
      %627 = vector.extract_strided_slice %620 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %628 = vector.insert %627, %626 [3] : vector<1xf32> into vector<4x1xf32>
      %629 = vector.extract %arg1[0, 0, 4, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %630 = vector.insert_strided_slice %629, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %631 = vector.extract %arg1[0, 0, 4, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %632 = vector.insert_strided_slice %631, %630 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %633 = vector.extract %arg1[0, 0, 4, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %634 = vector.insert_strided_slice %633, %632 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %635 = vector.extract %arg1[0, 0, 4, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %636 = vector.insert_strided_slice %635, %634 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %637 = amdgpu.mfma %542 * %554 + %636 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %638 = vector.extract_strided_slice %637 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %639 = vector.insert %638, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %640 = vector.extract_strided_slice %637 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %641 = vector.insert %640, %639 [1] : vector<1xf32> into vector<4x1xf32>
      %642 = vector.extract_strided_slice %637 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %643 = vector.insert %642, %641 [2] : vector<1xf32> into vector<4x1xf32>
      %644 = vector.extract_strided_slice %637 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %645 = vector.insert %644, %643 [3] : vector<1xf32> into vector<4x1xf32>
      %646 = vector.extract %arg1[0, 0, 5, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %647 = vector.insert_strided_slice %646, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %648 = vector.extract %arg1[0, 0, 5, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %649 = vector.insert_strided_slice %648, %647 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %650 = vector.extract %arg1[0, 0, 5, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %651 = vector.insert_strided_slice %650, %649 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %652 = vector.extract %arg1[0, 0, 5, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %653 = vector.insert_strided_slice %652, %651 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %654 = amdgpu.mfma %542 * %556 + %653 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %655 = vector.extract_strided_slice %654 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %656 = vector.insert %655, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %657 = vector.extract_strided_slice %654 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %658 = vector.insert %657, %656 [1] : vector<1xf32> into vector<4x1xf32>
      %659 = vector.extract_strided_slice %654 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %660 = vector.insert %659, %658 [2] : vector<1xf32> into vector<4x1xf32>
      %661 = vector.extract_strided_slice %654 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %662 = vector.insert %661, %660 [3] : vector<1xf32> into vector<4x1xf32>
      %663 = vector.extract %arg1[0, 0, 6, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %664 = vector.insert_strided_slice %663, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %665 = vector.extract %arg1[0, 0, 6, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %666 = vector.insert_strided_slice %665, %664 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %667 = vector.extract %arg1[0, 0, 6, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %668 = vector.insert_strided_slice %667, %666 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %669 = vector.extract %arg1[0, 0, 6, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %670 = vector.insert_strided_slice %669, %668 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %671 = amdgpu.mfma %542 * %558 + %670 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %672 = vector.extract_strided_slice %671 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %673 = vector.insert %672, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %674 = vector.extract_strided_slice %671 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %675 = vector.insert %674, %673 [1] : vector<1xf32> into vector<4x1xf32>
      %676 = vector.extract_strided_slice %671 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %677 = vector.insert %676, %675 [2] : vector<1xf32> into vector<4x1xf32>
      %678 = vector.extract_strided_slice %671 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %679 = vector.insert %678, %677 [3] : vector<1xf32> into vector<4x1xf32>
      %680 = vector.extract %arg1[0, 0, 7, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %681 = vector.insert_strided_slice %680, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %682 = vector.extract %arg1[0, 0, 7, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %683 = vector.insert_strided_slice %682, %681 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %684 = vector.extract %arg1[0, 0, 7, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %685 = vector.insert_strided_slice %684, %683 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %686 = vector.extract %arg1[0, 0, 7, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %687 = vector.insert_strided_slice %686, %685 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %688 = amdgpu.mfma %542 * %560 + %687 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %689 = vector.extract_strided_slice %688 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %690 = vector.insert %689, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %691 = vector.extract_strided_slice %688 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %692 = vector.insert %691, %690 [1] : vector<1xf32> into vector<4x1xf32>
      %693 = vector.extract_strided_slice %688 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %694 = vector.insert %693, %692 [2] : vector<1xf32> into vector<4x1xf32>
      %695 = vector.extract_strided_slice %688 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %696 = vector.insert %695, %694 [3] : vector<1xf32> into vector<4x1xf32>
      %697 = vector.extract %arg1[0, 1, 0, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %698 = vector.insert_strided_slice %697, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %699 = vector.extract %arg1[0, 1, 0, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %700 = vector.insert_strided_slice %699, %698 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %701 = vector.extract %arg1[0, 1, 0, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %702 = vector.insert_strided_slice %701, %700 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %703 = vector.extract %arg1[0, 1, 0, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %704 = vector.insert_strided_slice %703, %702 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %705 = amdgpu.mfma %544 * %546 + %704 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %706 = vector.extract_strided_slice %705 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %707 = vector.insert %706, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %708 = vector.extract_strided_slice %705 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %709 = vector.insert %708, %707 [1] : vector<1xf32> into vector<4x1xf32>
      %710 = vector.extract_strided_slice %705 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %711 = vector.insert %710, %709 [2] : vector<1xf32> into vector<4x1xf32>
      %712 = vector.extract_strided_slice %705 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %713 = vector.insert %712, %711 [3] : vector<1xf32> into vector<4x1xf32>
      %714 = vector.extract %arg1[0, 1, 1, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %715 = vector.insert_strided_slice %714, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %716 = vector.extract %arg1[0, 1, 1, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %717 = vector.insert_strided_slice %716, %715 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %718 = vector.extract %arg1[0, 1, 1, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %719 = vector.insert_strided_slice %718, %717 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %720 = vector.extract %arg1[0, 1, 1, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %721 = vector.insert_strided_slice %720, %719 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %722 = amdgpu.mfma %544 * %548 + %721 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %723 = vector.extract_strided_slice %722 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %724 = vector.insert %723, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %725 = vector.extract_strided_slice %722 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %726 = vector.insert %725, %724 [1] : vector<1xf32> into vector<4x1xf32>
      %727 = vector.extract_strided_slice %722 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %728 = vector.insert %727, %726 [2] : vector<1xf32> into vector<4x1xf32>
      %729 = vector.extract_strided_slice %722 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %730 = vector.insert %729, %728 [3] : vector<1xf32> into vector<4x1xf32>
      %731 = vector.extract %arg1[0, 1, 2, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %732 = vector.insert_strided_slice %731, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %733 = vector.extract %arg1[0, 1, 2, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %734 = vector.insert_strided_slice %733, %732 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %735 = vector.extract %arg1[0, 1, 2, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %736 = vector.insert_strided_slice %735, %734 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %737 = vector.extract %arg1[0, 1, 2, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %738 = vector.insert_strided_slice %737, %736 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %739 = amdgpu.mfma %544 * %550 + %738 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %740 = vector.extract_strided_slice %739 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %741 = vector.insert %740, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %742 = vector.extract_strided_slice %739 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %743 = vector.insert %742, %741 [1] : vector<1xf32> into vector<4x1xf32>
      %744 = vector.extract_strided_slice %739 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %745 = vector.insert %744, %743 [2] : vector<1xf32> into vector<4x1xf32>
      %746 = vector.extract_strided_slice %739 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %747 = vector.insert %746, %745 [3] : vector<1xf32> into vector<4x1xf32>
      %748 = vector.extract %arg1[0, 1, 3, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %749 = vector.insert_strided_slice %748, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %750 = vector.extract %arg1[0, 1, 3, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %751 = vector.insert_strided_slice %750, %749 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %752 = vector.extract %arg1[0, 1, 3, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %753 = vector.insert_strided_slice %752, %751 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %754 = vector.extract %arg1[0, 1, 3, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %755 = vector.insert_strided_slice %754, %753 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %756 = amdgpu.mfma %544 * %552 + %755 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %757 = vector.extract_strided_slice %756 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %758 = vector.insert %757, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %759 = vector.extract_strided_slice %756 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %760 = vector.insert %759, %758 [1] : vector<1xf32> into vector<4x1xf32>
      %761 = vector.extract_strided_slice %756 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %762 = vector.insert %761, %760 [2] : vector<1xf32> into vector<4x1xf32>
      %763 = vector.extract_strided_slice %756 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %764 = vector.insert %763, %762 [3] : vector<1xf32> into vector<4x1xf32>
      %765 = vector.extract %arg1[0, 1, 4, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %766 = vector.insert_strided_slice %765, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %767 = vector.extract %arg1[0, 1, 4, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %768 = vector.insert_strided_slice %767, %766 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %769 = vector.extract %arg1[0, 1, 4, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %770 = vector.insert_strided_slice %769, %768 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %771 = vector.extract %arg1[0, 1, 4, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %772 = vector.insert_strided_slice %771, %770 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %773 = amdgpu.mfma %544 * %554 + %772 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %774 = vector.extract_strided_slice %773 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %775 = vector.insert %774, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %776 = vector.extract_strided_slice %773 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %777 = vector.insert %776, %775 [1] : vector<1xf32> into vector<4x1xf32>
      %778 = vector.extract_strided_slice %773 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %779 = vector.insert %778, %777 [2] : vector<1xf32> into vector<4x1xf32>
      %780 = vector.extract_strided_slice %773 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %781 = vector.insert %780, %779 [3] : vector<1xf32> into vector<4x1xf32>
      %782 = vector.extract %arg1[0, 1, 5, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %783 = vector.insert_strided_slice %782, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %784 = vector.extract %arg1[0, 1, 5, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %785 = vector.insert_strided_slice %784, %783 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %786 = vector.extract %arg1[0, 1, 5, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %787 = vector.insert_strided_slice %786, %785 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %788 = vector.extract %arg1[0, 1, 5, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %789 = vector.insert_strided_slice %788, %787 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %790 = amdgpu.mfma %544 * %556 + %789 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %791 = vector.extract_strided_slice %790 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %792 = vector.insert %791, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %793 = vector.extract_strided_slice %790 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %794 = vector.insert %793, %792 [1] : vector<1xf32> into vector<4x1xf32>
      %795 = vector.extract_strided_slice %790 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %796 = vector.insert %795, %794 [2] : vector<1xf32> into vector<4x1xf32>
      %797 = vector.extract_strided_slice %790 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %798 = vector.insert %797, %796 [3] : vector<1xf32> into vector<4x1xf32>
      %799 = vector.extract %arg1[0, 1, 6, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %800 = vector.insert_strided_slice %799, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %801 = vector.extract %arg1[0, 1, 6, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %802 = vector.insert_strided_slice %801, %800 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %803 = vector.extract %arg1[0, 1, 6, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %804 = vector.insert_strided_slice %803, %802 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %805 = vector.extract %arg1[0, 1, 6, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %806 = vector.insert_strided_slice %805, %804 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %807 = amdgpu.mfma %544 * %558 + %806 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %808 = vector.extract_strided_slice %807 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %809 = vector.insert %808, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %810 = vector.extract_strided_slice %807 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %811 = vector.insert %810, %809 [1] : vector<1xf32> into vector<4x1xf32>
      %812 = vector.extract_strided_slice %807 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %813 = vector.insert %812, %811 [2] : vector<1xf32> into vector<4x1xf32>
      %814 = vector.extract_strided_slice %807 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %815 = vector.insert %814, %813 [3] : vector<1xf32> into vector<4x1xf32>
      %816 = vector.extract %arg1[0, 1, 7, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %817 = vector.insert_strided_slice %816, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
      %818 = vector.extract %arg1[0, 1, 7, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %819 = vector.insert_strided_slice %818, %817 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
      %820 = vector.extract %arg1[0, 1, 7, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %821 = vector.insert_strided_slice %820, %819 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
      %822 = vector.extract %arg1[0, 1, 7, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
      %823 = vector.insert_strided_slice %822, %821 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
      %824 = amdgpu.mfma %544 * %560 + %823 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
      %825 = vector.extract_strided_slice %824 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %826 = vector.insert %825, %2 [0] : vector<1xf32> into vector<4x1xf32>
      %827 = vector.extract_strided_slice %824 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %828 = vector.insert %827, %826 [1] : vector<1xf32> into vector<4x1xf32>
      %829 = vector.extract_strided_slice %824 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %830 = vector.insert %829, %828 [2] : vector<1xf32> into vector<4x1xf32>
      %831 = vector.extract_strided_slice %824 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
      %832 = vector.insert %831, %830 [3] : vector<1xf32> into vector<4x1xf32>
      %833 = vector.insert_strided_slice %577, %cst_0 {offsets = [0, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %834 = vector.insert_strided_slice %594, %833 {offsets = [0, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %835 = vector.insert_strided_slice %611, %834 {offsets = [0, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %836 = vector.insert_strided_slice %628, %835 {offsets = [0, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %837 = vector.insert_strided_slice %645, %836 {offsets = [0, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %838 = vector.insert_strided_slice %662, %837 {offsets = [0, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %839 = vector.insert_strided_slice %679, %838 {offsets = [0, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %840 = vector.insert_strided_slice %696, %839 {offsets = [0, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %841 = vector.insert_strided_slice %713, %840 {offsets = [1, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %842 = vector.insert_strided_slice %730, %841 {offsets = [1, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %843 = vector.insert_strided_slice %747, %842 {offsets = [1, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %844 = vector.insert_strided_slice %764, %843 {offsets = [1, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %845 = vector.insert_strided_slice %781, %844 {offsets = [1, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %846 = vector.insert_strided_slice %798, %845 {offsets = [1, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %847 = vector.insert_strided_slice %815, %846 {offsets = [1, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %848 = vector.insert_strided_slice %832, %847 {offsets = [1, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
      %849 = vector.broadcast %848 : vector<2x8x4x1xf32> to vector<1x2x8x4x1xf32>
      gpu.barrier
      vector.store %533, %alloc_2[%c0, %29, %57] : memref<1x32x20xf16, #gpu.address_space<workgroup>>, vector<2xf16>
      vector.store %535, %alloc[%34, %68] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      vector.store %537, %alloc[%40, %69] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      vector.store %539, %alloc[%46, %70] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      vector.store %541, %alloc[%52, %71] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
      scf.yield %849 : vector<1x2x8x4x1xf32>
    }
    gpu.barrier
    %89 = vector.load %alloc_2[%c0, %25, %26] : memref<1x32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    %90 = affine.apply #map()[%25]
    %91 = vector.load %alloc_2[%c0, %90, %26] : memref<1x32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    %92 = affine.apply #map1()[%15, %25]
    %93 = vector.load %alloc[%92, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    %94 = affine.apply #map2()[%15, %25]
    %95 = vector.load %alloc[%94, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    %96 = affine.apply #map3()[%15, %25]
    %97 = vector.load %alloc[%96, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    %98 = affine.apply #map4()[%15, %25]
    %99 = vector.load %alloc[%98, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    %100 = affine.apply #map5()[%15, %25]
    %101 = vector.load %alloc[%100, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    %102 = affine.apply #map6()[%15, %25]
    %103 = vector.load %alloc[%102, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    %104 = affine.apply #map7()[%15, %25]
    %105 = vector.load %alloc[%104, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    %106 = affine.apply #map8()[%15, %25]
    %107 = vector.load %alloc[%106, %26] : memref<512x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    %108 = vector.extract %88[0, 0, 0, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %109 = vector.insert_strided_slice %108, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %110 = vector.extract %88[0, 0, 0, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %111 = vector.insert_strided_slice %110, %109 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %112 = vector.extract %88[0, 0, 0, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %113 = vector.insert_strided_slice %112, %111 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %114 = vector.extract %88[0, 0, 0, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %115 = vector.insert_strided_slice %114, %113 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %116 = amdgpu.mfma %89 * %93 + %115 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %117 = vector.extract_strided_slice %116 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %118 = vector.insert %117, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %119 = vector.extract_strided_slice %116 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %120 = vector.insert %119, %118 [1] : vector<1xf32> into vector<4x1xf32>
    %121 = vector.extract_strided_slice %116 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %122 = vector.insert %121, %120 [2] : vector<1xf32> into vector<4x1xf32>
    %123 = vector.extract_strided_slice %116 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %124 = vector.insert %123, %122 [3] : vector<1xf32> into vector<4x1xf32>
    %125 = vector.extract %88[0, 0, 1, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %126 = vector.insert_strided_slice %125, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %127 = vector.extract %88[0, 0, 1, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %128 = vector.insert_strided_slice %127, %126 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %129 = vector.extract %88[0, 0, 1, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %130 = vector.insert_strided_slice %129, %128 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %131 = vector.extract %88[0, 0, 1, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %132 = vector.insert_strided_slice %131, %130 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %133 = amdgpu.mfma %89 * %95 + %132 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %134 = vector.extract_strided_slice %133 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %135 = vector.insert %134, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %136 = vector.extract_strided_slice %133 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %137 = vector.insert %136, %135 [1] : vector<1xf32> into vector<4x1xf32>
    %138 = vector.extract_strided_slice %133 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %139 = vector.insert %138, %137 [2] : vector<1xf32> into vector<4x1xf32>
    %140 = vector.extract_strided_slice %133 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %141 = vector.insert %140, %139 [3] : vector<1xf32> into vector<4x1xf32>
    %142 = vector.extract %88[0, 0, 2, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %143 = vector.insert_strided_slice %142, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %144 = vector.extract %88[0, 0, 2, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %145 = vector.insert_strided_slice %144, %143 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %146 = vector.extract %88[0, 0, 2, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %147 = vector.insert_strided_slice %146, %145 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %148 = vector.extract %88[0, 0, 2, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %149 = vector.insert_strided_slice %148, %147 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %150 = amdgpu.mfma %89 * %97 + %149 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %151 = vector.extract_strided_slice %150 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %152 = vector.insert %151, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %153 = vector.extract_strided_slice %150 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %154 = vector.insert %153, %152 [1] : vector<1xf32> into vector<4x1xf32>
    %155 = vector.extract_strided_slice %150 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %156 = vector.insert %155, %154 [2] : vector<1xf32> into vector<4x1xf32>
    %157 = vector.extract_strided_slice %150 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %158 = vector.insert %157, %156 [3] : vector<1xf32> into vector<4x1xf32>
    %159 = vector.extract %88[0, 0, 3, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %160 = vector.insert_strided_slice %159, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %161 = vector.extract %88[0, 0, 3, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %162 = vector.insert_strided_slice %161, %160 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %163 = vector.extract %88[0, 0, 3, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %164 = vector.insert_strided_slice %163, %162 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %165 = vector.extract %88[0, 0, 3, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %166 = vector.insert_strided_slice %165, %164 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %167 = amdgpu.mfma %89 * %99 + %166 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %168 = vector.extract_strided_slice %167 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %169 = vector.insert %168, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %170 = vector.extract_strided_slice %167 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %171 = vector.insert %170, %169 [1] : vector<1xf32> into vector<4x1xf32>
    %172 = vector.extract_strided_slice %167 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %173 = vector.insert %172, %171 [2] : vector<1xf32> into vector<4x1xf32>
    %174 = vector.extract_strided_slice %167 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %175 = vector.insert %174, %173 [3] : vector<1xf32> into vector<4x1xf32>
    %176 = vector.extract %88[0, 0, 4, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %177 = vector.insert_strided_slice %176, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %178 = vector.extract %88[0, 0, 4, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %179 = vector.insert_strided_slice %178, %177 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %180 = vector.extract %88[0, 0, 4, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %181 = vector.insert_strided_slice %180, %179 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %182 = vector.extract %88[0, 0, 4, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %183 = vector.insert_strided_slice %182, %181 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %184 = amdgpu.mfma %89 * %101 + %183 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %185 = vector.extract_strided_slice %184 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %186 = vector.insert %185, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %187 = vector.extract_strided_slice %184 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %188 = vector.insert %187, %186 [1] : vector<1xf32> into vector<4x1xf32>
    %189 = vector.extract_strided_slice %184 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %190 = vector.insert %189, %188 [2] : vector<1xf32> into vector<4x1xf32>
    %191 = vector.extract_strided_slice %184 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %192 = vector.insert %191, %190 [3] : vector<1xf32> into vector<4x1xf32>
    %193 = vector.extract %88[0, 0, 5, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %194 = vector.insert_strided_slice %193, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %195 = vector.extract %88[0, 0, 5, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %196 = vector.insert_strided_slice %195, %194 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %197 = vector.extract %88[0, 0, 5, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %198 = vector.insert_strided_slice %197, %196 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %199 = vector.extract %88[0, 0, 5, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %200 = vector.insert_strided_slice %199, %198 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %201 = amdgpu.mfma %89 * %103 + %200 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %202 = vector.extract_strided_slice %201 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %203 = vector.insert %202, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %204 = vector.extract_strided_slice %201 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %205 = vector.insert %204, %203 [1] : vector<1xf32> into vector<4x1xf32>
    %206 = vector.extract_strided_slice %201 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %207 = vector.insert %206, %205 [2] : vector<1xf32> into vector<4x1xf32>
    %208 = vector.extract_strided_slice %201 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %209 = vector.insert %208, %207 [3] : vector<1xf32> into vector<4x1xf32>
    %210 = vector.extract %88[0, 0, 6, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %211 = vector.insert_strided_slice %210, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %212 = vector.extract %88[0, 0, 6, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %213 = vector.insert_strided_slice %212, %211 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %214 = vector.extract %88[0, 0, 6, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %215 = vector.insert_strided_slice %214, %213 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %216 = vector.extract %88[0, 0, 6, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %217 = vector.insert_strided_slice %216, %215 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %218 = amdgpu.mfma %89 * %105 + %217 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %219 = vector.extract_strided_slice %218 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %220 = vector.insert %219, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %221 = vector.extract_strided_slice %218 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %222 = vector.insert %221, %220 [1] : vector<1xf32> into vector<4x1xf32>
    %223 = vector.extract_strided_slice %218 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %224 = vector.insert %223, %222 [2] : vector<1xf32> into vector<4x1xf32>
    %225 = vector.extract_strided_slice %218 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %226 = vector.insert %225, %224 [3] : vector<1xf32> into vector<4x1xf32>
    %227 = vector.extract %88[0, 0, 7, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %228 = vector.insert_strided_slice %227, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %229 = vector.extract %88[0, 0, 7, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %230 = vector.insert_strided_slice %229, %228 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %231 = vector.extract %88[0, 0, 7, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %232 = vector.insert_strided_slice %231, %230 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %233 = vector.extract %88[0, 0, 7, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %234 = vector.insert_strided_slice %233, %232 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %235 = amdgpu.mfma %89 * %107 + %234 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %236 = vector.extract_strided_slice %235 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %237 = vector.insert %236, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %238 = vector.extract_strided_slice %235 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %239 = vector.insert %238, %237 [1] : vector<1xf32> into vector<4x1xf32>
    %240 = vector.extract_strided_slice %235 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %241 = vector.insert %240, %239 [2] : vector<1xf32> into vector<4x1xf32>
    %242 = vector.extract_strided_slice %235 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %243 = vector.insert %242, %241 [3] : vector<1xf32> into vector<4x1xf32>
    %244 = vector.extract %88[0, 1, 0, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %245 = vector.insert_strided_slice %244, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %246 = vector.extract %88[0, 1, 0, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %247 = vector.insert_strided_slice %246, %245 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %248 = vector.extract %88[0, 1, 0, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %249 = vector.insert_strided_slice %248, %247 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %250 = vector.extract %88[0, 1, 0, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %251 = vector.insert_strided_slice %250, %249 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %252 = amdgpu.mfma %91 * %93 + %251 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %253 = vector.extract_strided_slice %252 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %254 = vector.insert %253, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %255 = vector.extract_strided_slice %252 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %256 = vector.insert %255, %254 [1] : vector<1xf32> into vector<4x1xf32>
    %257 = vector.extract_strided_slice %252 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %258 = vector.insert %257, %256 [2] : vector<1xf32> into vector<4x1xf32>
    %259 = vector.extract_strided_slice %252 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %260 = vector.insert %259, %258 [3] : vector<1xf32> into vector<4x1xf32>
    %261 = vector.extract %88[0, 1, 1, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %262 = vector.insert_strided_slice %261, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %263 = vector.extract %88[0, 1, 1, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %264 = vector.insert_strided_slice %263, %262 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %265 = vector.extract %88[0, 1, 1, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %266 = vector.insert_strided_slice %265, %264 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %267 = vector.extract %88[0, 1, 1, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %268 = vector.insert_strided_slice %267, %266 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %269 = amdgpu.mfma %91 * %95 + %268 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %270 = vector.extract_strided_slice %269 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %271 = vector.insert %270, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %272 = vector.extract_strided_slice %269 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %273 = vector.insert %272, %271 [1] : vector<1xf32> into vector<4x1xf32>
    %274 = vector.extract_strided_slice %269 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %275 = vector.insert %274, %273 [2] : vector<1xf32> into vector<4x1xf32>
    %276 = vector.extract_strided_slice %269 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %277 = vector.insert %276, %275 [3] : vector<1xf32> into vector<4x1xf32>
    %278 = vector.extract %88[0, 1, 2, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %279 = vector.insert_strided_slice %278, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %280 = vector.extract %88[0, 1, 2, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %281 = vector.insert_strided_slice %280, %279 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %282 = vector.extract %88[0, 1, 2, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %283 = vector.insert_strided_slice %282, %281 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %284 = vector.extract %88[0, 1, 2, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %285 = vector.insert_strided_slice %284, %283 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %286 = amdgpu.mfma %91 * %97 + %285 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %287 = vector.extract_strided_slice %286 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %288 = vector.insert %287, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %289 = vector.extract_strided_slice %286 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %290 = vector.insert %289, %288 [1] : vector<1xf32> into vector<4x1xf32>
    %291 = vector.extract_strided_slice %286 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %292 = vector.insert %291, %290 [2] : vector<1xf32> into vector<4x1xf32>
    %293 = vector.extract_strided_slice %286 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %294 = vector.insert %293, %292 [3] : vector<1xf32> into vector<4x1xf32>
    %295 = vector.extract %88[0, 1, 3, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %296 = vector.insert_strided_slice %295, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %297 = vector.extract %88[0, 1, 3, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %298 = vector.insert_strided_slice %297, %296 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %299 = vector.extract %88[0, 1, 3, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %300 = vector.insert_strided_slice %299, %298 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %301 = vector.extract %88[0, 1, 3, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %302 = vector.insert_strided_slice %301, %300 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %303 = amdgpu.mfma %91 * %99 + %302 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %304 = vector.extract_strided_slice %303 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %305 = vector.insert %304, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %306 = vector.extract_strided_slice %303 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %307 = vector.insert %306, %305 [1] : vector<1xf32> into vector<4x1xf32>
    %308 = vector.extract_strided_slice %303 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %309 = vector.insert %308, %307 [2] : vector<1xf32> into vector<4x1xf32>
    %310 = vector.extract_strided_slice %303 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %311 = vector.insert %310, %309 [3] : vector<1xf32> into vector<4x1xf32>
    %312 = vector.extract %88[0, 1, 4, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %313 = vector.insert_strided_slice %312, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %314 = vector.extract %88[0, 1, 4, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %315 = vector.insert_strided_slice %314, %313 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %316 = vector.extract %88[0, 1, 4, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %317 = vector.insert_strided_slice %316, %315 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %318 = vector.extract %88[0, 1, 4, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %319 = vector.insert_strided_slice %318, %317 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %320 = amdgpu.mfma %91 * %101 + %319 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %321 = vector.extract_strided_slice %320 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %322 = vector.insert %321, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %323 = vector.extract_strided_slice %320 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %324 = vector.insert %323, %322 [1] : vector<1xf32> into vector<4x1xf32>
    %325 = vector.extract_strided_slice %320 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %326 = vector.insert %325, %324 [2] : vector<1xf32> into vector<4x1xf32>
    %327 = vector.extract_strided_slice %320 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %328 = vector.insert %327, %326 [3] : vector<1xf32> into vector<4x1xf32>
    %329 = vector.extract %88[0, 1, 5, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %330 = vector.insert_strided_slice %329, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %331 = vector.extract %88[0, 1, 5, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %332 = vector.insert_strided_slice %331, %330 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %333 = vector.extract %88[0, 1, 5, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %334 = vector.insert_strided_slice %333, %332 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %335 = vector.extract %88[0, 1, 5, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %336 = vector.insert_strided_slice %335, %334 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %337 = amdgpu.mfma %91 * %103 + %336 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %338 = vector.extract_strided_slice %337 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %339 = vector.insert %338, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %340 = vector.extract_strided_slice %337 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %341 = vector.insert %340, %339 [1] : vector<1xf32> into vector<4x1xf32>
    %342 = vector.extract_strided_slice %337 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %343 = vector.insert %342, %341 [2] : vector<1xf32> into vector<4x1xf32>
    %344 = vector.extract_strided_slice %337 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %345 = vector.insert %344, %343 [3] : vector<1xf32> into vector<4x1xf32>
    %346 = vector.extract %88[0, 1, 6, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %347 = vector.insert_strided_slice %346, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %348 = vector.extract %88[0, 1, 6, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %349 = vector.insert_strided_slice %348, %347 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %350 = vector.extract %88[0, 1, 6, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %351 = vector.insert_strided_slice %350, %349 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %352 = vector.extract %88[0, 1, 6, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %353 = vector.insert_strided_slice %352, %351 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %354 = amdgpu.mfma %91 * %105 + %353 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %355 = vector.extract_strided_slice %354 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %356 = vector.insert %355, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %357 = vector.extract_strided_slice %354 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %358 = vector.insert %357, %356 [1] : vector<1xf32> into vector<4x1xf32>
    %359 = vector.extract_strided_slice %354 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %360 = vector.insert %359, %358 [2] : vector<1xf32> into vector<4x1xf32>
    %361 = vector.extract_strided_slice %354 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %362 = vector.insert %361, %360 [3] : vector<1xf32> into vector<4x1xf32>
    %363 = vector.extract %88[0, 1, 7, 0] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %364 = vector.insert_strided_slice %363, %3 {offsets = [0], strides = [1]} : vector<1xf32> into vector<4xf32>
    %365 = vector.extract %88[0, 1, 7, 1] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %366 = vector.insert_strided_slice %365, %364 {offsets = [1], strides = [1]} : vector<1xf32> into vector<4xf32>
    %367 = vector.extract %88[0, 1, 7, 2] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %368 = vector.insert_strided_slice %367, %366 {offsets = [2], strides = [1]} : vector<1xf32> into vector<4xf32>
    %369 = vector.extract %88[0, 1, 7, 3] : vector<1xf32> from vector<1x2x8x4x1xf32>
    %370 = vector.insert_strided_slice %369, %368 {offsets = [3], strides = [1]} : vector<1xf32> into vector<4xf32>
    %371 = amdgpu.mfma %91 * %107 + %370 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %372 = vector.extract_strided_slice %371 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %373 = vector.insert %372, %2 [0] : vector<1xf32> into vector<4x1xf32>
    %374 = vector.extract_strided_slice %371 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %375 = vector.insert %374, %373 [1] : vector<1xf32> into vector<4x1xf32>
    %376 = vector.extract_strided_slice %371 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %377 = vector.insert %376, %375 [2] : vector<1xf32> into vector<4x1xf32>
    %378 = vector.extract_strided_slice %371 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    %379 = vector.insert %378, %377 [3] : vector<1xf32> into vector<4x1xf32>
    %380 = vector.insert_strided_slice %124, %cst_0 {offsets = [0, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %381 = vector.insert_strided_slice %141, %380 {offsets = [0, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %382 = vector.insert_strided_slice %158, %381 {offsets = [0, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %383 = vector.insert_strided_slice %175, %382 {offsets = [0, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %384 = vector.insert_strided_slice %192, %383 {offsets = [0, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %385 = vector.insert_strided_slice %209, %384 {offsets = [0, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %386 = vector.insert_strided_slice %226, %385 {offsets = [0, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %387 = vector.insert_strided_slice %243, %386 {offsets = [0, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %388 = vector.insert_strided_slice %260, %387 {offsets = [1, 0, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %389 = vector.insert_strided_slice %277, %388 {offsets = [1, 1, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %390 = vector.insert_strided_slice %294, %389 {offsets = [1, 2, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %391 = vector.insert_strided_slice %311, %390 {offsets = [1, 3, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %392 = vector.insert_strided_slice %328, %391 {offsets = [1, 4, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %393 = vector.insert_strided_slice %345, %392 {offsets = [1, 5, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %394 = vector.insert_strided_slice %362, %393 {offsets = [1, 6, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %395 = vector.insert_strided_slice %379, %394 {offsets = [1, 7, 0, 0], strides = [1, 1]} : vector<4x1xf32> into vector<2x8x4x1xf32>
    %396 = vector.broadcast %395 : vector<2x8x4x1xf32> to vector<1x2x8x4x1xf32>
    %397 = vector.transpose %396, [0, 1, 3, 2, 4] : vector<1x2x8x4x1xf32> to vector<1x2x4x8x1xf32>
    %398 = vector.extract %397[0] : vector<2x4x8x1xf32> from vector<1x2x4x8x1xf32>
    %399 = vector.load %9[%15, %25] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %400 = vector.insert %399, %cst [0] : vector<1xf16> into vector<8x1xf16>
    %401 = affine.apply #map9()[%15]
    %402 = vector.load %9[%401, %25] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %403 = vector.insert %402, %400 [1] : vector<1xf16> into vector<8x1xf16>
    %404 = affine.apply #map10()[%15]
    %405 = vector.load %9[%404, %25] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %406 = vector.insert %405, %403 [2] : vector<1xf16> into vector<8x1xf16>
    %407 = affine.apply #map11()[%15]
    %408 = vector.load %9[%407, %25] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %409 = vector.insert %408, %406 [3] : vector<1xf16> into vector<8x1xf16>
    %410 = affine.apply #map12()[%15]
    %411 = vector.load %9[%410, %25] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %412 = vector.insert %411, %409 [4] : vector<1xf16> into vector<8x1xf16>
    %413 = affine.apply #map13()[%15]
    %414 = vector.load %9[%413, %25] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %415 = vector.insert %414, %412 [5] : vector<1xf16> into vector<8x1xf16>
    %416 = affine.apply #map14()[%15]
    %417 = vector.load %9[%416, %25] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %418 = vector.insert %417, %415 [6] : vector<1xf16> into vector<8x1xf16>
    %419 = affine.apply #map15()[%15]
    %420 = vector.load %9[%419, %25] : memref<32x16xf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %421 = vector.insert %420, %418 [7] : vector<1xf16> into vector<8x1xf16>
    %422 = arith.extf %421 : vector<8x1xf16> to vector<8x1xf32>
    %423 = vector.insert %422, %1 [0] : vector<8x1xf32> into vector<4x8x1xf32>
    %424 = vector.insert %422, %423 [1] : vector<8x1xf32> into vector<4x8x1xf32>
    %425 = vector.insert %422, %424 [2] : vector<8x1xf32> into vector<4x8x1xf32>
    %426 = vector.insert %422, %425 [3] : vector<8x1xf32> into vector<4x8x1xf32>
    %427 = vector.insert %426, %0 [0] : vector<4x8x1xf32> into vector<2x4x8x1xf32>
    %428 = vector.insert %426, %427 [1] : vector<4x8x1xf32> into vector<2x4x8x1xf32>
    %429 = arith.addf %398, %428 : vector<2x4x8x1xf32>
    %430 = arith.truncf %429 : vector<2x4x8x1xf32> to vector<2x4x8x1xf16>
    %431 = vector.broadcast %430 : vector<2x4x8x1xf16> to vector<1x2x4x8x1xf16>
    %432 = arith.muli %workgroup_id_x, %c2 overflow<nsw> : index
    %433 = vector.extract %430[0, 0, 0] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %433, %11[%workgroup_id_y, %432, %26, %15, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %434 = vector.extract %430[0, 0, 1] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %434, %11[%workgroup_id_y, %432, %26, %401, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %435 = vector.extract %430[0, 0, 2] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %435, %11[%workgroup_id_y, %432, %26, %404, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %436 = vector.extract %430[0, 0, 3] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %436, %11[%workgroup_id_y, %432, %26, %407, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %437 = vector.extract %430[0, 0, 4] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %437, %11[%workgroup_id_y, %432, %26, %410, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %438 = vector.extract %430[0, 0, 5] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %438, %11[%workgroup_id_y, %432, %26, %413, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %439 = vector.extract %430[0, 0, 6] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %439, %11[%workgroup_id_y, %432, %26, %416, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %440 = vector.extract %430[0, 0, 7] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %440, %11[%workgroup_id_y, %432, %26, %419, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %441 = affine.apply #map9()[%26]
    %442 = vector.extract %430[0, 1, 0] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %442, %11[%workgroup_id_y, %432, %441, %15, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %443 = vector.extract %430[0, 1, 1] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %443, %11[%workgroup_id_y, %432, %441, %401, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %444 = vector.extract %430[0, 1, 2] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %444, %11[%workgroup_id_y, %432, %441, %404, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %445 = vector.extract %430[0, 1, 3] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %445, %11[%workgroup_id_y, %432, %441, %407, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %446 = vector.extract %430[0, 1, 4] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %446, %11[%workgroup_id_y, %432, %441, %410, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %447 = vector.extract %430[0, 1, 5] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %447, %11[%workgroup_id_y, %432, %441, %413, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %448 = vector.extract %430[0, 1, 6] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %448, %11[%workgroup_id_y, %432, %441, %416, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %449 = vector.extract %430[0, 1, 7] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %449, %11[%workgroup_id_y, %432, %441, %419, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %450 = affine.apply #map10()[%26]
    %451 = vector.extract %430[0, 2, 0] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %451, %11[%workgroup_id_y, %432, %450, %15, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %452 = vector.extract %430[0, 2, 1] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %452, %11[%workgroup_id_y, %432, %450, %401, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %453 = vector.extract %430[0, 2, 2] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %453, %11[%workgroup_id_y, %432, %450, %404, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %454 = vector.extract %430[0, 2, 3] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %454, %11[%workgroup_id_y, %432, %450, %407, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %455 = vector.extract %430[0, 2, 4] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %455, %11[%workgroup_id_y, %432, %450, %410, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %456 = vector.extract %430[0, 2, 5] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %456, %11[%workgroup_id_y, %432, %450, %413, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %457 = vector.extract %430[0, 2, 6] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %457, %11[%workgroup_id_y, %432, %450, %416, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %458 = vector.extract %430[0, 2, 7] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %458, %11[%workgroup_id_y, %432, %450, %419, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %459 = affine.apply #map11()[%26]
    %460 = vector.extract %430[0, 3, 0] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %460, %11[%workgroup_id_y, %432, %459, %15, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %461 = vector.extract %430[0, 3, 1] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %461, %11[%workgroup_id_y, %432, %459, %401, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %462 = vector.extract %430[0, 3, 2] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %462, %11[%workgroup_id_y, %432, %459, %404, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %463 = vector.extract %430[0, 3, 3] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %463, %11[%workgroup_id_y, %432, %459, %407, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %464 = vector.extract %430[0, 3, 4] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %464, %11[%workgroup_id_y, %432, %459, %410, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %465 = vector.extract %430[0, 3, 5] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %465, %11[%workgroup_id_y, %432, %459, %413, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %466 = vector.extract %430[0, 3, 6] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %466, %11[%workgroup_id_y, %432, %459, %416, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %467 = vector.extract %430[0, 3, 7] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %467, %11[%workgroup_id_y, %432, %459, %419, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %468 = affine.apply #map9()[%432]
    %469 = vector.extract %430[1, 0, 0] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %469, %11[%workgroup_id_y, %468, %26, %15, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %470 = vector.extract %430[1, 0, 1] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %470, %11[%workgroup_id_y, %468, %26, %401, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %471 = vector.extract %430[1, 0, 2] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %471, %11[%workgroup_id_y, %468, %26, %404, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %472 = vector.extract %430[1, 0, 3] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %472, %11[%workgroup_id_y, %468, %26, %407, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %473 = vector.extract %430[1, 0, 4] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %473, %11[%workgroup_id_y, %468, %26, %410, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %474 = vector.extract %430[1, 0, 5] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %474, %11[%workgroup_id_y, %468, %26, %413, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %475 = vector.extract %430[1, 0, 6] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %475, %11[%workgroup_id_y, %468, %26, %416, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %476 = vector.extract %430[1, 0, 7] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %476, %11[%workgroup_id_y, %468, %26, %419, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %477 = vector.extract %430[1, 1, 0] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %477, %11[%workgroup_id_y, %468, %441, %15, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %478 = vector.extract %430[1, 1, 1] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %478, %11[%workgroup_id_y, %468, %441, %401, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %479 = vector.extract %430[1, 1, 2] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %479, %11[%workgroup_id_y, %468, %441, %404, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %480 = vector.extract %430[1, 1, 3] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %480, %11[%workgroup_id_y, %468, %441, %407, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %481 = vector.extract %430[1, 1, 4] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %481, %11[%workgroup_id_y, %468, %441, %410, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %482 = vector.extract %430[1, 1, 5] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %482, %11[%workgroup_id_y, %468, %441, %413, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %483 = vector.extract %430[1, 1, 6] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %483, %11[%workgroup_id_y, %468, %441, %416, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %484 = vector.extract %430[1, 1, 7] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %484, %11[%workgroup_id_y, %468, %441, %419, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %485 = vector.extract %430[1, 2, 0] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %485, %11[%workgroup_id_y, %468, %450, %15, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %486 = vector.extract %430[1, 2, 1] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %486, %11[%workgroup_id_y, %468, %450, %401, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %487 = vector.extract %430[1, 2, 2] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %487, %11[%workgroup_id_y, %468, %450, %404, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %488 = vector.extract %430[1, 2, 3] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %488, %11[%workgroup_id_y, %468, %450, %407, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %489 = vector.extract %430[1, 2, 4] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %489, %11[%workgroup_id_y, %468, %450, %410, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %490 = vector.extract %430[1, 2, 5] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %490, %11[%workgroup_id_y, %468, %450, %413, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %491 = vector.extract %430[1, 2, 6] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %491, %11[%workgroup_id_y, %468, %450, %416, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %492 = vector.extract %430[1, 2, 7] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %492, %11[%workgroup_id_y, %468, %450, %419, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %493 = vector.extract %430[1, 3, 0] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %493, %11[%workgroup_id_y, %468, %459, %15, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %494 = vector.extract %430[1, 3, 1] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %494, %11[%workgroup_id_y, %468, %459, %401, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %495 = vector.extract %430[1, 3, 2] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %495, %11[%workgroup_id_y, %468, %459, %404, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %496 = vector.extract %430[1, 3, 3] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %496, %11[%workgroup_id_y, %468, %459, %407, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %497 = vector.extract %430[1, 3, 4] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %497, %11[%workgroup_id_y, %468, %459, %410, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %498 = vector.extract %430[1, 3, 5] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %498, %11[%workgroup_id_y, %468, %459, %413, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %499 = vector.extract %430[1, 3, 6] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %499, %11[%workgroup_id_y, %468, %459, %416, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %500 = vector.extract %430[1, 3, 7] : vector<1xf16> from vector<2x4x8x1xf16>
    vector.store %500, %11[%workgroup_id_y, %468, %459, %419, %25] : memref<128x8x16x32x16xf16, strided<[65536, 8192, 512, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<1xf16>
    %501 = vector.transpose %431, [3, 4, 0, 1, 2] : vector<1x2x4x8x1xf16> to vector<8x1x1x2x4xf16>
    %502 = vector.extract %501[0, 0, 0, 0] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %502, %13[%15, %25, %workgroup_id_y, %432, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %503 = vector.extract %501[0, 0, 0, 1] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %503, %13[%15, %25, %workgroup_id_y, %468, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %504 = vector.extract %501[1, 0, 0, 0] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %504, %13[%401, %25, %workgroup_id_y, %432, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %505 = vector.extract %501[1, 0, 0, 1] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %505, %13[%401, %25, %workgroup_id_y, %468, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %506 = vector.extract %501[2, 0, 0, 0] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %506, %13[%404, %25, %workgroup_id_y, %432, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %507 = vector.extract %501[2, 0, 0, 1] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %507, %13[%404, %25, %workgroup_id_y, %468, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %508 = vector.extract %501[3, 0, 0, 0] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %508, %13[%407, %25, %workgroup_id_y, %432, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %509 = vector.extract %501[3, 0, 0, 1] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %509, %13[%407, %25, %workgroup_id_y, %468, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %510 = vector.extract %501[4, 0, 0, 0] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %510, %13[%410, %25, %workgroup_id_y, %432, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %511 = vector.extract %501[4, 0, 0, 1] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %511, %13[%410, %25, %workgroup_id_y, %468, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %512 = vector.extract %501[5, 0, 0, 0] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %512, %13[%413, %25, %workgroup_id_y, %432, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %513 = vector.extract %501[5, 0, 0, 1] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %513, %13[%413, %25, %workgroup_id_y, %468, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %514 = vector.extract %501[6, 0, 0, 0] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %514, %13[%416, %25, %workgroup_id_y, %432, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %515 = vector.extract %501[6, 0, 0, 1] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %515, %13[%416, %25, %workgroup_id_y, %468, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %516 = vector.extract %501[7, 0, 0, 0] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %516, %13[%419, %25, %workgroup_id_y, %432, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    %517 = vector.extract %501[7, 0, 0, 1] : vector<4xf16> from vector<8x1x1x2x4xf16>
    vector.store %517, %13[%419, %25, %workgroup_id_y, %468, %26] : memref<32x16x128x8x16xf16, strided<[262144, 16384, 128, 16, 1]>, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf16>
    gpu.barrier
    memref.dealloc %alloc_2 : memref<1x32x20xf16, #gpu.address_space<workgroup>>
    memref.dealloc %alloc : memref<512x20xf16, #gpu.address_space<workgroup>>
    return
  }
}

