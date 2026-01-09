// RUN: iree-opt -pass-pipeline="builtin.module(func.func(iree-llvmgpu-prefetch-shared-memory{num-stages=2}))" %s --split-input-file --debug

// -----// IR Dump After RemoveSingleIterationLoopPass (iree-codegen-remove-single-iteration-loop) //----- //
func.func @_matmul_32x64x32_f32_dispatch_1_matmul_8192x8192x8192_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>} {
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant dense<0.000000e+00> : vector<4x4x4x1xf32>
  %0 = ub.poison : f32
  %c4 = arith.constant 4 : index
  %c2048 = arith.constant 2048 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<16x128xf32, #gpu.address_space<workgroup>>
  %alloc_0 = memref.alloc() : memref<128x16xf32, #gpu.address_space<workgroup>>
  %thread_id_x = gpu.thread_id  x upper_bound 256
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : memref<8192x8192xf32, #hal.descriptor_type<storage_buffer>>
  %assume_align = memref.assume_alignment %1, 64 : memref<8192x8192xf32, #hal.descriptor_type<storage_buffer>>
  %2 = amdgpu.fat_raw_buffer_cast %assume_align resetOffset : memref<8192x8192xf32, #hal.descriptor_type<storage_buffer>> to memref<8192x8192xf32, #amdgpu.address_space<fat_raw_buffer>>
  %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : memref<8192x8192xf32, #hal.descriptor_type<storage_buffer>>
  %assume_align_1 = memref.assume_alignment %3, 64 : memref<8192x8192xf32, #hal.descriptor_type<storage_buffer>>
  %4 = amdgpu.fat_raw_buffer_cast %assume_align_1 resetOffset : memref<8192x8192xf32, #hal.descriptor_type<storage_buffer>> to memref<8192x8192xf32, #amdgpu.address_space<fat_raw_buffer>>
  %5 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : memref<8192x8192xf32, #hal.descriptor_type<storage_buffer>>
  %assume_align_2 = memref.assume_alignment %5, 64 : memref<8192x8192xf32, #hal.descriptor_type<storage_buffer>>
  %6 = amdgpu.fat_raw_buffer_cast %assume_align_2 resetOffset : memref<8192x8192xf32, #hal.descriptor_type<storage_buffer>> to memref<8192x8192xf32, #amdgpu.address_space<fat_raw_buffer>>
  %expand_shape = memref.expand_shape %6 [[0, 1], [2, 3]] output_shape [512, 16, 512, 16] : memref<8192x8192xf32, #amdgpu.address_space<fat_raw_buffer>> into memref<512x16x512x16xf32, #amdgpu.address_space<fat_raw_buffer>>
  scf.forall (%arg0, %arg1) in (64, 64) {
    %7 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg1)
    %8 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg1)
    %9 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg0)
    %subview = memref.subview %expand_shape[%9, 0, %8, 0] [8, 16, 8, 16] [1, 1, 1, 1] : memref<512x16x512x16xf32, #amdgpu.address_space<fat_raw_buffer>> to memref<8x16x8x16xf32, strided<[131072, 8192, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    gpu.barrier
    %10:3 = affine.delinearize_index %thread_id_x into (2, 2, 64) : index, index, index
    %11:3 = affine.delinearize_index %10#2 into (4, 16) : index, index, index
    %12 = affine.linearize_index disjoint [%11#1, %c0] by (4, 4) : index
    %13 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%10#1]
    %14 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%10#0]
    %subview_3 = memref.subview %subview[%14, %12, %13, %11#2] [4, 4, 4, 1] [1, 1, 1, 1] : memref<8x16x8x16xf32, strided<[131072, 8192, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>> to memref<4x4x4x1xf32, strided<[131072, 8192, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    %15:2 = affine.delinearize_index %thread_id_x into (4, 64) : index, index
    %16 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%15#0]
    %subview_4 = memref.subview %alloc_0[%16, 0] [32, 16] [1, 1] : memref<128x16xf32, #gpu.address_space<workgroup>> to memref<32x16xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
    %17 = affine.apply affine_map<(d0)[s0] -> (d0 * 128 + s0 * 32)>(%arg0)[%15#0]
    %18 = arith.muli %15#1, %c4 : index
    %19 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%15#0]
    %subview_5 = memref.subview %alloc[%19, 0] [4, 128] [1, 1] : memref<16x128xf32, #gpu.address_space<workgroup>> to memref<4x128xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
    %expand_shape_6 = memref.expand_shape %alloc [[0, 1], [2, 3]] output_shape [4, 4, 8, 16] : memref<16x128xf32, #gpu.address_space<workgroup>> into memref<4x4x8x16xf32, #gpu.address_space<workgroup>>
    %expand_shape_7 = memref.expand_shape %alloc_0 [[0, 1], [2, 3]] output_shape [8, 16, 4, 4] : memref<128x16xf32, #gpu.address_space<workgroup>> into memref<8x16x4x4xf32, #gpu.address_space<workgroup>>
    %20 = scf.for %arg2 = %c0 to %c2048 step %c4 iter_args(%arg3 = %cst) -> (vector<4x4x4x1xf32>) {
      gpu.barrier
      %22 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
      %subview_8 = memref.subview %2[%17, %22] [32, 16] [1, 1] : memref<8192x8192xf32, #amdgpu.address_space<fat_raw_buffer>> to memref<32x16xf32, strided<[8192, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
      amdgpu.gather_to_lds %subview_8[%c0, %18], %subview_4[%c0, %c0] : vector<4xf32>, memref<32x16xf32, strided<[8192, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, memref<32x16xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
      amdgpu.gather_to_lds %subview_8[%c16, %18], %subview_4[%c16, %c0] : vector<4xf32>, memref<32x16xf32, strided<[8192, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, memref<32x16xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
      %23 = affine.apply affine_map<(d0)[s0] -> (d0 * 4 + s0 * 4)>(%arg2)[%15#0]
      %subview_9 = memref.subview %4[%23, %7] [4, 128] [1, 1] : memref<8192x8192xf32, #amdgpu.address_space<fat_raw_buffer>> to memref<4x128xf32, strided<[8192, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
      amdgpu.gather_to_lds %subview_9[%c0, %18], %subview_5[%c0, %c0] : vector<4xf32>, memref<4x128xf32, strided<[8192, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, memref<4x128xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      amdgpu.gather_to_lds %subview_9[%c2, %18], %subview_5[%c2, %c0] : vector<4xf32>, memref<4x128xf32, strided<[8192, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>, memref<4x128xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
      gpu.barrier
      %24 = vector.transfer_read %expand_shape_7[%14, %11#2, %c0, %11#1], %0 {in_bounds = [true, true, true, true]} : memref<8x16x4x4xf32, #gpu.address_space<workgroup>>, vector<4x1x4x1xf32>
      %25 = vector.transfer_read %expand_shape_6[%c0, %11#1, %13, %11#2], %0 {in_bounds = [true, true, true, true]} : memref<4x4x8x16xf32, #gpu.address_space<workgroup>>, vector<4x1x4x1xf32>
      %26 = vector.extract %arg3[0, 0] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %27 = vector.shape_cast %26 : vector<4x1xf32> to vector<4xf32>
      %28 = vector.extract %24[0, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %29 = vector.extract %25[0, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %30 = amdgpu.mfma 16x16x4 %28 * %29 + %27 blgp =  none : f32, f32, vector<4xf32>
      %31 = vector.extract %arg3[0, 1] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %32 = vector.shape_cast %31 : vector<4x1xf32> to vector<4xf32>
      %33 = vector.extract %24[0, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %34 = vector.extract %25[0, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %35 = amdgpu.mfma 16x16x4 %33 * %34 + %32 blgp =  none : f32, f32, vector<4xf32>
      %36 = vector.extract %arg3[0, 2] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %37 = vector.shape_cast %36 : vector<4x1xf32> to vector<4xf32>
      %38 = vector.extract %24[0, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %39 = vector.extract %25[0, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %40 = amdgpu.mfma 16x16x4 %38 * %39 + %37 blgp =  none : f32, f32, vector<4xf32>
      %41 = vector.extract %arg3[0, 3] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %42 = vector.shape_cast %41 : vector<4x1xf32> to vector<4xf32>
      %43 = vector.extract %24[0, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %44 = vector.extract %25[0, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %45 = amdgpu.mfma 16x16x4 %43 * %44 + %42 blgp =  none : f32, f32, vector<4xf32>
      %46 = vector.extract %arg3[1, 0] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %47 = vector.shape_cast %46 : vector<4x1xf32> to vector<4xf32>
      %48 = vector.extract %24[1, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %49 = vector.extract %25[0, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %50 = amdgpu.mfma 16x16x4 %48 * %49 + %47 blgp =  none : f32, f32, vector<4xf32>
      %51 = vector.extract %arg3[1, 1] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %52 = vector.shape_cast %51 : vector<4x1xf32> to vector<4xf32>
      %53 = vector.extract %24[1, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %54 = vector.extract %25[0, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %55 = amdgpu.mfma 16x16x4 %53 * %54 + %52 blgp =  none : f32, f32, vector<4xf32>
      %56 = vector.extract %arg3[1, 2] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %57 = vector.shape_cast %56 : vector<4x1xf32> to vector<4xf32>
      %58 = vector.extract %24[1, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %59 = vector.extract %25[0, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %60 = amdgpu.mfma 16x16x4 %58 * %59 + %57 blgp =  none : f32, f32, vector<4xf32>
      %61 = vector.extract %arg3[1, 3] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %62 = vector.shape_cast %61 : vector<4x1xf32> to vector<4xf32>
      %63 = vector.extract %24[1, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %64 = vector.extract %25[0, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %65 = amdgpu.mfma 16x16x4 %63 * %64 + %62 blgp =  none : f32, f32, vector<4xf32>
      %66 = vector.extract %arg3[2, 0] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %67 = vector.shape_cast %66 : vector<4x1xf32> to vector<4xf32>
      %68 = vector.extract %24[2, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %69 = vector.extract %25[0, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %70 = amdgpu.mfma 16x16x4 %68 * %69 + %67 blgp =  none : f32, f32, vector<4xf32>
      %71 = vector.extract %arg3[2, 1] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %72 = vector.shape_cast %71 : vector<4x1xf32> to vector<4xf32>
      %73 = vector.extract %24[2, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %74 = vector.extract %25[0, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %75 = amdgpu.mfma 16x16x4 %73 * %74 + %72 blgp =  none : f32, f32, vector<4xf32>
      %76 = vector.extract %arg3[2, 2] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %77 = vector.shape_cast %76 : vector<4x1xf32> to vector<4xf32>
      %78 = vector.extract %24[2, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %79 = vector.extract %25[0, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %80 = amdgpu.mfma 16x16x4 %78 * %79 + %77 blgp =  none : f32, f32, vector<4xf32>
      %81 = vector.extract %arg3[2, 3] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %82 = vector.shape_cast %81 : vector<4x1xf32> to vector<4xf32>
      %83 = vector.extract %24[2, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %84 = vector.extract %25[0, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %85 = amdgpu.mfma 16x16x4 %83 * %84 + %82 blgp =  none : f32, f32, vector<4xf32>
      %86 = vector.extract %arg3[3, 0] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %87 = vector.shape_cast %86 : vector<4x1xf32> to vector<4xf32>
      %88 = vector.extract %24[3, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %89 = vector.extract %25[0, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %90 = amdgpu.mfma 16x16x4 %88 * %89 + %87 blgp =  none : f32, f32, vector<4xf32>
      %91 = vector.extract %arg3[3, 1] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %92 = vector.shape_cast %91 : vector<4x1xf32> to vector<4xf32>
      %93 = vector.extract %24[3, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %94 = vector.extract %25[0, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %95 = amdgpu.mfma 16x16x4 %93 * %94 + %92 blgp =  none : f32, f32, vector<4xf32>
      %96 = vector.extract %arg3[3, 2] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %97 = vector.shape_cast %96 : vector<4x1xf32> to vector<4xf32>
      %98 = vector.extract %24[3, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %99 = vector.extract %25[0, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %100 = amdgpu.mfma 16x16x4 %98 * %99 + %97 blgp =  none : f32, f32, vector<4xf32>
      %101 = vector.extract %arg3[3, 3] : vector<4x1xf32> from vector<4x4x4x1xf32>
      %102 = vector.shape_cast %101 : vector<4x1xf32> to vector<4xf32>
      %103 = vector.extract %24[3, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %104 = vector.extract %25[0, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %105 = amdgpu.mfma 16x16x4 %103 * %104 + %102 blgp =  none : f32, f32, vector<4xf32>
      %106 = vector.extract %24[0, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %107 = vector.extract %25[1, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %108 = amdgpu.mfma 16x16x4 %106 * %107 + %30 blgp =  none : f32, f32, vector<4xf32>
      %109 = vector.extract %24[0, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %110 = vector.extract %25[1, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %111 = amdgpu.mfma 16x16x4 %109 * %110 + %35 blgp =  none : f32, f32, vector<4xf32>
      %112 = vector.extract %24[0, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %113 = vector.extract %25[1, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %114 = amdgpu.mfma 16x16x4 %112 * %113 + %40 blgp =  none : f32, f32, vector<4xf32>
      %115 = vector.extract %24[0, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %116 = vector.extract %25[1, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %117 = amdgpu.mfma 16x16x4 %115 * %116 + %45 blgp =  none : f32, f32, vector<4xf32>
      %118 = vector.extract %24[1, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %119 = vector.extract %25[1, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %120 = amdgpu.mfma 16x16x4 %118 * %119 + %50 blgp =  none : f32, f32, vector<4xf32>
      %121 = vector.extract %24[1, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %122 = vector.extract %25[1, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %123 = amdgpu.mfma 16x16x4 %121 * %122 + %55 blgp =  none : f32, f32, vector<4xf32>
      %124 = vector.extract %24[1, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %125 = vector.extract %25[1, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %126 = amdgpu.mfma 16x16x4 %124 * %125 + %60 blgp =  none : f32, f32, vector<4xf32>
      %127 = vector.extract %24[1, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %128 = vector.extract %25[1, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %129 = amdgpu.mfma 16x16x4 %127 * %128 + %65 blgp =  none : f32, f32, vector<4xf32>
      %130 = vector.extract %24[2, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %131 = vector.extract %25[1, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %132 = amdgpu.mfma 16x16x4 %130 * %131 + %70 blgp =  none : f32, f32, vector<4xf32>
      %133 = vector.extract %24[2, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %134 = vector.extract %25[1, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %135 = amdgpu.mfma 16x16x4 %133 * %134 + %75 blgp =  none : f32, f32, vector<4xf32>
      %136 = vector.extract %24[2, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %137 = vector.extract %25[1, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %138 = amdgpu.mfma 16x16x4 %136 * %137 + %80 blgp =  none : f32, f32, vector<4xf32>
      %139 = vector.extract %24[2, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %140 = vector.extract %25[1, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %141 = amdgpu.mfma 16x16x4 %139 * %140 + %85 blgp =  none : f32, f32, vector<4xf32>
      %142 = vector.extract %24[3, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %143 = vector.extract %25[1, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %144 = amdgpu.mfma 16x16x4 %142 * %143 + %90 blgp =  none : f32, f32, vector<4xf32>
      %145 = vector.extract %24[3, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %146 = vector.extract %25[1, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %147 = amdgpu.mfma 16x16x4 %145 * %146 + %95 blgp =  none : f32, f32, vector<4xf32>
      %148 = vector.extract %24[3, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %149 = vector.extract %25[1, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %150 = amdgpu.mfma 16x16x4 %148 * %149 + %100 blgp =  none : f32, f32, vector<4xf32>
      %151 = vector.extract %24[3, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %152 = vector.extract %25[1, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %153 = amdgpu.mfma 16x16x4 %151 * %152 + %105 blgp =  none : f32, f32, vector<4xf32>
      %154 = vector.extract %24[0, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %155 = vector.extract %25[2, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %156 = amdgpu.mfma 16x16x4 %154 * %155 + %108 blgp =  none : f32, f32, vector<4xf32>
      %157 = vector.extract %24[0, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %158 = vector.extract %25[2, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %159 = amdgpu.mfma 16x16x4 %157 * %158 + %111 blgp =  none : f32, f32, vector<4xf32>
      %160 = vector.extract %24[0, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %161 = vector.extract %25[2, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %162 = amdgpu.mfma 16x16x4 %160 * %161 + %114 blgp =  none : f32, f32, vector<4xf32>
      %163 = vector.extract %24[0, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %164 = vector.extract %25[2, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %165 = amdgpu.mfma 16x16x4 %163 * %164 + %117 blgp =  none : f32, f32, vector<4xf32>
      %166 = vector.extract %24[1, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %167 = vector.extract %25[2, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %168 = amdgpu.mfma 16x16x4 %166 * %167 + %120 blgp =  none : f32, f32, vector<4xf32>
      %169 = vector.extract %24[1, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %170 = vector.extract %25[2, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %171 = amdgpu.mfma 16x16x4 %169 * %170 + %123 blgp =  none : f32, f32, vector<4xf32>
      %172 = vector.extract %24[1, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %173 = vector.extract %25[2, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %174 = amdgpu.mfma 16x16x4 %172 * %173 + %126 blgp =  none : f32, f32, vector<4xf32>
      %175 = vector.extract %24[1, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %176 = vector.extract %25[2, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %177 = amdgpu.mfma 16x16x4 %175 * %176 + %129 blgp =  none : f32, f32, vector<4xf32>
      %178 = vector.extract %24[2, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %179 = vector.extract %25[2, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %180 = amdgpu.mfma 16x16x4 %178 * %179 + %132 blgp =  none : f32, f32, vector<4xf32>
      %181 = vector.extract %24[2, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %182 = vector.extract %25[2, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %183 = amdgpu.mfma 16x16x4 %181 * %182 + %135 blgp =  none : f32, f32, vector<4xf32>
      %184 = vector.extract %24[2, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %185 = vector.extract %25[2, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %186 = amdgpu.mfma 16x16x4 %184 * %185 + %138 blgp =  none : f32, f32, vector<4xf32>
      %187 = vector.extract %24[2, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %188 = vector.extract %25[2, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %189 = amdgpu.mfma 16x16x4 %187 * %188 + %141 blgp =  none : f32, f32, vector<4xf32>
      %190 = vector.extract %24[3, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %191 = vector.extract %25[2, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %192 = amdgpu.mfma 16x16x4 %190 * %191 + %144 blgp =  none : f32, f32, vector<4xf32>
      %193 = vector.extract %24[3, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %194 = vector.extract %25[2, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %195 = amdgpu.mfma 16x16x4 %193 * %194 + %147 blgp =  none : f32, f32, vector<4xf32>
      %196 = vector.extract %24[3, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %197 = vector.extract %25[2, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %198 = amdgpu.mfma 16x16x4 %196 * %197 + %150 blgp =  none : f32, f32, vector<4xf32>
      %199 = vector.extract %24[3, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %200 = vector.extract %25[2, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %201 = amdgpu.mfma 16x16x4 %199 * %200 + %153 blgp =  none : f32, f32, vector<4xf32>
      %202 = vector.extract %24[0, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %203 = vector.extract %25[3, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %204 = amdgpu.mfma 16x16x4 %202 * %203 + %156 blgp =  none : f32, f32, vector<4xf32>
      %205 = vector.shape_cast %204 : vector<4xf32> to vector<4x1xf32>
      %206 = vector.broadcast %205 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %207 = vector.extract %24[0, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %208 = vector.extract %25[3, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %209 = amdgpu.mfma 16x16x4 %207 * %208 + %159 blgp =  none : f32, f32, vector<4xf32>
      %210 = vector.shape_cast %209 : vector<4xf32> to vector<4x1xf32>
      %211 = vector.broadcast %210 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %212 = vector.extract %24[0, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %213 = vector.extract %25[3, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %214 = amdgpu.mfma 16x16x4 %212 * %213 + %162 blgp =  none : f32, f32, vector<4xf32>
      %215 = vector.shape_cast %214 : vector<4xf32> to vector<4x1xf32>
      %216 = vector.broadcast %215 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %217 = vector.extract %24[0, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %218 = vector.extract %25[3, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %219 = amdgpu.mfma 16x16x4 %217 * %218 + %165 blgp =  none : f32, f32, vector<4xf32>
      %220 = vector.shape_cast %219 : vector<4xf32> to vector<4x1xf32>
      %221 = vector.broadcast %220 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %222 = vector.extract %24[1, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %223 = vector.extract %25[3, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %224 = amdgpu.mfma 16x16x4 %222 * %223 + %168 blgp =  none : f32, f32, vector<4xf32>
      %225 = vector.shape_cast %224 : vector<4xf32> to vector<4x1xf32>
      %226 = vector.broadcast %225 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %227 = vector.extract %24[1, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %228 = vector.extract %25[3, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %229 = amdgpu.mfma 16x16x4 %227 * %228 + %171 blgp =  none : f32, f32, vector<4xf32>
      %230 = vector.shape_cast %229 : vector<4xf32> to vector<4x1xf32>
      %231 = vector.broadcast %230 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %232 = vector.extract %24[1, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %233 = vector.extract %25[3, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %234 = amdgpu.mfma 16x16x4 %232 * %233 + %174 blgp =  none : f32, f32, vector<4xf32>
      %235 = vector.shape_cast %234 : vector<4xf32> to vector<4x1xf32>
      %236 = vector.broadcast %235 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %237 = vector.extract %24[1, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %238 = vector.extract %25[3, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %239 = amdgpu.mfma 16x16x4 %237 * %238 + %177 blgp =  none : f32, f32, vector<4xf32>
      %240 = vector.shape_cast %239 : vector<4xf32> to vector<4x1xf32>
      %241 = vector.broadcast %240 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %242 = vector.extract %24[2, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %243 = vector.extract %25[3, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %244 = amdgpu.mfma 16x16x4 %242 * %243 + %180 blgp =  none : f32, f32, vector<4xf32>
      %245 = vector.shape_cast %244 : vector<4xf32> to vector<4x1xf32>
      %246 = vector.broadcast %245 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %247 = vector.extract %24[2, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %248 = vector.extract %25[3, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %249 = amdgpu.mfma 16x16x4 %247 * %248 + %183 blgp =  none : f32, f32, vector<4xf32>
      %250 = vector.shape_cast %249 : vector<4xf32> to vector<4x1xf32>
      %251 = vector.broadcast %250 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %252 = vector.extract %24[2, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %253 = vector.extract %25[3, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %254 = amdgpu.mfma 16x16x4 %252 * %253 + %186 blgp =  none : f32, f32, vector<4xf32>
      %255 = vector.shape_cast %254 : vector<4xf32> to vector<4x1xf32>
      %256 = vector.broadcast %255 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %257 = vector.extract %24[2, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %258 = vector.extract %25[3, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %259 = amdgpu.mfma 16x16x4 %257 * %258 + %189 blgp =  none : f32, f32, vector<4xf32>
      %260 = vector.shape_cast %259 : vector<4xf32> to vector<4x1xf32>
      %261 = vector.broadcast %260 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %262 = vector.extract %24[3, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %263 = vector.extract %25[3, 0, 0, 0] : f32 from vector<4x1x4x1xf32>
      %264 = amdgpu.mfma 16x16x4 %262 * %263 + %192 blgp =  none : f32, f32, vector<4xf32>
      %265 = vector.shape_cast %264 : vector<4xf32> to vector<4x1xf32>
      %266 = vector.broadcast %265 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %267 = vector.extract %24[3, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %268 = vector.extract %25[3, 0, 1, 0] : f32 from vector<4x1x4x1xf32>
      %269 = amdgpu.mfma 16x16x4 %267 * %268 + %195 blgp =  none : f32, f32, vector<4xf32>
      %270 = vector.shape_cast %269 : vector<4xf32> to vector<4x1xf32>
      %271 = vector.broadcast %270 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %272 = vector.extract %24[3, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %273 = vector.extract %25[3, 0, 2, 0] : f32 from vector<4x1x4x1xf32>
      %274 = amdgpu.mfma 16x16x4 %272 * %273 + %198 blgp =  none : f32, f32, vector<4xf32>
      %275 = vector.shape_cast %274 : vector<4xf32> to vector<4x1xf32>
      %276 = vector.broadcast %275 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %277 = vector.extract %24[3, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %278 = vector.extract %25[3, 0, 3, 0] : f32 from vector<4x1x4x1xf32>
      %279 = amdgpu.mfma 16x16x4 %277 * %278 + %201 blgp =  none : f32, f32, vector<4xf32>
      %280 = vector.shape_cast %279 : vector<4xf32> to vector<4x1xf32>
      %281 = vector.broadcast %280 : vector<4x1xf32> to vector<1x1x4x1xf32>
      %282 = vector.insert_strided_slice %206, %cst {offsets = [0, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %283 = vector.insert_strided_slice %211, %282 {offsets = [0, 1, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %284 = vector.insert_strided_slice %216, %283 {offsets = [0, 2, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %285 = vector.insert_strided_slice %221, %284 {offsets = [0, 3, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %286 = vector.insert_strided_slice %226, %285 {offsets = [1, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %287 = vector.insert_strided_slice %231, %286 {offsets = [1, 1, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %288 = vector.insert_strided_slice %236, %287 {offsets = [1, 2, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %289 = vector.insert_strided_slice %241, %288 {offsets = [1, 3, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %290 = vector.insert_strided_slice %246, %289 {offsets = [2, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %291 = vector.insert_strided_slice %251, %290 {offsets = [2, 1, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %292 = vector.insert_strided_slice %256, %291 {offsets = [2, 2, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %293 = vector.insert_strided_slice %261, %292 {offsets = [2, 3, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %294 = vector.insert_strided_slice %266, %293 {offsets = [3, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %295 = vector.insert_strided_slice %271, %294 {offsets = [3, 1, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %296 = vector.insert_strided_slice %276, %295 {offsets = [3, 2, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      %297 = vector.insert_strided_slice %281, %296 {offsets = [3, 3, 0, 0], strides = [1, 1, 1, 1]} : vector<1x1x4x1xf32> into vector<4x4x4x1xf32>
      scf.yield %297 : vector<4x4x4x1xf32>
    }
    %21 = vector.transpose %20, [0, 2, 1, 3] : vector<4x4x4x1xf32> to vector<4x4x4x1xf32>
    vector.transfer_write %21, %subview_3[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<4x4x4x1xf32>, memref<4x4x4x1xf32, strided<[131072, 8192, 16, 1], offset: ?>, #amdgpu.address_space<fat_raw_buffer>>
    gpu.barrier
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  memref.dealloc %alloc_0 : memref<128x16xf32, #gpu.address_space<workgroup>>
  memref.dealloc %alloc : memref<16x128xf32, #gpu.address_space<workgroup>>
  return
}
