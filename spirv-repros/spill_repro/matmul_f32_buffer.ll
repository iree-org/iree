; IREE f32 matmul 2048x2048x2048, AOT path (with buffer instructions).
; Uses AMDGPU address space 7 (buffer fat pointers) for memory access.
;
; Compile:
;   llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=asm matmul_f32_buffer.ll -o buffer.s
;   grep -c scratch_ buffer.s  # => 0 (no spilling)

; ModuleID = 'matmul_f32_dispatch_0'
source_filename = "matmul_f32_dispatch_0"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

; Function Attrs: alwaysinline nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: write)
define amdgpu_kernel void @matmul_f32_dispatch_0_matmul_2048x2048x2048_f32(ptr addrspace(1) noalias noundef nonnull readonly align 16 captures(none) %0, ptr addrspace(1) noalias noundef nonnull readonly align 16 captures(none) %1, ptr addrspace(1) noalias noundef nonnull writeonly align 16 captures(none) %2) local_unnamed_addr #0 !reqd_work_group_size !2 {
  %4 = tail call range(i32 0, 8) i32 @llvm.amdgcn.workitem.id.y()
  %5 = tail call range(i32 0, 32) i32 @llvm.amdgcn.workitem.id.x()
  %6 = shl nuw nsw i32 %4, 5
  %7 = or disjoint i32 %6, %5
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %0, i64 64) ]
  %8 = tail call ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) nonnull %0, i16 0, i64 16777216, i32 822243328)
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %1, i64 64) ]
  %9 = tail call ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) nonnull %1, i16 0, i64 16777216, i32 822243328)
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %2, i64 64) ]
  %10 = lshr i32 %7, 3
  %11 = tail call range(i32 0, 1024) i32 @llvm.amdgcn.workgroup.id.x()
  %12 = shl nuw nsw i32 %11, 1
  %13 = and i32 %12, 2016
  %14 = or disjoint i32 %10, %13
  %15 = shl nuw nsw i32 %5, 4
  %16 = and i32 %15, 112
  %17 = shl nuw nsw i32 %11, 7
  %18 = and i32 %17, 1920
  %19 = or disjoint i32 %16, %18
  %.idx1 = shl nuw nsw i32 %14, 13
  %20 = getelementptr i8, ptr addrspace(7) %8, i32 %.idx1
  %invariant.gep = getelementptr [4 x i8], ptr addrspace(7) %9, i32 %19
  br label %21

21:                                               ; preds = %3, %21
  %22 = phi <16 x float> [ zeroinitializer, %3 ], [ %152, %21 ]
  %23 = phi i32 [ 0, %3 ], [ %153, %21 ]
  %24 = getelementptr [4 x i8], ptr addrspace(7) %20, i32 %23
  %25 = load <32 x float>, ptr addrspace(7) %24, align 4
  %.idx2 = shl nuw nsw i32 %23, 13
  %gep = getelementptr i8, ptr addrspace(7) %invariant.gep, i32 %.idx2
  %26 = load <16 x float>, ptr addrspace(7) %gep, align 4
  %27 = getelementptr i8, ptr addrspace(7) %gep, i32 8192
  %28 = load <16 x float>, ptr addrspace(7) %27, align 4
  %29 = getelementptr i8, ptr addrspace(7) %gep, i32 16384
  %30 = load <16 x float>, ptr addrspace(7) %29, align 4
  %31 = getelementptr i8, ptr addrspace(7) %gep, i32 24576
  %32 = load <16 x float>, ptr addrspace(7) %31, align 4
  %33 = getelementptr i8, ptr addrspace(7) %gep, i32 32768
  %34 = load <16 x float>, ptr addrspace(7) %33, align 4
  %35 = getelementptr i8, ptr addrspace(7) %gep, i32 40960
  %36 = load <16 x float>, ptr addrspace(7) %35, align 4
  %37 = getelementptr i8, ptr addrspace(7) %gep, i32 49152
  %38 = load <16 x float>, ptr addrspace(7) %37, align 4
  %39 = getelementptr i8, ptr addrspace(7) %gep, i32 57344
  %40 = load <16 x float>, ptr addrspace(7) %39, align 4
  %41 = getelementptr i8, ptr addrspace(7) %gep, i32 65536
  %42 = load <16 x float>, ptr addrspace(7) %41, align 4
  %43 = getelementptr i8, ptr addrspace(7) %gep, i32 73728
  %44 = load <16 x float>, ptr addrspace(7) %43, align 4
  %45 = getelementptr i8, ptr addrspace(7) %gep, i32 81920
  %46 = load <16 x float>, ptr addrspace(7) %45, align 4
  %47 = getelementptr i8, ptr addrspace(7) %gep, i32 90112
  %48 = load <16 x float>, ptr addrspace(7) %47, align 4
  %49 = getelementptr i8, ptr addrspace(7) %gep, i32 98304
  %50 = load <16 x float>, ptr addrspace(7) %49, align 4
  %51 = getelementptr i8, ptr addrspace(7) %gep, i32 106496
  %52 = load <16 x float>, ptr addrspace(7) %51, align 4
  %53 = getelementptr i8, ptr addrspace(7) %gep, i32 114688
  %54 = load <16 x float>, ptr addrspace(7) %53, align 4
  %55 = getelementptr i8, ptr addrspace(7) %gep, i32 122880
  %56 = load <16 x float>, ptr addrspace(7) %55, align 4
  %57 = getelementptr i8, ptr addrspace(7) %gep, i32 131072
  %58 = load <16 x float>, ptr addrspace(7) %57, align 4
  %59 = getelementptr i8, ptr addrspace(7) %gep, i32 139264
  %60 = load <16 x float>, ptr addrspace(7) %59, align 4
  %61 = getelementptr i8, ptr addrspace(7) %gep, i32 147456
  %62 = load <16 x float>, ptr addrspace(7) %61, align 4
  %63 = getelementptr i8, ptr addrspace(7) %gep, i32 155648
  %64 = load <16 x float>, ptr addrspace(7) %63, align 4
  %65 = getelementptr i8, ptr addrspace(7) %gep, i32 163840
  %66 = load <16 x float>, ptr addrspace(7) %65, align 4
  %67 = getelementptr i8, ptr addrspace(7) %gep, i32 172032
  %68 = load <16 x float>, ptr addrspace(7) %67, align 4
  %69 = getelementptr i8, ptr addrspace(7) %gep, i32 180224
  %70 = load <16 x float>, ptr addrspace(7) %69, align 4
  %71 = getelementptr i8, ptr addrspace(7) %gep, i32 188416
  %72 = load <16 x float>, ptr addrspace(7) %71, align 4
  %73 = getelementptr i8, ptr addrspace(7) %gep, i32 196608
  %74 = load <16 x float>, ptr addrspace(7) %73, align 4
  %75 = getelementptr i8, ptr addrspace(7) %gep, i32 204800
  %76 = load <16 x float>, ptr addrspace(7) %75, align 4
  %77 = getelementptr i8, ptr addrspace(7) %gep, i32 212992
  %78 = load <16 x float>, ptr addrspace(7) %77, align 4
  %79 = getelementptr i8, ptr addrspace(7) %gep, i32 221184
  %80 = load <16 x float>, ptr addrspace(7) %79, align 4
  %81 = getelementptr i8, ptr addrspace(7) %gep, i32 229376
  %82 = load <16 x float>, ptr addrspace(7) %81, align 4
  %83 = getelementptr i8, ptr addrspace(7) %gep, i32 237568
  %84 = load <16 x float>, ptr addrspace(7) %83, align 4
  %85 = getelementptr i8, ptr addrspace(7) %gep, i32 245760
  %86 = load <16 x float>, ptr addrspace(7) %85, align 4
  %87 = getelementptr i8, ptr addrspace(7) %gep, i32 253952
  %88 = load <16 x float>, ptr addrspace(7) %87, align 4
  %89 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  %90 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %89, <16 x float> %88, <16 x float> %22)
  %91 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 30, i32 30, i32 30, i32 30, i32 30, i32 30, i32 30, i32 30, i32 30, i32 30, i32 30, i32 30, i32 30, i32 30, i32 30, i32 30>
  %92 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %91, <16 x float> %86, <16 x float> %90)
  %93 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29>
  %94 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %93, <16 x float> %84, <16 x float> %92)
  %95 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 28, i32 28, i32 28, i32 28, i32 28, i32 28, i32 28, i32 28, i32 28, i32 28, i32 28, i32 28, i32 28, i32 28, i32 28, i32 28>
  %96 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %95, <16 x float> %82, <16 x float> %94)
  %97 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 27, i32 27, i32 27, i32 27, i32 27, i32 27, i32 27, i32 27, i32 27, i32 27, i32 27, i32 27, i32 27, i32 27, i32 27, i32 27>
  %98 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %97, <16 x float> %80, <16 x float> %96)
  %99 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 26, i32 26, i32 26, i32 26, i32 26, i32 26, i32 26, i32 26, i32 26, i32 26, i32 26, i32 26, i32 26, i32 26, i32 26, i32 26>
  %100 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %99, <16 x float> %78, <16 x float> %98)
  %101 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 25, i32 25, i32 25, i32 25, i32 25, i32 25, i32 25, i32 25, i32 25, i32 25, i32 25, i32 25, i32 25, i32 25, i32 25, i32 25>
  %102 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %101, <16 x float> %76, <16 x float> %100)
  %103 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %104 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %103, <16 x float> %74, <16 x float> %102)
  %105 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>
  %106 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %105, <16 x float> %72, <16 x float> %104)
  %107 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22>
  %108 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %107, <16 x float> %70, <16 x float> %106)
  %109 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21, i32 21>
  %110 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %109, <16 x float> %68, <16 x float> %108)
  %111 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 20, i32 20, i32 20, i32 20, i32 20, i32 20, i32 20, i32 20, i32 20, i32 20, i32 20, i32 20, i32 20, i32 20, i32 20, i32 20>
  %112 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %111, <16 x float> %66, <16 x float> %110)
  %113 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19, i32 19>
  %114 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %113, <16 x float> %64, <16 x float> %112)
  %115 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 18, i32 18, i32 18, i32 18, i32 18, i32 18, i32 18, i32 18, i32 18, i32 18, i32 18, i32 18, i32 18, i32 18, i32 18, i32 18>
  %116 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %115, <16 x float> %62, <16 x float> %114)
  %117 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17>
  %118 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %117, <16 x float> %60, <16 x float> %116)
  %119 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %120 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %119, <16 x float> %58, <16 x float> %118)
  %121 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
  %122 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %121, <16 x float> %56, <16 x float> %120)
  %123 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14>
  %124 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %123, <16 x float> %54, <16 x float> %122)
  %125 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13>
  %126 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %125, <16 x float> %52, <16 x float> %124)
  %127 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12>
  %128 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %127, <16 x float> %50, <16 x float> %126)
  %129 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11>
  %130 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %129, <16 x float> %48, <16 x float> %128)
  %131 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10>
  %132 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %131, <16 x float> %46, <16 x float> %130)
  %133 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9>
  %134 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %133, <16 x float> %44, <16 x float> %132)
  %135 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  %136 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %135, <16 x float> %42, <16 x float> %134)
  %137 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %138 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %137, <16 x float> %40, <16 x float> %136)
  %139 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6>
  %140 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %139, <16 x float> %38, <16 x float> %138)
  %141 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  %142 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %141, <16 x float> %36, <16 x float> %140)
  %143 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %144 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %143, <16 x float> %34, <16 x float> %142)
  %145 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %146 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %145, <16 x float> %32, <16 x float> %144)
  %147 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %148 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %147, <16 x float> %30, <16 x float> %146)
  %149 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %150 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %149, <16 x float> %28, <16 x float> %148)
  %151 = shufflevector <32 x float> %25, <32 x float> poison, <16 x i32> zeroinitializer
  %152 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %151, <16 x float> %26, <16 x float> %150)
  %153 = add nuw nsw i32 %23, 32
  %154 = icmp samesign ult i32 %23, 2016
  br i1 %154, label %21, label %155

155:                                              ; preds = %21
  %156 = tail call ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) nonnull %2, i16 0, i64 16777216, i32 822243328)
  %157 = getelementptr i8, ptr addrspace(7) %156, i32 %.idx1
  %158 = getelementptr [4 x i8], ptr addrspace(7) %157, i32 %19
  store <16 x float> %152, ptr addrspace(7) %158, align 4
  ret void
}

; Function Attrs: alwaysinline mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.y() #1

; Function Attrs: alwaysinline mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: alwaysinline mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #2

; Function Attrs: alwaysinline mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) readnone, i16, i64, i32) #3

; Function Attrs: alwaysinline mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: alwaysinline mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <16 x float> @llvm.fma.v16f32(<16 x float>, <16 x float>, <16 x float>) #3

attributes #0 = { alwaysinline nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: write) "amdgpu-flat-work-group-size"="256,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-z" "uniform-work-group-size" }
attributes #1 = { alwaysinline mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { alwaysinline mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #3 = { alwaysinline mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!2 = !{i32 32, i32 8, i32 1}
