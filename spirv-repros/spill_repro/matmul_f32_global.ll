; IREE f32 matmul 2048x2048x2048, SPIR-V round-tripped (no buffer instructions).
; This is the output of `amd-llvm-spirv -r` (reverse translation from SPIR-V).
; Uses global_load/global_store instead of buffer_load/buffer_store.
;
; The ROCm comgr JIT compiles this at -O0, causing massive register spilling.
;
; Compile (reproduces JIT spilling):
;   clang -target amdgcn-amd-amdhsa -mcpu=gfx1201 -S -O0 matmul_f32_global.ll -o global_O0.s
;   grep -c scratch_ global_O0.s  # => ~1229 (massive spilling!)
;
; Compile (with -O2, no spilling):
;   llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=asm matmul_f32_global.ll -o global_O2.s
;   grep -c scratch_ global_O2.s  # => 0 (no spilling)
;
; ModuleID = 'hip_code_object.spv.bc'
target datalayout = "m:e-e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@0 = internal addrspace(1) constant i32 0
@__oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 500

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #0

; Function Attrs: alwaysinline convergent mustprogress nounwind
define amdgpu_kernel void @matmul_f32_dispatch_0_matmul_2048x2048x2048_f32(ptr addrspace(1) noalias readonly align 16 %0, ptr addrspace(1) noalias readonly align 16 %1, ptr addrspace(1) noalias align 16 %2) #1 !kernel_arg_addr_space !7 !kernel_arg_access_qual !8 !kernel_arg_type !9 !kernel_arg_type_qual !10 !kernel_arg_base_type !9 !spirv.ParameterDecorations !11 {
  %4 = call i32 @llvm.amdgcn.workitem.id.y()
  %5 = call i32 @llvm.amdgcn.workitem.id.x()
  %6 = mul nuw nsw i32 %4, 32
  %7 = add i32 %6, %5
  %8 = udiv i32 %7, 8
  %9 = urem i32 %7, 8
  %10 = call i32 @llvm.amdgcn.workgroup.id.x()
  %11 = udiv i32 %10, 16
  %12 = urem i32 %10, 16
  %13 = mul nuw nsw i32 %11, 32
  %14 = add i32 %8, %13
  %15 = zext i32 %14 to i64
  %16 = mul nuw nsw i32 %9, 16
  %17 = mul nuw nsw i32 %12, 128
  %18 = add i32 %16, %17
  %19 = zext i32 %18 to i64
  %20 = shl nsw i64 %19, 2
  %21 = add nuw nsw i64 %20, 253952
  %22 = getelementptr i8, ptr addrspace(1) %1, i64 %21
  %23 = shl nsw i64 %15, 13
  %24 = getelementptr i8, ptr addrspace(1) %0, i64 %23
  br label %25

25:                                               ; preds = %35, %3
  %26 = phi ptr addrspace(1) [ %261, %35 ], [ %24, %3 ]
  %27 = phi ptr addrspace(1) [ %260, %35 ], [ %22, %3 ]
  %28 = phi i32 [ %259, %35 ], [ 0, %3 ]
  %29 = phi <16 x float> [ %258, %35 ], [ zeroinitializer, %3 ]
  %30 = icmp slt i32 %28, 2048
  br i1 %30, label %35, label %31

31:                                               ; preds = %25
  %32 = mul i64 %15, 2048
  %33 = add i64 %32, %19
  %34 = getelementptr float, ptr addrspace(1) %2, i64 %33
  store <16 x float> %29, ptr addrspace(1) %34, align 4
  ret void

35:                                               ; preds = %25
  %36 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 0
  %37 = load float, ptr addrspace(1) %36, align 4
  %38 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 1
  %39 = load float, ptr addrspace(1) %38, align 4
  %40 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 2
  %41 = load float, ptr addrspace(1) %40, align 4
  %42 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 3
  %43 = load float, ptr addrspace(1) %42, align 4
  %44 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 4
  %45 = load float, ptr addrspace(1) %44, align 4
  %46 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 5
  %47 = load float, ptr addrspace(1) %46, align 4
  %48 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 6
  %49 = load float, ptr addrspace(1) %48, align 4
  %50 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 7
  %51 = load float, ptr addrspace(1) %50, align 4
  %52 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 8
  %53 = load float, ptr addrspace(1) %52, align 4
  %54 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 9
  %55 = load float, ptr addrspace(1) %54, align 4
  %56 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 10
  %57 = load float, ptr addrspace(1) %56, align 4
  %58 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 11
  %59 = load float, ptr addrspace(1) %58, align 4
  %60 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 12
  %61 = load float, ptr addrspace(1) %60, align 4
  %62 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 13
  %63 = load float, ptr addrspace(1) %62, align 4
  %64 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 14
  %65 = load float, ptr addrspace(1) %64, align 4
  %66 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 15
  %67 = load float, ptr addrspace(1) %66, align 4
  %68 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 16
  %69 = load float, ptr addrspace(1) %68, align 4
  %70 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 17
  %71 = load float, ptr addrspace(1) %70, align 4
  %72 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 18
  %73 = load float, ptr addrspace(1) %72, align 4
  %74 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 19
  %75 = load float, ptr addrspace(1) %74, align 4
  %76 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 20
  %77 = load float, ptr addrspace(1) %76, align 4
  %78 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 21
  %79 = load float, ptr addrspace(1) %78, align 4
  %80 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 22
  %81 = load float, ptr addrspace(1) %80, align 4
  %82 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 23
  %83 = load float, ptr addrspace(1) %82, align 4
  %84 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 24
  %85 = load float, ptr addrspace(1) %84, align 4
  %86 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 25
  %87 = load float, ptr addrspace(1) %86, align 4
  %88 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 26
  %89 = load float, ptr addrspace(1) %88, align 4
  %90 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 27
  %91 = load float, ptr addrspace(1) %90, align 4
  %92 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 28
  %93 = load float, ptr addrspace(1) %92, align 4
  %94 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 29
  %95 = load float, ptr addrspace(1) %94, align 4
  %96 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 30
  %97 = load float, ptr addrspace(1) %96, align 4
  %98 = getelementptr inbounds <32 x float>, ptr addrspace(1) %26, i32 0, i32 31
  %99 = load float, ptr addrspace(1) %98, align 4
  %100 = getelementptr i8, ptr addrspace(1) %27, i64 -253952
  %101 = load <16 x float>, ptr addrspace(1) %100, align 4
  %102 = getelementptr i8, ptr addrspace(1) %27, i64 -245760
  %103 = load <16 x float>, ptr addrspace(1) %102, align 4
  %104 = getelementptr i8, ptr addrspace(1) %27, i64 -237568
  %105 = load <16 x float>, ptr addrspace(1) %104, align 4
  %106 = getelementptr i8, ptr addrspace(1) %27, i64 -229376
  %107 = load <16 x float>, ptr addrspace(1) %106, align 4
  %108 = getelementptr i8, ptr addrspace(1) %27, i64 -221184
  %109 = load <16 x float>, ptr addrspace(1) %108, align 4
  %110 = getelementptr i8, ptr addrspace(1) %27, i64 -212992
  %111 = load <16 x float>, ptr addrspace(1) %110, align 4
  %112 = getelementptr i8, ptr addrspace(1) %27, i64 -204800
  %113 = load <16 x float>, ptr addrspace(1) %112, align 4
  %114 = getelementptr i8, ptr addrspace(1) %27, i64 -196608
  %115 = load <16 x float>, ptr addrspace(1) %114, align 4
  %116 = getelementptr i8, ptr addrspace(1) %27, i64 -188416
  %117 = load <16 x float>, ptr addrspace(1) %116, align 4
  %118 = getelementptr i8, ptr addrspace(1) %27, i64 -180224
  %119 = load <16 x float>, ptr addrspace(1) %118, align 4
  %120 = getelementptr i8, ptr addrspace(1) %27, i64 -172032
  %121 = load <16 x float>, ptr addrspace(1) %120, align 4
  %122 = getelementptr i8, ptr addrspace(1) %27, i64 -163840
  %123 = load <16 x float>, ptr addrspace(1) %122, align 4
  %124 = getelementptr i8, ptr addrspace(1) %27, i64 -155648
  %125 = load <16 x float>, ptr addrspace(1) %124, align 4
  %126 = getelementptr i8, ptr addrspace(1) %27, i64 -147456
  %127 = load <16 x float>, ptr addrspace(1) %126, align 4
  %128 = getelementptr i8, ptr addrspace(1) %27, i64 -139264
  %129 = load <16 x float>, ptr addrspace(1) %128, align 4
  %130 = getelementptr i8, ptr addrspace(1) %27, i64 -131072
  %131 = load <16 x float>, ptr addrspace(1) %130, align 4
  %132 = getelementptr i8, ptr addrspace(1) %27, i64 -122880
  %133 = load <16 x float>, ptr addrspace(1) %132, align 4
  %134 = getelementptr i8, ptr addrspace(1) %27, i64 -114688
  %135 = load <16 x float>, ptr addrspace(1) %134, align 4
  %136 = getelementptr i8, ptr addrspace(1) %27, i64 -106496
  %137 = load <16 x float>, ptr addrspace(1) %136, align 4
  %138 = getelementptr i8, ptr addrspace(1) %27, i64 -98304
  %139 = load <16 x float>, ptr addrspace(1) %138, align 4
  %140 = getelementptr i8, ptr addrspace(1) %27, i64 -90112
  %141 = load <16 x float>, ptr addrspace(1) %140, align 4
  %142 = getelementptr i8, ptr addrspace(1) %27, i64 -81920
  %143 = load <16 x float>, ptr addrspace(1) %142, align 4
  %144 = getelementptr i8, ptr addrspace(1) %27, i64 -73728
  %145 = load <16 x float>, ptr addrspace(1) %144, align 4
  %146 = getelementptr i8, ptr addrspace(1) %27, i64 -65536
  %147 = load <16 x float>, ptr addrspace(1) %146, align 4
  %148 = getelementptr i8, ptr addrspace(1) %27, i64 -57344
  %149 = load <16 x float>, ptr addrspace(1) %148, align 4
  %150 = getelementptr i8, ptr addrspace(1) %27, i64 -49152
  %151 = load <16 x float>, ptr addrspace(1) %150, align 4
  %152 = getelementptr i8, ptr addrspace(1) %27, i64 -40960
  %153 = load <16 x float>, ptr addrspace(1) %152, align 4
  %154 = getelementptr i8, ptr addrspace(1) %27, i64 -32768
  %155 = load <16 x float>, ptr addrspace(1) %154, align 4
  %156 = getelementptr i8, ptr addrspace(1) %27, i64 -24576
  %157 = load <16 x float>, ptr addrspace(1) %156, align 4
  %158 = getelementptr i8, ptr addrspace(1) %27, i64 -16384
  %159 = load <16 x float>, ptr addrspace(1) %158, align 4
  %160 = getelementptr i8, ptr addrspace(1) %27, i64 -8192
  %161 = load <16 x float>, ptr addrspace(1) %160, align 4
  %162 = load <16 x float>, ptr addrspace(1) %27, align 4
  %163 = insertelement <16 x float> undef, float %99, i32 0
  %164 = shufflevector <16 x float> %163, <16 x float> undef, <16 x i32> zeroinitializer
  %165 = call <16 x float> @llvm.fma.v16f32(<16 x float> %164, <16 x float> %162, <16 x float> %29) #3
  %166 = insertelement <16 x float> undef, float %97, i32 0
  %167 = shufflevector <16 x float> %166, <16 x float> undef, <16 x i32> zeroinitializer
  %168 = call <16 x float> @llvm.fma.v16f32(<16 x float> %167, <16 x float> %161, <16 x float> %165) #3
  %169 = insertelement <16 x float> undef, float %95, i32 0
  %170 = shufflevector <16 x float> %169, <16 x float> undef, <16 x i32> zeroinitializer
  %171 = call <16 x float> @llvm.fma.v16f32(<16 x float> %170, <16 x float> %159, <16 x float> %168) #3
  %172 = insertelement <16 x float> undef, float %93, i32 0
  %173 = shufflevector <16 x float> %172, <16 x float> undef, <16 x i32> zeroinitializer
  %174 = call <16 x float> @llvm.fma.v16f32(<16 x float> %173, <16 x float> %157, <16 x float> %171) #3
  %175 = insertelement <16 x float> undef, float %91, i32 0
  %176 = shufflevector <16 x float> %175, <16 x float> undef, <16 x i32> zeroinitializer
  %177 = call <16 x float> @llvm.fma.v16f32(<16 x float> %176, <16 x float> %155, <16 x float> %174) #3
  %178 = insertelement <16 x float> undef, float %89, i32 0
  %179 = shufflevector <16 x float> %178, <16 x float> undef, <16 x i32> zeroinitializer
  %180 = call <16 x float> @llvm.fma.v16f32(<16 x float> %179, <16 x float> %153, <16 x float> %177) #3
  %181 = insertelement <16 x float> undef, float %87, i32 0
  %182 = shufflevector <16 x float> %181, <16 x float> undef, <16 x i32> zeroinitializer
  %183 = call <16 x float> @llvm.fma.v16f32(<16 x float> %182, <16 x float> %151, <16 x float> %180) #3
  %184 = insertelement <16 x float> undef, float %85, i32 0
  %185 = shufflevector <16 x float> %184, <16 x float> undef, <16 x i32> zeroinitializer
  %186 = call <16 x float> @llvm.fma.v16f32(<16 x float> %185, <16 x float> %149, <16 x float> %183) #3
  %187 = insertelement <16 x float> undef, float %83, i32 0
  %188 = shufflevector <16 x float> %187, <16 x float> undef, <16 x i32> zeroinitializer
  %189 = call <16 x float> @llvm.fma.v16f32(<16 x float> %188, <16 x float> %147, <16 x float> %186) #3
  %190 = insertelement <16 x float> undef, float %81, i32 0
  %191 = shufflevector <16 x float> %190, <16 x float> undef, <16 x i32> zeroinitializer
  %192 = call <16 x float> @llvm.fma.v16f32(<16 x float> %191, <16 x float> %145, <16 x float> %189) #3
  %193 = insertelement <16 x float> undef, float %79, i32 0
  %194 = shufflevector <16 x float> %193, <16 x float> undef, <16 x i32> zeroinitializer
  %195 = call <16 x float> @llvm.fma.v16f32(<16 x float> %194, <16 x float> %143, <16 x float> %192) #3
  %196 = insertelement <16 x float> undef, float %77, i32 0
  %197 = shufflevector <16 x float> %196, <16 x float> undef, <16 x i32> zeroinitializer
  %198 = call <16 x float> @llvm.fma.v16f32(<16 x float> %197, <16 x float> %141, <16 x float> %195) #3
  %199 = insertelement <16 x float> undef, float %75, i32 0
  %200 = shufflevector <16 x float> %199, <16 x float> undef, <16 x i32> zeroinitializer
  %201 = call <16 x float> @llvm.fma.v16f32(<16 x float> %200, <16 x float> %139, <16 x float> %198) #3
  %202 = insertelement <16 x float> undef, float %73, i32 0
  %203 = shufflevector <16 x float> %202, <16 x float> undef, <16 x i32> zeroinitializer
  %204 = call <16 x float> @llvm.fma.v16f32(<16 x float> %203, <16 x float> %137, <16 x float> %201) #3
  %205 = insertelement <16 x float> undef, float %71, i32 0
  %206 = shufflevector <16 x float> %205, <16 x float> undef, <16 x i32> zeroinitializer
  %207 = call <16 x float> @llvm.fma.v16f32(<16 x float> %206, <16 x float> %135, <16 x float> %204) #3
  %208 = insertelement <16 x float> undef, float %69, i32 0
  %209 = shufflevector <16 x float> %208, <16 x float> undef, <16 x i32> zeroinitializer
  %210 = call <16 x float> @llvm.fma.v16f32(<16 x float> %209, <16 x float> %133, <16 x float> %207) #3
  %211 = insertelement <16 x float> undef, float %67, i32 0
  %212 = shufflevector <16 x float> %211, <16 x float> undef, <16 x i32> zeroinitializer
  %213 = call <16 x float> @llvm.fma.v16f32(<16 x float> %212, <16 x float> %131, <16 x float> %210) #3
  %214 = insertelement <16 x float> undef, float %65, i32 0
  %215 = shufflevector <16 x float> %214, <16 x float> undef, <16 x i32> zeroinitializer
  %216 = call <16 x float> @llvm.fma.v16f32(<16 x float> %215, <16 x float> %129, <16 x float> %213) #3
  %217 = insertelement <16 x float> undef, float %63, i32 0
  %218 = shufflevector <16 x float> %217, <16 x float> undef, <16 x i32> zeroinitializer
  %219 = call <16 x float> @llvm.fma.v16f32(<16 x float> %218, <16 x float> %127, <16 x float> %216) #3
  %220 = insertelement <16 x float> undef, float %61, i32 0
  %221 = shufflevector <16 x float> %220, <16 x float> undef, <16 x i32> zeroinitializer
  %222 = call <16 x float> @llvm.fma.v16f32(<16 x float> %221, <16 x float> %125, <16 x float> %219) #3
  %223 = insertelement <16 x float> undef, float %59, i32 0
  %224 = shufflevector <16 x float> %223, <16 x float> undef, <16 x i32> zeroinitializer
  %225 = call <16 x float> @llvm.fma.v16f32(<16 x float> %224, <16 x float> %123, <16 x float> %222) #3
  %226 = insertelement <16 x float> undef, float %57, i32 0
  %227 = shufflevector <16 x float> %226, <16 x float> undef, <16 x i32> zeroinitializer
  %228 = call <16 x float> @llvm.fma.v16f32(<16 x float> %227, <16 x float> %121, <16 x float> %225) #3
  %229 = insertelement <16 x float> undef, float %55, i32 0
  %230 = shufflevector <16 x float> %229, <16 x float> undef, <16 x i32> zeroinitializer
  %231 = call <16 x float> @llvm.fma.v16f32(<16 x float> %230, <16 x float> %119, <16 x float> %228) #3
  %232 = insertelement <16 x float> undef, float %53, i32 0
  %233 = shufflevector <16 x float> %232, <16 x float> undef, <16 x i32> zeroinitializer
  %234 = call <16 x float> @llvm.fma.v16f32(<16 x float> %233, <16 x float> %117, <16 x float> %231) #3
  %235 = insertelement <16 x float> undef, float %51, i32 0
  %236 = shufflevector <16 x float> %235, <16 x float> undef, <16 x i32> zeroinitializer
  %237 = call <16 x float> @llvm.fma.v16f32(<16 x float> %236, <16 x float> %115, <16 x float> %234) #3
  %238 = insertelement <16 x float> undef, float %49, i32 0
  %239 = shufflevector <16 x float> %238, <16 x float> undef, <16 x i32> zeroinitializer
  %240 = call <16 x float> @llvm.fma.v16f32(<16 x float> %239, <16 x float> %113, <16 x float> %237) #3
  %241 = insertelement <16 x float> undef, float %47, i32 0
  %242 = shufflevector <16 x float> %241, <16 x float> undef, <16 x i32> zeroinitializer
  %243 = call <16 x float> @llvm.fma.v16f32(<16 x float> %242, <16 x float> %111, <16 x float> %240) #3
  %244 = insertelement <16 x float> undef, float %45, i32 0
  %245 = shufflevector <16 x float> %244, <16 x float> undef, <16 x i32> zeroinitializer
  %246 = call <16 x float> @llvm.fma.v16f32(<16 x float> %245, <16 x float> %109, <16 x float> %243) #3
  %247 = insertelement <16 x float> undef, float %43, i32 0
  %248 = shufflevector <16 x float> %247, <16 x float> undef, <16 x i32> zeroinitializer
  %249 = call <16 x float> @llvm.fma.v16f32(<16 x float> %248, <16 x float> %107, <16 x float> %246) #3
  %250 = insertelement <16 x float> undef, float %41, i32 0
  %251 = shufflevector <16 x float> %250, <16 x float> undef, <16 x i32> zeroinitializer
  %252 = call <16 x float> @llvm.fma.v16f32(<16 x float> %251, <16 x float> %105, <16 x float> %249) #3
  %253 = insertelement <16 x float> undef, float %39, i32 0
  %254 = shufflevector <16 x float> %253, <16 x float> undef, <16 x i32> zeroinitializer
  %255 = call <16 x float> @llvm.fma.v16f32(<16 x float> %254, <16 x float> %103, <16 x float> %252) #3
  %256 = insertelement <16 x float> undef, float %37, i32 0
  %257 = shufflevector <16 x float> %256, <16 x float> undef, <16 x i32> zeroinitializer
  %258 = call <16 x float> @llvm.fma.v16f32(<16 x float> %257, <16 x float> %101, <16 x float> %255) #3
  %259 = add i32 %28, 32
  %260 = getelementptr i8, ptr addrspace(1) %27, i64 262144
  %261 = getelementptr i8, ptr addrspace(1) %26, i64 128
  br label %25
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <16 x float> @llvm.fma.v16f32(<16 x float>, <16 x float>, <16 x float>) #2

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { alwaysinline convergent mustprogress nounwind "uniform-work-group-size"="true" }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nounwind }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!spirv.Generator = !{!4}
!llvm.module.flags = !{!5, !6}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i32 1, i32 2}
!3 = !{}
!4 = !{i16 0, i16 -1}
!5 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!6 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!7 = !{i32 1, i32 1, i32 1}
!8 = !{!"none", !"none", !"none"}
!9 = !{!"char*", !"char*", !"float*"}
!10 = !{!"restrict", !"restrict", !"restrict"}
!11 = !{!12, !12, !16}
!12 = !{!13, !14, !15}
!13 = !{i32 38, i32 6}
!14 = !{i32 38, i32 4}
!15 = !{i32 44, i32 16}
!16 = !{!14, !15}
