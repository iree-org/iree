; To reproduce the .optimized.ll from the .linked.ll, run:
; opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 --passes='verify,memprof-remove-attributes,annotation2metadata,forceattrs,inferattrs,coro-early,function<eager-inv>(ee-instrument<>,lower-expect,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;no-switch-range-to-icmp;no-switch-to-arithmetic;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,sroa<modify-cfg>,early-cse<>),openmp-opt,amdgpu-printf-runtime-binding,ipsccp,called-value-propagation,globalopt,function<eager-inv>(mem2reg,instcombine<max-iterations=1;no-verify-fixpoint>,amdgpu-usenative,amdgpu-simplifylib,amdgpu-uniform-intrinsic-combine,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-arithmetic;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>),always-inline,require<globals-aa>,function(invalidate<aa>),require<profile-summary>,cgscc(devirt<4>(inline,function-attrs<skip-non-recursive-function-attrs>,openmp-opt-cgscc,function(amdgpu-promote-kernel-arguments,infer-address-spaces,amdgpu-lower-kernel-attributes,amdgpu-promote-alloca-to-vector),function<eager-inv;no-rerun>(sroa<modify-cfg>,early-cse<memssa>,speculative-execution<only-if-divergent-target>,jump-threading,correlated-propagation,jump-table-to-switch,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-arithmetic;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,instcombine<max-iterations=1;no-verify-fixpoint>,aggressive-instcombine,libcalls-shrinkwrap,amdgpu-usenative,amdgpu-simplifylib,amdgpu-uniform-intrinsic-combine,tailcallelim,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-arithmetic;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,reassociate,constraint-elimination,loop-mssa(loop-instsimplify,loop-simplifycfg,licm<no-allowspeculation>,loop-rotate<header-duplication;no-prepare-for-lto;no-check-exit-count>,licm<allowspeculation>,simple-loop-unswitch<no-nontrivial;trivial>),simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-arithmetic;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,instcombine<max-iterations=1;no-verify-fixpoint>,loop(loop-idiom,indvars,extra-simple-loop-unswitch-passes,loop-deletion,loop-unroll-full),sroa<modify-cfg>,vector-combine,mldst-motion<no-split-footer-bb>,gvn<>,sccp,bdce,instcombine<max-iterations=1;no-verify-fixpoint>,amdgpu-usenative,amdgpu-simplifylib,amdgpu-uniform-intrinsic-combine,jump-threading,correlated-propagation,adce,memcpyopt,dse,move-auto-init,loop-mssa(licm<allowspeculation>),coro-elide,infer-address-spaces,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;switch-to-arithmetic;no-switch-to-lookup;keep-loops;hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,instcombine<max-iterations=1;no-verify-fixpoint>,amdgpu-usenative,amdgpu-simplifylib,amdgpu-uniform-intrinsic-combine),function-attrs,function(require<should-not-run-function-passes>),coro-split,coro-annotation-elide)),deadargelim,coro-cleanup,globalopt,globaldce,elim-avail-extern,rpo-function-attrs,recompute-globalsaa,function<eager-inv>(drop-unnecessary-assumes,float2int,lower-constant-intrinsics,loop(loop-rotate<header-duplication;no-prepare-for-lto;check-exit-count>,loop-deletion),loop-distribute,inject-tli-mappings,loop-vectorize<no-interleave-forced-only;no-vectorize-forced-only;>,drop-unnecessary-assumes,infer-alignment,loop-load-elim,instcombine<max-iterations=1;no-verify-fixpoint>,simplifycfg<bonus-inst-threshold=1;forward-switch-cond;switch-range-to-icmp;switch-to-arithmetic;switch-to-lookup;no-keep-loops;hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,slp-vectorizer,vector-combine,instcombine<max-iterations=1;no-verify-fixpoint>,loop-unroll<O2>,transform-warning,sroa<preserve-cfg>,infer-alignment,instcombine<max-iterations=1;no-verify-fixpoint>,loop-mssa(licm<allowspeculation>),alignment-from-assumptions,infer-address-spaces,loop-sink,instsimplify,div-rem-pairs,mergeicmps,expand-memcmp,tailcallelim,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;switch-to-arithmetic;no-switch-to-lookup;keep-loops;no-hoist-common-insts;hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;speculate-unpredictables>),alloc-token,amdgpu-attributor,globaldce,constmerge,cg-profile,rel-lookup-table-converter,function(annotation-remarks),verify' <.linked.ll>
; The flag '-S' is to emit LLVMIR.
; The behavior of some passes depends on '-mtriple' and '-mcpu'.

; ModuleID = 'matmul_2048x2048x2048_dispatch_0'
source_filename = "matmul_2048x2048x2048_dispatch_0"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@__shared_memory___0 = private unnamed_addr addrspace(3) global [64 x [132 x half]] undef, align 16
@__shared_memory__ = private unnamed_addr addrspace(3) global [256 x [68 x half]] undef, align 16

; Function Attrs: alwaysinline nofree norecurse nounwind
define amdgpu_kernel void @matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32(ptr addrspace(1) noalias noundef nonnull readonly align 16 captures(none) %0, ptr addrspace(1) noalias noundef nonnull readonly align 16 captures(none) %1, ptr addrspace(1) noalias noundef nonnull writeonly align 16 %2) local_unnamed_addr #0 !reqd_work_group_size !2 {
  %4 = tail call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x()
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %0, i64 64) ]
  %5 = tail call ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) nonnull %0, i16 0, i64 8388608, i32 822243328)
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %1, i64 64) ]
  %6 = tail call ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) nonnull %1, i16 0, i64 8388608, i32 822243328)
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %2, i64 64) ]
  %7 = and i32 %4, 15
  %8 = lshr i32 %4, 1
  %9 = and i32 %8, 8
  %10 = lshr i32 %4, 3
  %11 = or disjoint i32 %4, 256
  %12 = lshr i32 %11, 3
  %13 = or disjoint i32 %4, 512
  %14 = lshr i32 %13, 3
  %15 = or disjoint i32 %4, 768
  %16 = lshr i32 %15, 3
  %17 = or disjoint i32 %10, 128
  %18 = or disjoint i32 %10, 160
  %19 = or disjoint i32 %10, 192
  %20 = or disjoint i32 %10, 224
  %21 = lshr i32 %4, 4
  %22 = lshr i32 %11, 4
  %23 = lshr i32 %13, 4
  %24 = lshr i32 %15, 4
  %25 = shl nuw nsw i32 %4, 3
  %26 = and i32 %25, 56
  %27 = and i32 %25, 120
  %28 = tail call range(i32 0, 128) i32 @llvm.amdgcn.workgroup.id.x()
  %29 = shl nuw nsw i32 %28, 4
  %30 = and i32 %29, 1792
  %31 = or disjoint i32 %30, %10
  %32 = or disjoint i32 %12, %30
  %.idx32 = shl nuw nsw i32 %31, 12
  %33 = getelementptr i8, ptr addrspace(7) %5, i32 %.idx32
  %34 = getelementptr [2 x i8], ptr addrspace(7) %33, i32 %26
  %35 = load <8 x half>, ptr addrspace(7) %34, align 2
  %36 = or disjoint i32 %14, %30
  %.idx33 = shl nuw nsw i32 %32, 12
  %37 = getelementptr i8, ptr addrspace(7) %5, i32 %.idx33
  %38 = getelementptr [2 x i8], ptr addrspace(7) %37, i32 %26
  %39 = load <8 x half>, ptr addrspace(7) %38, align 2
  %40 = or disjoint i32 %16, %30
  %.idx34 = shl nuw nsw i32 %36, 12
  %41 = getelementptr i8, ptr addrspace(7) %5, i32 %.idx34
  %42 = getelementptr [2 x i8], ptr addrspace(7) %41, i32 %26
  %43 = load <8 x half>, ptr addrspace(7) %42, align 2
  %44 = or disjoint i32 %17, %30
  %.idx35 = shl nuw nsw i32 %40, 12
  %45 = getelementptr i8, ptr addrspace(7) %5, i32 %.idx35
  %46 = getelementptr [2 x i8], ptr addrspace(7) %45, i32 %26
  %47 = load <8 x half>, ptr addrspace(7) %46, align 2
  %48 = or disjoint i32 %18, %30
  %.idx36 = shl nuw nsw i32 %44, 12
  %49 = getelementptr i8, ptr addrspace(7) %5, i32 %.idx36
  %50 = getelementptr [2 x i8], ptr addrspace(7) %49, i32 %26
  %51 = load <8 x half>, ptr addrspace(7) %50, align 2
  %52 = or disjoint i32 %19, %30
  %.idx37 = shl nuw nsw i32 %48, 12
  %53 = getelementptr i8, ptr addrspace(7) %5, i32 %.idx37
  %54 = getelementptr [2 x i8], ptr addrspace(7) %53, i32 %26
  %55 = load <8 x half>, ptr addrspace(7) %54, align 2
  %56 = or disjoint i32 %20, %30
  %.idx38 = shl nuw nsw i32 %52, 12
  %57 = getelementptr i8, ptr addrspace(7) %5, i32 %.idx38
  %58 = getelementptr [2 x i8], ptr addrspace(7) %57, i32 %26
  %59 = load <8 x half>, ptr addrspace(7) %58, align 2
  %60 = shl nuw nsw i32 %28, 7
  %61 = and i32 %60, 1920
  %62 = or disjoint i32 %27, %61
  %.idx39 = shl nuw nsw i32 %56, 12
  %63 = getelementptr i8, ptr addrspace(7) %5, i32 %.idx39
  %64 = getelementptr [2 x i8], ptr addrspace(7) %63, i32 %26
  %65 = load <8 x half>, ptr addrspace(7) %64, align 2
  %.idx = shl nuw nsw i32 %21, 12
  %66 = getelementptr i8, ptr addrspace(7) %6, i32 %.idx
  %67 = getelementptr [2 x i8], ptr addrspace(7) %66, i32 %62
  %68 = load <8 x half>, ptr addrspace(7) %67, align 2
  %.idx1 = shl nuw nsw i32 %22, 12
  %69 = getelementptr i8, ptr addrspace(7) %6, i32 %.idx1
  %70 = getelementptr [2 x i8], ptr addrspace(7) %69, i32 %62
  %71 = load <8 x half>, ptr addrspace(7) %70, align 2
  %.idx2 = shl nuw nsw i32 %23, 12
  %72 = getelementptr i8, ptr addrspace(7) %6, i32 %.idx2
  %73 = getelementptr [2 x i8], ptr addrspace(7) %72, i32 %62
  %74 = load <8 x half>, ptr addrspace(7) %73, align 2
  %.idx3 = shl nuw nsw i32 %24, 12
  %75 = getelementptr i8, ptr addrspace(7) %6, i32 %.idx3
  %76 = getelementptr [2 x i8], ptr addrspace(7) %75, i32 %62
  %77 = load <8 x half>, ptr addrspace(7) %76, align 2
  %.idx5 = mul nuw nsw i32 %10, 136
  %78 = getelementptr i8, ptr addrspace(3) @__shared_memory__, i32 %.idx5
  %79 = getelementptr [2 x i8], ptr addrspace(3) %78, i32 %26
  store <8 x half> %35, ptr addrspace(3) %79, align 8
  %.idx7 = mul nuw nsw i32 %12, 136
  %80 = getelementptr i8, ptr addrspace(3) @__shared_memory__, i32 %.idx7
  %81 = getelementptr [2 x i8], ptr addrspace(3) %80, i32 %26
  store <8 x half> %39, ptr addrspace(3) %81, align 8
  %.idx9 = mul nuw nsw i32 %14, 136
  %82 = getelementptr i8, ptr addrspace(3) @__shared_memory__, i32 %.idx9
  %83 = getelementptr [2 x i8], ptr addrspace(3) %82, i32 %26
  store <8 x half> %43, ptr addrspace(3) %83, align 8
  %.idx11 = mul nuw nsw i32 %16, 136
  %84 = getelementptr i8, ptr addrspace(3) @__shared_memory__, i32 %.idx11
  %85 = getelementptr [2 x i8], ptr addrspace(3) %84, i32 %26
  store <8 x half> %47, ptr addrspace(3) %85, align 8
  %.idx13 = mul nuw nsw i32 %17, 136
  %86 = getelementptr i8, ptr addrspace(3) @__shared_memory__, i32 %.idx13
  %87 = getelementptr [2 x i8], ptr addrspace(3) %86, i32 %26
  store <8 x half> %51, ptr addrspace(3) %87, align 8
  %.idx15 = mul nuw nsw i32 %18, 136
  %88 = getelementptr i8, ptr addrspace(3) @__shared_memory__, i32 %.idx15
  %89 = getelementptr [2 x i8], ptr addrspace(3) %88, i32 %26
  store <8 x half> %55, ptr addrspace(3) %89, align 8
  %.idx17 = mul nuw nsw i32 %19, 136
  %90 = getelementptr i8, ptr addrspace(3) @__shared_memory__, i32 %.idx17
  %91 = getelementptr [2 x i8], ptr addrspace(3) %90, i32 %26
  store <8 x half> %59, ptr addrspace(3) %91, align 8
  %.idx19 = mul nuw nsw i32 %20, 136
  %92 = getelementptr i8, ptr addrspace(3) @__shared_memory__, i32 %.idx19
  %93 = getelementptr [2 x i8], ptr addrspace(3) %92, i32 %26
  store <8 x half> %65, ptr addrspace(3) %93, align 8
  %.idx21 = mul nuw nsw i32 %21, 264
  %94 = getelementptr i8, ptr addrspace(3) @__shared_memory___0, i32 %.idx21
  %95 = getelementptr [2 x i8], ptr addrspace(3) %94, i32 %27
  store <8 x half> %68, ptr addrspace(3) %95, align 8
  %.idx23 = mul nuw nsw i32 %22, 264
  %96 = getelementptr i8, ptr addrspace(3) @__shared_memory___0, i32 %.idx23
  %97 = getelementptr [2 x i8], ptr addrspace(3) %96, i32 %27
  store <8 x half> %71, ptr addrspace(3) %97, align 8
  %.idx25 = mul nuw nsw i32 %23, 264
  %98 = getelementptr i8, ptr addrspace(3) @__shared_memory___0, i32 %.idx25
  %99 = getelementptr [2 x i8], ptr addrspace(3) %98, i32 %27
  store <8 x half> %74, ptr addrspace(3) %99, align 8
  %.idx27 = mul nuw nsw i32 %24, 264
  %100 = getelementptr i8, ptr addrspace(3) @__shared_memory___0, i32 %.idx27
  %101 = getelementptr [2 x i8], ptr addrspace(3) %100, i32 %27
  store <8 x half> %77, ptr addrspace(3) %101, align 8
  %102 = and i32 %4, 207
  %103 = or disjoint i32 %9, 16
  %104 = or disjoint i32 %9, 32
  %105 = or disjoint i32 %9, 48
  %106 = or i32 %4, 48
  %107 = shl nuw nsw i32 %4, 1
  %108 = and i32 %107, 64
  %109 = or disjoint i32 %108, %7
  %invariant.gep = getelementptr [2 x i8], ptr addrspace(7) %6, i32 %62
  %narrow = mul nuw nsw i32 %102, 68
  %110 = getelementptr [2 x i8], ptr addrspace(3) @__shared_memory__, i32 %9
  %111 = getelementptr [2 x i8], ptr addrspace(3) %110, i32 %narrow
  %112 = getelementptr [2 x i8], ptr addrspace(3) @__shared_memory__, i32 %103
  %113 = getelementptr [2 x i8], ptr addrspace(3) %112, i32 %narrow
  %114 = getelementptr [2 x i8], ptr addrspace(3) @__shared_memory__, i32 %104
  %115 = getelementptr [2 x i8], ptr addrspace(3) %114, i32 %narrow
  %116 = getelementptr [2 x i8], ptr addrspace(3) @__shared_memory__, i32 %105
  %117 = getelementptr [2 x i8], ptr addrspace(3) %116, i32 %narrow
  %118 = mul nuw nsw i32 %102, 68
  %narrow40 = add nuw nsw i32 %118, 1088
  %119 = getelementptr [2 x i8], ptr addrspace(3) %110, i32 %narrow40
  %120 = getelementptr [2 x i8], ptr addrspace(3) %112, i32 %narrow40
  %121 = getelementptr [2 x i8], ptr addrspace(3) %114, i32 %narrow40
  %122 = getelementptr [2 x i8], ptr addrspace(3) %116, i32 %narrow40
  %123 = mul nuw nsw i32 %102, 68
  %narrow41 = add nuw nsw i32 %123, 2176
  %124 = getelementptr [2 x i8], ptr addrspace(3) %110, i32 %narrow41
  %125 = getelementptr [2 x i8], ptr addrspace(3) %112, i32 %narrow41
  %126 = getelementptr [2 x i8], ptr addrspace(3) %114, i32 %narrow41
  %127 = getelementptr [2 x i8], ptr addrspace(3) %116, i32 %narrow41
  %narrow42 = mul nuw nsw i32 %106, 68
  %128 = getelementptr [2 x i8], ptr addrspace(3) %110, i32 %narrow42
  %129 = getelementptr [2 x i8], ptr addrspace(3) %112, i32 %narrow42
  %130 = getelementptr [2 x i8], ptr addrspace(3) %114, i32 %narrow42
  %131 = getelementptr [2 x i8], ptr addrspace(3) %116, i32 %narrow42
  %narrow43 = mul nuw nsw i32 %9, 132
  %132 = getelementptr [2 x i8], ptr addrspace(3) @__shared_memory___0, i32 %109
  %133 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow43
  %134 = getelementptr [2 x i8], ptr addrspace(3) @__shared_memory___0, i32 %109
  %135 = getelementptr i8, ptr addrspace(3) %134, i32 32
  %136 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow43
  %137 = getelementptr [2 x i8], ptr addrspace(3) @__shared_memory___0, i32 %109
  %138 = getelementptr i8, ptr addrspace(3) %137, i32 64
  %139 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow43
  %140 = getelementptr [2 x i8], ptr addrspace(3) @__shared_memory___0, i32 %109
  %141 = getelementptr i8, ptr addrspace(3) %140, i32 96
  %142 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow43
  %143 = mul nuw nsw i32 %9, 132
  %narrow44 = add nuw nsw i32 %143, 132
  %144 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow44
  %145 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow44
  %146 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow44
  %147 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow44
  %148 = mul nuw nsw i32 %9, 132
  %narrow45 = add nuw nsw i32 %148, 264
  %149 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow45
  %150 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow45
  %151 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow45
  %152 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow45
  %153 = mul nuw nsw i32 %9, 132
  %narrow46 = add nuw nsw i32 %153, 396
  %154 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow46
  %155 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow46
  %156 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow46
  %157 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow46
  %158 = mul nuw nsw i32 %9, 132
  %narrow47 = add nuw nsw i32 %158, 528
  %159 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow47
  %160 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow47
  %161 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow47
  %162 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow47
  %163 = mul nuw nsw i32 %9, 132
  %narrow48 = add nuw nsw i32 %163, 660
  %164 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow48
  %165 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow48
  %166 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow48
  %167 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow48
  %168 = mul nuw nsw i32 %9, 132
  %narrow49 = add nuw nsw i32 %168, 792
  %169 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow49
  %170 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow49
  %171 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow49
  %172 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow49
  %173 = mul nuw nsw i32 %9, 132
  %narrow50 = add nuw nsw i32 %173, 924
  %174 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow50
  %175 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow50
  %176 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow50
  %177 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow50
  %narrow51 = mul nuw nsw i32 %103, 132
  %178 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow51
  %179 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow51
  %180 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow51
  %181 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow51
  %182 = mul nuw nsw i32 %9, 132
  %narrow52 = add nuw nsw i32 %182, 2244
  %183 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow52
  %184 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow52
  %185 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow52
  %186 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow52
  %187 = mul nuw nsw i32 %9, 132
  %narrow53 = add nuw nsw i32 %187, 2376
  %188 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow53
  %189 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow53
  %190 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow53
  %191 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow53
  %192 = mul nuw nsw i32 %9, 132
  %narrow54 = add nuw nsw i32 %192, 2508
  %193 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow54
  %194 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow54
  %195 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow54
  %196 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow54
  %197 = mul nuw nsw i32 %9, 132
  %narrow55 = add nuw nsw i32 %197, 2640
  %198 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow55
  %199 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow55
  %200 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow55
  %201 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow55
  %202 = mul nuw nsw i32 %9, 132
  %narrow56 = add nuw nsw i32 %202, 2772
  %203 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow56
  %204 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow56
  %205 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow56
  %206 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow56
  %207 = mul nuw nsw i32 %9, 132
  %narrow57 = add nuw nsw i32 %207, 2904
  %208 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow57
  %209 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow57
  %210 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow57
  %211 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow57
  %212 = mul nuw nsw i32 %9, 132
  %narrow58 = add nuw nsw i32 %212, 3036
  %213 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow58
  %214 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow58
  %215 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow58
  %216 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow58
  %narrow59 = mul nuw nsw i32 %104, 132
  %217 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow59
  %218 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow59
  %219 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow59
  %220 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow59
  %221 = mul nuw nsw i32 %9, 132
  %narrow60 = add nuw nsw i32 %221, 4356
  %222 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow60
  %223 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow60
  %224 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow60
  %225 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow60
  %226 = mul nuw nsw i32 %9, 132
  %narrow61 = add nuw nsw i32 %226, 4488
  %227 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow61
  %228 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow61
  %229 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow61
  %230 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow61
  %231 = mul nuw nsw i32 %9, 132
  %narrow62 = add nuw nsw i32 %231, 4620
  %232 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow62
  %233 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow62
  %234 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow62
  %235 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow62
  %236 = mul nuw nsw i32 %9, 132
  %narrow63 = add nuw nsw i32 %236, 4752
  %237 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow63
  %238 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow63
  %239 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow63
  %240 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow63
  %241 = mul nuw nsw i32 %9, 132
  %narrow64 = add nuw nsw i32 %241, 4884
  %242 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow64
  %243 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow64
  %244 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow64
  %245 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow64
  %246 = mul nuw nsw i32 %9, 132
  %narrow65 = add nuw nsw i32 %246, 5016
  %247 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow65
  %248 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow65
  %249 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow65
  %250 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow65
  %251 = mul nuw nsw i32 %9, 132
  %narrow66 = add nuw nsw i32 %251, 5148
  %252 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow66
  %253 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow66
  %254 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow66
  %255 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow66
  %narrow67 = mul nuw nsw i32 %105, 132
  %256 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow67
  %257 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow67
  %258 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow67
  %259 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow67
  %260 = mul nuw nsw i32 %9, 132
  %narrow68 = add nuw nsw i32 %260, 6468
  %261 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow68
  %262 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow68
  %263 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow68
  %264 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow68
  %265 = mul nuw nsw i32 %9, 132
  %narrow69 = add nuw nsw i32 %265, 6600
  %266 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow69
  %267 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow69
  %268 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow69
  %269 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow69
  %270 = mul nuw nsw i32 %9, 132
  %narrow70 = add nuw nsw i32 %270, 6732
  %271 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow70
  %272 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow70
  %273 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow70
  %274 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow70
  %275 = mul nuw nsw i32 %9, 132
  %narrow71 = add nuw nsw i32 %275, 6864
  %276 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow71
  %277 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow71
  %278 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow71
  %279 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow71
  %280 = mul nuw nsw i32 %9, 132
  %narrow72 = add nuw nsw i32 %280, 6996
  %281 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow72
  %282 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow72
  %283 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow72
  %284 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow72
  %285 = mul nuw nsw i32 %9, 132
  %narrow73 = add nuw nsw i32 %285, 7128
  %286 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow73
  %287 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow73
  %288 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow73
  %289 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow73
  %290 = mul nuw nsw i32 %9, 132
  %narrow74 = add nuw nsw i32 %290, 7260
  %291 = getelementptr [2 x i8], ptr addrspace(3) %132, i32 %narrow74
  %292 = getelementptr [2 x i8], ptr addrspace(3) %135, i32 %narrow74
  %293 = getelementptr [2 x i8], ptr addrspace(3) %138, i32 %narrow74
  %294 = getelementptr [2 x i8], ptr addrspace(3) %141, i32 %narrow74
  br label %295

295:                                              ; preds = %3, %295
  %296 = phi <8 x float> [ zeroinitializer, %3 ], [ %739, %295 ]
  %297 = phi <8 x float> [ zeroinitializer, %3 ], [ %740, %295 ]
  %298 = phi <8 x float> [ zeroinitializer, %3 ], [ %741, %295 ]
  %299 = phi <8 x float> [ zeroinitializer, %3 ], [ %742, %295 ]
  %300 = phi <8 x float> [ zeroinitializer, %3 ], [ %743, %295 ]
  %301 = phi <8 x float> [ zeroinitializer, %3 ], [ %744, %295 ]
  %302 = phi <8 x float> [ zeroinitializer, %3 ], [ %745, %295 ]
  %303 = phi <8 x float> [ zeroinitializer, %3 ], [ %746, %295 ]
  %304 = phi <8 x float> [ zeroinitializer, %3 ], [ %747, %295 ]
  %305 = phi <8 x float> [ zeroinitializer, %3 ], [ %748, %295 ]
  %306 = phi <8 x float> [ zeroinitializer, %3 ], [ %749, %295 ]
  %307 = phi <8 x float> [ zeroinitializer, %3 ], [ %750, %295 ]
  %308 = phi <8 x float> [ zeroinitializer, %3 ], [ %751, %295 ]
  %309 = phi <8 x float> [ zeroinitializer, %3 ], [ %752, %295 ]
  %310 = phi <8 x float> [ zeroinitializer, %3 ], [ %753, %295 ]
  %311 = phi <8 x float> [ zeroinitializer, %3 ], [ %754, %295 ]
  %312 = phi i32 [ 0, %3 ], [ %313, %295 ]
  %313 = add nuw nsw i32 %312, 4
  %314 = shl nuw nsw i32 %313, 4
  %315 = getelementptr [2 x i8], ptr addrspace(7) %34, i32 %314
  %316 = load <8 x half>, ptr addrspace(7) %315, align 2
  %317 = getelementptr [2 x i8], ptr addrspace(7) %38, i32 %314
  %318 = load <8 x half>, ptr addrspace(7) %317, align 2
  %319 = getelementptr [2 x i8], ptr addrspace(7) %42, i32 %314
  %320 = load <8 x half>, ptr addrspace(7) %319, align 2
  %321 = getelementptr [2 x i8], ptr addrspace(7) %46, i32 %314
  %322 = load <8 x half>, ptr addrspace(7) %321, align 2
  %323 = getelementptr [2 x i8], ptr addrspace(7) %50, i32 %314
  %324 = load <8 x half>, ptr addrspace(7) %323, align 2
  %325 = getelementptr [2 x i8], ptr addrspace(7) %54, i32 %314
  %326 = load <8 x half>, ptr addrspace(7) %325, align 2
  %327 = getelementptr [2 x i8], ptr addrspace(7) %58, i32 %314
  %328 = load <8 x half>, ptr addrspace(7) %327, align 2
  %329 = or disjoint i32 %314, %21
  %330 = getelementptr [2 x i8], ptr addrspace(7) %64, i32 %314
  %331 = load <8 x half>, ptr addrspace(7) %330, align 2
  %332 = or disjoint i32 %314, %22
  %.idx28 = shl nuw nsw i32 %329, 12
  %gep = getelementptr i8, ptr addrspace(7) %invariant.gep, i32 %.idx28
  %333 = load <8 x half>, ptr addrspace(7) %gep, align 2
  %334 = or disjoint i32 %314, %23
  %.idx29 = shl nuw nsw i32 %332, 12
  %gep91 = getelementptr i8, ptr addrspace(7) %invariant.gep, i32 %.idx29
  %335 = load <8 x half>, ptr addrspace(7) %gep91, align 2
  %336 = or disjoint i32 %314, %24
  %.idx30 = shl nuw nsw i32 %334, 12
  %gep93 = getelementptr i8, ptr addrspace(7) %invariant.gep, i32 %.idx30
  %337 = load <8 x half>, ptr addrspace(7) %gep93, align 2
  %.idx31 = shl nuw nsw i32 %336, 12
  %gep95 = getelementptr i8, ptr addrspace(7) %invariant.gep, i32 %.idx31
  %338 = load <8 x half>, ptr addrspace(7) %gep95, align 2
  fence syncscope("workgroup") release, !mmra !3
  tail call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  tail call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire, !mmra !3
  %339 = load <8 x half>, ptr addrspace(3) %111, align 8
  %340 = load <8 x half>, ptr addrspace(3) %113, align 8
  %341 = load <8 x half>, ptr addrspace(3) %115, align 8
  %342 = load <8 x half>, ptr addrspace(3) %117, align 8
  %343 = load <8 x half>, ptr addrspace(3) %119, align 8
  %344 = load <8 x half>, ptr addrspace(3) %120, align 8
  %345 = load <8 x half>, ptr addrspace(3) %121, align 8
  %346 = load <8 x half>, ptr addrspace(3) %122, align 8
  %347 = load <8 x half>, ptr addrspace(3) %124, align 8
  %348 = load <8 x half>, ptr addrspace(3) %125, align 8
  %349 = load <8 x half>, ptr addrspace(3) %126, align 8
  %350 = load <8 x half>, ptr addrspace(3) %127, align 8
  %351 = load <8 x half>, ptr addrspace(3) %128, align 8
  %352 = load <8 x half>, ptr addrspace(3) %129, align 8
  %353 = load <8 x half>, ptr addrspace(3) %130, align 8
  %354 = load <8 x half>, ptr addrspace(3) %131, align 8
  %355 = load <1 x half>, ptr addrspace(3) %133, align 2
  %356 = load <1 x half>, ptr addrspace(3) %136, align 2
  %357 = load <1 x half>, ptr addrspace(3) %139, align 2
  %358 = load <1 x half>, ptr addrspace(3) %142, align 2
  %359 = load <1 x half>, ptr addrspace(3) %144, align 2
  %360 = load <1 x half>, ptr addrspace(3) %145, align 2
  %361 = load <1 x half>, ptr addrspace(3) %146, align 2
  %362 = load <1 x half>, ptr addrspace(3) %147, align 2
  %363 = load <1 x half>, ptr addrspace(3) %149, align 2
  %364 = shufflevector <1 x half> %363, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %365 = load <1 x half>, ptr addrspace(3) %150, align 2
  %366 = shufflevector <1 x half> %365, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %367 = load <1 x half>, ptr addrspace(3) %151, align 2
  %368 = shufflevector <1 x half> %367, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %369 = load <1 x half>, ptr addrspace(3) %152, align 2
  %370 = shufflevector <1 x half> %369, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %371 = load <1 x half>, ptr addrspace(3) %154, align 2
  %372 = shufflevector <1 x half> %371, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %373 = load <1 x half>, ptr addrspace(3) %155, align 2
  %374 = shufflevector <1 x half> %373, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %375 = load <1 x half>, ptr addrspace(3) %156, align 2
  %376 = shufflevector <1 x half> %375, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %377 = load <1 x half>, ptr addrspace(3) %157, align 2
  %378 = shufflevector <1 x half> %377, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %379 = load <1 x half>, ptr addrspace(3) %159, align 2
  %380 = shufflevector <1 x half> %379, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %381 = load <1 x half>, ptr addrspace(3) %160, align 2
  %382 = shufflevector <1 x half> %381, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %383 = load <1 x half>, ptr addrspace(3) %161, align 2
  %384 = shufflevector <1 x half> %383, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %385 = load <1 x half>, ptr addrspace(3) %162, align 2
  %386 = shufflevector <1 x half> %385, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %387 = load <1 x half>, ptr addrspace(3) %164, align 2
  %388 = shufflevector <1 x half> %387, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %389 = load <1 x half>, ptr addrspace(3) %165, align 2
  %390 = shufflevector <1 x half> %389, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %391 = load <1 x half>, ptr addrspace(3) %166, align 2
  %392 = shufflevector <1 x half> %391, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %393 = load <1 x half>, ptr addrspace(3) %167, align 2
  %394 = shufflevector <1 x half> %393, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %395 = load <1 x half>, ptr addrspace(3) %169, align 2
  %396 = shufflevector <1 x half> %395, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %397 = load <1 x half>, ptr addrspace(3) %170, align 2
  %398 = shufflevector <1 x half> %397, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %399 = load <1 x half>, ptr addrspace(3) %171, align 2
  %400 = shufflevector <1 x half> %399, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %401 = load <1 x half>, ptr addrspace(3) %172, align 2
  %402 = shufflevector <1 x half> %401, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %403 = load <1 x half>, ptr addrspace(3) %174, align 2
  %404 = shufflevector <1 x half> %403, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %405 = load <1 x half>, ptr addrspace(3) %175, align 2
  %406 = shufflevector <1 x half> %405, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %407 = load <1 x half>, ptr addrspace(3) %176, align 2
  %408 = shufflevector <1 x half> %407, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %409 = load <1 x half>, ptr addrspace(3) %177, align 2
  %410 = shufflevector <1 x half> %409, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %411 = load <1 x half>, ptr addrspace(3) %178, align 2
  %412 = load <1 x half>, ptr addrspace(3) %179, align 2
  %413 = load <1 x half>, ptr addrspace(3) %180, align 2
  %414 = load <1 x half>, ptr addrspace(3) %181, align 2
  %415 = load <1 x half>, ptr addrspace(3) %183, align 2
  %416 = load <1 x half>, ptr addrspace(3) %184, align 2
  %417 = load <1 x half>, ptr addrspace(3) %185, align 2
  %418 = load <1 x half>, ptr addrspace(3) %186, align 2
  %419 = load <1 x half>, ptr addrspace(3) %188, align 2
  %420 = shufflevector <1 x half> %419, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %421 = load <1 x half>, ptr addrspace(3) %189, align 2
  %422 = shufflevector <1 x half> %421, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %423 = load <1 x half>, ptr addrspace(3) %190, align 2
  %424 = shufflevector <1 x half> %423, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %425 = load <1 x half>, ptr addrspace(3) %191, align 2
  %426 = shufflevector <1 x half> %425, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %427 = load <1 x half>, ptr addrspace(3) %193, align 2
  %428 = shufflevector <1 x half> %427, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %429 = load <1 x half>, ptr addrspace(3) %194, align 2
  %430 = shufflevector <1 x half> %429, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %431 = load <1 x half>, ptr addrspace(3) %195, align 2
  %432 = shufflevector <1 x half> %431, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %433 = load <1 x half>, ptr addrspace(3) %196, align 2
  %434 = shufflevector <1 x half> %433, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %435 = load <1 x half>, ptr addrspace(3) %198, align 2
  %436 = shufflevector <1 x half> %435, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %437 = load <1 x half>, ptr addrspace(3) %199, align 2
  %438 = shufflevector <1 x half> %437, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %439 = load <1 x half>, ptr addrspace(3) %200, align 2
  %440 = shufflevector <1 x half> %439, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %441 = load <1 x half>, ptr addrspace(3) %201, align 2
  %442 = shufflevector <1 x half> %441, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %443 = load <1 x half>, ptr addrspace(3) %203, align 2
  %444 = shufflevector <1 x half> %443, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %445 = load <1 x half>, ptr addrspace(3) %204, align 2
  %446 = shufflevector <1 x half> %445, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %447 = load <1 x half>, ptr addrspace(3) %205, align 2
  %448 = shufflevector <1 x half> %447, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %449 = load <1 x half>, ptr addrspace(3) %206, align 2
  %450 = shufflevector <1 x half> %449, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %451 = load <1 x half>, ptr addrspace(3) %208, align 2
  %452 = shufflevector <1 x half> %451, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %453 = load <1 x half>, ptr addrspace(3) %209, align 2
  %454 = shufflevector <1 x half> %453, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %455 = load <1 x half>, ptr addrspace(3) %210, align 2
  %456 = shufflevector <1 x half> %455, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %457 = load <1 x half>, ptr addrspace(3) %211, align 2
  %458 = shufflevector <1 x half> %457, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %459 = load <1 x half>, ptr addrspace(3) %213, align 2
  %460 = shufflevector <1 x half> %459, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %461 = load <1 x half>, ptr addrspace(3) %214, align 2
  %462 = shufflevector <1 x half> %461, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %463 = load <1 x half>, ptr addrspace(3) %215, align 2
  %464 = shufflevector <1 x half> %463, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %465 = load <1 x half>, ptr addrspace(3) %216, align 2
  %466 = shufflevector <1 x half> %465, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %467 = load <1 x half>, ptr addrspace(3) %217, align 2
  %468 = load <1 x half>, ptr addrspace(3) %218, align 2
  %469 = load <1 x half>, ptr addrspace(3) %219, align 2
  %470 = load <1 x half>, ptr addrspace(3) %220, align 2
  %471 = load <1 x half>, ptr addrspace(3) %222, align 2
  %472 = load <1 x half>, ptr addrspace(3) %223, align 2
  %473 = load <1 x half>, ptr addrspace(3) %224, align 2
  %474 = load <1 x half>, ptr addrspace(3) %225, align 2
  %475 = load <1 x half>, ptr addrspace(3) %227, align 2
  %476 = shufflevector <1 x half> %475, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %477 = load <1 x half>, ptr addrspace(3) %228, align 2
  %478 = shufflevector <1 x half> %477, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %479 = load <1 x half>, ptr addrspace(3) %229, align 2
  %480 = shufflevector <1 x half> %479, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %481 = load <1 x half>, ptr addrspace(3) %230, align 2
  %482 = shufflevector <1 x half> %481, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %483 = load <1 x half>, ptr addrspace(3) %232, align 2
  %484 = shufflevector <1 x half> %483, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %485 = load <1 x half>, ptr addrspace(3) %233, align 2
  %486 = shufflevector <1 x half> %485, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %487 = load <1 x half>, ptr addrspace(3) %234, align 2
  %488 = shufflevector <1 x half> %487, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %489 = load <1 x half>, ptr addrspace(3) %235, align 2
  %490 = shufflevector <1 x half> %489, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %491 = load <1 x half>, ptr addrspace(3) %237, align 2
  %492 = shufflevector <1 x half> %491, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %493 = load <1 x half>, ptr addrspace(3) %238, align 2
  %494 = shufflevector <1 x half> %493, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %495 = load <1 x half>, ptr addrspace(3) %239, align 2
  %496 = shufflevector <1 x half> %495, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %497 = load <1 x half>, ptr addrspace(3) %240, align 2
  %498 = shufflevector <1 x half> %497, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %499 = load <1 x half>, ptr addrspace(3) %242, align 2
  %500 = shufflevector <1 x half> %499, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %501 = load <1 x half>, ptr addrspace(3) %243, align 2
  %502 = shufflevector <1 x half> %501, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %503 = load <1 x half>, ptr addrspace(3) %244, align 2
  %504 = shufflevector <1 x half> %503, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %505 = load <1 x half>, ptr addrspace(3) %245, align 2
  %506 = shufflevector <1 x half> %505, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %507 = load <1 x half>, ptr addrspace(3) %247, align 2
  %508 = shufflevector <1 x half> %507, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %509 = load <1 x half>, ptr addrspace(3) %248, align 2
  %510 = shufflevector <1 x half> %509, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %511 = load <1 x half>, ptr addrspace(3) %249, align 2
  %512 = shufflevector <1 x half> %511, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %513 = load <1 x half>, ptr addrspace(3) %250, align 2
  %514 = shufflevector <1 x half> %513, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %515 = load <1 x half>, ptr addrspace(3) %252, align 2
  %516 = shufflevector <1 x half> %515, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %517 = load <1 x half>, ptr addrspace(3) %253, align 2
  %518 = shufflevector <1 x half> %517, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %519 = load <1 x half>, ptr addrspace(3) %254, align 2
  %520 = shufflevector <1 x half> %519, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %521 = load <1 x half>, ptr addrspace(3) %255, align 2
  %522 = shufflevector <1 x half> %521, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %523 = load <1 x half>, ptr addrspace(3) %256, align 2
  %524 = load <1 x half>, ptr addrspace(3) %257, align 2
  %525 = load <1 x half>, ptr addrspace(3) %258, align 2
  %526 = load <1 x half>, ptr addrspace(3) %259, align 2
  %527 = load <1 x half>, ptr addrspace(3) %261, align 2
  %528 = load <1 x half>, ptr addrspace(3) %262, align 2
  %529 = load <1 x half>, ptr addrspace(3) %263, align 2
  %530 = load <1 x half>, ptr addrspace(3) %264, align 2
  %531 = load <1 x half>, ptr addrspace(3) %266, align 2
  %532 = shufflevector <1 x half> %531, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %533 = load <1 x half>, ptr addrspace(3) %267, align 2
  %534 = shufflevector <1 x half> %533, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %535 = load <1 x half>, ptr addrspace(3) %268, align 2
  %536 = shufflevector <1 x half> %535, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %537 = load <1 x half>, ptr addrspace(3) %269, align 2
  %538 = shufflevector <1 x half> %537, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %539 = load <1 x half>, ptr addrspace(3) %271, align 2
  %540 = shufflevector <1 x half> %539, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %541 = load <1 x half>, ptr addrspace(3) %272, align 2
  %542 = shufflevector <1 x half> %541, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %543 = load <1 x half>, ptr addrspace(3) %273, align 2
  %544 = shufflevector <1 x half> %543, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %545 = load <1 x half>, ptr addrspace(3) %274, align 2
  %546 = shufflevector <1 x half> %545, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %547 = load <1 x half>, ptr addrspace(3) %276, align 2
  %548 = shufflevector <1 x half> %547, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %549 = load <1 x half>, ptr addrspace(3) %277, align 2
  %550 = shufflevector <1 x half> %549, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %551 = load <1 x half>, ptr addrspace(3) %278, align 2
  %552 = shufflevector <1 x half> %551, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %553 = load <1 x half>, ptr addrspace(3) %279, align 2
  %554 = shufflevector <1 x half> %553, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %555 = load <1 x half>, ptr addrspace(3) %281, align 2
  %556 = shufflevector <1 x half> %555, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %557 = load <1 x half>, ptr addrspace(3) %282, align 2
  %558 = shufflevector <1 x half> %557, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %559 = load <1 x half>, ptr addrspace(3) %283, align 2
  %560 = shufflevector <1 x half> %559, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %561 = load <1 x half>, ptr addrspace(3) %284, align 2
  %562 = shufflevector <1 x half> %561, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %563 = load <1 x half>, ptr addrspace(3) %286, align 2
  %564 = shufflevector <1 x half> %563, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %565 = load <1 x half>, ptr addrspace(3) %287, align 2
  %566 = shufflevector <1 x half> %565, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %567 = load <1 x half>, ptr addrspace(3) %288, align 2
  %568 = shufflevector <1 x half> %567, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %569 = load <1 x half>, ptr addrspace(3) %289, align 2
  %570 = shufflevector <1 x half> %569, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %571 = load <1 x half>, ptr addrspace(3) %291, align 2
  %572 = shufflevector <1 x half> %571, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %573 = load <1 x half>, ptr addrspace(3) %292, align 2
  %574 = shufflevector <1 x half> %573, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %575 = load <1 x half>, ptr addrspace(3) %293, align 2
  %576 = shufflevector <1 x half> %575, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %577 = load <1 x half>, ptr addrspace(3) %294, align 2
  %578 = shufflevector <1 x half> %577, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %579 = shufflevector <1 x half> %355, <1 x half> %359, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %580 = shufflevector <8 x half> %579, <8 x half> %364, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %581 = shufflevector <8 x half> %580, <8 x half> %372, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %582 = shufflevector <8 x half> %581, <8 x half> %380, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %583 = shufflevector <8 x half> %582, <8 x half> %388, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %584 = shufflevector <8 x half> %583, <8 x half> %396, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %585 = shufflevector <8 x half> %584, <8 x half> %404, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %586 = shufflevector <1 x half> %356, <1 x half> %360, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %587 = shufflevector <8 x half> %586, <8 x half> %366, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %588 = shufflevector <8 x half> %587, <8 x half> %374, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %589 = shufflevector <8 x half> %588, <8 x half> %382, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %590 = shufflevector <8 x half> %589, <8 x half> %390, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %591 = shufflevector <8 x half> %590, <8 x half> %398, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %592 = shufflevector <8 x half> %591, <8 x half> %406, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %593 = shufflevector <1 x half> %357, <1 x half> %361, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %594 = shufflevector <8 x half> %593, <8 x half> %368, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %595 = shufflevector <8 x half> %594, <8 x half> %376, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %596 = shufflevector <8 x half> %595, <8 x half> %384, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %597 = shufflevector <8 x half> %596, <8 x half> %392, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %598 = shufflevector <8 x half> %597, <8 x half> %400, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %599 = shufflevector <8 x half> %598, <8 x half> %408, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %600 = shufflevector <1 x half> %358, <1 x half> %362, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %601 = shufflevector <8 x half> %600, <8 x half> %370, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %602 = shufflevector <8 x half> %601, <8 x half> %378, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %603 = shufflevector <8 x half> %602, <8 x half> %386, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %604 = shufflevector <8 x half> %603, <8 x half> %394, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %605 = shufflevector <8 x half> %604, <8 x half> %402, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %606 = shufflevector <8 x half> %605, <8 x half> %410, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %607 = shufflevector <1 x half> %411, <1 x half> %415, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %608 = shufflevector <8 x half> %607, <8 x half> %420, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %609 = shufflevector <8 x half> %608, <8 x half> %428, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %610 = shufflevector <8 x half> %609, <8 x half> %436, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %611 = shufflevector <8 x half> %610, <8 x half> %444, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %612 = shufflevector <8 x half> %611, <8 x half> %452, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %613 = shufflevector <8 x half> %612, <8 x half> %460, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %614 = shufflevector <1 x half> %412, <1 x half> %416, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %615 = shufflevector <8 x half> %614, <8 x half> %422, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %616 = shufflevector <8 x half> %615, <8 x half> %430, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %617 = shufflevector <8 x half> %616, <8 x half> %438, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %618 = shufflevector <8 x half> %617, <8 x half> %446, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %619 = shufflevector <8 x half> %618, <8 x half> %454, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %620 = shufflevector <8 x half> %619, <8 x half> %462, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %621 = shufflevector <1 x half> %413, <1 x half> %417, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %622 = shufflevector <8 x half> %621, <8 x half> %424, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %623 = shufflevector <8 x half> %622, <8 x half> %432, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %624 = shufflevector <8 x half> %623, <8 x half> %440, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %625 = shufflevector <8 x half> %624, <8 x half> %448, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %626 = shufflevector <8 x half> %625, <8 x half> %456, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %627 = shufflevector <8 x half> %626, <8 x half> %464, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %628 = shufflevector <1 x half> %414, <1 x half> %418, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %629 = shufflevector <8 x half> %628, <8 x half> %426, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %630 = shufflevector <8 x half> %629, <8 x half> %434, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %631 = shufflevector <8 x half> %630, <8 x half> %442, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %632 = shufflevector <8 x half> %631, <8 x half> %450, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %633 = shufflevector <8 x half> %632, <8 x half> %458, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %634 = shufflevector <8 x half> %633, <8 x half> %466, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %635 = shufflevector <1 x half> %467, <1 x half> %471, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %636 = shufflevector <8 x half> %635, <8 x half> %476, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %637 = shufflevector <8 x half> %636, <8 x half> %484, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %638 = shufflevector <8 x half> %637, <8 x half> %492, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %639 = shufflevector <8 x half> %638, <8 x half> %500, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %640 = shufflevector <8 x half> %639, <8 x half> %508, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %641 = shufflevector <8 x half> %640, <8 x half> %516, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %642 = shufflevector <1 x half> %468, <1 x half> %472, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %643 = shufflevector <8 x half> %642, <8 x half> %478, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %644 = shufflevector <8 x half> %643, <8 x half> %486, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %645 = shufflevector <8 x half> %644, <8 x half> %494, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %646 = shufflevector <8 x half> %645, <8 x half> %502, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %647 = shufflevector <8 x half> %646, <8 x half> %510, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %648 = shufflevector <8 x half> %647, <8 x half> %518, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %649 = shufflevector <1 x half> %469, <1 x half> %473, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %650 = shufflevector <8 x half> %649, <8 x half> %480, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %651 = shufflevector <8 x half> %650, <8 x half> %488, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %652 = shufflevector <8 x half> %651, <8 x half> %496, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %653 = shufflevector <8 x half> %652, <8 x half> %504, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %654 = shufflevector <8 x half> %653, <8 x half> %512, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %655 = shufflevector <8 x half> %654, <8 x half> %520, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %656 = shufflevector <1 x half> %470, <1 x half> %474, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %657 = shufflevector <8 x half> %656, <8 x half> %482, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %658 = shufflevector <8 x half> %657, <8 x half> %490, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %659 = shufflevector <8 x half> %658, <8 x half> %498, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %660 = shufflevector <8 x half> %659, <8 x half> %506, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %661 = shufflevector <8 x half> %660, <8 x half> %514, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %662 = shufflevector <8 x half> %661, <8 x half> %522, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %663 = shufflevector <1 x half> %523, <1 x half> %527, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %664 = shufflevector <8 x half> %663, <8 x half> %532, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %665 = shufflevector <8 x half> %664, <8 x half> %540, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %666 = shufflevector <8 x half> %665, <8 x half> %548, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %667 = shufflevector <8 x half> %666, <8 x half> %556, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %668 = shufflevector <8 x half> %667, <8 x half> %564, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %669 = shufflevector <8 x half> %668, <8 x half> %572, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %670 = shufflevector <1 x half> %524, <1 x half> %528, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %671 = shufflevector <8 x half> %670, <8 x half> %534, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %672 = shufflevector <8 x half> %671, <8 x half> %542, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %673 = shufflevector <8 x half> %672, <8 x half> %550, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %674 = shufflevector <8 x half> %673, <8 x half> %558, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %675 = shufflevector <8 x half> %674, <8 x half> %566, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %676 = shufflevector <8 x half> %675, <8 x half> %574, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %677 = shufflevector <1 x half> %525, <1 x half> %529, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %678 = shufflevector <8 x half> %677, <8 x half> %536, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %679 = shufflevector <8 x half> %678, <8 x half> %544, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %680 = shufflevector <8 x half> %679, <8 x half> %552, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %681 = shufflevector <8 x half> %680, <8 x half> %560, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %682 = shufflevector <8 x half> %681, <8 x half> %568, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %683 = shufflevector <8 x half> %682, <8 x half> %576, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %684 = shufflevector <1 x half> %526, <1 x half> %530, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %685 = shufflevector <8 x half> %684, <8 x half> %538, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %686 = shufflevector <8 x half> %685, <8 x half> %546, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %687 = shufflevector <8 x half> %686, <8 x half> %554, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %688 = shufflevector <8 x half> %687, <8 x half> %562, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %689 = shufflevector <8 x half> %688, <8 x half> %570, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %690 = shufflevector <8 x half> %689, <8 x half> %578, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %691 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %339, <8 x half> %585, <8 x float> %296)
  %692 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %339, <8 x half> %592, <8 x float> %297)
  %693 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %339, <8 x half> %599, <8 x float> %298)
  %694 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %339, <8 x half> %606, <8 x float> %299)
  %695 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %343, <8 x half> %585, <8 x float> %300)
  %696 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %343, <8 x half> %592, <8 x float> %301)
  %697 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %343, <8 x half> %599, <8 x float> %302)
  %698 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %343, <8 x half> %606, <8 x float> %303)
  %699 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %347, <8 x half> %585, <8 x float> %304)
  %700 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %347, <8 x half> %592, <8 x float> %305)
  %701 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %347, <8 x half> %599, <8 x float> %306)
  %702 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %347, <8 x half> %606, <8 x float> %307)
  %703 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %351, <8 x half> %585, <8 x float> %308)
  %704 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %351, <8 x half> %592, <8 x float> %309)
  %705 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %351, <8 x half> %599, <8 x float> %310)
  %706 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %351, <8 x half> %606, <8 x float> %311)
  %707 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %340, <8 x half> %613, <8 x float> %691)
  %708 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %340, <8 x half> %620, <8 x float> %692)
  %709 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %340, <8 x half> %627, <8 x float> %693)
  %710 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %340, <8 x half> %634, <8 x float> %694)
  %711 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %344, <8 x half> %613, <8 x float> %695)
  %712 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %344, <8 x half> %620, <8 x float> %696)
  %713 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %344, <8 x half> %627, <8 x float> %697)
  %714 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %344, <8 x half> %634, <8 x float> %698)
  %715 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %348, <8 x half> %613, <8 x float> %699)
  %716 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %348, <8 x half> %620, <8 x float> %700)
  %717 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %348, <8 x half> %627, <8 x float> %701)
  %718 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %348, <8 x half> %634, <8 x float> %702)
  %719 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %352, <8 x half> %613, <8 x float> %703)
  %720 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %352, <8 x half> %620, <8 x float> %704)
  %721 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %352, <8 x half> %627, <8 x float> %705)
  %722 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %352, <8 x half> %634, <8 x float> %706)
  %723 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %341, <8 x half> %641, <8 x float> %707)
  %724 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %341, <8 x half> %648, <8 x float> %708)
  %725 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %341, <8 x half> %655, <8 x float> %709)
  %726 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %341, <8 x half> %662, <8 x float> %710)
  %727 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %345, <8 x half> %641, <8 x float> %711)
  %728 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %345, <8 x half> %648, <8 x float> %712)
  %729 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %345, <8 x half> %655, <8 x float> %713)
  %730 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %345, <8 x half> %662, <8 x float> %714)
  %731 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %349, <8 x half> %641, <8 x float> %715)
  %732 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %349, <8 x half> %648, <8 x float> %716)
  %733 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %349, <8 x half> %655, <8 x float> %717)
  %734 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %349, <8 x half> %662, <8 x float> %718)
  %735 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %353, <8 x half> %641, <8 x float> %719)
  %736 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %353, <8 x half> %648, <8 x float> %720)
  %737 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %353, <8 x half> %655, <8 x float> %721)
  %738 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %353, <8 x half> %662, <8 x float> %722)
  %739 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %342, <8 x half> %669, <8 x float> %723)
  %740 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %342, <8 x half> %676, <8 x float> %724)
  %741 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %342, <8 x half> %683, <8 x float> %725)
  %742 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %342, <8 x half> %690, <8 x float> %726)
  %743 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %346, <8 x half> %669, <8 x float> %727)
  %744 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %346, <8 x half> %676, <8 x float> %728)
  %745 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %346, <8 x half> %683, <8 x float> %729)
  %746 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %346, <8 x half> %690, <8 x float> %730)
  %747 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %350, <8 x half> %669, <8 x float> %731)
  %748 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %350, <8 x half> %676, <8 x float> %732)
  %749 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %350, <8 x half> %683, <8 x float> %733)
  %750 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %350, <8 x half> %690, <8 x float> %734)
  %751 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %354, <8 x half> %669, <8 x float> %735)
  %752 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %354, <8 x half> %676, <8 x float> %736)
  %753 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %354, <8 x half> %683, <8 x float> %737)
  %754 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %354, <8 x half> %690, <8 x float> %738)
  fence syncscope("workgroup") release, !mmra !3
  tail call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  tail call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire, !mmra !3
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  store <8 x half> %316, ptr addrspace(3) %79, align 8
  store <8 x half> %318, ptr addrspace(3) %81, align 8
  store <8 x half> %320, ptr addrspace(3) %83, align 8
  store <8 x half> %322, ptr addrspace(3) %85, align 8
  store <8 x half> %324, ptr addrspace(3) %87, align 8
  store <8 x half> %326, ptr addrspace(3) %89, align 8
  store <8 x half> %328, ptr addrspace(3) %91, align 8
  store <8 x half> %331, ptr addrspace(3) %93, align 8
  store <8 x half> %333, ptr addrspace(3) %95, align 8
  store <8 x half> %335, ptr addrspace(3) %97, align 8
  store <8 x half> %337, ptr addrspace(3) %99, align 8
  store <8 x half> %338, ptr addrspace(3) %101, align 8
  %755 = icmp samesign ult i32 %312, 120
  br i1 %755, label %295, label %756

756:                                              ; preds = %295
  %757 = tail call ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) nonnull %2, i16 0, i64 16777216, i32 822243328)
  fence syncscope("workgroup") release, !mmra !3
  tail call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  tail call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire, !mmra !3
  %758 = load <8 x half>, ptr addrspace(3) %111, align 8
  %759 = load <8 x half>, ptr addrspace(3) %113, align 8
  %760 = load <8 x half>, ptr addrspace(3) %115, align 8
  %761 = load <8 x half>, ptr addrspace(3) %117, align 8
  %762 = load <8 x half>, ptr addrspace(3) %119, align 8
  %763 = load <8 x half>, ptr addrspace(3) %120, align 8
  %764 = load <8 x half>, ptr addrspace(3) %121, align 8
  %765 = load <8 x half>, ptr addrspace(3) %122, align 8
  %766 = load <8 x half>, ptr addrspace(3) %124, align 8
  %767 = load <8 x half>, ptr addrspace(3) %125, align 8
  %768 = load <8 x half>, ptr addrspace(3) %126, align 8
  %769 = load <8 x half>, ptr addrspace(3) %127, align 8
  %770 = load <8 x half>, ptr addrspace(3) %128, align 8
  %771 = load <8 x half>, ptr addrspace(3) %129, align 8
  %772 = load <8 x half>, ptr addrspace(3) %130, align 8
  %773 = load <8 x half>, ptr addrspace(3) %131, align 8
  %774 = load <1 x half>, ptr addrspace(3) %133, align 2
  %775 = load <1 x half>, ptr addrspace(3) %136, align 2
  %776 = load <1 x half>, ptr addrspace(3) %139, align 2
  %777 = load <1 x half>, ptr addrspace(3) %142, align 2
  %778 = load <1 x half>, ptr addrspace(3) %144, align 2
  %779 = load <1 x half>, ptr addrspace(3) %145, align 2
  %780 = load <1 x half>, ptr addrspace(3) %146, align 2
  %781 = load <1 x half>, ptr addrspace(3) %147, align 2
  %782 = load <1 x half>, ptr addrspace(3) %149, align 2
  %783 = shufflevector <1 x half> %782, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %784 = load <1 x half>, ptr addrspace(3) %150, align 2
  %785 = shufflevector <1 x half> %784, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %786 = load <1 x half>, ptr addrspace(3) %151, align 2
  %787 = shufflevector <1 x half> %786, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %788 = load <1 x half>, ptr addrspace(3) %152, align 2
  %789 = shufflevector <1 x half> %788, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %790 = load <1 x half>, ptr addrspace(3) %154, align 2
  %791 = shufflevector <1 x half> %790, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %792 = load <1 x half>, ptr addrspace(3) %155, align 2
  %793 = shufflevector <1 x half> %792, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %794 = load <1 x half>, ptr addrspace(3) %156, align 2
  %795 = shufflevector <1 x half> %794, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %796 = load <1 x half>, ptr addrspace(3) %157, align 2
  %797 = shufflevector <1 x half> %796, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %798 = load <1 x half>, ptr addrspace(3) %159, align 2
  %799 = shufflevector <1 x half> %798, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %800 = load <1 x half>, ptr addrspace(3) %160, align 2
  %801 = shufflevector <1 x half> %800, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %802 = load <1 x half>, ptr addrspace(3) %161, align 2
  %803 = shufflevector <1 x half> %802, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %804 = load <1 x half>, ptr addrspace(3) %162, align 2
  %805 = shufflevector <1 x half> %804, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %806 = load <1 x half>, ptr addrspace(3) %164, align 2
  %807 = shufflevector <1 x half> %806, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %808 = load <1 x half>, ptr addrspace(3) %165, align 2
  %809 = shufflevector <1 x half> %808, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %810 = load <1 x half>, ptr addrspace(3) %166, align 2
  %811 = shufflevector <1 x half> %810, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %812 = load <1 x half>, ptr addrspace(3) %167, align 2
  %813 = shufflevector <1 x half> %812, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %814 = load <1 x half>, ptr addrspace(3) %169, align 2
  %815 = shufflevector <1 x half> %814, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %816 = load <1 x half>, ptr addrspace(3) %170, align 2
  %817 = shufflevector <1 x half> %816, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %818 = load <1 x half>, ptr addrspace(3) %171, align 2
  %819 = shufflevector <1 x half> %818, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %820 = load <1 x half>, ptr addrspace(3) %172, align 2
  %821 = shufflevector <1 x half> %820, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %822 = load <1 x half>, ptr addrspace(3) %174, align 2
  %823 = shufflevector <1 x half> %822, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %824 = load <1 x half>, ptr addrspace(3) %175, align 2
  %825 = shufflevector <1 x half> %824, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %826 = load <1 x half>, ptr addrspace(3) %176, align 2
  %827 = shufflevector <1 x half> %826, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %828 = load <1 x half>, ptr addrspace(3) %177, align 2
  %829 = shufflevector <1 x half> %828, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %830 = load <1 x half>, ptr addrspace(3) %178, align 2
  %831 = load <1 x half>, ptr addrspace(3) %179, align 2
  %832 = load <1 x half>, ptr addrspace(3) %180, align 2
  %833 = load <1 x half>, ptr addrspace(3) %181, align 2
  %834 = load <1 x half>, ptr addrspace(3) %183, align 2
  %835 = load <1 x half>, ptr addrspace(3) %184, align 2
  %836 = load <1 x half>, ptr addrspace(3) %185, align 2
  %837 = load <1 x half>, ptr addrspace(3) %186, align 2
  %838 = load <1 x half>, ptr addrspace(3) %188, align 2
  %839 = shufflevector <1 x half> %838, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %840 = load <1 x half>, ptr addrspace(3) %189, align 2
  %841 = shufflevector <1 x half> %840, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %842 = load <1 x half>, ptr addrspace(3) %190, align 2
  %843 = shufflevector <1 x half> %842, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %844 = load <1 x half>, ptr addrspace(3) %191, align 2
  %845 = shufflevector <1 x half> %844, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %846 = load <1 x half>, ptr addrspace(3) %193, align 2
  %847 = shufflevector <1 x half> %846, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %848 = load <1 x half>, ptr addrspace(3) %194, align 2
  %849 = shufflevector <1 x half> %848, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %850 = load <1 x half>, ptr addrspace(3) %195, align 2
  %851 = shufflevector <1 x half> %850, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %852 = load <1 x half>, ptr addrspace(3) %196, align 2
  %853 = shufflevector <1 x half> %852, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %854 = load <1 x half>, ptr addrspace(3) %198, align 2
  %855 = shufflevector <1 x half> %854, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %856 = load <1 x half>, ptr addrspace(3) %199, align 2
  %857 = shufflevector <1 x half> %856, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %858 = load <1 x half>, ptr addrspace(3) %200, align 2
  %859 = shufflevector <1 x half> %858, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %860 = load <1 x half>, ptr addrspace(3) %201, align 2
  %861 = shufflevector <1 x half> %860, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %862 = load <1 x half>, ptr addrspace(3) %203, align 2
  %863 = shufflevector <1 x half> %862, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %864 = load <1 x half>, ptr addrspace(3) %204, align 2
  %865 = shufflevector <1 x half> %864, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %866 = load <1 x half>, ptr addrspace(3) %205, align 2
  %867 = shufflevector <1 x half> %866, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %868 = load <1 x half>, ptr addrspace(3) %206, align 2
  %869 = shufflevector <1 x half> %868, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %870 = load <1 x half>, ptr addrspace(3) %208, align 2
  %871 = shufflevector <1 x half> %870, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %872 = load <1 x half>, ptr addrspace(3) %209, align 2
  %873 = shufflevector <1 x half> %872, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %874 = load <1 x half>, ptr addrspace(3) %210, align 2
  %875 = shufflevector <1 x half> %874, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %876 = load <1 x half>, ptr addrspace(3) %211, align 2
  %877 = shufflevector <1 x half> %876, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %878 = load <1 x half>, ptr addrspace(3) %213, align 2
  %879 = shufflevector <1 x half> %878, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %880 = load <1 x half>, ptr addrspace(3) %214, align 2
  %881 = shufflevector <1 x half> %880, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %882 = load <1 x half>, ptr addrspace(3) %215, align 2
  %883 = shufflevector <1 x half> %882, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %884 = load <1 x half>, ptr addrspace(3) %216, align 2
  %885 = shufflevector <1 x half> %884, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %886 = load <1 x half>, ptr addrspace(3) %217, align 2
  %887 = load <1 x half>, ptr addrspace(3) %218, align 2
  %888 = load <1 x half>, ptr addrspace(3) %219, align 2
  %889 = load <1 x half>, ptr addrspace(3) %220, align 2
  %890 = load <1 x half>, ptr addrspace(3) %222, align 2
  %891 = load <1 x half>, ptr addrspace(3) %223, align 2
  %892 = load <1 x half>, ptr addrspace(3) %224, align 2
  %893 = load <1 x half>, ptr addrspace(3) %225, align 2
  %894 = load <1 x half>, ptr addrspace(3) %227, align 2
  %895 = shufflevector <1 x half> %894, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %896 = load <1 x half>, ptr addrspace(3) %228, align 2
  %897 = shufflevector <1 x half> %896, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %898 = load <1 x half>, ptr addrspace(3) %229, align 2
  %899 = shufflevector <1 x half> %898, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %900 = load <1 x half>, ptr addrspace(3) %230, align 2
  %901 = shufflevector <1 x half> %900, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %902 = load <1 x half>, ptr addrspace(3) %232, align 2
  %903 = shufflevector <1 x half> %902, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %904 = load <1 x half>, ptr addrspace(3) %233, align 2
  %905 = shufflevector <1 x half> %904, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %906 = load <1 x half>, ptr addrspace(3) %234, align 2
  %907 = shufflevector <1 x half> %906, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %908 = load <1 x half>, ptr addrspace(3) %235, align 2
  %909 = shufflevector <1 x half> %908, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %910 = load <1 x half>, ptr addrspace(3) %237, align 2
  %911 = shufflevector <1 x half> %910, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %912 = load <1 x half>, ptr addrspace(3) %238, align 2
  %913 = shufflevector <1 x half> %912, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %914 = load <1 x half>, ptr addrspace(3) %239, align 2
  %915 = shufflevector <1 x half> %914, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %916 = load <1 x half>, ptr addrspace(3) %240, align 2
  %917 = shufflevector <1 x half> %916, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %918 = load <1 x half>, ptr addrspace(3) %242, align 2
  %919 = shufflevector <1 x half> %918, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %920 = load <1 x half>, ptr addrspace(3) %243, align 2
  %921 = shufflevector <1 x half> %920, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %922 = load <1 x half>, ptr addrspace(3) %244, align 2
  %923 = shufflevector <1 x half> %922, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %924 = load <1 x half>, ptr addrspace(3) %245, align 2
  %925 = shufflevector <1 x half> %924, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %926 = load <1 x half>, ptr addrspace(3) %247, align 2
  %927 = shufflevector <1 x half> %926, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %928 = load <1 x half>, ptr addrspace(3) %248, align 2
  %929 = shufflevector <1 x half> %928, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %930 = load <1 x half>, ptr addrspace(3) %249, align 2
  %931 = shufflevector <1 x half> %930, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %932 = load <1 x half>, ptr addrspace(3) %250, align 2
  %933 = shufflevector <1 x half> %932, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %934 = load <1 x half>, ptr addrspace(3) %252, align 2
  %935 = shufflevector <1 x half> %934, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %936 = load <1 x half>, ptr addrspace(3) %253, align 2
  %937 = shufflevector <1 x half> %936, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %938 = load <1 x half>, ptr addrspace(3) %254, align 2
  %939 = shufflevector <1 x half> %938, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %940 = load <1 x half>, ptr addrspace(3) %255, align 2
  %941 = shufflevector <1 x half> %940, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %942 = load <1 x half>, ptr addrspace(3) %256, align 2
  %943 = load <1 x half>, ptr addrspace(3) %257, align 2
  %944 = load <1 x half>, ptr addrspace(3) %258, align 2
  %945 = load <1 x half>, ptr addrspace(3) %259, align 2
  %946 = load <1 x half>, ptr addrspace(3) %261, align 2
  %947 = load <1 x half>, ptr addrspace(3) %262, align 2
  %948 = load <1 x half>, ptr addrspace(3) %263, align 2
  %949 = load <1 x half>, ptr addrspace(3) %264, align 2
  %950 = load <1 x half>, ptr addrspace(3) %266, align 2
  %951 = shufflevector <1 x half> %950, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %952 = load <1 x half>, ptr addrspace(3) %267, align 2
  %953 = shufflevector <1 x half> %952, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %954 = load <1 x half>, ptr addrspace(3) %268, align 2
  %955 = shufflevector <1 x half> %954, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %956 = load <1 x half>, ptr addrspace(3) %269, align 2
  %957 = shufflevector <1 x half> %956, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %958 = load <1 x half>, ptr addrspace(3) %271, align 2
  %959 = shufflevector <1 x half> %958, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %960 = load <1 x half>, ptr addrspace(3) %272, align 2
  %961 = shufflevector <1 x half> %960, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %962 = load <1 x half>, ptr addrspace(3) %273, align 2
  %963 = shufflevector <1 x half> %962, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %964 = load <1 x half>, ptr addrspace(3) %274, align 2
  %965 = shufflevector <1 x half> %964, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %966 = load <1 x half>, ptr addrspace(3) %276, align 2
  %967 = shufflevector <1 x half> %966, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %968 = load <1 x half>, ptr addrspace(3) %277, align 2
  %969 = shufflevector <1 x half> %968, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %970 = load <1 x half>, ptr addrspace(3) %278, align 2
  %971 = shufflevector <1 x half> %970, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %972 = load <1 x half>, ptr addrspace(3) %279, align 2
  %973 = shufflevector <1 x half> %972, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %974 = load <1 x half>, ptr addrspace(3) %281, align 2
  %975 = shufflevector <1 x half> %974, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %976 = load <1 x half>, ptr addrspace(3) %282, align 2
  %977 = shufflevector <1 x half> %976, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %978 = load <1 x half>, ptr addrspace(3) %283, align 2
  %979 = shufflevector <1 x half> %978, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %980 = load <1 x half>, ptr addrspace(3) %284, align 2
  %981 = shufflevector <1 x half> %980, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %982 = load <1 x half>, ptr addrspace(3) %286, align 2
  %983 = shufflevector <1 x half> %982, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %984 = load <1 x half>, ptr addrspace(3) %287, align 2
  %985 = shufflevector <1 x half> %984, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %986 = load <1 x half>, ptr addrspace(3) %288, align 2
  %987 = shufflevector <1 x half> %986, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %988 = load <1 x half>, ptr addrspace(3) %289, align 2
  %989 = shufflevector <1 x half> %988, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %990 = load <1 x half>, ptr addrspace(3) %291, align 2
  %991 = shufflevector <1 x half> %990, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %992 = load <1 x half>, ptr addrspace(3) %292, align 2
  %993 = shufflevector <1 x half> %992, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %994 = load <1 x half>, ptr addrspace(3) %293, align 2
  %995 = shufflevector <1 x half> %994, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %996 = load <1 x half>, ptr addrspace(3) %294, align 2
  %997 = shufflevector <1 x half> %996, <1 x half> poison, <8 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %998 = shufflevector <1 x half> %774, <1 x half> %778, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %999 = shufflevector <8 x half> %998, <8 x half> %783, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1000 = shufflevector <8 x half> %999, <8 x half> %791, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1001 = shufflevector <8 x half> %1000, <8 x half> %799, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1002 = shufflevector <8 x half> %1001, <8 x half> %807, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1003 = shufflevector <8 x half> %1002, <8 x half> %815, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1004 = shufflevector <8 x half> %1003, <8 x half> %823, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1005 = shufflevector <1 x half> %775, <1 x half> %779, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1006 = shufflevector <8 x half> %1005, <8 x half> %785, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1007 = shufflevector <8 x half> %1006, <8 x half> %793, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1008 = shufflevector <8 x half> %1007, <8 x half> %801, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1009 = shufflevector <8 x half> %1008, <8 x half> %809, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1010 = shufflevector <8 x half> %1009, <8 x half> %817, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1011 = shufflevector <8 x half> %1010, <8 x half> %825, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1012 = shufflevector <1 x half> %776, <1 x half> %780, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1013 = shufflevector <8 x half> %1012, <8 x half> %787, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1014 = shufflevector <8 x half> %1013, <8 x half> %795, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1015 = shufflevector <8 x half> %1014, <8 x half> %803, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1016 = shufflevector <8 x half> %1015, <8 x half> %811, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1017 = shufflevector <8 x half> %1016, <8 x half> %819, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1018 = shufflevector <8 x half> %1017, <8 x half> %827, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1019 = shufflevector <1 x half> %777, <1 x half> %781, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1020 = shufflevector <8 x half> %1019, <8 x half> %789, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1021 = shufflevector <8 x half> %1020, <8 x half> %797, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1022 = shufflevector <8 x half> %1021, <8 x half> %805, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1023 = shufflevector <8 x half> %1022, <8 x half> %813, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1024 = shufflevector <8 x half> %1023, <8 x half> %821, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1025 = shufflevector <8 x half> %1024, <8 x half> %829, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1026 = shufflevector <1 x half> %830, <1 x half> %834, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1027 = shufflevector <8 x half> %1026, <8 x half> %839, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1028 = shufflevector <8 x half> %1027, <8 x half> %847, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1029 = shufflevector <8 x half> %1028, <8 x half> %855, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1030 = shufflevector <8 x half> %1029, <8 x half> %863, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1031 = shufflevector <8 x half> %1030, <8 x half> %871, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1032 = shufflevector <8 x half> %1031, <8 x half> %879, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1033 = shufflevector <1 x half> %831, <1 x half> %835, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1034 = shufflevector <8 x half> %1033, <8 x half> %841, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1035 = shufflevector <8 x half> %1034, <8 x half> %849, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1036 = shufflevector <8 x half> %1035, <8 x half> %857, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1037 = shufflevector <8 x half> %1036, <8 x half> %865, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1038 = shufflevector <8 x half> %1037, <8 x half> %873, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1039 = shufflevector <8 x half> %1038, <8 x half> %881, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1040 = shufflevector <1 x half> %832, <1 x half> %836, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1041 = shufflevector <8 x half> %1040, <8 x half> %843, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1042 = shufflevector <8 x half> %1041, <8 x half> %851, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1043 = shufflevector <8 x half> %1042, <8 x half> %859, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1044 = shufflevector <8 x half> %1043, <8 x half> %867, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1045 = shufflevector <8 x half> %1044, <8 x half> %875, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1046 = shufflevector <8 x half> %1045, <8 x half> %883, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1047 = shufflevector <1 x half> %833, <1 x half> %837, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1048 = shufflevector <8 x half> %1047, <8 x half> %845, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1049 = shufflevector <8 x half> %1048, <8 x half> %853, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1050 = shufflevector <8 x half> %1049, <8 x half> %861, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1051 = shufflevector <8 x half> %1050, <8 x half> %869, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1052 = shufflevector <8 x half> %1051, <8 x half> %877, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1053 = shufflevector <8 x half> %1052, <8 x half> %885, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1054 = shufflevector <1 x half> %886, <1 x half> %890, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1055 = shufflevector <8 x half> %1054, <8 x half> %895, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1056 = shufflevector <8 x half> %1055, <8 x half> %903, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1057 = shufflevector <8 x half> %1056, <8 x half> %911, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1058 = shufflevector <8 x half> %1057, <8 x half> %919, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1059 = shufflevector <8 x half> %1058, <8 x half> %927, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1060 = shufflevector <8 x half> %1059, <8 x half> %935, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1061 = shufflevector <1 x half> %887, <1 x half> %891, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1062 = shufflevector <8 x half> %1061, <8 x half> %897, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1063 = shufflevector <8 x half> %1062, <8 x half> %905, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1064 = shufflevector <8 x half> %1063, <8 x half> %913, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1065 = shufflevector <8 x half> %1064, <8 x half> %921, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1066 = shufflevector <8 x half> %1065, <8 x half> %929, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1067 = shufflevector <8 x half> %1066, <8 x half> %937, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1068 = shufflevector <1 x half> %888, <1 x half> %892, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1069 = shufflevector <8 x half> %1068, <8 x half> %899, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1070 = shufflevector <8 x half> %1069, <8 x half> %907, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1071 = shufflevector <8 x half> %1070, <8 x half> %915, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1072 = shufflevector <8 x half> %1071, <8 x half> %923, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1073 = shufflevector <8 x half> %1072, <8 x half> %931, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1074 = shufflevector <8 x half> %1073, <8 x half> %939, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1075 = shufflevector <1 x half> %889, <1 x half> %893, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1076 = shufflevector <8 x half> %1075, <8 x half> %901, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1077 = shufflevector <8 x half> %1076, <8 x half> %909, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1078 = shufflevector <8 x half> %1077, <8 x half> %917, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1079 = shufflevector <8 x half> %1078, <8 x half> %925, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1080 = shufflevector <8 x half> %1079, <8 x half> %933, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1081 = shufflevector <8 x half> %1080, <8 x half> %941, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1082 = shufflevector <1 x half> %942, <1 x half> %946, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1083 = shufflevector <8 x half> %1082, <8 x half> %951, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1084 = shufflevector <8 x half> %1083, <8 x half> %959, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1085 = shufflevector <8 x half> %1084, <8 x half> %967, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1086 = shufflevector <8 x half> %1085, <8 x half> %975, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1087 = shufflevector <8 x half> %1086, <8 x half> %983, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1088 = shufflevector <8 x half> %1087, <8 x half> %991, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1089 = shufflevector <1 x half> %943, <1 x half> %947, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1090 = shufflevector <8 x half> %1089, <8 x half> %953, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1091 = shufflevector <8 x half> %1090, <8 x half> %961, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1092 = shufflevector <8 x half> %1091, <8 x half> %969, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1093 = shufflevector <8 x half> %1092, <8 x half> %977, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1094 = shufflevector <8 x half> %1093, <8 x half> %985, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1095 = shufflevector <8 x half> %1094, <8 x half> %993, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1096 = shufflevector <1 x half> %944, <1 x half> %948, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1097 = shufflevector <8 x half> %1096, <8 x half> %955, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1098 = shufflevector <8 x half> %1097, <8 x half> %963, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1099 = shufflevector <8 x half> %1098, <8 x half> %971, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1100 = shufflevector <8 x half> %1099, <8 x half> %979, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1101 = shufflevector <8 x half> %1100, <8 x half> %987, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1102 = shufflevector <8 x half> %1101, <8 x half> %995, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1103 = shufflevector <1 x half> %945, <1 x half> %949, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1104 = shufflevector <8 x half> %1103, <8 x half> %957, <8 x i32> <i32 0, i32 1, i32 8, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1105 = shufflevector <8 x half> %1104, <8 x half> %965, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 poison, i32 poison, i32 poison, i32 poison>
  %1106 = shufflevector <8 x half> %1105, <8 x half> %973, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %1107 = shufflevector <8 x half> %1106, <8 x half> %981, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 8, i32 poison, i32 poison>
  %1108 = shufflevector <8 x half> %1107, <8 x half> %989, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 poison>
  %1109 = shufflevector <8 x half> %1108, <8 x half> %997, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8>
  %1110 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %758, <8 x half> %1004, <8 x float> %739)
  %1111 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %758, <8 x half> %1011, <8 x float> %740)
  %1112 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %758, <8 x half> %1018, <8 x float> %741)
  %1113 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %758, <8 x half> %1025, <8 x float> %742)
  %1114 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %762, <8 x half> %1004, <8 x float> %743)
  %1115 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %762, <8 x half> %1011, <8 x float> %744)
  %1116 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %762, <8 x half> %1018, <8 x float> %745)
  %1117 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %762, <8 x half> %1025, <8 x float> %746)
  %1118 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %766, <8 x half> %1004, <8 x float> %747)
  %1119 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %766, <8 x half> %1011, <8 x float> %748)
  %1120 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %766, <8 x half> %1018, <8 x float> %749)
  %1121 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %766, <8 x half> %1025, <8 x float> %750)
  %1122 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %770, <8 x half> %1004, <8 x float> %751)
  %1123 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %770, <8 x half> %1011, <8 x float> %752)
  %1124 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %770, <8 x half> %1018, <8 x float> %753)
  %1125 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %770, <8 x half> %1025, <8 x float> %754)
  %1126 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %759, <8 x half> %1032, <8 x float> %1110)
  %1127 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %759, <8 x half> %1039, <8 x float> %1111)
  %1128 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %759, <8 x half> %1046, <8 x float> %1112)
  %1129 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %759, <8 x half> %1053, <8 x float> %1113)
  %1130 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %763, <8 x half> %1032, <8 x float> %1114)
  %1131 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %763, <8 x half> %1039, <8 x float> %1115)
  %1132 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %763, <8 x half> %1046, <8 x float> %1116)
  %1133 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %763, <8 x half> %1053, <8 x float> %1117)
  %1134 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %767, <8 x half> %1032, <8 x float> %1118)
  %1135 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %767, <8 x half> %1039, <8 x float> %1119)
  %1136 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %767, <8 x half> %1046, <8 x float> %1120)
  %1137 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %767, <8 x half> %1053, <8 x float> %1121)
  %1138 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %771, <8 x half> %1032, <8 x float> %1122)
  %1139 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %771, <8 x half> %1039, <8 x float> %1123)
  %1140 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %771, <8 x half> %1046, <8 x float> %1124)
  %1141 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %771, <8 x half> %1053, <8 x float> %1125)
  %1142 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %760, <8 x half> %1060, <8 x float> %1126)
  %1143 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %760, <8 x half> %1067, <8 x float> %1127)
  %1144 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %760, <8 x half> %1074, <8 x float> %1128)
  %1145 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %760, <8 x half> %1081, <8 x float> %1129)
  %1146 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %764, <8 x half> %1060, <8 x float> %1130)
  %1147 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %764, <8 x half> %1067, <8 x float> %1131)
  %1148 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %764, <8 x half> %1074, <8 x float> %1132)
  %1149 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %764, <8 x half> %1081, <8 x float> %1133)
  %1150 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %768, <8 x half> %1060, <8 x float> %1134)
  %1151 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %768, <8 x half> %1067, <8 x float> %1135)
  %1152 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %768, <8 x half> %1074, <8 x float> %1136)
  %1153 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %768, <8 x half> %1081, <8 x float> %1137)
  %1154 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %772, <8 x half> %1060, <8 x float> %1138)
  %1155 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %772, <8 x half> %1067, <8 x float> %1139)
  %1156 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %772, <8 x half> %1074, <8 x float> %1140)
  %1157 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %772, <8 x half> %1081, <8 x float> %1141)
  %1158 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %761, <8 x half> %1088, <8 x float> %1142)
  %1159 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %761, <8 x half> %1095, <8 x float> %1143)
  %1160 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %761, <8 x half> %1102, <8 x float> %1144)
  %1161 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %761, <8 x half> %1109, <8 x float> %1145)
  %1162 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %765, <8 x half> %1088, <8 x float> %1146)
  %1163 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %765, <8 x half> %1095, <8 x float> %1147)
  %1164 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %765, <8 x half> %1102, <8 x float> %1148)
  %1165 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %765, <8 x half> %1109, <8 x float> %1149)
  %1166 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %769, <8 x half> %1088, <8 x float> %1150)
  %1167 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %769, <8 x half> %1095, <8 x float> %1151)
  %1168 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %769, <8 x half> %1102, <8 x float> %1152)
  %1169 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %769, <8 x half> %1109, <8 x float> %1153)
  %1170 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %773, <8 x half> %1088, <8 x float> %1154)
  %1171 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %773, <8 x half> %1095, <8 x float> %1155)
  %1172 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %773, <8 x half> %1102, <8 x float> %1156)
  %1173 = tail call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %773, <8 x half> %1109, <8 x float> %1157)
  %1174 = shufflevector <8 x float> %1158, <8 x float> poison, <1 x i32> zeroinitializer
  %1175 = shufflevector <8 x float> %1158, <8 x float> poison, <1 x i32> <i32 1>
  %1176 = shufflevector <8 x float> %1158, <8 x float> poison, <1 x i32> <i32 2>
  %1177 = shufflevector <8 x float> %1158, <8 x float> poison, <1 x i32> <i32 3>
  %1178 = shufflevector <8 x float> %1158, <8 x float> poison, <1 x i32> <i32 4>
  %1179 = shufflevector <8 x float> %1158, <8 x float> poison, <1 x i32> <i32 5>
  %1180 = shufflevector <8 x float> %1158, <8 x float> poison, <1 x i32> <i32 6>
  %1181 = shufflevector <8 x float> %1158, <8 x float> poison, <1 x i32> <i32 7>
  %1182 = shufflevector <8 x float> %1159, <8 x float> poison, <1 x i32> zeroinitializer
  %1183 = shufflevector <8 x float> %1159, <8 x float> poison, <1 x i32> <i32 1>
  %1184 = shufflevector <8 x float> %1159, <8 x float> poison, <1 x i32> <i32 2>
  %1185 = shufflevector <8 x float> %1159, <8 x float> poison, <1 x i32> <i32 3>
  %1186 = shufflevector <8 x float> %1159, <8 x float> poison, <1 x i32> <i32 4>
  %1187 = shufflevector <8 x float> %1159, <8 x float> poison, <1 x i32> <i32 5>
  %1188 = shufflevector <8 x float> %1159, <8 x float> poison, <1 x i32> <i32 6>
  %1189 = shufflevector <8 x float> %1159, <8 x float> poison, <1 x i32> <i32 7>
  %1190 = shufflevector <8 x float> %1160, <8 x float> poison, <1 x i32> zeroinitializer
  %1191 = shufflevector <8 x float> %1160, <8 x float> poison, <1 x i32> <i32 1>
  %1192 = shufflevector <8 x float> %1160, <8 x float> poison, <1 x i32> <i32 2>
  %1193 = shufflevector <8 x float> %1160, <8 x float> poison, <1 x i32> <i32 3>
  %1194 = shufflevector <8 x float> %1160, <8 x float> poison, <1 x i32> <i32 4>
  %1195 = shufflevector <8 x float> %1160, <8 x float> poison, <1 x i32> <i32 5>
  %1196 = shufflevector <8 x float> %1160, <8 x float> poison, <1 x i32> <i32 6>
  %1197 = shufflevector <8 x float> %1160, <8 x float> poison, <1 x i32> <i32 7>
  %1198 = shufflevector <8 x float> %1161, <8 x float> poison, <1 x i32> zeroinitializer
  %1199 = shufflevector <8 x float> %1161, <8 x float> poison, <1 x i32> <i32 1>
  %1200 = shufflevector <8 x float> %1161, <8 x float> poison, <1 x i32> <i32 2>
  %1201 = shufflevector <8 x float> %1161, <8 x float> poison, <1 x i32> <i32 3>
  %1202 = shufflevector <8 x float> %1161, <8 x float> poison, <1 x i32> <i32 4>
  %1203 = shufflevector <8 x float> %1161, <8 x float> poison, <1 x i32> <i32 5>
  %1204 = shufflevector <8 x float> %1161, <8 x float> poison, <1 x i32> <i32 6>
  %1205 = shufflevector <8 x float> %1161, <8 x float> poison, <1 x i32> <i32 7>
  %1206 = shufflevector <8 x float> %1162, <8 x float> poison, <1 x i32> zeroinitializer
  %1207 = shufflevector <8 x float> %1162, <8 x float> poison, <1 x i32> <i32 1>
  %1208 = shufflevector <8 x float> %1162, <8 x float> poison, <1 x i32> <i32 2>
  %1209 = shufflevector <8 x float> %1162, <8 x float> poison, <1 x i32> <i32 3>
  %1210 = shufflevector <8 x float> %1162, <8 x float> poison, <1 x i32> <i32 4>
  %1211 = shufflevector <8 x float> %1162, <8 x float> poison, <1 x i32> <i32 5>
  %1212 = shufflevector <8 x float> %1162, <8 x float> poison, <1 x i32> <i32 6>
  %1213 = shufflevector <8 x float> %1162, <8 x float> poison, <1 x i32> <i32 7>
  %1214 = shufflevector <8 x float> %1163, <8 x float> poison, <1 x i32> zeroinitializer
  %1215 = shufflevector <8 x float> %1163, <8 x float> poison, <1 x i32> <i32 1>
  %1216 = shufflevector <8 x float> %1163, <8 x float> poison, <1 x i32> <i32 2>
  %1217 = shufflevector <8 x float> %1163, <8 x float> poison, <1 x i32> <i32 3>
  %1218 = shufflevector <8 x float> %1163, <8 x float> poison, <1 x i32> <i32 4>
  %1219 = shufflevector <8 x float> %1163, <8 x float> poison, <1 x i32> <i32 5>
  %1220 = shufflevector <8 x float> %1163, <8 x float> poison, <1 x i32> <i32 6>
  %1221 = shufflevector <8 x float> %1163, <8 x float> poison, <1 x i32> <i32 7>
  %1222 = shufflevector <8 x float> %1164, <8 x float> poison, <1 x i32> zeroinitializer
  %1223 = shufflevector <8 x float> %1164, <8 x float> poison, <1 x i32> <i32 1>
  %1224 = shufflevector <8 x float> %1164, <8 x float> poison, <1 x i32> <i32 2>
  %1225 = shufflevector <8 x float> %1164, <8 x float> poison, <1 x i32> <i32 3>
  %1226 = shufflevector <8 x float> %1164, <8 x float> poison, <1 x i32> <i32 4>
  %1227 = shufflevector <8 x float> %1164, <8 x float> poison, <1 x i32> <i32 5>
  %1228 = shufflevector <8 x float> %1164, <8 x float> poison, <1 x i32> <i32 6>
  %1229 = shufflevector <8 x float> %1164, <8 x float> poison, <1 x i32> <i32 7>
  %1230 = shufflevector <8 x float> %1165, <8 x float> poison, <1 x i32> zeroinitializer
  %1231 = shufflevector <8 x float> %1165, <8 x float> poison, <1 x i32> <i32 1>
  %1232 = shufflevector <8 x float> %1165, <8 x float> poison, <1 x i32> <i32 2>
  %1233 = shufflevector <8 x float> %1165, <8 x float> poison, <1 x i32> <i32 3>
  %1234 = shufflevector <8 x float> %1165, <8 x float> poison, <1 x i32> <i32 4>
  %1235 = shufflevector <8 x float> %1165, <8 x float> poison, <1 x i32> <i32 5>
  %1236 = shufflevector <8 x float> %1165, <8 x float> poison, <1 x i32> <i32 6>
  %1237 = shufflevector <8 x float> %1165, <8 x float> poison, <1 x i32> <i32 7>
  %1238 = shufflevector <8 x float> %1166, <8 x float> poison, <1 x i32> zeroinitializer
  %1239 = shufflevector <8 x float> %1166, <8 x float> poison, <1 x i32> <i32 1>
  %1240 = shufflevector <8 x float> %1166, <8 x float> poison, <1 x i32> <i32 2>
  %1241 = shufflevector <8 x float> %1166, <8 x float> poison, <1 x i32> <i32 3>
  %1242 = shufflevector <8 x float> %1166, <8 x float> poison, <1 x i32> <i32 4>
  %1243 = shufflevector <8 x float> %1166, <8 x float> poison, <1 x i32> <i32 5>
  %1244 = shufflevector <8 x float> %1166, <8 x float> poison, <1 x i32> <i32 6>
  %1245 = shufflevector <8 x float> %1166, <8 x float> poison, <1 x i32> <i32 7>
  %1246 = shufflevector <8 x float> %1167, <8 x float> poison, <1 x i32> zeroinitializer
  %1247 = shufflevector <8 x float> %1167, <8 x float> poison, <1 x i32> <i32 1>
  %1248 = shufflevector <8 x float> %1167, <8 x float> poison, <1 x i32> <i32 2>
  %1249 = shufflevector <8 x float> %1167, <8 x float> poison, <1 x i32> <i32 3>
  %1250 = shufflevector <8 x float> %1167, <8 x float> poison, <1 x i32> <i32 4>
  %1251 = shufflevector <8 x float> %1167, <8 x float> poison, <1 x i32> <i32 5>
  %1252 = shufflevector <8 x float> %1167, <8 x float> poison, <1 x i32> <i32 6>
  %1253 = shufflevector <8 x float> %1167, <8 x float> poison, <1 x i32> <i32 7>
  %1254 = shufflevector <8 x float> %1168, <8 x float> poison, <1 x i32> zeroinitializer
  %1255 = shufflevector <8 x float> %1168, <8 x float> poison, <1 x i32> <i32 1>
  %1256 = shufflevector <8 x float> %1168, <8 x float> poison, <1 x i32> <i32 2>
  %1257 = shufflevector <8 x float> %1168, <8 x float> poison, <1 x i32> <i32 3>
  %1258 = shufflevector <8 x float> %1168, <8 x float> poison, <1 x i32> <i32 4>
  %1259 = shufflevector <8 x float> %1168, <8 x float> poison, <1 x i32> <i32 5>
  %1260 = shufflevector <8 x float> %1168, <8 x float> poison, <1 x i32> <i32 6>
  %1261 = shufflevector <8 x float> %1168, <8 x float> poison, <1 x i32> <i32 7>
  %1262 = shufflevector <8 x float> %1169, <8 x float> poison, <1 x i32> zeroinitializer
  %1263 = shufflevector <8 x float> %1169, <8 x float> poison, <1 x i32> <i32 1>
  %1264 = shufflevector <8 x float> %1169, <8 x float> poison, <1 x i32> <i32 2>
  %1265 = shufflevector <8 x float> %1169, <8 x float> poison, <1 x i32> <i32 3>
  %1266 = shufflevector <8 x float> %1169, <8 x float> poison, <1 x i32> <i32 4>
  %1267 = shufflevector <8 x float> %1169, <8 x float> poison, <1 x i32> <i32 5>
  %1268 = shufflevector <8 x float> %1169, <8 x float> poison, <1 x i32> <i32 6>
  %1269 = shufflevector <8 x float> %1169, <8 x float> poison, <1 x i32> <i32 7>
  %1270 = shufflevector <8 x float> %1170, <8 x float> poison, <1 x i32> zeroinitializer
  %1271 = shufflevector <8 x float> %1170, <8 x float> poison, <1 x i32> <i32 1>
  %1272 = shufflevector <8 x float> %1170, <8 x float> poison, <1 x i32> <i32 2>
  %1273 = shufflevector <8 x float> %1170, <8 x float> poison, <1 x i32> <i32 3>
  %1274 = shufflevector <8 x float> %1170, <8 x float> poison, <1 x i32> <i32 4>
  %1275 = shufflevector <8 x float> %1170, <8 x float> poison, <1 x i32> <i32 5>
  %1276 = shufflevector <8 x float> %1170, <8 x float> poison, <1 x i32> <i32 6>
  %1277 = shufflevector <8 x float> %1170, <8 x float> poison, <1 x i32> <i32 7>
  %1278 = shufflevector <8 x float> %1171, <8 x float> poison, <1 x i32> zeroinitializer
  %1279 = shufflevector <8 x float> %1171, <8 x float> poison, <1 x i32> <i32 1>
  %1280 = shufflevector <8 x float> %1171, <8 x float> poison, <1 x i32> <i32 2>
  %1281 = shufflevector <8 x float> %1171, <8 x float> poison, <1 x i32> <i32 3>
  %1282 = shufflevector <8 x float> %1171, <8 x float> poison, <1 x i32> <i32 4>
  %1283 = shufflevector <8 x float> %1171, <8 x float> poison, <1 x i32> <i32 5>
  %1284 = shufflevector <8 x float> %1171, <8 x float> poison, <1 x i32> <i32 6>
  %1285 = shufflevector <8 x float> %1171, <8 x float> poison, <1 x i32> <i32 7>
  %1286 = shufflevector <8 x float> %1172, <8 x float> poison, <1 x i32> zeroinitializer
  %1287 = shufflevector <8 x float> %1172, <8 x float> poison, <1 x i32> <i32 1>
  %1288 = shufflevector <8 x float> %1172, <8 x float> poison, <1 x i32> <i32 2>
  %1289 = shufflevector <8 x float> %1172, <8 x float> poison, <1 x i32> <i32 3>
  %1290 = shufflevector <8 x float> %1172, <8 x float> poison, <1 x i32> <i32 4>
  %1291 = shufflevector <8 x float> %1172, <8 x float> poison, <1 x i32> <i32 5>
  %1292 = shufflevector <8 x float> %1172, <8 x float> poison, <1 x i32> <i32 6>
  %1293 = shufflevector <8 x float> %1172, <8 x float> poison, <1 x i32> <i32 7>
  %1294 = shufflevector <8 x float> %1173, <8 x float> poison, <1 x i32> zeroinitializer
  %1295 = shufflevector <8 x float> %1173, <8 x float> poison, <1 x i32> <i32 1>
  %1296 = shufflevector <8 x float> %1173, <8 x float> poison, <1 x i32> <i32 2>
  %1297 = shufflevector <8 x float> %1173, <8 x float> poison, <1 x i32> <i32 3>
  %1298 = shufflevector <8 x float> %1173, <8 x float> poison, <1 x i32> <i32 4>
  %1299 = shufflevector <8 x float> %1173, <8 x float> poison, <1 x i32> <i32 5>
  %1300 = shufflevector <8 x float> %1173, <8 x float> poison, <1 x i32> <i32 6>
  %1301 = shufflevector <8 x float> %1173, <8 x float> poison, <1 x i32> <i32 7>
  %.mask = and i32 %4, 192
  %1302 = or disjoint i32 %30, %.mask
  %1303 = or disjoint i32 %1302, %9
  %1304 = or disjoint i32 %109, %61
  %.idx111 = shl nuw nsw i32 %1303, 13
  %1305 = getelementptr i8, ptr addrspace(7) %757, i32 %.idx111
  %1306 = getelementptr [4 x i8], ptr addrspace(7) %1305, i32 %1304
  store <1 x float> %1174, ptr addrspace(7) %1306, align 4
  %1307 = or disjoint i32 %1304, 16
  %1308 = getelementptr [4 x i8], ptr addrspace(7) %1305, i32 %1307
  store <1 x float> %1182, ptr addrspace(7) %1308, align 4
  %1309 = or disjoint i32 %1304, 32
  %1310 = getelementptr [4 x i8], ptr addrspace(7) %1305, i32 %1309
  store <1 x float> %1190, ptr addrspace(7) %1310, align 4
  %1311 = or disjoint i32 %1304, 48
  %1312 = getelementptr [4 x i8], ptr addrspace(7) %1305, i32 %1311
  store <1 x float> %1198, ptr addrspace(7) %1312, align 4
  %1313 = getelementptr i8, ptr addrspace(7) %1305, i32 8192
  %1314 = getelementptr [4 x i8], ptr addrspace(7) %1313, i32 %1304
  store <1 x float> %1175, ptr addrspace(7) %1314, align 4
  %1315 = getelementptr [4 x i8], ptr addrspace(7) %1313, i32 %1307
  store <1 x float> %1183, ptr addrspace(7) %1315, align 4
  %1316 = getelementptr [4 x i8], ptr addrspace(7) %1313, i32 %1309
  store <1 x float> %1191, ptr addrspace(7) %1316, align 4
  %1317 = getelementptr [4 x i8], ptr addrspace(7) %1313, i32 %1311
  store <1 x float> %1199, ptr addrspace(7) %1317, align 4
  %1318 = getelementptr i8, ptr addrspace(7) %1305, i32 16384
  %1319 = getelementptr [4 x i8], ptr addrspace(7) %1318, i32 %1304
  store <1 x float> %1176, ptr addrspace(7) %1319, align 4
  %1320 = getelementptr [4 x i8], ptr addrspace(7) %1318, i32 %1307
  store <1 x float> %1184, ptr addrspace(7) %1320, align 4
  %1321 = getelementptr [4 x i8], ptr addrspace(7) %1318, i32 %1309
  store <1 x float> %1192, ptr addrspace(7) %1321, align 4
  %1322 = getelementptr [4 x i8], ptr addrspace(7) %1318, i32 %1311
  store <1 x float> %1200, ptr addrspace(7) %1322, align 4
  %1323 = getelementptr i8, ptr addrspace(7) %1305, i32 24576
  %1324 = getelementptr [4 x i8], ptr addrspace(7) %1323, i32 %1304
  store <1 x float> %1177, ptr addrspace(7) %1324, align 4
  %1325 = getelementptr [4 x i8], ptr addrspace(7) %1323, i32 %1307
  store <1 x float> %1185, ptr addrspace(7) %1325, align 4
  %1326 = getelementptr [4 x i8], ptr addrspace(7) %1323, i32 %1309
  store <1 x float> %1193, ptr addrspace(7) %1326, align 4
  %1327 = getelementptr [4 x i8], ptr addrspace(7) %1323, i32 %1311
  store <1 x float> %1201, ptr addrspace(7) %1327, align 4
  %1328 = getelementptr i8, ptr addrspace(7) %1305, i32 32768
  %1329 = getelementptr [4 x i8], ptr addrspace(7) %1328, i32 %1304
  store <1 x float> %1178, ptr addrspace(7) %1329, align 4
  %1330 = getelementptr [4 x i8], ptr addrspace(7) %1328, i32 %1307
  store <1 x float> %1186, ptr addrspace(7) %1330, align 4
  %1331 = getelementptr [4 x i8], ptr addrspace(7) %1328, i32 %1309
  store <1 x float> %1194, ptr addrspace(7) %1331, align 4
  %1332 = getelementptr [4 x i8], ptr addrspace(7) %1328, i32 %1311
  store <1 x float> %1202, ptr addrspace(7) %1332, align 4
  %1333 = getelementptr i8, ptr addrspace(7) %1305, i32 40960
  %1334 = getelementptr [4 x i8], ptr addrspace(7) %1333, i32 %1304
  store <1 x float> %1179, ptr addrspace(7) %1334, align 4
  %1335 = getelementptr [4 x i8], ptr addrspace(7) %1333, i32 %1307
  store <1 x float> %1187, ptr addrspace(7) %1335, align 4
  %1336 = getelementptr [4 x i8], ptr addrspace(7) %1333, i32 %1309
  store <1 x float> %1195, ptr addrspace(7) %1336, align 4
  %1337 = getelementptr [4 x i8], ptr addrspace(7) %1333, i32 %1311
  store <1 x float> %1203, ptr addrspace(7) %1337, align 4
  %1338 = getelementptr i8, ptr addrspace(7) %1305, i32 49152
  %1339 = getelementptr [4 x i8], ptr addrspace(7) %1338, i32 %1304
  store <1 x float> %1180, ptr addrspace(7) %1339, align 4
  %1340 = getelementptr [4 x i8], ptr addrspace(7) %1338, i32 %1307
  store <1 x float> %1188, ptr addrspace(7) %1340, align 4
  %1341 = getelementptr [4 x i8], ptr addrspace(7) %1338, i32 %1309
  store <1 x float> %1196, ptr addrspace(7) %1341, align 4
  %1342 = getelementptr [4 x i8], ptr addrspace(7) %1338, i32 %1311
  store <1 x float> %1204, ptr addrspace(7) %1342, align 4
  %1343 = getelementptr i8, ptr addrspace(7) %1305, i32 57344
  %1344 = getelementptr [4 x i8], ptr addrspace(7) %1343, i32 %1304
  store <1 x float> %1181, ptr addrspace(7) %1344, align 4
  %1345 = getelementptr [4 x i8], ptr addrspace(7) %1343, i32 %1307
  store <1 x float> %1189, ptr addrspace(7) %1345, align 4
  %1346 = getelementptr [4 x i8], ptr addrspace(7) %1343, i32 %1309
  store <1 x float> %1197, ptr addrspace(7) %1346, align 4
  %1347 = getelementptr [4 x i8], ptr addrspace(7) %1343, i32 %1311
  store <1 x float> %1205, ptr addrspace(7) %1347, align 4
  %1348 = getelementptr i8, ptr addrspace(7) %1305, i32 131072
  %1349 = getelementptr [4 x i8], ptr addrspace(7) %1348, i32 %1304
  store <1 x float> %1206, ptr addrspace(7) %1349, align 4
  %1350 = getelementptr [4 x i8], ptr addrspace(7) %1348, i32 %1307
  store <1 x float> %1214, ptr addrspace(7) %1350, align 4
  %1351 = getelementptr [4 x i8], ptr addrspace(7) %1348, i32 %1309
  store <1 x float> %1222, ptr addrspace(7) %1351, align 4
  %1352 = getelementptr [4 x i8], ptr addrspace(7) %1348, i32 %1311
  store <1 x float> %1230, ptr addrspace(7) %1352, align 4
  %1353 = getelementptr i8, ptr addrspace(7) %1305, i32 139264
  %1354 = getelementptr [4 x i8], ptr addrspace(7) %1353, i32 %1304
  store <1 x float> %1207, ptr addrspace(7) %1354, align 4
  %1355 = getelementptr [4 x i8], ptr addrspace(7) %1353, i32 %1307
  store <1 x float> %1215, ptr addrspace(7) %1355, align 4
  %1356 = getelementptr [4 x i8], ptr addrspace(7) %1353, i32 %1309
  store <1 x float> %1223, ptr addrspace(7) %1356, align 4
  %1357 = getelementptr [4 x i8], ptr addrspace(7) %1353, i32 %1311
  store <1 x float> %1231, ptr addrspace(7) %1357, align 4
  %1358 = getelementptr i8, ptr addrspace(7) %1305, i32 147456
  %1359 = getelementptr [4 x i8], ptr addrspace(7) %1358, i32 %1304
  store <1 x float> %1208, ptr addrspace(7) %1359, align 4
  %1360 = getelementptr [4 x i8], ptr addrspace(7) %1358, i32 %1307
  store <1 x float> %1216, ptr addrspace(7) %1360, align 4
  %1361 = getelementptr [4 x i8], ptr addrspace(7) %1358, i32 %1309
  store <1 x float> %1224, ptr addrspace(7) %1361, align 4
  %1362 = getelementptr [4 x i8], ptr addrspace(7) %1358, i32 %1311
  store <1 x float> %1232, ptr addrspace(7) %1362, align 4
  %1363 = getelementptr i8, ptr addrspace(7) %1305, i32 155648
  %1364 = getelementptr [4 x i8], ptr addrspace(7) %1363, i32 %1304
  store <1 x float> %1209, ptr addrspace(7) %1364, align 4
  %1365 = getelementptr [4 x i8], ptr addrspace(7) %1363, i32 %1307
  store <1 x float> %1217, ptr addrspace(7) %1365, align 4
  %1366 = getelementptr [4 x i8], ptr addrspace(7) %1363, i32 %1309
  store <1 x float> %1225, ptr addrspace(7) %1366, align 4
  %1367 = getelementptr [4 x i8], ptr addrspace(7) %1363, i32 %1311
  store <1 x float> %1233, ptr addrspace(7) %1367, align 4
  %1368 = getelementptr i8, ptr addrspace(7) %1305, i32 163840
  %1369 = getelementptr [4 x i8], ptr addrspace(7) %1368, i32 %1304
  store <1 x float> %1210, ptr addrspace(7) %1369, align 4
  %1370 = getelementptr [4 x i8], ptr addrspace(7) %1368, i32 %1307
  store <1 x float> %1218, ptr addrspace(7) %1370, align 4
  %1371 = getelementptr [4 x i8], ptr addrspace(7) %1368, i32 %1309
  store <1 x float> %1226, ptr addrspace(7) %1371, align 4
  %1372 = getelementptr [4 x i8], ptr addrspace(7) %1368, i32 %1311
  store <1 x float> %1234, ptr addrspace(7) %1372, align 4
  %1373 = getelementptr i8, ptr addrspace(7) %1305, i32 172032
  %1374 = getelementptr [4 x i8], ptr addrspace(7) %1373, i32 %1304
  store <1 x float> %1211, ptr addrspace(7) %1374, align 4
  %1375 = getelementptr [4 x i8], ptr addrspace(7) %1373, i32 %1307
  store <1 x float> %1219, ptr addrspace(7) %1375, align 4
  %1376 = getelementptr [4 x i8], ptr addrspace(7) %1373, i32 %1309
  store <1 x float> %1227, ptr addrspace(7) %1376, align 4
  %1377 = getelementptr [4 x i8], ptr addrspace(7) %1373, i32 %1311
  store <1 x float> %1235, ptr addrspace(7) %1377, align 4
  %1378 = getelementptr i8, ptr addrspace(7) %1305, i32 180224
  %1379 = getelementptr [4 x i8], ptr addrspace(7) %1378, i32 %1304
  store <1 x float> %1212, ptr addrspace(7) %1379, align 4
  %1380 = getelementptr [4 x i8], ptr addrspace(7) %1378, i32 %1307
  store <1 x float> %1220, ptr addrspace(7) %1380, align 4
  %1381 = getelementptr [4 x i8], ptr addrspace(7) %1378, i32 %1309
  store <1 x float> %1228, ptr addrspace(7) %1381, align 4
  %1382 = getelementptr [4 x i8], ptr addrspace(7) %1378, i32 %1311
  store <1 x float> %1236, ptr addrspace(7) %1382, align 4
  %1383 = getelementptr i8, ptr addrspace(7) %1305, i32 188416
  %1384 = getelementptr [4 x i8], ptr addrspace(7) %1383, i32 %1304
  store <1 x float> %1213, ptr addrspace(7) %1384, align 4
  %1385 = getelementptr [4 x i8], ptr addrspace(7) %1383, i32 %1307
  store <1 x float> %1221, ptr addrspace(7) %1385, align 4
  %1386 = getelementptr [4 x i8], ptr addrspace(7) %1383, i32 %1309
  store <1 x float> %1229, ptr addrspace(7) %1386, align 4
  %1387 = getelementptr [4 x i8], ptr addrspace(7) %1383, i32 %1311
  store <1 x float> %1237, ptr addrspace(7) %1387, align 4
  %1388 = getelementptr i8, ptr addrspace(7) %1305, i32 262144
  %1389 = getelementptr [4 x i8], ptr addrspace(7) %1388, i32 %1304
  store <1 x float> %1238, ptr addrspace(7) %1389, align 4
  %1390 = getelementptr [4 x i8], ptr addrspace(7) %1388, i32 %1307
  store <1 x float> %1246, ptr addrspace(7) %1390, align 4
  %1391 = getelementptr [4 x i8], ptr addrspace(7) %1388, i32 %1309
  store <1 x float> %1254, ptr addrspace(7) %1391, align 4
  %1392 = getelementptr [4 x i8], ptr addrspace(7) %1388, i32 %1311
  store <1 x float> %1262, ptr addrspace(7) %1392, align 4
  %1393 = getelementptr i8, ptr addrspace(7) %1305, i32 270336
  %1394 = getelementptr [4 x i8], ptr addrspace(7) %1393, i32 %1304
  store <1 x float> %1239, ptr addrspace(7) %1394, align 4
  %1395 = getelementptr [4 x i8], ptr addrspace(7) %1393, i32 %1307
  store <1 x float> %1247, ptr addrspace(7) %1395, align 4
  %1396 = getelementptr [4 x i8], ptr addrspace(7) %1393, i32 %1309
  store <1 x float> %1255, ptr addrspace(7) %1396, align 4
  %1397 = getelementptr [4 x i8], ptr addrspace(7) %1393, i32 %1311
  store <1 x float> %1263, ptr addrspace(7) %1397, align 4
  %1398 = getelementptr i8, ptr addrspace(7) %1305, i32 278528
  %1399 = getelementptr [4 x i8], ptr addrspace(7) %1398, i32 %1304
  store <1 x float> %1240, ptr addrspace(7) %1399, align 4
  %1400 = getelementptr [4 x i8], ptr addrspace(7) %1398, i32 %1307
  store <1 x float> %1248, ptr addrspace(7) %1400, align 4
  %1401 = getelementptr [4 x i8], ptr addrspace(7) %1398, i32 %1309
  store <1 x float> %1256, ptr addrspace(7) %1401, align 4
  %1402 = getelementptr [4 x i8], ptr addrspace(7) %1398, i32 %1311
  store <1 x float> %1264, ptr addrspace(7) %1402, align 4
  %1403 = getelementptr i8, ptr addrspace(7) %1305, i32 286720
  %1404 = getelementptr [4 x i8], ptr addrspace(7) %1403, i32 %1304
  store <1 x float> %1241, ptr addrspace(7) %1404, align 4
  %1405 = getelementptr [4 x i8], ptr addrspace(7) %1403, i32 %1307
  store <1 x float> %1249, ptr addrspace(7) %1405, align 4
  %1406 = getelementptr [4 x i8], ptr addrspace(7) %1403, i32 %1309
  store <1 x float> %1257, ptr addrspace(7) %1406, align 4
  %1407 = getelementptr [4 x i8], ptr addrspace(7) %1403, i32 %1311
  store <1 x float> %1265, ptr addrspace(7) %1407, align 4
  %1408 = getelementptr i8, ptr addrspace(7) %1305, i32 294912
  %1409 = getelementptr [4 x i8], ptr addrspace(7) %1408, i32 %1304
  store <1 x float> %1242, ptr addrspace(7) %1409, align 4
  %1410 = getelementptr [4 x i8], ptr addrspace(7) %1408, i32 %1307
  store <1 x float> %1250, ptr addrspace(7) %1410, align 4
  %1411 = getelementptr [4 x i8], ptr addrspace(7) %1408, i32 %1309
  store <1 x float> %1258, ptr addrspace(7) %1411, align 4
  %1412 = getelementptr [4 x i8], ptr addrspace(7) %1408, i32 %1311
  store <1 x float> %1266, ptr addrspace(7) %1412, align 4
  %1413 = getelementptr i8, ptr addrspace(7) %1305, i32 303104
  %1414 = getelementptr [4 x i8], ptr addrspace(7) %1413, i32 %1304
  store <1 x float> %1243, ptr addrspace(7) %1414, align 4
  %1415 = getelementptr [4 x i8], ptr addrspace(7) %1413, i32 %1307
  store <1 x float> %1251, ptr addrspace(7) %1415, align 4
  %1416 = getelementptr [4 x i8], ptr addrspace(7) %1413, i32 %1309
  store <1 x float> %1259, ptr addrspace(7) %1416, align 4
  %1417 = getelementptr [4 x i8], ptr addrspace(7) %1413, i32 %1311
  store <1 x float> %1267, ptr addrspace(7) %1417, align 4
  %1418 = getelementptr i8, ptr addrspace(7) %1305, i32 311296
  %1419 = getelementptr [4 x i8], ptr addrspace(7) %1418, i32 %1304
  store <1 x float> %1244, ptr addrspace(7) %1419, align 4
  %1420 = getelementptr [4 x i8], ptr addrspace(7) %1418, i32 %1307
  store <1 x float> %1252, ptr addrspace(7) %1420, align 4
  %1421 = getelementptr [4 x i8], ptr addrspace(7) %1418, i32 %1309
  store <1 x float> %1260, ptr addrspace(7) %1421, align 4
  %1422 = getelementptr [4 x i8], ptr addrspace(7) %1418, i32 %1311
  store <1 x float> %1268, ptr addrspace(7) %1422, align 4
  %1423 = getelementptr i8, ptr addrspace(7) %1305, i32 319488
  %1424 = getelementptr [4 x i8], ptr addrspace(7) %1423, i32 %1304
  store <1 x float> %1245, ptr addrspace(7) %1424, align 4
  %1425 = getelementptr [4 x i8], ptr addrspace(7) %1423, i32 %1307
  store <1 x float> %1253, ptr addrspace(7) %1425, align 4
  %1426 = getelementptr [4 x i8], ptr addrspace(7) %1423, i32 %1309
  store <1 x float> %1261, ptr addrspace(7) %1426, align 4
  %1427 = getelementptr [4 x i8], ptr addrspace(7) %1423, i32 %1311
  store <1 x float> %1269, ptr addrspace(7) %1427, align 4
  %1428 = getelementptr i8, ptr addrspace(7) %1305, i32 393216
  %1429 = getelementptr [4 x i8], ptr addrspace(7) %1428, i32 %1304
  store <1 x float> %1270, ptr addrspace(7) %1429, align 4
  %1430 = getelementptr [4 x i8], ptr addrspace(7) %1428, i32 %1307
  store <1 x float> %1278, ptr addrspace(7) %1430, align 4
  %1431 = getelementptr [4 x i8], ptr addrspace(7) %1428, i32 %1309
  store <1 x float> %1286, ptr addrspace(7) %1431, align 4
  %1432 = getelementptr [4 x i8], ptr addrspace(7) %1428, i32 %1311
  store <1 x float> %1294, ptr addrspace(7) %1432, align 4
  %1433 = getelementptr i8, ptr addrspace(7) %1305, i32 401408
  %1434 = getelementptr [4 x i8], ptr addrspace(7) %1433, i32 %1304
  store <1 x float> %1271, ptr addrspace(7) %1434, align 4
  %1435 = getelementptr [4 x i8], ptr addrspace(7) %1433, i32 %1307
  store <1 x float> %1279, ptr addrspace(7) %1435, align 4
  %1436 = getelementptr [4 x i8], ptr addrspace(7) %1433, i32 %1309
  store <1 x float> %1287, ptr addrspace(7) %1436, align 4
  %1437 = getelementptr [4 x i8], ptr addrspace(7) %1433, i32 %1311
  store <1 x float> %1295, ptr addrspace(7) %1437, align 4
  %1438 = getelementptr i8, ptr addrspace(7) %1305, i32 409600
  %1439 = getelementptr [4 x i8], ptr addrspace(7) %1438, i32 %1304
  store <1 x float> %1272, ptr addrspace(7) %1439, align 4
  %1440 = getelementptr [4 x i8], ptr addrspace(7) %1438, i32 %1307
  store <1 x float> %1280, ptr addrspace(7) %1440, align 4
  %1441 = getelementptr [4 x i8], ptr addrspace(7) %1438, i32 %1309
  store <1 x float> %1288, ptr addrspace(7) %1441, align 4
  %1442 = getelementptr [4 x i8], ptr addrspace(7) %1438, i32 %1311
  store <1 x float> %1296, ptr addrspace(7) %1442, align 4
  %1443 = getelementptr i8, ptr addrspace(7) %1305, i32 417792
  %1444 = getelementptr [4 x i8], ptr addrspace(7) %1443, i32 %1304
  store <1 x float> %1273, ptr addrspace(7) %1444, align 4
  %1445 = getelementptr [4 x i8], ptr addrspace(7) %1443, i32 %1307
  store <1 x float> %1281, ptr addrspace(7) %1445, align 4
  %1446 = getelementptr [4 x i8], ptr addrspace(7) %1443, i32 %1309
  store <1 x float> %1289, ptr addrspace(7) %1446, align 4
  %1447 = getelementptr [4 x i8], ptr addrspace(7) %1443, i32 %1311
  store <1 x float> %1297, ptr addrspace(7) %1447, align 4
  %1448 = getelementptr i8, ptr addrspace(7) %1305, i32 425984
  %1449 = getelementptr [4 x i8], ptr addrspace(7) %1448, i32 %1304
  store <1 x float> %1274, ptr addrspace(7) %1449, align 4
  %1450 = getelementptr [4 x i8], ptr addrspace(7) %1448, i32 %1307
  store <1 x float> %1282, ptr addrspace(7) %1450, align 4
  %1451 = getelementptr [4 x i8], ptr addrspace(7) %1448, i32 %1309
  store <1 x float> %1290, ptr addrspace(7) %1451, align 4
  %1452 = getelementptr [4 x i8], ptr addrspace(7) %1448, i32 %1311
  store <1 x float> %1298, ptr addrspace(7) %1452, align 4
  %1453 = getelementptr i8, ptr addrspace(7) %1305, i32 434176
  %1454 = getelementptr [4 x i8], ptr addrspace(7) %1453, i32 %1304
  store <1 x float> %1275, ptr addrspace(7) %1454, align 4
  %1455 = getelementptr [4 x i8], ptr addrspace(7) %1453, i32 %1307
  store <1 x float> %1283, ptr addrspace(7) %1455, align 4
  %1456 = getelementptr [4 x i8], ptr addrspace(7) %1453, i32 %1309
  store <1 x float> %1291, ptr addrspace(7) %1456, align 4
  %1457 = getelementptr [4 x i8], ptr addrspace(7) %1453, i32 %1311
  store <1 x float> %1299, ptr addrspace(7) %1457, align 4
  %1458 = getelementptr i8, ptr addrspace(7) %1305, i32 442368
  %1459 = getelementptr [4 x i8], ptr addrspace(7) %1458, i32 %1304
  store <1 x float> %1276, ptr addrspace(7) %1459, align 4
  %1460 = getelementptr [4 x i8], ptr addrspace(7) %1458, i32 %1307
  store <1 x float> %1284, ptr addrspace(7) %1460, align 4
  %1461 = getelementptr [4 x i8], ptr addrspace(7) %1458, i32 %1309
  store <1 x float> %1292, ptr addrspace(7) %1461, align 4
  %1462 = getelementptr [4 x i8], ptr addrspace(7) %1458, i32 %1311
  store <1 x float> %1300, ptr addrspace(7) %1462, align 4
  %1463 = getelementptr i8, ptr addrspace(7) %1305, i32 450560
  %1464 = getelementptr [4 x i8], ptr addrspace(7) %1463, i32 %1304
  store <1 x float> %1277, ptr addrspace(7) %1464, align 4
  %1465 = getelementptr [4 x i8], ptr addrspace(7) %1463, i32 %1307
  store <1 x float> %1285, ptr addrspace(7) %1465, align 4
  %1466 = getelementptr [4 x i8], ptr addrspace(7) %1463, i32 %1309
  store <1 x float> %1293, ptr addrspace(7) %1466, align 4
  %1467 = getelementptr [4 x i8], ptr addrspace(7) %1463, i32 %1311
  store <1 x float> %1301, ptr addrspace(7) %1467, align 4
  fence syncscope("workgroup") release, !mmra !3
  tail call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  tail call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire, !mmra !3
  ret void
}

; Function Attrs: alwaysinline mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: alwaysinline mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #2

; Function Attrs: alwaysinline mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(7) @llvm.amdgcn.make.buffer.rsrc.p7.p1(ptr addrspace(1) readnone, i16, i64, i32) #3

; Function Attrs: alwaysinline mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: alwaysinline convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.signal(i32 immarg) #4

; Function Attrs: alwaysinline convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.wait(i16 immarg) #4

; Function Attrs: alwaysinline convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half>, <8 x half>, <8 x float>) #5

; Function Attrs: alwaysinline convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #4

attributes #0 = { alwaysinline nofree norecurse nounwind "amdgpu-flat-work-group-size"="256,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "uniform-work-group-size" }
attributes #1 = { alwaysinline mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { alwaysinline mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #3 = { alwaysinline mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { alwaysinline convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { alwaysinline convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!2 = !{i32 256, i32 1, i32 1}
!3 = !{!"amdgpu-synchronize-as", !"local"}
