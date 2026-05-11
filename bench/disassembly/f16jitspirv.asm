; SPIR-V
; Version: 1.6
; Generator: Khronos; 65535
; Bound: 13942
; Schema: 0
               OpCapability Kernel
               OpCapability Addresses
               OpCapability Float16Buffer
               OpCapability Int16
               OpCapability Vector16
               OpCapability Int8
               OpCapability Int64
               OpCapability Float16
               OpCapability Linkage
               OpExtension "SPV_KHR_no_integer_wrap_decoration"
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32 "matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32" %__shared_memory__ %97 %__shared_memory___0 %llvm_cmdline
               OpEntryPoint Kernel %matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32 "matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32" %__shared_memory__ %97 %__shared_memory___0 %llvm_cmdline
               OpEntryPoint Kernel %matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32 "matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32" %__shared_memory__ %97 %__shared_memory___0 %llvm_cmdline
               OpExecutionModeId %matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32 FPFastMathDefault %half %90
               OpExecutionModeId %matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32 FPFastMathDefault %float %90
               OpExecutionModeId %matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32 FPFastMathDefault %half %90
               OpExecutionModeId %matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32 FPFastMathDefault %float %90
               OpExecutionModeId %matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32 FPFastMathDefault %half %90
               OpExecutionModeId %matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32 FPFastMathDefault %float %90
               OpSource OpenCL_CPP 100000
               OpName %matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32 "matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32"
               OpName %llvm_cmdline "llvm.cmdline"
               OpName %spirv_llvm_amdgcn_workitem_id_x "spirv.llvm_amdgcn_workitem_id_x"
               OpName %spirv_llvm_amdgcn_workgroup_id_x "spirv.llvm_amdgcn_workgroup_id_x"
               OpName %__shared_memory__ "__shared_memory__"
               OpName %__shared_memory___0 "__shared_memory___0"
               OpName %spirv_llvm_amdgcn_s_barrier_signal "spirv.llvm_amdgcn_s_barrier_signal"
               OpName %spirv_llvm_amdgcn_s_barrier_wait "spirv.llvm_amdgcn_s_barrier_wait"
               OpName %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 "spirv.llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16"
               OpName %spirv_llvm_amdgcn_sched_barrier "spirv.llvm_amdgcn_sched_barrier"
               OpName %matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32 "matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32"
               OpName %matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32 "matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32"
               OpDecorate %122 Alignment 16
               OpDecorate %122 FuncParamAttr NoWrite
               OpDecorate %122 FuncParamAttr NoAlias
               OpDecorate %123 Alignment 16
               OpDecorate %123 FuncParamAttr NoWrite
               OpDecorate %123 FuncParamAttr NoAlias
               OpDecorate %124 Alignment 16
               OpDecorate %124 FuncParamAttr NoAlias
               OpDecorate %llvm_cmdline Constant
               OpDecorate %97 Constant
               OpDecorate %spirv_llvm_amdgcn_workitem_id_x LinkageAttributes "spirv.llvm_amdgcn_workitem_id_x" Import
               OpDecorate %spirv_llvm_amdgcn_workgroup_id_x LinkageAttributes "spirv.llvm_amdgcn_workgroup_id_x" Import
               OpDecorate %__shared_memory__ Alignment 16
               OpDecorate %__shared_memory___0 Alignment 16
               OpDecorate %spirv_llvm_amdgcn_s_barrier_signal LinkageAttributes "spirv.llvm_amdgcn_s_barrier_signal" Import
               OpDecorate %spirv_llvm_amdgcn_s_barrier_wait LinkageAttributes "spirv.llvm_amdgcn_s_barrier_wait" Import
               OpDecorate %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 LinkageAttributes "spirv.llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16" Import
               OpDecorate %spirv_llvm_amdgcn_sched_barrier LinkageAttributes "spirv.llvm_amdgcn_sched_barrier" Import
               OpDecorate %134 NoSignedWrap
               OpDecorate %139 NoSignedWrap
               OpDecorate %143 NoSignedWrap
               OpDecorate %147 NoSignedWrap
               OpDecorate %151 NoSignedWrap
               OpDecorate %155 NoSignedWrap
               OpDecorate %159 NoSignedWrap
               OpDecorate %163 NoSignedWrap
               OpDecorate %179 NoSignedWrap
               OpDecorate %181 NoSignedWrap
               OpDecorate %183 NoSignedWrap
               OpDecorate %185 NoSignedWrap
               OpDecorate %187 NoSignedWrap
               OpDecorate %189 NoSignedWrap
               OpDecorate %191 NoSignedWrap
               OpDecorate %193 NoSignedWrap
               OpDecorate %195 NoSignedWrap
               OpDecorate %197 NoSignedWrap
               OpDecorate %199 NoSignedWrap
               OpDecorate %201 NoSignedWrap
               OpDecorate %206 NoSignedWrap
               OpDecorate %258 NoSignedWrap
               OpDecorate %342 NoSignedWrap
               OpDecorate %343 NoSignedWrap
               OpDecorate %344 NoSignedWrap
               OpDecorate %346 NoSignedWrap
               OpDecorate %348 NoSignedWrap
               OpDecorate %350 NoSignedWrap
               OpDecorate %352 NoSignedWrap
               OpDecorate %354 NoSignedWrap
               OpDecorate %356 NoSignedWrap
               OpDecorate %358 NoSignedWrap
               OpDecorate %359 NoSignedWrap
               OpDecorate %360 NoSignedWrap
               OpDecorate %362 NoSignedWrap
               OpDecorate %364 NoSignedWrap
               OpDecorate %366 NoSignedWrap
               OpDecorate %368 NoSignedWrap
               OpDecorate %370 NoSignedWrap
               OpDecorate %372 NoSignedWrap
               OpDecorate %374 NoSignedWrap
               OpDecorate %376 NoSignedWrap
               OpDecorate %378 NoSignedWrap
               OpDecorate %380 NoSignedWrap
               OpDecorate %382 NoSignedWrap
               OpDecorate %384 NoSignedWrap
               OpDecorate %386 NoSignedWrap
               OpDecorate %388 NoSignedWrap
               OpDecorate %390 NoSignedWrap
               OpDecorate %392 NoSignedWrap
               OpDecorate %394 NoSignedWrap
               OpDecorate %396 NoSignedWrap
               OpDecorate %398 NoSignedWrap
               OpDecorate %400 NoSignedWrap
               OpDecorate %402 NoSignedWrap
               OpDecorate %404 NoSignedWrap
               OpDecorate %406 NoSignedWrap
               OpDecorate %408 NoSignedWrap
               OpDecorate %410 NoSignedWrap
               OpDecorate %412 NoSignedWrap
               OpDecorate %414 NoSignedWrap
               OpDecorate %416 NoSignedWrap
               OpDecorate %418 NoSignedWrap
               OpDecorate %420 NoSignedWrap
               OpDecorate %422 NoSignedWrap
               OpDecorate %424 NoSignedWrap
               OpDecorate %424 NoUnsignedWrap
               OpDecorate %426 NoSignedWrap
               OpDecorate %427 NoSignedWrap
               OpDecorate %432 NoSignedWrap
               OpDecorate %432 NoUnsignedWrap
               OpDecorate %434 NoSignedWrap
               OpDecorate %438 NoSignedWrap
               OpDecorate %438 NoUnsignedWrap
               OpDecorate %440 NoSignedWrap
               OpDecorate %445 NoSignedWrap
               OpDecorate %451 NoSignedWrap
               OpDecorate %452 NoSignedWrap
               OpDecorate %453 NoSignedWrap
               OpDecorate %453 NoUnsignedWrap
               OpDecorate %456 NoSignedWrap
               OpDecorate %457 NoSignedWrap
               OpDecorate %457 NoUnsignedWrap
               OpDecorate %459 NoSignedWrap
               OpDecorate %460 NoSignedWrap
               OpDecorate %460 NoUnsignedWrap
               OpDecorate %462 NoSignedWrap
               OpDecorate %463 NoSignedWrap
               OpDecorate %463 NoUnsignedWrap
               OpDecorate %465 NoSignedWrap
               OpDecorate %466 NoSignedWrap
               OpDecorate %466 NoUnsignedWrap
               OpDecorate %468 NoSignedWrap
               OpDecorate %469 NoSignedWrap
               OpDecorate %469 NoUnsignedWrap
               OpDecorate %471 NoSignedWrap
               OpDecorate %472 NoSignedWrap
               OpDecorate %472 NoUnsignedWrap
               OpDecorate %474 NoSignedWrap
               OpDecorate %475 NoSignedWrap
               OpDecorate %475 NoUnsignedWrap
               OpDecorate %1344 NoSignedWrap
               OpDecorate %1346 NoSignedWrap
               OpDecorate %1348 NoSignedWrap
               OpDecorate %1349 NoSignedWrap
               OpDecorate %1351 NoSignedWrap
               OpDecorate %1352 NoSignedWrap
               OpDecorate %1357 NoSignedWrap
               OpDecorate %1361 NoSignedWrap
               OpDecorate %1365 NoSignedWrap
               OpDecorate %1369 NoSignedWrap
               OpDecorate %1380 NoSignedWrap
               OpDecorate %1391 NoSignedWrap
               OpDecorate %1402 NoSignedWrap
               OpDecorate %1413 NoSignedWrap
               OpDecorate %1424 NoSignedWrap
               OpDecorate %1435 NoSignedWrap
               OpDecorate %1446 NoSignedWrap
               OpDecorate %1457 NoSignedWrap
               OpDecorate %1468 NoSignedWrap
               OpDecorate %1479 NoSignedWrap
               OpDecorate %1490 NoSignedWrap
               OpDecorate %1501 NoSignedWrap
               OpDecorate %1512 NoSignedWrap
               OpDecorate %1523 NoSignedWrap
               OpDecorate %1534 NoSignedWrap
               OpDecorate %1545 NoSignedWrap
               OpDecorate %1556 NoSignedWrap
               OpDecorate %1567 NoSignedWrap
               OpDecorate %1578 NoSignedWrap
               OpDecorate %1589 NoSignedWrap
               OpDecorate %1600 NoSignedWrap
               OpDecorate %1611 NoSignedWrap
               OpDecorate %1622 NoSignedWrap
               OpDecorate %1633 NoSignedWrap
               OpDecorate %1644 NoSignedWrap
               OpDecorate %1655 NoSignedWrap
               OpDecorate %1666 NoSignedWrap
               OpDecorate %1677 NoSignedWrap
               OpDecorate %1688 NoSignedWrap
               OpDecorate %1699 NoSignedWrap
               OpDecorate %2399 Alignment 16
               OpDecorate %2399 FuncParamAttr NoWrite
               OpDecorate %2399 FuncParamAttr NoAlias
               OpDecorate %2400 Alignment 16
               OpDecorate %2400 FuncParamAttr NoWrite
               OpDecorate %2400 FuncParamAttr NoAlias
               OpDecorate %2401 Alignment 16
               OpDecorate %2401 FuncParamAttr NoAlias
               OpDecorate %2411 NoSignedWrap
               OpDecorate %2416 NoSignedWrap
               OpDecorate %2420 NoSignedWrap
               OpDecorate %2424 NoSignedWrap
               OpDecorate %2428 NoSignedWrap
               OpDecorate %2432 NoSignedWrap
               OpDecorate %2436 NoSignedWrap
               OpDecorate %2440 NoSignedWrap
               OpDecorate %2456 NoSignedWrap
               OpDecorate %2458 NoSignedWrap
               OpDecorate %2460 NoSignedWrap
               OpDecorate %2462 NoSignedWrap
               OpDecorate %2464 NoSignedWrap
               OpDecorate %2466 NoSignedWrap
               OpDecorate %2468 NoSignedWrap
               OpDecorate %2470 NoSignedWrap
               OpDecorate %2472 NoSignedWrap
               OpDecorate %2474 NoSignedWrap
               OpDecorate %2476 NoSignedWrap
               OpDecorate %2478 NoSignedWrap
               OpDecorate %2483 NoSignedWrap
               OpDecorate %2535 NoSignedWrap
               OpDecorate %2619 NoSignedWrap
               OpDecorate %2620 NoSignedWrap
               OpDecorate %2621 NoSignedWrap
               OpDecorate %2623 NoSignedWrap
               OpDecorate %2625 NoSignedWrap
               OpDecorate %2627 NoSignedWrap
               OpDecorate %2629 NoSignedWrap
               OpDecorate %2631 NoSignedWrap
               OpDecorate %2633 NoSignedWrap
               OpDecorate %2635 NoSignedWrap
               OpDecorate %2636 NoSignedWrap
               OpDecorate %2637 NoSignedWrap
               OpDecorate %2639 NoSignedWrap
               OpDecorate %2641 NoSignedWrap
               OpDecorate %2643 NoSignedWrap
               OpDecorate %2645 NoSignedWrap
               OpDecorate %2647 NoSignedWrap
               OpDecorate %2649 NoSignedWrap
               OpDecorate %2651 NoSignedWrap
               OpDecorate %2653 NoSignedWrap
               OpDecorate %2655 NoSignedWrap
               OpDecorate %2657 NoSignedWrap
               OpDecorate %2659 NoSignedWrap
               OpDecorate %2661 NoSignedWrap
               OpDecorate %2663 NoSignedWrap
               OpDecorate %2665 NoSignedWrap
               OpDecorate %2667 NoSignedWrap
               OpDecorate %2669 NoSignedWrap
               OpDecorate %2671 NoSignedWrap
               OpDecorate %2673 NoSignedWrap
               OpDecorate %2675 NoSignedWrap
               OpDecorate %2677 NoSignedWrap
               OpDecorate %2679 NoSignedWrap
               OpDecorate %2681 NoSignedWrap
               OpDecorate %2683 NoSignedWrap
               OpDecorate %2685 NoSignedWrap
               OpDecorate %2687 NoSignedWrap
               OpDecorate %2689 NoSignedWrap
               OpDecorate %2691 NoSignedWrap
               OpDecorate %2693 NoSignedWrap
               OpDecorate %2695 NoSignedWrap
               OpDecorate %2697 NoSignedWrap
               OpDecorate %2699 NoSignedWrap
               OpDecorate %2701 NoSignedWrap
               OpDecorate %2701 NoUnsignedWrap
               OpDecorate %2703 NoSignedWrap
               OpDecorate %2704 NoSignedWrap
               OpDecorate %2709 NoSignedWrap
               OpDecorate %2709 NoUnsignedWrap
               OpDecorate %2711 NoSignedWrap
               OpDecorate %2715 NoSignedWrap
               OpDecorate %2715 NoUnsignedWrap
               OpDecorate %2717 NoSignedWrap
               OpDecorate %2722 NoSignedWrap
               OpDecorate %2728 NoSignedWrap
               OpDecorate %2729 NoSignedWrap
               OpDecorate %2730 NoSignedWrap
               OpDecorate %2730 NoUnsignedWrap
               OpDecorate %2733 NoSignedWrap
               OpDecorate %2734 NoSignedWrap
               OpDecorate %2734 NoUnsignedWrap
               OpDecorate %2736 NoSignedWrap
               OpDecorate %2737 NoSignedWrap
               OpDecorate %2737 NoUnsignedWrap
               OpDecorate %2739 NoSignedWrap
               OpDecorate %2740 NoSignedWrap
               OpDecorate %2740 NoUnsignedWrap
               OpDecorate %2742 NoSignedWrap
               OpDecorate %2743 NoSignedWrap
               OpDecorate %2743 NoUnsignedWrap
               OpDecorate %2745 NoSignedWrap
               OpDecorate %2746 NoSignedWrap
               OpDecorate %2746 NoUnsignedWrap
               OpDecorate %2748 NoSignedWrap
               OpDecorate %2749 NoSignedWrap
               OpDecorate %2749 NoUnsignedWrap
               OpDecorate %2751 NoSignedWrap
               OpDecorate %2752 NoSignedWrap
               OpDecorate %2752 NoUnsignedWrap
               OpDecorate %3621 NoSignedWrap
               OpDecorate %3623 NoSignedWrap
               OpDecorate %3625 NoSignedWrap
               OpDecorate %3626 NoSignedWrap
               OpDecorate %3628 NoSignedWrap
               OpDecorate %3629 NoSignedWrap
               OpDecorate %3634 NoSignedWrap
               OpDecorate %3638 NoSignedWrap
               OpDecorate %3642 NoSignedWrap
               OpDecorate %3646 NoSignedWrap
               OpDecorate %3657 NoSignedWrap
               OpDecorate %3668 NoSignedWrap
               OpDecorate %3679 NoSignedWrap
               OpDecorate %3690 NoSignedWrap
               OpDecorate %3701 NoSignedWrap
               OpDecorate %3712 NoSignedWrap
               OpDecorate %3723 NoSignedWrap
               OpDecorate %3734 NoSignedWrap
               OpDecorate %3745 NoSignedWrap
               OpDecorate %3756 NoSignedWrap
               OpDecorate %3767 NoSignedWrap
               OpDecorate %3778 NoSignedWrap
               OpDecorate %3789 NoSignedWrap
               OpDecorate %3800 NoSignedWrap
               OpDecorate %3811 NoSignedWrap
               OpDecorate %3822 NoSignedWrap
               OpDecorate %3833 NoSignedWrap
               OpDecorate %3844 NoSignedWrap
               OpDecorate %3855 NoSignedWrap
               OpDecorate %3866 NoSignedWrap
               OpDecorate %3877 NoSignedWrap
               OpDecorate %3888 NoSignedWrap
               OpDecorate %3899 NoSignedWrap
               OpDecorate %3910 NoSignedWrap
               OpDecorate %3921 NoSignedWrap
               OpDecorate %3932 NoSignedWrap
               OpDecorate %3943 NoSignedWrap
               OpDecorate %3954 NoSignedWrap
               OpDecorate %3965 NoSignedWrap
               OpDecorate %3976 NoSignedWrap
               OpDecorate %4676 Alignment 16
               OpDecorate %4676 FuncParamAttr NoWrite
               OpDecorate %4676 FuncParamAttr NoAlias
               OpDecorate %4677 Alignment 16
               OpDecorate %4677 FuncParamAttr NoWrite
               OpDecorate %4677 FuncParamAttr NoAlias
               OpDecorate %4678 Alignment 16
               OpDecorate %4678 FuncParamAttr NoAlias
               OpDecorate %4688 NoSignedWrap
               OpDecorate %4693 NoSignedWrap
               OpDecorate %4697 NoSignedWrap
               OpDecorate %4701 NoSignedWrap
               OpDecorate %4705 NoSignedWrap
               OpDecorate %4709 NoSignedWrap
               OpDecorate %4713 NoSignedWrap
               OpDecorate %4717 NoSignedWrap
               OpDecorate %4733 NoSignedWrap
               OpDecorate %4735 NoSignedWrap
               OpDecorate %4737 NoSignedWrap
               OpDecorate %4739 NoSignedWrap
               OpDecorate %4741 NoSignedWrap
               OpDecorate %4743 NoSignedWrap
               OpDecorate %4745 NoSignedWrap
               OpDecorate %4747 NoSignedWrap
               OpDecorate %4749 NoSignedWrap
               OpDecorate %4751 NoSignedWrap
               OpDecorate %4753 NoSignedWrap
               OpDecorate %4755 NoSignedWrap
               OpDecorate %4760 NoSignedWrap
               OpDecorate %4812 NoSignedWrap
               OpDecorate %4896 NoSignedWrap
               OpDecorate %4897 NoSignedWrap
               OpDecorate %4898 NoSignedWrap
               OpDecorate %4900 NoSignedWrap
               OpDecorate %4902 NoSignedWrap
               OpDecorate %4904 NoSignedWrap
               OpDecorate %4906 NoSignedWrap
               OpDecorate %4908 NoSignedWrap
               OpDecorate %4910 NoSignedWrap
               OpDecorate %4912 NoSignedWrap
               OpDecorate %4913 NoSignedWrap
               OpDecorate %4914 NoSignedWrap
               OpDecorate %4916 NoSignedWrap
               OpDecorate %4918 NoSignedWrap
               OpDecorate %4920 NoSignedWrap
               OpDecorate %4922 NoSignedWrap
               OpDecorate %4924 NoSignedWrap
               OpDecorate %4926 NoSignedWrap
               OpDecorate %4928 NoSignedWrap
               OpDecorate %4930 NoSignedWrap
               OpDecorate %4932 NoSignedWrap
               OpDecorate %4934 NoSignedWrap
               OpDecorate %4936 NoSignedWrap
               OpDecorate %4938 NoSignedWrap
               OpDecorate %4940 NoSignedWrap
               OpDecorate %4942 NoSignedWrap
               OpDecorate %4944 NoSignedWrap
               OpDecorate %4946 NoSignedWrap
               OpDecorate %4948 NoSignedWrap
               OpDecorate %4950 NoSignedWrap
               OpDecorate %4952 NoSignedWrap
               OpDecorate %4954 NoSignedWrap
               OpDecorate %4956 NoSignedWrap
               OpDecorate %4958 NoSignedWrap
               OpDecorate %4960 NoSignedWrap
               OpDecorate %4962 NoSignedWrap
               OpDecorate %4964 NoSignedWrap
               OpDecorate %4966 NoSignedWrap
               OpDecorate %4968 NoSignedWrap
               OpDecorate %4970 NoSignedWrap
               OpDecorate %4972 NoSignedWrap
               OpDecorate %4974 NoSignedWrap
               OpDecorate %4976 NoSignedWrap
               OpDecorate %4978 NoSignedWrap
               OpDecorate %4978 NoUnsignedWrap
               OpDecorate %4980 NoSignedWrap
               OpDecorate %4981 NoSignedWrap
               OpDecorate %4986 NoSignedWrap
               OpDecorate %4986 NoUnsignedWrap
               OpDecorate %4988 NoSignedWrap
               OpDecorate %4992 NoSignedWrap
               OpDecorate %4992 NoUnsignedWrap
               OpDecorate %4994 NoSignedWrap
               OpDecorate %4999 NoSignedWrap
               OpDecorate %5005 NoSignedWrap
               OpDecorate %5006 NoSignedWrap
               OpDecorate %5007 NoSignedWrap
               OpDecorate %5007 NoUnsignedWrap
               OpDecorate %5010 NoSignedWrap
               OpDecorate %5011 NoSignedWrap
               OpDecorate %5011 NoUnsignedWrap
               OpDecorate %5013 NoSignedWrap
               OpDecorate %5014 NoSignedWrap
               OpDecorate %5014 NoUnsignedWrap
               OpDecorate %5016 NoSignedWrap
               OpDecorate %5017 NoSignedWrap
               OpDecorate %5017 NoUnsignedWrap
               OpDecorate %5019 NoSignedWrap
               OpDecorate %5020 NoSignedWrap
               OpDecorate %5020 NoUnsignedWrap
               OpDecorate %5022 NoSignedWrap
               OpDecorate %5023 NoSignedWrap
               OpDecorate %5023 NoUnsignedWrap
               OpDecorate %5025 NoSignedWrap
               OpDecorate %5026 NoSignedWrap
               OpDecorate %5026 NoUnsignedWrap
               OpDecorate %5028 NoSignedWrap
               OpDecorate %5029 NoSignedWrap
               OpDecorate %5029 NoUnsignedWrap
               OpDecorate %5898 NoSignedWrap
               OpDecorate %5900 NoSignedWrap
               OpDecorate %5902 NoSignedWrap
               OpDecorate %5903 NoSignedWrap
               OpDecorate %5905 NoSignedWrap
               OpDecorate %5906 NoSignedWrap
               OpDecorate %5911 NoSignedWrap
               OpDecorate %5915 NoSignedWrap
               OpDecorate %5919 NoSignedWrap
               OpDecorate %5923 NoSignedWrap
               OpDecorate %5934 NoSignedWrap
               OpDecorate %5945 NoSignedWrap
               OpDecorate %5956 NoSignedWrap
               OpDecorate %5967 NoSignedWrap
               OpDecorate %5978 NoSignedWrap
               OpDecorate %5989 NoSignedWrap
               OpDecorate %6000 NoSignedWrap
               OpDecorate %6011 NoSignedWrap
               OpDecorate %6022 NoSignedWrap
               OpDecorate %6033 NoSignedWrap
               OpDecorate %6044 NoSignedWrap
               OpDecorate %6055 NoSignedWrap
               OpDecorate %6066 NoSignedWrap
               OpDecorate %6077 NoSignedWrap
               OpDecorate %6088 NoSignedWrap
               OpDecorate %6099 NoSignedWrap
               OpDecorate %6110 NoSignedWrap
               OpDecorate %6121 NoSignedWrap
               OpDecorate %6132 NoSignedWrap
               OpDecorate %6143 NoSignedWrap
               OpDecorate %6154 NoSignedWrap
               OpDecorate %6165 NoSignedWrap
               OpDecorate %6176 NoSignedWrap
               OpDecorate %6187 NoSignedWrap
               OpDecorate %6198 NoSignedWrap
               OpDecorate %6209 NoSignedWrap
               OpDecorate %6220 NoSignedWrap
               OpDecorate %6231 NoSignedWrap
               OpDecorate %6242 NoSignedWrap
               OpDecorate %6253 NoSignedWrap
       %half = OpTypeFloat 16
%_ptr_CrossWorkgroup_half = OpTypePointer CrossWorkgroup %half
      %float = OpTypeFloat 32
%_ptr_CrossWorkgroup_float = OpTypePointer CrossWorkgroup %float
       %void = OpTypeVoid
          %7 = OpTypeFunction %void %_ptr_CrossWorkgroup_half %_ptr_CrossWorkgroup_half %_ptr_CrossWorkgroup_float
       %uint = OpTypeInt 32 0
          %9 = OpTypeFunction %uint
         %10 = OpTypeFunction %void %uint
     %ushort = OpTypeInt 16 0
         %12 = OpTypeFunction %void %ushort
    %v8float = OpTypeVector %float 8
     %v8half = OpTypeVector %half 8
         %15 = OpTypeFunction %v8float %v8half %v8half %v8float
%_ptr_CrossWorkgroup_v8half = OpTypePointer CrossWorkgroup %v8half
%_ptr_Workgroup_half = OpTypePointer Workgroup %half
%_ptr_Workgroup_v8half = OpTypePointer Workgroup %v8half
      %uchar = OpTypeInt 8 0
%_ptr_CrossWorkgroup_uchar = OpTypePointer CrossWorkgroup %uchar
      %ulong = OpTypeInt 64 0
       %bool = OpTypeBool
     %uint_4 = OpConstant %uint 4
%_arr_uchar_uint_4 = OpTypeArray %uchar %uint_4
   %uint_132 = OpConstant %uint 132
%_arr_half_uint_132 = OpTypeArray %half %uint_132
    %uint_64 = OpConstant %uint 64
%_arr__arr_half_uint_132_uint_64 = OpTypeArray %_arr_half_uint_132 %uint_64
%_ptr_Workgroup__arr__arr_half_uint_132_uint_64 = OpTypePointer Workgroup %_arr__arr_half_uint_132_uint_64
    %uint_68 = OpConstant %uint 68
%_arr_half_uint_68 = OpTypeArray %half %uint_68
   %uint_256 = OpConstant %uint 256
%_arr__arr_half_uint_68_uint_256 = OpTypeArray %_arr_half_uint_68 %uint_256
%_ptr_Workgroup__arr__arr_half_uint_68_uint_256 = OpTypePointer Workgroup %_arr__arr_half_uint_68_uint_256
%_ptr_CrossWorkgroup_uint = OpTypePointer CrossWorkgroup %uint
%_ptr_CrossWorkgroup__arr_uchar_uint_4 = OpTypePointer CrossWorkgroup %_arr_uchar_uint_4
    %ulong_7 = OpConstant %ulong 7
         %38 = OpConstantNull %float
%ushort_65535 = OpConstant %ushort 65535
%uint_4294967295 = OpConstant %uint 4294967295
   %uint_124 = OpConstant %uint 124
  %ulong_128 = OpConstant %ulong 128
  %ulong_256 = OpConstant %ulong 256
  %ulong_512 = OpConstant %ulong 512
%ulong_262144 = OpConstant %ulong 262144
    %ulong_1 = OpConstant %ulong 1
   %ulong_12 = OpConstant %ulong 12
    %ulong_4 = OpConstant %ulong 4
  %ulong_768 = OpConstant %ulong 768
    %uint_55 = OpConstant %uint 55
    %uint_54 = OpConstant %uint 54
    %uint_53 = OpConstant %uint 53
    %uint_52 = OpConstant %uint 52
    %uint_51 = OpConstant %uint 51
    %uint_50 = OpConstant %uint 50
    %uint_49 = OpConstant %uint 49
    %uint_39 = OpConstant %uint 39
    %uint_38 = OpConstant %uint 38
    %uint_37 = OpConstant %uint 37
    %uint_36 = OpConstant %uint 36
    %uint_35 = OpConstant %uint 35
    %uint_34 = OpConstant %uint 34
    %uint_33 = OpConstant %uint 33
    %uint_23 = OpConstant %uint 23
    %uint_22 = OpConstant %uint 22
    %uint_21 = OpConstant %uint 21
    %uint_20 = OpConstant %uint 20
    %uint_19 = OpConstant %uint 19
    %uint_18 = OpConstant %uint 18
    %uint_17 = OpConstant %uint 17
     %uint_7 = OpConstant %uint 7
     %uint_6 = OpConstant %uint 6
     %uint_5 = OpConstant %uint 5
     %uint_3 = OpConstant %uint 3
     %uint_1 = OpConstant %uint 1
    %uint_48 = OpConstant %uint 48
  %ulong_132 = OpConstant %ulong 132
   %ulong_68 = OpConstant %ulong 68
   %uint_128 = OpConstant %uint 128
 %ulong_2048 = OpConstant %ulong 2048
  %uint_1792 = OpConstant %uint 1792
  %uint_1536 = OpConstant %uint 1536
  %uint_1280 = OpConstant %uint 1280
  %uint_1024 = OpConstant %uint 1024
   %uint_768 = OpConstant %uint 768
   %uint_512 = OpConstant %uint 512
     %uint_8 = OpConstant %uint 8
    %uint_16 = OpConstant %uint 16
    %uint_32 = OpConstant %uint 32
         %90 = OpConstantNull %uint
         %91 = OpConstantNull %uchar
   %uchar_51 = OpConstant %uchar 51
   %uchar_79 = OpConstant %uchar 79
   %uchar_45 = OpConstant %uchar 45
         %95 = OpConstantComposite %_arr_uchar_uint_4 %uchar_45 %uchar_79 %uchar_51 %91
%llvm_cmdline = OpVariable %_ptr_CrossWorkgroup__arr_uchar_uint_4 CrossWorkgroup %95
         %97 = OpVariable %_ptr_CrossWorkgroup_uint CrossWorkgroup %90
%__shared_memory__ = OpVariable %_ptr_Workgroup__arr__arr_half_uint_68_uint_256 Workgroup
%__shared_memory___0 = OpVariable %_ptr_Workgroup__arr__arr_half_uint_132_uint_64 Workgroup
        %106 = OpUndef %v8half
        %113 = OpConstantComposite %v8float %38 %38 %38 %38 %38 %38 %38 %38
     %uint_2 = OpConstant %uint 2
   %uint_252 = OpConstant %uint 252
   %ulong_13 = OpConstant %ulong 13
%ulong_131072 = OpConstant %ulong 131072
   %ulong_11 = OpConstant %ulong 11
 %ulong_1024 = OpConstant %ulong 1024
 %ulong_4096 = OpConstant %ulong 4096
%ulong_524288 = OpConstant %ulong 524288
%spirv_llvm_amdgcn_workitem_id_x = OpFunction %uint None %9
               OpFunctionEnd
%spirv_llvm_amdgcn_workgroup_id_x = OpFunction %uint None %9
               OpFunctionEnd
%spirv_llvm_amdgcn_s_barrier_signal = OpFunction %void None %10
        %103 = OpFunctionParameter %uint
               OpFunctionEnd
%spirv_llvm_amdgcn_s_barrier_wait = OpFunction %void None %12
        %105 = OpFunctionParameter %ushort
               OpFunctionEnd
%spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 = OpFunction %v8float None %15
        %108 = OpFunctionParameter %v8half
        %109 = OpFunctionParameter %v8half
        %110 = OpFunctionParameter %v8float
               OpFunctionEnd
%spirv_llvm_amdgcn_sched_barrier = OpFunction %void None %10
        %112 = OpFunctionParameter %uint
               OpFunctionEnd
%matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32 = OpFunction %void Inline %7
        %122 = OpFunctionParameter %_ptr_CrossWorkgroup_half
        %123 = OpFunctionParameter %_ptr_CrossWorkgroup_half
        %124 = OpFunctionParameter %_ptr_CrossWorkgroup_float
       %6953 = OpLabel
        %126 = OpFunctionCall %uint %spirv_llvm_amdgcn_workitem_id_x
        %127 = OpSConvert %ulong %126
        %128 = OpUDiv %uint %126 %uint_64
        %129 = OpUMod %uint %126 %uint_64
        %130 = OpUDiv %uint %129 %uint_32
        %131 = OpUMod %uint %126 %uint_32
        %132 = OpUDiv %uint %131 %uint_16
        %133 = OpUMod %uint %131 %uint_16
        %134 = OpIMul %uint %132 %uint_8
        %135 = OpUConvert %ulong %134
        %136 = OpUDiv %uint %126 %uint_8
        %137 = OpUConvert %ulong %136
        %138 = OpUMod %uint %126 %uint_8
        %139 = OpIAdd %uint %126 %uint_256
        %140 = OpUDiv %uint %139 %uint_8
        %141 = OpUConvert %ulong %140
        %142 = OpUMod %uint %139 %uint_8
        %143 = OpIAdd %uint %126 %uint_512
        %144 = OpUDiv %uint %143 %uint_8
        %145 = OpUConvert %ulong %144
        %146 = OpUMod %uint %143 %uint_8
        %147 = OpIAdd %uint %126 %uint_768
        %148 = OpUDiv %uint %147 %uint_8
        %149 = OpUConvert %ulong %148
        %150 = OpUMod %uint %147 %uint_8
        %151 = OpIAdd %uint %126 %uint_1024
        %152 = OpUDiv %uint %151 %uint_8
        %153 = OpUConvert %ulong %152
        %154 = OpUMod %uint %151 %uint_8
        %155 = OpIAdd %uint %126 %uint_1280
        %156 = OpUDiv %uint %155 %uint_8
        %157 = OpUConvert %ulong %156
        %158 = OpUMod %uint %155 %uint_8
        %159 = OpIAdd %uint %126 %uint_1536
        %160 = OpUDiv %uint %159 %uint_8
        %161 = OpUConvert %ulong %160
        %162 = OpUMod %uint %159 %uint_8
        %163 = OpIAdd %uint %126 %uint_1792
        %164 = OpUDiv %uint %163 %uint_8
        %165 = OpUConvert %ulong %164
        %166 = OpUMod %uint %163 %uint_8
        %167 = OpUDiv %uint %126 %uint_16
        %168 = OpUConvert %ulong %167
        %169 = OpUMod %uint %126 %uint_16
        %170 = OpUDiv %uint %139 %uint_16
        %171 = OpUConvert %ulong %170
        %172 = OpUMod %uint %139 %uint_16
        %173 = OpUDiv %uint %143 %uint_16
        %174 = OpUConvert %ulong %173
        %175 = OpUMod %uint %143 %uint_16
        %176 = OpUDiv %uint %147 %uint_16
        %177 = OpUConvert %ulong %176
        %178 = OpUMod %uint %147 %uint_16
        %179 = OpIMul %uint %138 %uint_8
        %180 = OpUConvert %ulong %179
        %181 = OpIMul %uint %142 %uint_8
        %182 = OpUConvert %ulong %181
        %183 = OpIMul %uint %146 %uint_8
        %184 = OpUConvert %ulong %183
        %185 = OpIMul %uint %150 %uint_8
        %186 = OpUConvert %ulong %185
        %187 = OpIMul %uint %154 %uint_8
        %188 = OpUConvert %ulong %187
        %189 = OpIMul %uint %158 %uint_8
        %190 = OpUConvert %ulong %189
        %191 = OpIMul %uint %162 %uint_8
        %192 = OpUConvert %ulong %191
        %193 = OpIMul %uint %166 %uint_8
        %194 = OpUConvert %ulong %193
        %195 = OpIMul %uint %169 %uint_8
        %196 = OpUConvert %ulong %195
        %197 = OpIMul %uint %172 %uint_8
        %198 = OpUConvert %ulong %197
        %199 = OpIMul %uint %175 %uint_8
        %200 = OpUConvert %ulong %199
        %201 = OpIMul %uint %178 %uint_8
        %202 = OpUConvert %ulong %201
        %203 = OpFunctionCall %uint %spirv_llvm_amdgcn_workgroup_id_x
        %204 = OpUDiv %uint %203 %uint_16
        %205 = OpUMod %uint %203 %uint_16
        %206 = OpIMul %uint %204 %uint_256
        %207 = OpIAdd %uint %136 %206
        %208 = OpUConvert %ulong %207
        %209 = OpIAdd %uint %140 %206
        %210 = OpUConvert %ulong %209
        %211 = OpIMul %ulong %208 %ulong_2048
        %212 = OpIAdd %ulong %211 %180
        %213 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %122 %212
        %214 = OpBitcast %_ptr_CrossWorkgroup_v8half %213
        %215 = OpLoad %v8half %214 Aligned 2
        %216 = OpIAdd %uint %144 %206
        %217 = OpUConvert %ulong %216
        %218 = OpIMul %ulong %210 %ulong_2048
        %219 = OpIAdd %ulong %218 %182
        %220 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %122 %219
        %221 = OpBitcast %_ptr_CrossWorkgroup_v8half %220
        %222 = OpLoad %v8half %221 Aligned 2
        %223 = OpIAdd %uint %148 %206
        %224 = OpUConvert %ulong %223
        %225 = OpIMul %ulong %217 %ulong_2048
        %226 = OpIAdd %ulong %225 %184
        %227 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %122 %226
        %228 = OpBitcast %_ptr_CrossWorkgroup_v8half %227
        %229 = OpLoad %v8half %228 Aligned 2
        %230 = OpIAdd %uint %152 %206
        %231 = OpUConvert %ulong %230
        %232 = OpIMul %ulong %224 %ulong_2048
        %233 = OpIAdd %ulong %232 %186
        %234 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %122 %233
        %235 = OpBitcast %_ptr_CrossWorkgroup_v8half %234
        %236 = OpLoad %v8half %235 Aligned 2
        %237 = OpIAdd %uint %156 %206
        %238 = OpUConvert %ulong %237
        %239 = OpIMul %ulong %231 %ulong_2048
        %240 = OpIAdd %ulong %239 %188
        %241 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %122 %240
        %242 = OpBitcast %_ptr_CrossWorkgroup_v8half %241
        %243 = OpLoad %v8half %242 Aligned 2
        %244 = OpIAdd %uint %160 %206
        %245 = OpUConvert %ulong %244
        %246 = OpIMul %ulong %238 %ulong_2048
        %247 = OpIAdd %ulong %246 %190
        %248 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %122 %247
        %249 = OpBitcast %_ptr_CrossWorkgroup_v8half %248
        %250 = OpLoad %v8half %249 Aligned 2
        %251 = OpIAdd %uint %164 %206
        %252 = OpUConvert %ulong %251
        %253 = OpIMul %ulong %245 %ulong_2048
        %254 = OpIAdd %ulong %253 %192
        %255 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %122 %254
        %256 = OpBitcast %_ptr_CrossWorkgroup_v8half %255
        %257 = OpLoad %v8half %256 Aligned 2
        %258 = OpIMul %uint %205 %uint_128
        %259 = OpIAdd %uint %195 %258
        %260 = OpUConvert %ulong %259
        %261 = OpIMul %ulong %252 %ulong_2048
        %262 = OpIAdd %ulong %261 %194
        %263 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %122 %262
        %264 = OpBitcast %_ptr_CrossWorkgroup_v8half %263
        %265 = OpLoad %v8half %264 Aligned 2
        %266 = OpIAdd %uint %197 %258
        %267 = OpUConvert %ulong %266
        %268 = OpIMul %ulong %168 %ulong_2048
        %269 = OpIAdd %ulong %268 %260
        %270 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %123 %269
        %271 = OpBitcast %_ptr_CrossWorkgroup_v8half %270
        %272 = OpLoad %v8half %271 Aligned 2
        %273 = OpIAdd %uint %199 %258
        %274 = OpUConvert %ulong %273
        %275 = OpIMul %ulong %171 %ulong_2048
        %276 = OpIAdd %ulong %275 %267
        %277 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %123 %276
        %278 = OpBitcast %_ptr_CrossWorkgroup_v8half %277
        %279 = OpLoad %v8half %278 Aligned 2
        %280 = OpIAdd %uint %201 %258
        %281 = OpUConvert %ulong %280
        %282 = OpIMul %ulong %174 %ulong_2048
        %283 = OpIAdd %ulong %282 %274
        %284 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %123 %283
        %285 = OpBitcast %_ptr_CrossWorkgroup_v8half %284
        %286 = OpLoad %v8half %285 Aligned 2
        %287 = OpIMul %ulong %177 %ulong_2048
        %288 = OpIAdd %ulong %287 %281
        %289 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %123 %288
        %290 = OpBitcast %_ptr_CrossWorkgroup_v8half %289
        %291 = OpLoad %v8half %290 Aligned 2
        %292 = OpIMul %ulong %137 %ulong_68
        %293 = OpIAdd %ulong %292 %180
        %294 = OpBitcast %_ptr_Workgroup_half %__shared_memory__
        %295 = OpPtrAccessChain %_ptr_Workgroup_half %294 %293
        %296 = OpBitcast %_ptr_Workgroup_v8half %295
               OpStore %296 %215 Aligned 2
        %297 = OpIMul %ulong %141 %ulong_68
        %298 = OpIAdd %ulong %297 %182
        %299 = OpPtrAccessChain %_ptr_Workgroup_half %294 %298
        %300 = OpBitcast %_ptr_Workgroup_v8half %299
               OpStore %300 %222 Aligned 2
        %301 = OpIMul %ulong %145 %ulong_68
        %302 = OpIAdd %ulong %301 %184
        %303 = OpPtrAccessChain %_ptr_Workgroup_half %294 %302
        %304 = OpBitcast %_ptr_Workgroup_v8half %303
               OpStore %304 %229 Aligned 2
        %305 = OpIMul %ulong %149 %ulong_68
        %306 = OpIAdd %ulong %305 %186
        %307 = OpPtrAccessChain %_ptr_Workgroup_half %294 %306
        %308 = OpBitcast %_ptr_Workgroup_v8half %307
               OpStore %308 %236 Aligned 2
        %309 = OpIMul %ulong %153 %ulong_68
        %310 = OpIAdd %ulong %309 %188
        %311 = OpPtrAccessChain %_ptr_Workgroup_half %294 %310
        %312 = OpBitcast %_ptr_Workgroup_v8half %311
               OpStore %312 %243 Aligned 2
        %313 = OpIMul %ulong %157 %ulong_68
        %314 = OpIAdd %ulong %313 %190
        %315 = OpPtrAccessChain %_ptr_Workgroup_half %294 %314
        %316 = OpBitcast %_ptr_Workgroup_v8half %315
               OpStore %316 %250 Aligned 2
        %317 = OpIMul %ulong %161 %ulong_68
        %318 = OpIAdd %ulong %317 %192
        %319 = OpPtrAccessChain %_ptr_Workgroup_half %294 %318
        %320 = OpBitcast %_ptr_Workgroup_v8half %319
               OpStore %320 %257 Aligned 2
        %321 = OpIMul %ulong %165 %ulong_68
        %322 = OpIAdd %ulong %321 %194
        %323 = OpPtrAccessChain %_ptr_Workgroup_half %294 %322
        %324 = OpBitcast %_ptr_Workgroup_v8half %323
               OpStore %324 %265 Aligned 2
        %325 = OpIMul %ulong %168 %ulong_132
        %326 = OpIAdd %ulong %325 %196
        %327 = OpBitcast %_ptr_Workgroup_half %__shared_memory___0
        %328 = OpPtrAccessChain %_ptr_Workgroup_half %327 %326
        %329 = OpBitcast %_ptr_Workgroup_v8half %328
               OpStore %329 %272 Aligned 2
        %330 = OpIMul %ulong %171 %ulong_132
        %331 = OpIAdd %ulong %330 %198
        %332 = OpPtrAccessChain %_ptr_Workgroup_half %327 %331
        %333 = OpBitcast %_ptr_Workgroup_v8half %332
               OpStore %333 %279 Aligned 2
        %334 = OpIMul %ulong %174 %ulong_132
        %335 = OpIAdd %ulong %334 %200
        %336 = OpPtrAccessChain %_ptr_Workgroup_half %327 %335
        %337 = OpBitcast %_ptr_Workgroup_v8half %336
               OpStore %337 %286 Aligned 2
        %338 = OpIMul %ulong %177 %ulong_132
        %339 = OpIAdd %ulong %338 %202
        %340 = OpPtrAccessChain %_ptr_Workgroup_half %327 %339
        %341 = OpBitcast %_ptr_Workgroup_v8half %340
               OpStore %341 %291 Aligned 2
        %342 = OpIMul %uint %128 %uint_4
        %343 = OpIMul %uint %128 %uint_64
        %344 = OpIAdd %uint %343 %133
        %345 = OpUConvert %ulong %344
        %346 = OpIAdd %uint %134 %uint_16
        %347 = OpUConvert %ulong %346
        %348 = OpIAdd %uint %134 %uint_32
        %349 = OpUConvert %ulong %348
        %350 = OpIAdd %uint %134 %uint_48
        %351 = OpUConvert %ulong %350
        %352 = OpIAdd %uint %344 %uint_16
        %353 = OpUConvert %ulong %352
        %354 = OpIAdd %uint %344 %uint_32
        %355 = OpUConvert %ulong %354
        %356 = OpIAdd %uint %344 %uint_48
        %357 = OpUConvert %ulong %356
        %358 = OpIMul %uint %130 %uint_4
        %359 = OpIMul %uint %130 %uint_64
        %360 = OpIAdd %uint %359 %133
        %361 = OpUConvert %ulong %360
        %362 = OpIAdd %uint %360 %uint_16
        %363 = OpUConvert %ulong %362
        %364 = OpIAdd %uint %360 %uint_32
        %365 = OpUConvert %ulong %364
        %366 = OpIAdd %uint %360 %uint_48
        %367 = OpUConvert %ulong %366
        %368 = OpIAdd %uint %134 %uint_1
        %369 = OpUConvert %ulong %368
        %370 = OpIAdd %uint %134 %uint_2
        %371 = OpUConvert %ulong %370
        %372 = OpIAdd %uint %134 %uint_3
        %373 = OpUConvert %ulong %372
        %374 = OpIAdd %uint %134 %uint_4
        %375 = OpUConvert %ulong %374
        %376 = OpIAdd %uint %134 %uint_5
        %377 = OpUConvert %ulong %376
        %378 = OpIAdd %uint %134 %uint_6
        %379 = OpUConvert %ulong %378
        %380 = OpIAdd %uint %134 %uint_7
        %381 = OpUConvert %ulong %380
        %382 = OpIAdd %uint %134 %uint_17
        %383 = OpUConvert %ulong %382
        %384 = OpIAdd %uint %134 %uint_18
        %385 = OpUConvert %ulong %384
        %386 = OpIAdd %uint %134 %uint_19
        %387 = OpUConvert %ulong %386
        %388 = OpIAdd %uint %134 %uint_20
        %389 = OpUConvert %ulong %388
        %390 = OpIAdd %uint %134 %uint_21
        %391 = OpUConvert %ulong %390
        %392 = OpIAdd %uint %134 %uint_22
        %393 = OpUConvert %ulong %392
        %394 = OpIAdd %uint %134 %uint_23
        %395 = OpUConvert %ulong %394
        %396 = OpIAdd %uint %134 %uint_33
        %397 = OpUConvert %ulong %396
        %398 = OpIAdd %uint %134 %uint_34
        %399 = OpUConvert %ulong %398
        %400 = OpIAdd %uint %134 %uint_35
        %401 = OpUConvert %ulong %400
        %402 = OpIAdd %uint %134 %uint_36
        %403 = OpUConvert %ulong %402
        %404 = OpIAdd %uint %134 %uint_37
        %405 = OpUConvert %ulong %404
        %406 = OpIAdd %uint %134 %uint_38
        %407 = OpUConvert %ulong %406
        %408 = OpIAdd %uint %134 %uint_39
        %409 = OpUConvert %ulong %408
        %410 = OpIAdd %uint %134 %uint_49
        %411 = OpUConvert %ulong %410
        %412 = OpIAdd %uint %134 %uint_50
        %413 = OpUConvert %ulong %412
        %414 = OpIAdd %uint %134 %uint_51
        %415 = OpUConvert %ulong %414
        %416 = OpIAdd %uint %134 %uint_52
        %417 = OpUConvert %ulong %416
        %418 = OpIAdd %uint %134 %uint_53
        %419 = OpUConvert %ulong %418
        %420 = OpIAdd %uint %134 %uint_54
        %421 = OpUConvert %ulong %420
        %422 = OpIAdd %uint %134 %uint_55
        %423 = OpUConvert %ulong %422
        %424 = OpIAdd %ulong %127 %ulong_768
        %425 = OpShiftRightLogical %ulong %424 %ulong_4
        %426 = OpShiftLeftLogical %ulong %425 %ulong_12
        %427 = OpShiftLeftLogical %ulong %281 %ulong_1
        %428 = OpIAdd %ulong %426 %427
        %429 = OpIAdd %ulong %428 %ulong_262144
        %430 = OpBitcast %_ptr_CrossWorkgroup_uchar %123
        %431 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %430 %429
        %432 = OpIAdd %ulong %127 %ulong_512
        %433 = OpShiftRightLogical %ulong %432 %ulong_4
        %434 = OpShiftLeftLogical %ulong %433 %ulong_12
        %435 = OpIAdd %ulong %434 %427
        %436 = OpIAdd %ulong %435 %ulong_262144
        %437 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %430 %436
        %438 = OpIAdd %ulong %127 %ulong_256
        %439 = OpShiftRightLogical %ulong %438 %ulong_4
        %440 = OpShiftLeftLogical %ulong %439 %ulong_12
        %441 = OpIAdd %ulong %440 %427
        %442 = OpIAdd %ulong %441 %ulong_262144
        %443 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %430 %442
        %444 = OpShiftRightLogical %ulong %127 %ulong_4
        %445 = OpShiftLeftLogical %ulong %444 %ulong_12
        %446 = OpIAdd %ulong %445 %427
        %447 = OpIAdd %ulong %446 %ulong_262144
        %448 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %430 %447
        %449 = OpUConvert %ulong %126
        %450 = OpBitwiseAnd %ulong %449 %ulong_7
        %451 = OpShiftLeftLogical %ulong %450 %ulong_4
        %452 = OpShiftLeftLogical %ulong %252 %ulong_12
        %453 = OpIAdd %ulong %452 %ulong_128
        %454 = OpBitcast %_ptr_CrossWorkgroup_uchar %122
        %455 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %454 %453
        %456 = OpShiftLeftLogical %ulong %245 %ulong_12
        %457 = OpIAdd %ulong %456 %ulong_128
        %458 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %454 %457
        %459 = OpShiftLeftLogical %ulong %238 %ulong_12
        %460 = OpIAdd %ulong %459 %ulong_128
        %461 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %454 %460
        %462 = OpShiftLeftLogical %ulong %231 %ulong_12
        %463 = OpIAdd %ulong %462 %ulong_128
        %464 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %454 %463
        %465 = OpShiftLeftLogical %ulong %224 %ulong_12
        %466 = OpIAdd %ulong %465 %ulong_128
        %467 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %454 %466
        %468 = OpShiftLeftLogical %ulong %217 %ulong_12
        %469 = OpIAdd %ulong %468 %ulong_128
        %470 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %454 %469
        %471 = OpShiftLeftLogical %ulong %210 %ulong_12
        %472 = OpIAdd %ulong %471 %ulong_128
        %473 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %454 %472
        %474 = OpShiftLeftLogical %ulong %208 %ulong_12
        %475 = OpIAdd %ulong %474 %ulong_128
        %476 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %454 %475
               OpBranch %6954
       %6954 = OpLabel
        %477 = OpPhi %_ptr_CrossWorkgroup_uchar %478 %6955 %476 %6953
        %479 = OpPhi %_ptr_CrossWorkgroup_uchar %480 %6955 %473 %6953
        %481 = OpPhi %_ptr_CrossWorkgroup_uchar %482 %6955 %470 %6953
        %483 = OpPhi %_ptr_CrossWorkgroup_uchar %484 %6955 %467 %6953
        %485 = OpPhi %_ptr_CrossWorkgroup_uchar %486 %6955 %464 %6953
        %487 = OpPhi %_ptr_CrossWorkgroup_uchar %488 %6955 %461 %6953
        %489 = OpPhi %_ptr_CrossWorkgroup_uchar %490 %6955 %458 %6953
        %491 = OpPhi %_ptr_CrossWorkgroup_uchar %492 %6955 %455 %6953
        %493 = OpPhi %_ptr_CrossWorkgroup_uchar %494 %6955 %448 %6953
        %495 = OpPhi %_ptr_CrossWorkgroup_uchar %496 %6955 %443 %6953
        %497 = OpPhi %_ptr_CrossWorkgroup_uchar %498 %6955 %437 %6953
        %499 = OpPhi %_ptr_CrossWorkgroup_uchar %500 %6955 %431 %6953
        %501 = OpPhi %uint %502 %6955 %90 %6953
        %503 = OpPhi %v8float %504 %6955 %113 %6953
        %505 = OpPhi %v8float %506 %6955 %113 %6953
        %507 = OpPhi %v8float %508 %6955 %113 %6953
        %509 = OpPhi %v8float %510 %6955 %113 %6953
        %511 = OpPhi %v8float %512 %6955 %113 %6953
        %513 = OpPhi %v8float %514 %6955 %113 %6953
        %515 = OpPhi %v8float %516 %6955 %113 %6953
        %517 = OpPhi %v8float %518 %6955 %113 %6953
        %519 = OpPhi %v8float %520 %6955 %113 %6953
        %521 = OpPhi %v8float %522 %6955 %113 %6953
        %523 = OpPhi %v8float %524 %6955 %113 %6953
        %525 = OpPhi %v8float %526 %6955 %113 %6953
        %527 = OpPhi %v8float %528 %6955 %113 %6953
        %529 = OpPhi %v8float %530 %6955 %113 %6953
        %531 = OpPhi %v8float %532 %6955 %113 %6953
        %533 = OpPhi %v8float %534 %6955 %113 %6953
        %535 = OpSLessThan %bool %501 %uint_124
        %536 = OpIMul %ulong %345 %ulong_68
        %537 = OpIMul %ulong %353 %ulong_68
        %538 = OpIMul %ulong %355 %ulong_68
        %539 = OpIMul %ulong %357 %ulong_68
        %540 = OpIMul %ulong %135 %ulong_132
        %541 = OpIMul %ulong %369 %ulong_132
        %542 = OpIMul %ulong %371 %ulong_132
        %543 = OpIMul %ulong %373 %ulong_132
        %544 = OpIMul %ulong %375 %ulong_132
        %545 = OpIMul %ulong %377 %ulong_132
        %546 = OpIMul %ulong %379 %ulong_132
        %547 = OpIMul %ulong %381 %ulong_132
        %548 = OpIMul %ulong %347 %ulong_132
        %549 = OpIMul %ulong %383 %ulong_132
        %550 = OpIMul %ulong %385 %ulong_132
        %551 = OpIMul %ulong %387 %ulong_132
        %552 = OpIMul %ulong %389 %ulong_132
        %553 = OpIMul %ulong %391 %ulong_132
        %554 = OpIMul %ulong %393 %ulong_132
        %555 = OpIMul %ulong %395 %ulong_132
        %556 = OpIMul %ulong %349 %ulong_132
        %557 = OpIMul %ulong %397 %ulong_132
        %558 = OpIMul %ulong %399 %ulong_132
        %559 = OpIMul %ulong %401 %ulong_132
        %560 = OpIMul %ulong %403 %ulong_132
        %561 = OpIMul %ulong %405 %ulong_132
        %562 = OpIMul %ulong %407 %ulong_132
        %563 = OpIMul %ulong %409 %ulong_132
        %564 = OpIMul %ulong %351 %ulong_132
        %565 = OpIMul %ulong %411 %ulong_132
        %566 = OpIMul %ulong %413 %ulong_132
        %567 = OpIMul %ulong %415 %ulong_132
        %568 = OpIMul %ulong %417 %ulong_132
        %569 = OpIMul %ulong %419 %ulong_132
        %570 = OpIMul %ulong %421 %ulong_132
        %571 = OpIMul %ulong %423 %ulong_132
               OpBranchConditional %535 %6955 %6956
       %6956 = OpLabel
               OpMemoryBarrier %uint_2 %uint_4
        %572 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_signal %uint_4294967295
        %573 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_wait %ushort_65535
               OpMemoryBarrier %uint_2 %uint_2
        %574 = OpIAdd %ulong %536 %135
        %575 = OpBitcast %_ptr_Workgroup_half %__shared_memory__
        %576 = OpPtrAccessChain %_ptr_Workgroup_half %575 %574
        %577 = OpBitcast %_ptr_Workgroup_v8half %576
        %578 = OpLoad %v8half %577 Aligned 2
        %579 = OpIAdd %ulong %536 %347
        %580 = OpPtrAccessChain %_ptr_Workgroup_half %575 %579
        %581 = OpBitcast %_ptr_Workgroup_v8half %580
        %582 = OpLoad %v8half %581 Aligned 2
        %583 = OpIAdd %ulong %536 %349
        %584 = OpPtrAccessChain %_ptr_Workgroup_half %575 %583
        %585 = OpBitcast %_ptr_Workgroup_v8half %584
        %586 = OpLoad %v8half %585 Aligned 2
        %587 = OpIAdd %ulong %536 %351
        %588 = OpPtrAccessChain %_ptr_Workgroup_half %575 %587
        %589 = OpBitcast %_ptr_Workgroup_v8half %588
        %590 = OpLoad %v8half %589 Aligned 2
        %591 = OpIAdd %ulong %537 %135
        %592 = OpPtrAccessChain %_ptr_Workgroup_half %575 %591
        %593 = OpBitcast %_ptr_Workgroup_v8half %592
        %594 = OpLoad %v8half %593 Aligned 2
        %595 = OpIAdd %ulong %537 %347
        %596 = OpPtrAccessChain %_ptr_Workgroup_half %575 %595
        %597 = OpBitcast %_ptr_Workgroup_v8half %596
        %598 = OpLoad %v8half %597 Aligned 2
        %599 = OpIAdd %ulong %537 %349
        %600 = OpPtrAccessChain %_ptr_Workgroup_half %575 %599
        %601 = OpBitcast %_ptr_Workgroup_v8half %600
        %602 = OpLoad %v8half %601 Aligned 2
        %603 = OpIAdd %ulong %537 %351
        %604 = OpPtrAccessChain %_ptr_Workgroup_half %575 %603
        %605 = OpBitcast %_ptr_Workgroup_v8half %604
        %606 = OpLoad %v8half %605 Aligned 2
        %607 = OpIAdd %ulong %538 %135
        %608 = OpPtrAccessChain %_ptr_Workgroup_half %575 %607
        %609 = OpBitcast %_ptr_Workgroup_v8half %608
        %610 = OpLoad %v8half %609 Aligned 2
        %611 = OpIAdd %ulong %538 %347
        %612 = OpPtrAccessChain %_ptr_Workgroup_half %575 %611
        %613 = OpBitcast %_ptr_Workgroup_v8half %612
        %614 = OpLoad %v8half %613 Aligned 2
        %615 = OpIAdd %ulong %538 %349
        %616 = OpPtrAccessChain %_ptr_Workgroup_half %575 %615
        %617 = OpBitcast %_ptr_Workgroup_v8half %616
        %618 = OpLoad %v8half %617 Aligned 2
        %619 = OpIAdd %ulong %538 %351
        %620 = OpPtrAccessChain %_ptr_Workgroup_half %575 %619
        %621 = OpBitcast %_ptr_Workgroup_v8half %620
        %622 = OpLoad %v8half %621 Aligned 2
        %623 = OpIAdd %ulong %539 %135
        %624 = OpPtrAccessChain %_ptr_Workgroup_half %575 %623
        %625 = OpBitcast %_ptr_Workgroup_v8half %624
        %626 = OpLoad %v8half %625 Aligned 2
        %627 = OpIAdd %ulong %539 %347
        %628 = OpPtrAccessChain %_ptr_Workgroup_half %575 %627
        %629 = OpBitcast %_ptr_Workgroup_v8half %628
        %630 = OpLoad %v8half %629 Aligned 2
        %631 = OpIAdd %ulong %539 %349
        %632 = OpPtrAccessChain %_ptr_Workgroup_half %575 %631
        %633 = OpBitcast %_ptr_Workgroup_v8half %632
        %634 = OpLoad %v8half %633 Aligned 2
        %635 = OpIAdd %ulong %539 %351
        %636 = OpPtrAccessChain %_ptr_Workgroup_half %575 %635
        %637 = OpBitcast %_ptr_Workgroup_v8half %636
        %638 = OpLoad %v8half %637 Aligned 2
        %639 = OpIAdd %ulong %540 %361
        %640 = OpBitcast %_ptr_Workgroup_half %__shared_memory___0
        %641 = OpPtrAccessChain %_ptr_Workgroup_half %640 %639
        %642 = OpLoad %half %641 Aligned 2
        %643 = OpIAdd %ulong %540 %363
        %644 = OpPtrAccessChain %_ptr_Workgroup_half %640 %643
        %645 = OpLoad %half %644 Aligned 2
        %646 = OpIAdd %ulong %540 %365
        %647 = OpPtrAccessChain %_ptr_Workgroup_half %640 %646
        %648 = OpLoad %half %647 Aligned 2
        %649 = OpIAdd %ulong %540 %367
        %650 = OpPtrAccessChain %_ptr_Workgroup_half %640 %649
        %651 = OpLoad %half %650 Aligned 2
        %652 = OpIAdd %ulong %541 %361
        %653 = OpPtrAccessChain %_ptr_Workgroup_half %640 %652
        %654 = OpLoad %half %653 Aligned 2
        %655 = OpIAdd %ulong %541 %363
        %656 = OpPtrAccessChain %_ptr_Workgroup_half %640 %655
        %657 = OpLoad %half %656 Aligned 2
        %658 = OpIAdd %ulong %541 %365
        %659 = OpPtrAccessChain %_ptr_Workgroup_half %640 %658
        %660 = OpLoad %half %659 Aligned 2
        %661 = OpIAdd %ulong %541 %367
        %662 = OpPtrAccessChain %_ptr_Workgroup_half %640 %661
        %663 = OpLoad %half %662 Aligned 2
        %664 = OpIAdd %ulong %542 %361
        %665 = OpPtrAccessChain %_ptr_Workgroup_half %640 %664
        %666 = OpLoad %half %665 Aligned 2
        %667 = OpIAdd %ulong %542 %363
        %668 = OpPtrAccessChain %_ptr_Workgroup_half %640 %667
        %669 = OpLoad %half %668 Aligned 2
        %670 = OpIAdd %ulong %542 %365
        %671 = OpPtrAccessChain %_ptr_Workgroup_half %640 %670
        %672 = OpLoad %half %671 Aligned 2
        %673 = OpIAdd %ulong %542 %367
        %674 = OpPtrAccessChain %_ptr_Workgroup_half %640 %673
        %675 = OpLoad %half %674 Aligned 2
        %676 = OpIAdd %ulong %543 %361
        %677 = OpPtrAccessChain %_ptr_Workgroup_half %640 %676
        %678 = OpLoad %half %677 Aligned 2
        %679 = OpIAdd %ulong %543 %363
        %680 = OpPtrAccessChain %_ptr_Workgroup_half %640 %679
        %681 = OpLoad %half %680 Aligned 2
        %682 = OpIAdd %ulong %543 %365
        %683 = OpPtrAccessChain %_ptr_Workgroup_half %640 %682
        %684 = OpLoad %half %683 Aligned 2
        %685 = OpIAdd %ulong %543 %367
        %686 = OpPtrAccessChain %_ptr_Workgroup_half %640 %685
        %687 = OpLoad %half %686 Aligned 2
        %688 = OpIAdd %ulong %544 %361
        %689 = OpPtrAccessChain %_ptr_Workgroup_half %640 %688
        %690 = OpLoad %half %689 Aligned 2
        %691 = OpIAdd %ulong %544 %363
        %692 = OpPtrAccessChain %_ptr_Workgroup_half %640 %691
        %693 = OpLoad %half %692 Aligned 2
        %694 = OpIAdd %ulong %544 %365
        %695 = OpPtrAccessChain %_ptr_Workgroup_half %640 %694
        %696 = OpLoad %half %695 Aligned 2
        %697 = OpIAdd %ulong %544 %367
        %698 = OpPtrAccessChain %_ptr_Workgroup_half %640 %697
        %699 = OpLoad %half %698 Aligned 2
        %700 = OpIAdd %ulong %545 %361
        %701 = OpPtrAccessChain %_ptr_Workgroup_half %640 %700
        %702 = OpLoad %half %701 Aligned 2
        %703 = OpIAdd %ulong %545 %363
        %704 = OpPtrAccessChain %_ptr_Workgroup_half %640 %703
        %705 = OpLoad %half %704 Aligned 2
        %706 = OpIAdd %ulong %545 %365
        %707 = OpPtrAccessChain %_ptr_Workgroup_half %640 %706
        %708 = OpLoad %half %707 Aligned 2
        %709 = OpIAdd %ulong %545 %367
        %710 = OpPtrAccessChain %_ptr_Workgroup_half %640 %709
        %711 = OpLoad %half %710 Aligned 2
        %712 = OpIAdd %ulong %546 %361
        %713 = OpPtrAccessChain %_ptr_Workgroup_half %640 %712
        %714 = OpLoad %half %713 Aligned 2
        %715 = OpIAdd %ulong %546 %363
        %716 = OpPtrAccessChain %_ptr_Workgroup_half %640 %715
        %717 = OpLoad %half %716 Aligned 2
        %718 = OpIAdd %ulong %546 %365
        %719 = OpPtrAccessChain %_ptr_Workgroup_half %640 %718
        %720 = OpLoad %half %719 Aligned 2
        %721 = OpIAdd %ulong %546 %367
        %722 = OpPtrAccessChain %_ptr_Workgroup_half %640 %721
        %723 = OpLoad %half %722 Aligned 2
        %724 = OpIAdd %ulong %547 %361
        %725 = OpPtrAccessChain %_ptr_Workgroup_half %640 %724
        %726 = OpLoad %half %725 Aligned 2
        %727 = OpIAdd %ulong %547 %363
        %728 = OpPtrAccessChain %_ptr_Workgroup_half %640 %727
        %729 = OpLoad %half %728 Aligned 2
        %730 = OpIAdd %ulong %547 %365
        %731 = OpPtrAccessChain %_ptr_Workgroup_half %640 %730
        %732 = OpLoad %half %731 Aligned 2
        %733 = OpIAdd %ulong %547 %367
        %734 = OpPtrAccessChain %_ptr_Workgroup_half %640 %733
        %735 = OpLoad %half %734 Aligned 2
        %736 = OpIAdd %ulong %548 %361
        %737 = OpPtrAccessChain %_ptr_Workgroup_half %640 %736
        %738 = OpLoad %half %737 Aligned 2
        %739 = OpIAdd %ulong %548 %363
        %740 = OpPtrAccessChain %_ptr_Workgroup_half %640 %739
        %741 = OpLoad %half %740 Aligned 2
        %742 = OpIAdd %ulong %548 %365
        %743 = OpPtrAccessChain %_ptr_Workgroup_half %640 %742
        %744 = OpLoad %half %743 Aligned 2
        %745 = OpIAdd %ulong %548 %367
        %746 = OpPtrAccessChain %_ptr_Workgroup_half %640 %745
        %747 = OpLoad %half %746 Aligned 2
        %748 = OpIAdd %ulong %549 %361
        %749 = OpPtrAccessChain %_ptr_Workgroup_half %640 %748
        %750 = OpLoad %half %749 Aligned 2
        %751 = OpIAdd %ulong %549 %363
        %752 = OpPtrAccessChain %_ptr_Workgroup_half %640 %751
        %753 = OpLoad %half %752 Aligned 2
        %754 = OpIAdd %ulong %549 %365
        %755 = OpPtrAccessChain %_ptr_Workgroup_half %640 %754
        %756 = OpLoad %half %755 Aligned 2
        %757 = OpIAdd %ulong %549 %367
        %758 = OpPtrAccessChain %_ptr_Workgroup_half %640 %757
        %759 = OpLoad %half %758 Aligned 2
        %760 = OpIAdd %ulong %550 %361
        %761 = OpPtrAccessChain %_ptr_Workgroup_half %640 %760
        %762 = OpLoad %half %761 Aligned 2
        %763 = OpIAdd %ulong %550 %363
        %764 = OpPtrAccessChain %_ptr_Workgroup_half %640 %763
        %765 = OpLoad %half %764 Aligned 2
        %766 = OpIAdd %ulong %550 %365
        %767 = OpPtrAccessChain %_ptr_Workgroup_half %640 %766
        %768 = OpLoad %half %767 Aligned 2
        %769 = OpIAdd %ulong %550 %367
        %770 = OpPtrAccessChain %_ptr_Workgroup_half %640 %769
        %771 = OpLoad %half %770 Aligned 2
        %772 = OpIAdd %ulong %551 %361
        %773 = OpPtrAccessChain %_ptr_Workgroup_half %640 %772
        %774 = OpLoad %half %773 Aligned 2
        %775 = OpIAdd %ulong %551 %363
        %776 = OpPtrAccessChain %_ptr_Workgroup_half %640 %775
        %777 = OpLoad %half %776 Aligned 2
        %778 = OpIAdd %ulong %551 %365
        %779 = OpPtrAccessChain %_ptr_Workgroup_half %640 %778
        %780 = OpLoad %half %779 Aligned 2
        %781 = OpIAdd %ulong %551 %367
        %782 = OpPtrAccessChain %_ptr_Workgroup_half %640 %781
        %783 = OpLoad %half %782 Aligned 2
        %784 = OpIAdd %ulong %552 %361
        %785 = OpPtrAccessChain %_ptr_Workgroup_half %640 %784
        %786 = OpLoad %half %785 Aligned 2
        %787 = OpIAdd %ulong %552 %363
        %788 = OpPtrAccessChain %_ptr_Workgroup_half %640 %787
        %789 = OpLoad %half %788 Aligned 2
        %790 = OpIAdd %ulong %552 %365
        %791 = OpPtrAccessChain %_ptr_Workgroup_half %640 %790
        %792 = OpLoad %half %791 Aligned 2
        %793 = OpIAdd %ulong %552 %367
        %794 = OpPtrAccessChain %_ptr_Workgroup_half %640 %793
        %795 = OpLoad %half %794 Aligned 2
        %796 = OpIAdd %ulong %553 %361
        %797 = OpPtrAccessChain %_ptr_Workgroup_half %640 %796
        %798 = OpLoad %half %797 Aligned 2
        %799 = OpIAdd %ulong %553 %363
        %800 = OpPtrAccessChain %_ptr_Workgroup_half %640 %799
        %801 = OpLoad %half %800 Aligned 2
        %802 = OpIAdd %ulong %553 %365
        %803 = OpPtrAccessChain %_ptr_Workgroup_half %640 %802
        %804 = OpLoad %half %803 Aligned 2
        %805 = OpIAdd %ulong %553 %367
        %806 = OpPtrAccessChain %_ptr_Workgroup_half %640 %805
        %807 = OpLoad %half %806 Aligned 2
        %808 = OpIAdd %ulong %554 %361
        %809 = OpPtrAccessChain %_ptr_Workgroup_half %640 %808
        %810 = OpLoad %half %809 Aligned 2
        %811 = OpIAdd %ulong %554 %363
        %812 = OpPtrAccessChain %_ptr_Workgroup_half %640 %811
        %813 = OpLoad %half %812 Aligned 2
        %814 = OpIAdd %ulong %554 %365
        %815 = OpPtrAccessChain %_ptr_Workgroup_half %640 %814
        %816 = OpLoad %half %815 Aligned 2
        %817 = OpIAdd %ulong %554 %367
        %818 = OpPtrAccessChain %_ptr_Workgroup_half %640 %817
        %819 = OpLoad %half %818 Aligned 2
        %820 = OpIAdd %ulong %555 %361
        %821 = OpPtrAccessChain %_ptr_Workgroup_half %640 %820
        %822 = OpLoad %half %821 Aligned 2
        %823 = OpIAdd %ulong %555 %363
        %824 = OpPtrAccessChain %_ptr_Workgroup_half %640 %823
        %825 = OpLoad %half %824 Aligned 2
        %826 = OpIAdd %ulong %555 %365
        %827 = OpPtrAccessChain %_ptr_Workgroup_half %640 %826
        %828 = OpLoad %half %827 Aligned 2
        %829 = OpIAdd %ulong %555 %367
        %830 = OpPtrAccessChain %_ptr_Workgroup_half %640 %829
        %831 = OpLoad %half %830 Aligned 2
        %832 = OpIAdd %ulong %556 %361
        %833 = OpPtrAccessChain %_ptr_Workgroup_half %640 %832
        %834 = OpLoad %half %833 Aligned 2
        %835 = OpIAdd %ulong %556 %363
        %836 = OpPtrAccessChain %_ptr_Workgroup_half %640 %835
        %837 = OpLoad %half %836 Aligned 2
        %838 = OpIAdd %ulong %556 %365
        %839 = OpPtrAccessChain %_ptr_Workgroup_half %640 %838
        %840 = OpLoad %half %839 Aligned 2
        %841 = OpIAdd %ulong %556 %367
        %842 = OpPtrAccessChain %_ptr_Workgroup_half %640 %841
        %843 = OpLoad %half %842 Aligned 2
        %844 = OpIAdd %ulong %557 %361
        %845 = OpPtrAccessChain %_ptr_Workgroup_half %640 %844
        %846 = OpLoad %half %845 Aligned 2
        %847 = OpIAdd %ulong %557 %363
        %848 = OpPtrAccessChain %_ptr_Workgroup_half %640 %847
        %849 = OpLoad %half %848 Aligned 2
        %850 = OpIAdd %ulong %557 %365
        %851 = OpPtrAccessChain %_ptr_Workgroup_half %640 %850
        %852 = OpLoad %half %851 Aligned 2
        %853 = OpIAdd %ulong %557 %367
        %854 = OpPtrAccessChain %_ptr_Workgroup_half %640 %853
        %855 = OpLoad %half %854 Aligned 2
        %856 = OpIAdd %ulong %558 %361
        %857 = OpPtrAccessChain %_ptr_Workgroup_half %640 %856
        %858 = OpLoad %half %857 Aligned 2
        %859 = OpIAdd %ulong %558 %363
        %860 = OpPtrAccessChain %_ptr_Workgroup_half %640 %859
        %861 = OpLoad %half %860 Aligned 2
        %862 = OpIAdd %ulong %558 %365
        %863 = OpPtrAccessChain %_ptr_Workgroup_half %640 %862
        %864 = OpLoad %half %863 Aligned 2
        %865 = OpIAdd %ulong %558 %367
        %866 = OpPtrAccessChain %_ptr_Workgroup_half %640 %865
        %867 = OpLoad %half %866 Aligned 2
        %868 = OpIAdd %ulong %559 %361
        %869 = OpPtrAccessChain %_ptr_Workgroup_half %640 %868
        %870 = OpLoad %half %869 Aligned 2
        %871 = OpIAdd %ulong %559 %363
        %872 = OpPtrAccessChain %_ptr_Workgroup_half %640 %871
        %873 = OpLoad %half %872 Aligned 2
        %874 = OpIAdd %ulong %559 %365
        %875 = OpPtrAccessChain %_ptr_Workgroup_half %640 %874
        %876 = OpLoad %half %875 Aligned 2
        %877 = OpIAdd %ulong %559 %367
        %878 = OpPtrAccessChain %_ptr_Workgroup_half %640 %877
        %879 = OpLoad %half %878 Aligned 2
        %880 = OpIAdd %ulong %560 %361
        %881 = OpPtrAccessChain %_ptr_Workgroup_half %640 %880
        %882 = OpLoad %half %881 Aligned 2
        %883 = OpIAdd %ulong %560 %363
        %884 = OpPtrAccessChain %_ptr_Workgroup_half %640 %883
        %885 = OpLoad %half %884 Aligned 2
        %886 = OpIAdd %ulong %560 %365
        %887 = OpPtrAccessChain %_ptr_Workgroup_half %640 %886
        %888 = OpLoad %half %887 Aligned 2
        %889 = OpIAdd %ulong %560 %367
        %890 = OpPtrAccessChain %_ptr_Workgroup_half %640 %889
        %891 = OpLoad %half %890 Aligned 2
        %892 = OpIAdd %ulong %561 %361
        %893 = OpPtrAccessChain %_ptr_Workgroup_half %640 %892
        %894 = OpLoad %half %893 Aligned 2
        %895 = OpIAdd %ulong %561 %363
        %896 = OpPtrAccessChain %_ptr_Workgroup_half %640 %895
        %897 = OpLoad %half %896 Aligned 2
        %898 = OpIAdd %ulong %561 %365
        %899 = OpPtrAccessChain %_ptr_Workgroup_half %640 %898
        %900 = OpLoad %half %899 Aligned 2
        %901 = OpIAdd %ulong %561 %367
        %902 = OpPtrAccessChain %_ptr_Workgroup_half %640 %901
        %903 = OpLoad %half %902 Aligned 2
        %904 = OpIAdd %ulong %562 %361
        %905 = OpPtrAccessChain %_ptr_Workgroup_half %640 %904
        %906 = OpLoad %half %905 Aligned 2
        %907 = OpIAdd %ulong %562 %363
        %908 = OpPtrAccessChain %_ptr_Workgroup_half %640 %907
        %909 = OpLoad %half %908 Aligned 2
        %910 = OpIAdd %ulong %562 %365
        %911 = OpPtrAccessChain %_ptr_Workgroup_half %640 %910
        %912 = OpLoad %half %911 Aligned 2
        %913 = OpIAdd %ulong %562 %367
        %914 = OpPtrAccessChain %_ptr_Workgroup_half %640 %913
        %915 = OpLoad %half %914 Aligned 2
        %916 = OpIAdd %ulong %563 %361
        %917 = OpPtrAccessChain %_ptr_Workgroup_half %640 %916
        %918 = OpLoad %half %917 Aligned 2
        %919 = OpIAdd %ulong %563 %363
        %920 = OpPtrAccessChain %_ptr_Workgroup_half %640 %919
        %921 = OpLoad %half %920 Aligned 2
        %922 = OpIAdd %ulong %563 %365
        %923 = OpPtrAccessChain %_ptr_Workgroup_half %640 %922
        %924 = OpLoad %half %923 Aligned 2
        %925 = OpIAdd %ulong %563 %367
        %926 = OpPtrAccessChain %_ptr_Workgroup_half %640 %925
        %927 = OpLoad %half %926 Aligned 2
        %928 = OpIAdd %ulong %564 %361
        %929 = OpPtrAccessChain %_ptr_Workgroup_half %640 %928
        %930 = OpLoad %half %929 Aligned 2
        %931 = OpIAdd %ulong %564 %363
        %932 = OpPtrAccessChain %_ptr_Workgroup_half %640 %931
        %933 = OpLoad %half %932 Aligned 2
        %934 = OpIAdd %ulong %564 %365
        %935 = OpPtrAccessChain %_ptr_Workgroup_half %640 %934
        %936 = OpLoad %half %935 Aligned 2
        %937 = OpIAdd %ulong %564 %367
        %938 = OpPtrAccessChain %_ptr_Workgroup_half %640 %937
        %939 = OpLoad %half %938 Aligned 2
        %940 = OpIAdd %ulong %565 %361
        %941 = OpPtrAccessChain %_ptr_Workgroup_half %640 %940
        %942 = OpLoad %half %941 Aligned 2
        %943 = OpIAdd %ulong %565 %363
        %944 = OpPtrAccessChain %_ptr_Workgroup_half %640 %943
        %945 = OpLoad %half %944 Aligned 2
        %946 = OpIAdd %ulong %565 %365
        %947 = OpPtrAccessChain %_ptr_Workgroup_half %640 %946
        %948 = OpLoad %half %947 Aligned 2
        %949 = OpIAdd %ulong %565 %367
        %950 = OpPtrAccessChain %_ptr_Workgroup_half %640 %949
        %951 = OpLoad %half %950 Aligned 2
        %952 = OpIAdd %ulong %566 %361
        %953 = OpPtrAccessChain %_ptr_Workgroup_half %640 %952
        %954 = OpLoad %half %953 Aligned 2
        %955 = OpIAdd %ulong %566 %363
        %956 = OpPtrAccessChain %_ptr_Workgroup_half %640 %955
        %957 = OpLoad %half %956 Aligned 2
        %958 = OpIAdd %ulong %566 %365
        %959 = OpPtrAccessChain %_ptr_Workgroup_half %640 %958
        %960 = OpLoad %half %959 Aligned 2
        %961 = OpIAdd %ulong %566 %367
        %962 = OpPtrAccessChain %_ptr_Workgroup_half %640 %961
        %963 = OpLoad %half %962 Aligned 2
        %964 = OpIAdd %ulong %567 %361
        %965 = OpPtrAccessChain %_ptr_Workgroup_half %640 %964
        %966 = OpLoad %half %965 Aligned 2
        %967 = OpIAdd %ulong %567 %363
        %968 = OpPtrAccessChain %_ptr_Workgroup_half %640 %967
        %969 = OpLoad %half %968 Aligned 2
        %970 = OpIAdd %ulong %567 %365
        %971 = OpPtrAccessChain %_ptr_Workgroup_half %640 %970
        %972 = OpLoad %half %971 Aligned 2
        %973 = OpIAdd %ulong %567 %367
        %974 = OpPtrAccessChain %_ptr_Workgroup_half %640 %973
        %975 = OpLoad %half %974 Aligned 2
        %976 = OpIAdd %ulong %568 %361
        %977 = OpPtrAccessChain %_ptr_Workgroup_half %640 %976
        %978 = OpLoad %half %977 Aligned 2
        %979 = OpIAdd %ulong %568 %363
        %980 = OpPtrAccessChain %_ptr_Workgroup_half %640 %979
        %981 = OpLoad %half %980 Aligned 2
        %982 = OpIAdd %ulong %568 %365
        %983 = OpPtrAccessChain %_ptr_Workgroup_half %640 %982
        %984 = OpLoad %half %983 Aligned 2
        %985 = OpIAdd %ulong %568 %367
        %986 = OpPtrAccessChain %_ptr_Workgroup_half %640 %985
        %987 = OpLoad %half %986 Aligned 2
        %988 = OpIAdd %ulong %569 %361
        %989 = OpPtrAccessChain %_ptr_Workgroup_half %640 %988
        %990 = OpLoad %half %989 Aligned 2
        %991 = OpIAdd %ulong %569 %363
        %992 = OpPtrAccessChain %_ptr_Workgroup_half %640 %991
        %993 = OpLoad %half %992 Aligned 2
        %994 = OpIAdd %ulong %569 %365
        %995 = OpPtrAccessChain %_ptr_Workgroup_half %640 %994
        %996 = OpLoad %half %995 Aligned 2
        %997 = OpIAdd %ulong %569 %367
        %998 = OpPtrAccessChain %_ptr_Workgroup_half %640 %997
        %999 = OpLoad %half %998 Aligned 2
       %1000 = OpIAdd %ulong %570 %361
       %1001 = OpPtrAccessChain %_ptr_Workgroup_half %640 %1000
       %1002 = OpLoad %half %1001 Aligned 2
       %1003 = OpIAdd %ulong %570 %363
       %1004 = OpPtrAccessChain %_ptr_Workgroup_half %640 %1003
       %1005 = OpLoad %half %1004 Aligned 2
       %1006 = OpIAdd %ulong %570 %365
       %1007 = OpPtrAccessChain %_ptr_Workgroup_half %640 %1006
       %1008 = OpLoad %half %1007 Aligned 2
       %1009 = OpIAdd %ulong %570 %367
       %1010 = OpPtrAccessChain %_ptr_Workgroup_half %640 %1009
       %1011 = OpLoad %half %1010 Aligned 2
       %1012 = OpIAdd %ulong %571 %361
       %1013 = OpPtrAccessChain %_ptr_Workgroup_half %640 %1012
       %1014 = OpLoad %half %1013 Aligned 2
       %1015 = OpIAdd %ulong %571 %363
       %1016 = OpPtrAccessChain %_ptr_Workgroup_half %640 %1015
       %1017 = OpLoad %half %1016 Aligned 2
       %1018 = OpIAdd %ulong %571 %365
       %1019 = OpPtrAccessChain %_ptr_Workgroup_half %640 %1018
       %1020 = OpLoad %half %1019 Aligned 2
       %1021 = OpIAdd %ulong %571 %367
       %1022 = OpPtrAccessChain %_ptr_Workgroup_half %640 %1021
       %1023 = OpLoad %half %1022 Aligned 2
       %1024 = OpCompositeInsert %v8half %642 %106 0
       %1025 = OpCompositeInsert %v8half %654 %1024 1
       %1026 = OpCompositeInsert %v8half %666 %1025 2
       %1027 = OpCompositeInsert %v8half %678 %1026 3
       %1028 = OpCompositeInsert %v8half %690 %1027 4
       %1029 = OpCompositeInsert %v8half %702 %1028 5
       %1030 = OpCompositeInsert %v8half %714 %1029 6
       %1031 = OpCompositeInsert %v8half %726 %1030 7
       %1032 = OpCompositeInsert %v8half %645 %106 0
       %1033 = OpCompositeInsert %v8half %657 %1032 1
       %1034 = OpCompositeInsert %v8half %669 %1033 2
       %1035 = OpCompositeInsert %v8half %681 %1034 3
       %1036 = OpCompositeInsert %v8half %693 %1035 4
       %1037 = OpCompositeInsert %v8half %705 %1036 5
       %1038 = OpCompositeInsert %v8half %717 %1037 6
       %1039 = OpCompositeInsert %v8half %729 %1038 7
       %1040 = OpCompositeInsert %v8half %648 %106 0
       %1041 = OpCompositeInsert %v8half %660 %1040 1
       %1042 = OpCompositeInsert %v8half %672 %1041 2
       %1043 = OpCompositeInsert %v8half %684 %1042 3
       %1044 = OpCompositeInsert %v8half %696 %1043 4
       %1045 = OpCompositeInsert %v8half %708 %1044 5
       %1046 = OpCompositeInsert %v8half %720 %1045 6
       %1047 = OpCompositeInsert %v8half %732 %1046 7
       %1048 = OpCompositeInsert %v8half %651 %106 0
       %1049 = OpCompositeInsert %v8half %663 %1048 1
       %1050 = OpCompositeInsert %v8half %675 %1049 2
       %1051 = OpCompositeInsert %v8half %687 %1050 3
       %1052 = OpCompositeInsert %v8half %699 %1051 4
       %1053 = OpCompositeInsert %v8half %711 %1052 5
       %1054 = OpCompositeInsert %v8half %723 %1053 6
       %1055 = OpCompositeInsert %v8half %735 %1054 7
       %1056 = OpCompositeInsert %v8half %738 %106 0
       %1057 = OpCompositeInsert %v8half %750 %1056 1
       %1058 = OpCompositeInsert %v8half %762 %1057 2
       %1059 = OpCompositeInsert %v8half %774 %1058 3
       %1060 = OpCompositeInsert %v8half %786 %1059 4
       %1061 = OpCompositeInsert %v8half %798 %1060 5
       %1062 = OpCompositeInsert %v8half %810 %1061 6
       %1063 = OpCompositeInsert %v8half %822 %1062 7
       %1064 = OpCompositeInsert %v8half %741 %106 0
       %1065 = OpCompositeInsert %v8half %753 %1064 1
       %1066 = OpCompositeInsert %v8half %765 %1065 2
       %1067 = OpCompositeInsert %v8half %777 %1066 3
       %1068 = OpCompositeInsert %v8half %789 %1067 4
       %1069 = OpCompositeInsert %v8half %801 %1068 5
       %1070 = OpCompositeInsert %v8half %813 %1069 6
       %1071 = OpCompositeInsert %v8half %825 %1070 7
       %1072 = OpCompositeInsert %v8half %744 %106 0
       %1073 = OpCompositeInsert %v8half %756 %1072 1
       %1074 = OpCompositeInsert %v8half %768 %1073 2
       %1075 = OpCompositeInsert %v8half %780 %1074 3
       %1076 = OpCompositeInsert %v8half %792 %1075 4
       %1077 = OpCompositeInsert %v8half %804 %1076 5
       %1078 = OpCompositeInsert %v8half %816 %1077 6
       %1079 = OpCompositeInsert %v8half %828 %1078 7
       %1080 = OpCompositeInsert %v8half %747 %106 0
       %1081 = OpCompositeInsert %v8half %759 %1080 1
       %1082 = OpCompositeInsert %v8half %771 %1081 2
       %1083 = OpCompositeInsert %v8half %783 %1082 3
       %1084 = OpCompositeInsert %v8half %795 %1083 4
       %1085 = OpCompositeInsert %v8half %807 %1084 5
       %1086 = OpCompositeInsert %v8half %819 %1085 6
       %1087 = OpCompositeInsert %v8half %831 %1086 7
       %1088 = OpCompositeInsert %v8half %834 %106 0
       %1089 = OpCompositeInsert %v8half %846 %1088 1
       %1090 = OpCompositeInsert %v8half %858 %1089 2
       %1091 = OpCompositeInsert %v8half %870 %1090 3
       %1092 = OpCompositeInsert %v8half %882 %1091 4
       %1093 = OpCompositeInsert %v8half %894 %1092 5
       %1094 = OpCompositeInsert %v8half %906 %1093 6
       %1095 = OpCompositeInsert %v8half %918 %1094 7
       %1096 = OpCompositeInsert %v8half %837 %106 0
       %1097 = OpCompositeInsert %v8half %849 %1096 1
       %1098 = OpCompositeInsert %v8half %861 %1097 2
       %1099 = OpCompositeInsert %v8half %873 %1098 3
       %1100 = OpCompositeInsert %v8half %885 %1099 4
       %1101 = OpCompositeInsert %v8half %897 %1100 5
       %1102 = OpCompositeInsert %v8half %909 %1101 6
       %1103 = OpCompositeInsert %v8half %921 %1102 7
       %1104 = OpCompositeInsert %v8half %840 %106 0
       %1105 = OpCompositeInsert %v8half %852 %1104 1
       %1106 = OpCompositeInsert %v8half %864 %1105 2
       %1107 = OpCompositeInsert %v8half %876 %1106 3
       %1108 = OpCompositeInsert %v8half %888 %1107 4
       %1109 = OpCompositeInsert %v8half %900 %1108 5
       %1110 = OpCompositeInsert %v8half %912 %1109 6
       %1111 = OpCompositeInsert %v8half %924 %1110 7
       %1112 = OpCompositeInsert %v8half %843 %106 0
       %1113 = OpCompositeInsert %v8half %855 %1112 1
       %1114 = OpCompositeInsert %v8half %867 %1113 2
       %1115 = OpCompositeInsert %v8half %879 %1114 3
       %1116 = OpCompositeInsert %v8half %891 %1115 4
       %1117 = OpCompositeInsert %v8half %903 %1116 5
       %1118 = OpCompositeInsert %v8half %915 %1117 6
       %1119 = OpCompositeInsert %v8half %927 %1118 7
       %1120 = OpCompositeInsert %v8half %930 %106 0
       %1121 = OpCompositeInsert %v8half %942 %1120 1
       %1122 = OpCompositeInsert %v8half %954 %1121 2
       %1123 = OpCompositeInsert %v8half %966 %1122 3
       %1124 = OpCompositeInsert %v8half %978 %1123 4
       %1125 = OpCompositeInsert %v8half %990 %1124 5
       %1126 = OpCompositeInsert %v8half %1002 %1125 6
       %1127 = OpCompositeInsert %v8half %1014 %1126 7
       %1128 = OpCompositeInsert %v8half %933 %106 0
       %1129 = OpCompositeInsert %v8half %945 %1128 1
       %1130 = OpCompositeInsert %v8half %957 %1129 2
       %1131 = OpCompositeInsert %v8half %969 %1130 3
       %1132 = OpCompositeInsert %v8half %981 %1131 4
       %1133 = OpCompositeInsert %v8half %993 %1132 5
       %1134 = OpCompositeInsert %v8half %1005 %1133 6
       %1135 = OpCompositeInsert %v8half %1017 %1134 7
       %1136 = OpCompositeInsert %v8half %936 %106 0
       %1137 = OpCompositeInsert %v8half %948 %1136 1
       %1138 = OpCompositeInsert %v8half %960 %1137 2
       %1139 = OpCompositeInsert %v8half %972 %1138 3
       %1140 = OpCompositeInsert %v8half %984 %1139 4
       %1141 = OpCompositeInsert %v8half %996 %1140 5
       %1142 = OpCompositeInsert %v8half %1008 %1141 6
       %1143 = OpCompositeInsert %v8half %1020 %1142 7
       %1144 = OpCompositeInsert %v8half %939 %106 0
       %1145 = OpCompositeInsert %v8half %951 %1144 1
       %1146 = OpCompositeInsert %v8half %963 %1145 2
       %1147 = OpCompositeInsert %v8half %975 %1146 3
       %1148 = OpCompositeInsert %v8half %987 %1147 4
       %1149 = OpCompositeInsert %v8half %999 %1148 5
       %1150 = OpCompositeInsert %v8half %1011 %1149 6
       %1151 = OpCompositeInsert %v8half %1023 %1150 7
       %1152 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %578 %1031 %533
       %1153 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %578 %1039 %531
       %1154 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %578 %1047 %529
       %1155 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %578 %1055 %527
       %1156 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %594 %1031 %525
       %1157 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %594 %1039 %523
       %1158 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %594 %1047 %521
       %1159 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %594 %1055 %519
       %1160 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %610 %1031 %517
       %1161 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %610 %1039 %515
       %1162 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %610 %1047 %513
       %1163 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %610 %1055 %511
       %1164 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %626 %1031 %509
       %1165 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %626 %1039 %507
       %1166 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %626 %1047 %505
       %1167 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %626 %1055 %503
       %1168 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %582 %1063 %1152
       %1169 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %582 %1071 %1153
       %1170 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %582 %1079 %1154
       %1171 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %582 %1087 %1155
       %1172 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %598 %1063 %1156
       %1173 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %598 %1071 %1157
       %1174 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %598 %1079 %1158
       %1175 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %598 %1087 %1159
       %1176 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %614 %1063 %1160
       %1177 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %614 %1071 %1161
       %1178 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %614 %1079 %1162
       %1179 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %614 %1087 %1163
       %1180 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %630 %1063 %1164
       %1181 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %630 %1071 %1165
       %1182 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %630 %1079 %1166
       %1183 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %630 %1087 %1167
       %1184 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %586 %1095 %1168
       %1185 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %586 %1103 %1169
       %1186 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %586 %1111 %1170
       %1187 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %586 %1119 %1171
       %1188 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %602 %1095 %1172
       %1189 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %602 %1103 %1173
       %1190 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %602 %1111 %1174
       %1191 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %602 %1119 %1175
       %1192 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %618 %1095 %1176
       %1193 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %618 %1103 %1177
       %1194 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %618 %1111 %1178
       %1195 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %618 %1119 %1179
       %1196 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %634 %1095 %1180
       %1197 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %634 %1103 %1181
       %1198 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %634 %1111 %1182
       %1199 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %634 %1119 %1183
       %1200 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %590 %1127 %1184
       %1201 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %590 %1135 %1185
       %1202 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %590 %1143 %1186
       %1203 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %590 %1151 %1187
       %1204 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %606 %1127 %1188
       %1205 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %606 %1135 %1189
       %1206 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %606 %1143 %1190
       %1207 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %606 %1151 %1191
       %1208 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %622 %1127 %1192
       %1209 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %622 %1135 %1193
       %1210 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %622 %1143 %1194
       %1211 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %622 %1151 %1195
       %1212 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %638 %1127 %1196
       %1213 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %638 %1135 %1197
       %1214 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %638 %1143 %1198
       %1215 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %638 %1151 %1199
       %1216 = OpCompositeExtract %float %1200 0
       %1217 = OpCompositeExtract %float %1200 1
       %1218 = OpCompositeExtract %float %1200 2
       %1219 = OpCompositeExtract %float %1200 3
       %1220 = OpCompositeExtract %float %1200 4
       %1221 = OpCompositeExtract %float %1200 5
       %1222 = OpCompositeExtract %float %1200 6
       %1223 = OpCompositeExtract %float %1200 7
       %1224 = OpCompositeExtract %float %1201 0
       %1225 = OpCompositeExtract %float %1201 1
       %1226 = OpCompositeExtract %float %1201 2
       %1227 = OpCompositeExtract %float %1201 3
       %1228 = OpCompositeExtract %float %1201 4
       %1229 = OpCompositeExtract %float %1201 5
       %1230 = OpCompositeExtract %float %1201 6
       %1231 = OpCompositeExtract %float %1201 7
       %1232 = OpCompositeExtract %float %1202 0
       %1233 = OpCompositeExtract %float %1202 1
       %1234 = OpCompositeExtract %float %1202 2
       %1235 = OpCompositeExtract %float %1202 3
       %1236 = OpCompositeExtract %float %1202 4
       %1237 = OpCompositeExtract %float %1202 5
       %1238 = OpCompositeExtract %float %1202 6
       %1239 = OpCompositeExtract %float %1202 7
       %1240 = OpCompositeExtract %float %1203 0
       %1241 = OpCompositeExtract %float %1203 1
       %1242 = OpCompositeExtract %float %1203 2
       %1243 = OpCompositeExtract %float %1203 3
       %1244 = OpCompositeExtract %float %1203 4
       %1245 = OpCompositeExtract %float %1203 5
       %1246 = OpCompositeExtract %float %1203 6
       %1247 = OpCompositeExtract %float %1203 7
       %1248 = OpCompositeExtract %float %1204 0
       %1249 = OpCompositeExtract %float %1204 1
       %1250 = OpCompositeExtract %float %1204 2
       %1251 = OpCompositeExtract %float %1204 3
       %1252 = OpCompositeExtract %float %1204 4
       %1253 = OpCompositeExtract %float %1204 5
       %1254 = OpCompositeExtract %float %1204 6
       %1255 = OpCompositeExtract %float %1204 7
       %1256 = OpCompositeExtract %float %1205 0
       %1257 = OpCompositeExtract %float %1205 1
       %1258 = OpCompositeExtract %float %1205 2
       %1259 = OpCompositeExtract %float %1205 3
       %1260 = OpCompositeExtract %float %1205 4
       %1261 = OpCompositeExtract %float %1205 5
       %1262 = OpCompositeExtract %float %1205 6
       %1263 = OpCompositeExtract %float %1205 7
       %1264 = OpCompositeExtract %float %1206 0
       %1265 = OpCompositeExtract %float %1206 1
       %1266 = OpCompositeExtract %float %1206 2
       %1267 = OpCompositeExtract %float %1206 3
       %1268 = OpCompositeExtract %float %1206 4
       %1269 = OpCompositeExtract %float %1206 5
       %1270 = OpCompositeExtract %float %1206 6
       %1271 = OpCompositeExtract %float %1206 7
       %1272 = OpCompositeExtract %float %1207 0
       %1273 = OpCompositeExtract %float %1207 1
       %1274 = OpCompositeExtract %float %1207 2
       %1275 = OpCompositeExtract %float %1207 3
       %1276 = OpCompositeExtract %float %1207 4
       %1277 = OpCompositeExtract %float %1207 5
       %1278 = OpCompositeExtract %float %1207 6
       %1279 = OpCompositeExtract %float %1207 7
       %1280 = OpCompositeExtract %float %1208 0
       %1281 = OpCompositeExtract %float %1208 1
       %1282 = OpCompositeExtract %float %1208 2
       %1283 = OpCompositeExtract %float %1208 3
       %1284 = OpCompositeExtract %float %1208 4
       %1285 = OpCompositeExtract %float %1208 5
       %1286 = OpCompositeExtract %float %1208 6
       %1287 = OpCompositeExtract %float %1208 7
       %1288 = OpCompositeExtract %float %1209 0
       %1289 = OpCompositeExtract %float %1209 1
       %1290 = OpCompositeExtract %float %1209 2
       %1291 = OpCompositeExtract %float %1209 3
       %1292 = OpCompositeExtract %float %1209 4
       %1293 = OpCompositeExtract %float %1209 5
       %1294 = OpCompositeExtract %float %1209 6
       %1295 = OpCompositeExtract %float %1209 7
       %1296 = OpCompositeExtract %float %1210 0
       %1297 = OpCompositeExtract %float %1210 1
       %1298 = OpCompositeExtract %float %1210 2
       %1299 = OpCompositeExtract %float %1210 3
       %1300 = OpCompositeExtract %float %1210 4
       %1301 = OpCompositeExtract %float %1210 5
       %1302 = OpCompositeExtract %float %1210 6
       %1303 = OpCompositeExtract %float %1210 7
       %1304 = OpCompositeExtract %float %1211 0
       %1305 = OpCompositeExtract %float %1211 1
       %1306 = OpCompositeExtract %float %1211 2
       %1307 = OpCompositeExtract %float %1211 3
       %1308 = OpCompositeExtract %float %1211 4
       %1309 = OpCompositeExtract %float %1211 5
       %1310 = OpCompositeExtract %float %1211 6
       %1311 = OpCompositeExtract %float %1211 7
       %1312 = OpCompositeExtract %float %1212 0
       %1313 = OpCompositeExtract %float %1212 1
       %1314 = OpCompositeExtract %float %1212 2
       %1315 = OpCompositeExtract %float %1212 3
       %1316 = OpCompositeExtract %float %1212 4
       %1317 = OpCompositeExtract %float %1212 5
       %1318 = OpCompositeExtract %float %1212 6
       %1319 = OpCompositeExtract %float %1212 7
       %1320 = OpCompositeExtract %float %1213 0
       %1321 = OpCompositeExtract %float %1213 1
       %1322 = OpCompositeExtract %float %1213 2
       %1323 = OpCompositeExtract %float %1213 3
       %1324 = OpCompositeExtract %float %1213 4
       %1325 = OpCompositeExtract %float %1213 5
       %1326 = OpCompositeExtract %float %1213 6
       %1327 = OpCompositeExtract %float %1213 7
       %1328 = OpCompositeExtract %float %1214 0
       %1329 = OpCompositeExtract %float %1214 1
       %1330 = OpCompositeExtract %float %1214 2
       %1331 = OpCompositeExtract %float %1214 3
       %1332 = OpCompositeExtract %float %1214 4
       %1333 = OpCompositeExtract %float %1214 5
       %1334 = OpCompositeExtract %float %1214 6
       %1335 = OpCompositeExtract %float %1214 7
       %1336 = OpCompositeExtract %float %1215 0
       %1337 = OpCompositeExtract %float %1215 1
       %1338 = OpCompositeExtract %float %1215 2
       %1339 = OpCompositeExtract %float %1215 3
       %1340 = OpCompositeExtract %float %1215 4
       %1341 = OpCompositeExtract %float %1215 5
       %1342 = OpCompositeExtract %float %1215 6
       %1343 = OpCompositeExtract %float %1215 7
       %1344 = OpIMul %uint %204 %uint_16
       %1345 = OpIAdd %uint %342 %1344
       %1346 = OpIMul %uint %205 %uint_8
       %1347 = OpIAdd %uint %358 %1346
       %1348 = OpIMul %uint %1345 %uint_16
       %1349 = OpIAdd %uint %1348 %134
       %1350 = OpUConvert %ulong %1349
       %1351 = OpIMul %uint %1347 %uint_16
       %1352 = OpIAdd %uint %1351 %133
       %1353 = OpUConvert %ulong %1352
       %1354 = OpIMul %ulong %1350 %ulong_2048
       %1355 = OpIAdd %ulong %1354 %1353
       %1356 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1355
               OpStore %1356 %1216 Aligned 4
       %1357 = OpIAdd %uint %1352 %uint_16
       %1358 = OpUConvert %ulong %1357
       %1359 = OpIAdd %ulong %1354 %1358
       %1360 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1359
               OpStore %1360 %1224 Aligned 4
       %1361 = OpIAdd %uint %1352 %uint_32
       %1362 = OpUConvert %ulong %1361
       %1363 = OpIAdd %ulong %1354 %1362
       %1364 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1363
               OpStore %1364 %1232 Aligned 4
       %1365 = OpIAdd %uint %1352 %uint_48
       %1366 = OpUConvert %ulong %1365
       %1367 = OpIAdd %ulong %1354 %1366
       %1368 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1367
               OpStore %1368 %1240 Aligned 4
       %1369 = OpIAdd %uint %1349 %uint_1
       %1370 = OpUConvert %ulong %1369
       %1371 = OpIMul %ulong %1370 %ulong_2048
       %1372 = OpIAdd %ulong %1371 %1353
       %1373 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1372
               OpStore %1373 %1217 Aligned 4
       %1374 = OpIAdd %ulong %1371 %1358
       %1375 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1374
               OpStore %1375 %1225 Aligned 4
       %1376 = OpIAdd %ulong %1371 %1362
       %1377 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1376
               OpStore %1377 %1233 Aligned 4
       %1378 = OpIAdd %ulong %1371 %1366
       %1379 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1378
               OpStore %1379 %1241 Aligned 4
       %1380 = OpIAdd %uint %1349 %uint_2
       %1381 = OpUConvert %ulong %1380
       %1382 = OpIMul %ulong %1381 %ulong_2048
       %1383 = OpIAdd %ulong %1382 %1353
       %1384 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1383
               OpStore %1384 %1218 Aligned 4
       %1385 = OpIAdd %ulong %1382 %1358
       %1386 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1385
               OpStore %1386 %1226 Aligned 4
       %1387 = OpIAdd %ulong %1382 %1362
       %1388 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1387
               OpStore %1388 %1234 Aligned 4
       %1389 = OpIAdd %ulong %1382 %1366
       %1390 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1389
               OpStore %1390 %1242 Aligned 4
       %1391 = OpIAdd %uint %1349 %uint_3
       %1392 = OpUConvert %ulong %1391
       %1393 = OpIMul %ulong %1392 %ulong_2048
       %1394 = OpIAdd %ulong %1393 %1353
       %1395 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1394
               OpStore %1395 %1219 Aligned 4
       %1396 = OpIAdd %ulong %1393 %1358
       %1397 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1396
               OpStore %1397 %1227 Aligned 4
       %1398 = OpIAdd %ulong %1393 %1362
       %1399 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1398
               OpStore %1399 %1235 Aligned 4
       %1400 = OpIAdd %ulong %1393 %1366
       %1401 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1400
               OpStore %1401 %1243 Aligned 4
       %1402 = OpIAdd %uint %1349 %uint_4
       %1403 = OpUConvert %ulong %1402
       %1404 = OpIMul %ulong %1403 %ulong_2048
       %1405 = OpIAdd %ulong %1404 %1353
       %1406 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1405
               OpStore %1406 %1220 Aligned 4
       %1407 = OpIAdd %ulong %1404 %1358
       %1408 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1407
               OpStore %1408 %1228 Aligned 4
       %1409 = OpIAdd %ulong %1404 %1362
       %1410 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1409
               OpStore %1410 %1236 Aligned 4
       %1411 = OpIAdd %ulong %1404 %1366
       %1412 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1411
               OpStore %1412 %1244 Aligned 4
       %1413 = OpIAdd %uint %1349 %uint_5
       %1414 = OpUConvert %ulong %1413
       %1415 = OpIMul %ulong %1414 %ulong_2048
       %1416 = OpIAdd %ulong %1415 %1353
       %1417 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1416
               OpStore %1417 %1221 Aligned 4
       %1418 = OpIAdd %ulong %1415 %1358
       %1419 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1418
               OpStore %1419 %1229 Aligned 4
       %1420 = OpIAdd %ulong %1415 %1362
       %1421 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1420
               OpStore %1421 %1237 Aligned 4
       %1422 = OpIAdd %ulong %1415 %1366
       %1423 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1422
               OpStore %1423 %1245 Aligned 4
       %1424 = OpIAdd %uint %1349 %uint_6
       %1425 = OpUConvert %ulong %1424
       %1426 = OpIMul %ulong %1425 %ulong_2048
       %1427 = OpIAdd %ulong %1426 %1353
       %1428 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1427
               OpStore %1428 %1222 Aligned 4
       %1429 = OpIAdd %ulong %1426 %1358
       %1430 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1429
               OpStore %1430 %1230 Aligned 4
       %1431 = OpIAdd %ulong %1426 %1362
       %1432 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1431
               OpStore %1432 %1238 Aligned 4
       %1433 = OpIAdd %ulong %1426 %1366
       %1434 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1433
               OpStore %1434 %1246 Aligned 4
       %1435 = OpIAdd %uint %1349 %uint_7
       %1436 = OpUConvert %ulong %1435
       %1437 = OpIMul %ulong %1436 %ulong_2048
       %1438 = OpIAdd %ulong %1437 %1353
       %1439 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1438
               OpStore %1439 %1223 Aligned 4
       %1440 = OpIAdd %ulong %1437 %1358
       %1441 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1440
               OpStore %1441 %1231 Aligned 4
       %1442 = OpIAdd %ulong %1437 %1362
       %1443 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1442
               OpStore %1443 %1239 Aligned 4
       %1444 = OpIAdd %ulong %1437 %1366
       %1445 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1444
               OpStore %1445 %1247 Aligned 4
       %1446 = OpIAdd %uint %1349 %uint_16
       %1447 = OpUConvert %ulong %1446
       %1448 = OpIMul %ulong %1447 %ulong_2048
       %1449 = OpIAdd %ulong %1448 %1353
       %1450 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1449
               OpStore %1450 %1248 Aligned 4
       %1451 = OpIAdd %ulong %1448 %1358
       %1452 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1451
               OpStore %1452 %1256 Aligned 4
       %1453 = OpIAdd %ulong %1448 %1362
       %1454 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1453
               OpStore %1454 %1264 Aligned 4
       %1455 = OpIAdd %ulong %1448 %1366
       %1456 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1455
               OpStore %1456 %1272 Aligned 4
       %1457 = OpIAdd %uint %1349 %uint_17
       %1458 = OpUConvert %ulong %1457
       %1459 = OpIMul %ulong %1458 %ulong_2048
       %1460 = OpIAdd %ulong %1459 %1353
       %1461 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1460
               OpStore %1461 %1249 Aligned 4
       %1462 = OpIAdd %ulong %1459 %1358
       %1463 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1462
               OpStore %1463 %1257 Aligned 4
       %1464 = OpIAdd %ulong %1459 %1362
       %1465 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1464
               OpStore %1465 %1265 Aligned 4
       %1466 = OpIAdd %ulong %1459 %1366
       %1467 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1466
               OpStore %1467 %1273 Aligned 4
       %1468 = OpIAdd %uint %1349 %uint_18
       %1469 = OpUConvert %ulong %1468
       %1470 = OpIMul %ulong %1469 %ulong_2048
       %1471 = OpIAdd %ulong %1470 %1353
       %1472 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1471
               OpStore %1472 %1250 Aligned 4
       %1473 = OpIAdd %ulong %1470 %1358
       %1474 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1473
               OpStore %1474 %1258 Aligned 4
       %1475 = OpIAdd %ulong %1470 %1362
       %1476 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1475
               OpStore %1476 %1266 Aligned 4
       %1477 = OpIAdd %ulong %1470 %1366
       %1478 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1477
               OpStore %1478 %1274 Aligned 4
       %1479 = OpIAdd %uint %1349 %uint_19
       %1480 = OpUConvert %ulong %1479
       %1481 = OpIMul %ulong %1480 %ulong_2048
       %1482 = OpIAdd %ulong %1481 %1353
       %1483 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1482
               OpStore %1483 %1251 Aligned 4
       %1484 = OpIAdd %ulong %1481 %1358
       %1485 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1484
               OpStore %1485 %1259 Aligned 4
       %1486 = OpIAdd %ulong %1481 %1362
       %1487 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1486
               OpStore %1487 %1267 Aligned 4
       %1488 = OpIAdd %ulong %1481 %1366
       %1489 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1488
               OpStore %1489 %1275 Aligned 4
       %1490 = OpIAdd %uint %1349 %uint_20
       %1491 = OpUConvert %ulong %1490
       %1492 = OpIMul %ulong %1491 %ulong_2048
       %1493 = OpIAdd %ulong %1492 %1353
       %1494 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1493
               OpStore %1494 %1252 Aligned 4
       %1495 = OpIAdd %ulong %1492 %1358
       %1496 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1495
               OpStore %1496 %1260 Aligned 4
       %1497 = OpIAdd %ulong %1492 %1362
       %1498 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1497
               OpStore %1498 %1268 Aligned 4
       %1499 = OpIAdd %ulong %1492 %1366
       %1500 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1499
               OpStore %1500 %1276 Aligned 4
       %1501 = OpIAdd %uint %1349 %uint_21
       %1502 = OpUConvert %ulong %1501
       %1503 = OpIMul %ulong %1502 %ulong_2048
       %1504 = OpIAdd %ulong %1503 %1353
       %1505 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1504
               OpStore %1505 %1253 Aligned 4
       %1506 = OpIAdd %ulong %1503 %1358
       %1507 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1506
               OpStore %1507 %1261 Aligned 4
       %1508 = OpIAdd %ulong %1503 %1362
       %1509 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1508
               OpStore %1509 %1269 Aligned 4
       %1510 = OpIAdd %ulong %1503 %1366
       %1511 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1510
               OpStore %1511 %1277 Aligned 4
       %1512 = OpIAdd %uint %1349 %uint_22
       %1513 = OpUConvert %ulong %1512
       %1514 = OpIMul %ulong %1513 %ulong_2048
       %1515 = OpIAdd %ulong %1514 %1353
       %1516 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1515
               OpStore %1516 %1254 Aligned 4
       %1517 = OpIAdd %ulong %1514 %1358
       %1518 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1517
               OpStore %1518 %1262 Aligned 4
       %1519 = OpIAdd %ulong %1514 %1362
       %1520 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1519
               OpStore %1520 %1270 Aligned 4
       %1521 = OpIAdd %ulong %1514 %1366
       %1522 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1521
               OpStore %1522 %1278 Aligned 4
       %1523 = OpIAdd %uint %1349 %uint_23
       %1524 = OpUConvert %ulong %1523
       %1525 = OpIMul %ulong %1524 %ulong_2048
       %1526 = OpIAdd %ulong %1525 %1353
       %1527 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1526
               OpStore %1527 %1255 Aligned 4
       %1528 = OpIAdd %ulong %1525 %1358
       %1529 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1528
               OpStore %1529 %1263 Aligned 4
       %1530 = OpIAdd %ulong %1525 %1362
       %1531 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1530
               OpStore %1531 %1271 Aligned 4
       %1532 = OpIAdd %ulong %1525 %1366
       %1533 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1532
               OpStore %1533 %1279 Aligned 4
       %1534 = OpIAdd %uint %1349 %uint_32
       %1535 = OpUConvert %ulong %1534
       %1536 = OpIMul %ulong %1535 %ulong_2048
       %1537 = OpIAdd %ulong %1536 %1353
       %1538 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1537
               OpStore %1538 %1280 Aligned 4
       %1539 = OpIAdd %ulong %1536 %1358
       %1540 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1539
               OpStore %1540 %1288 Aligned 4
       %1541 = OpIAdd %ulong %1536 %1362
       %1542 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1541
               OpStore %1542 %1296 Aligned 4
       %1543 = OpIAdd %ulong %1536 %1366
       %1544 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1543
               OpStore %1544 %1304 Aligned 4
       %1545 = OpIAdd %uint %1349 %uint_33
       %1546 = OpUConvert %ulong %1545
       %1547 = OpIMul %ulong %1546 %ulong_2048
       %1548 = OpIAdd %ulong %1547 %1353
       %1549 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1548
               OpStore %1549 %1281 Aligned 4
       %1550 = OpIAdd %ulong %1547 %1358
       %1551 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1550
               OpStore %1551 %1289 Aligned 4
       %1552 = OpIAdd %ulong %1547 %1362
       %1553 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1552
               OpStore %1553 %1297 Aligned 4
       %1554 = OpIAdd %ulong %1547 %1366
       %1555 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1554
               OpStore %1555 %1305 Aligned 4
       %1556 = OpIAdd %uint %1349 %uint_34
       %1557 = OpUConvert %ulong %1556
       %1558 = OpIMul %ulong %1557 %ulong_2048
       %1559 = OpIAdd %ulong %1558 %1353
       %1560 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1559
               OpStore %1560 %1282 Aligned 4
       %1561 = OpIAdd %ulong %1558 %1358
       %1562 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1561
               OpStore %1562 %1290 Aligned 4
       %1563 = OpIAdd %ulong %1558 %1362
       %1564 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1563
               OpStore %1564 %1298 Aligned 4
       %1565 = OpIAdd %ulong %1558 %1366
       %1566 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1565
               OpStore %1566 %1306 Aligned 4
       %1567 = OpIAdd %uint %1349 %uint_35
       %1568 = OpUConvert %ulong %1567
       %1569 = OpIMul %ulong %1568 %ulong_2048
       %1570 = OpIAdd %ulong %1569 %1353
       %1571 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1570
               OpStore %1571 %1283 Aligned 4
       %1572 = OpIAdd %ulong %1569 %1358
       %1573 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1572
               OpStore %1573 %1291 Aligned 4
       %1574 = OpIAdd %ulong %1569 %1362
       %1575 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1574
               OpStore %1575 %1299 Aligned 4
       %1576 = OpIAdd %ulong %1569 %1366
       %1577 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1576
               OpStore %1577 %1307 Aligned 4
       %1578 = OpIAdd %uint %1349 %uint_36
       %1579 = OpUConvert %ulong %1578
       %1580 = OpIMul %ulong %1579 %ulong_2048
       %1581 = OpIAdd %ulong %1580 %1353
       %1582 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1581
               OpStore %1582 %1284 Aligned 4
       %1583 = OpIAdd %ulong %1580 %1358
       %1584 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1583
               OpStore %1584 %1292 Aligned 4
       %1585 = OpIAdd %ulong %1580 %1362
       %1586 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1585
               OpStore %1586 %1300 Aligned 4
       %1587 = OpIAdd %ulong %1580 %1366
       %1588 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1587
               OpStore %1588 %1308 Aligned 4
       %1589 = OpIAdd %uint %1349 %uint_37
       %1590 = OpUConvert %ulong %1589
       %1591 = OpIMul %ulong %1590 %ulong_2048
       %1592 = OpIAdd %ulong %1591 %1353
       %1593 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1592
               OpStore %1593 %1285 Aligned 4
       %1594 = OpIAdd %ulong %1591 %1358
       %1595 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1594
               OpStore %1595 %1293 Aligned 4
       %1596 = OpIAdd %ulong %1591 %1362
       %1597 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1596
               OpStore %1597 %1301 Aligned 4
       %1598 = OpIAdd %ulong %1591 %1366
       %1599 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1598
               OpStore %1599 %1309 Aligned 4
       %1600 = OpIAdd %uint %1349 %uint_38
       %1601 = OpUConvert %ulong %1600
       %1602 = OpIMul %ulong %1601 %ulong_2048
       %1603 = OpIAdd %ulong %1602 %1353
       %1604 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1603
               OpStore %1604 %1286 Aligned 4
       %1605 = OpIAdd %ulong %1602 %1358
       %1606 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1605
               OpStore %1606 %1294 Aligned 4
       %1607 = OpIAdd %ulong %1602 %1362
       %1608 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1607
               OpStore %1608 %1302 Aligned 4
       %1609 = OpIAdd %ulong %1602 %1366
       %1610 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1609
               OpStore %1610 %1310 Aligned 4
       %1611 = OpIAdd %uint %1349 %uint_39
       %1612 = OpUConvert %ulong %1611
       %1613 = OpIMul %ulong %1612 %ulong_2048
       %1614 = OpIAdd %ulong %1613 %1353
       %1615 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1614
               OpStore %1615 %1287 Aligned 4
       %1616 = OpIAdd %ulong %1613 %1358
       %1617 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1616
               OpStore %1617 %1295 Aligned 4
       %1618 = OpIAdd %ulong %1613 %1362
       %1619 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1618
               OpStore %1619 %1303 Aligned 4
       %1620 = OpIAdd %ulong %1613 %1366
       %1621 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1620
               OpStore %1621 %1311 Aligned 4
       %1622 = OpIAdd %uint %1349 %uint_48
       %1623 = OpUConvert %ulong %1622
       %1624 = OpIMul %ulong %1623 %ulong_2048
       %1625 = OpIAdd %ulong %1624 %1353
       %1626 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1625
               OpStore %1626 %1312 Aligned 4
       %1627 = OpIAdd %ulong %1624 %1358
       %1628 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1627
               OpStore %1628 %1320 Aligned 4
       %1629 = OpIAdd %ulong %1624 %1362
       %1630 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1629
               OpStore %1630 %1328 Aligned 4
       %1631 = OpIAdd %ulong %1624 %1366
       %1632 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1631
               OpStore %1632 %1336 Aligned 4
       %1633 = OpIAdd %uint %1349 %uint_49
       %1634 = OpUConvert %ulong %1633
       %1635 = OpIMul %ulong %1634 %ulong_2048
       %1636 = OpIAdd %ulong %1635 %1353
       %1637 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1636
               OpStore %1637 %1313 Aligned 4
       %1638 = OpIAdd %ulong %1635 %1358
       %1639 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1638
               OpStore %1639 %1321 Aligned 4
       %1640 = OpIAdd %ulong %1635 %1362
       %1641 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1640
               OpStore %1641 %1329 Aligned 4
       %1642 = OpIAdd %ulong %1635 %1366
       %1643 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1642
               OpStore %1643 %1337 Aligned 4
       %1644 = OpIAdd %uint %1349 %uint_50
       %1645 = OpUConvert %ulong %1644
       %1646 = OpIMul %ulong %1645 %ulong_2048
       %1647 = OpIAdd %ulong %1646 %1353
       %1648 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1647
               OpStore %1648 %1314 Aligned 4
       %1649 = OpIAdd %ulong %1646 %1358
       %1650 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1649
               OpStore %1650 %1322 Aligned 4
       %1651 = OpIAdd %ulong %1646 %1362
       %1652 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1651
               OpStore %1652 %1330 Aligned 4
       %1653 = OpIAdd %ulong %1646 %1366
       %1654 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1653
               OpStore %1654 %1338 Aligned 4
       %1655 = OpIAdd %uint %1349 %uint_51
       %1656 = OpUConvert %ulong %1655
       %1657 = OpIMul %ulong %1656 %ulong_2048
       %1658 = OpIAdd %ulong %1657 %1353
       %1659 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1658
               OpStore %1659 %1315 Aligned 4
       %1660 = OpIAdd %ulong %1657 %1358
       %1661 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1660
               OpStore %1661 %1323 Aligned 4
       %1662 = OpIAdd %ulong %1657 %1362
       %1663 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1662
               OpStore %1663 %1331 Aligned 4
       %1664 = OpIAdd %ulong %1657 %1366
       %1665 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1664
               OpStore %1665 %1339 Aligned 4
       %1666 = OpIAdd %uint %1349 %uint_52
       %1667 = OpUConvert %ulong %1666
       %1668 = OpIMul %ulong %1667 %ulong_2048
       %1669 = OpIAdd %ulong %1668 %1353
       %1670 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1669
               OpStore %1670 %1316 Aligned 4
       %1671 = OpIAdd %ulong %1668 %1358
       %1672 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1671
               OpStore %1672 %1324 Aligned 4
       %1673 = OpIAdd %ulong %1668 %1362
       %1674 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1673
               OpStore %1674 %1332 Aligned 4
       %1675 = OpIAdd %ulong %1668 %1366
       %1676 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1675
               OpStore %1676 %1340 Aligned 4
       %1677 = OpIAdd %uint %1349 %uint_53
       %1678 = OpUConvert %ulong %1677
       %1679 = OpIMul %ulong %1678 %ulong_2048
       %1680 = OpIAdd %ulong %1679 %1353
       %1681 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1680
               OpStore %1681 %1317 Aligned 4
       %1682 = OpIAdd %ulong %1679 %1358
       %1683 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1682
               OpStore %1683 %1325 Aligned 4
       %1684 = OpIAdd %ulong %1679 %1362
       %1685 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1684
               OpStore %1685 %1333 Aligned 4
       %1686 = OpIAdd %ulong %1679 %1366
       %1687 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1686
               OpStore %1687 %1341 Aligned 4
       %1688 = OpIAdd %uint %1349 %uint_54
       %1689 = OpUConvert %ulong %1688
       %1690 = OpIMul %ulong %1689 %ulong_2048
       %1691 = OpIAdd %ulong %1690 %1353
       %1692 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1691
               OpStore %1692 %1318 Aligned 4
       %1693 = OpIAdd %ulong %1690 %1358
       %1694 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1693
               OpStore %1694 %1326 Aligned 4
       %1695 = OpIAdd %ulong %1690 %1362
       %1696 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1695
               OpStore %1696 %1334 Aligned 4
       %1697 = OpIAdd %ulong %1690 %1366
       %1698 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1697
               OpStore %1698 %1342 Aligned 4
       %1699 = OpIAdd %uint %1349 %uint_55
       %1700 = OpUConvert %ulong %1699
       %1701 = OpIMul %ulong %1700 %ulong_2048
       %1702 = OpIAdd %ulong %1701 %1353
       %1703 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1702
               OpStore %1703 %1319 Aligned 4
       %1704 = OpIAdd %ulong %1701 %1358
       %1705 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1704
               OpStore %1705 %1327 Aligned 4
       %1706 = OpIAdd %ulong %1701 %1362
       %1707 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1706
               OpStore %1707 %1335 Aligned 4
       %1708 = OpIAdd %ulong %1701 %1366
       %1709 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %124 %1708
               OpStore %1709 %1343 Aligned 4
               OpMemoryBarrier %uint_2 %uint_4
       %1710 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_signal %uint_4294967295
       %1711 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_wait %ushort_65535
               OpMemoryBarrier %uint_2 %uint_2
               OpReturn
       %6955 = OpLabel
        %502 = OpIAdd %uint %501 %uint_4
       %1712 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %477 %451
       %1713 = OpBitcast %_ptr_CrossWorkgroup_v8half %1712
       %1714 = OpLoad %v8half %1713 Aligned 2
       %1715 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %479 %451
       %1716 = OpBitcast %_ptr_CrossWorkgroup_v8half %1715
       %1717 = OpLoad %v8half %1716 Aligned 2
       %1718 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %481 %451
       %1719 = OpBitcast %_ptr_CrossWorkgroup_v8half %1718
       %1720 = OpLoad %v8half %1719 Aligned 2
       %1721 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %483 %451
       %1722 = OpBitcast %_ptr_CrossWorkgroup_v8half %1721
       %1723 = OpLoad %v8half %1722 Aligned 2
       %1724 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %485 %451
       %1725 = OpBitcast %_ptr_CrossWorkgroup_v8half %1724
       %1726 = OpLoad %v8half %1725 Aligned 2
       %1727 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %487 %451
       %1728 = OpBitcast %_ptr_CrossWorkgroup_v8half %1727
       %1729 = OpLoad %v8half %1728 Aligned 2
       %1730 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %489 %451
       %1731 = OpBitcast %_ptr_CrossWorkgroup_v8half %1730
       %1732 = OpLoad %v8half %1731 Aligned 2
       %1733 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %491 %451
       %1734 = OpBitcast %_ptr_CrossWorkgroup_v8half %1733
       %1735 = OpLoad %v8half %1734 Aligned 2
       %1736 = OpBitcast %_ptr_CrossWorkgroup_v8half %493
       %1737 = OpLoad %v8half %1736 Aligned 2
       %1738 = OpBitcast %_ptr_CrossWorkgroup_v8half %495
       %1739 = OpLoad %v8half %1738 Aligned 2
       %1740 = OpBitcast %_ptr_CrossWorkgroup_v8half %497
       %1741 = OpLoad %v8half %1740 Aligned 2
       %1742 = OpBitcast %_ptr_CrossWorkgroup_v8half %499
       %1743 = OpLoad %v8half %1742 Aligned 2
               OpMemoryBarrier %uint_2 %uint_4
       %1744 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_signal %uint_4294967295
       %1745 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_wait %ushort_65535
               OpMemoryBarrier %uint_2 %uint_2
       %1746 = OpIAdd %ulong %536 %135
       %1747 = OpBitcast %_ptr_Workgroup_half %__shared_memory__
       %1748 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1746
       %1749 = OpBitcast %_ptr_Workgroup_v8half %1748
       %1750 = OpLoad %v8half %1749 Aligned 2
       %1751 = OpIAdd %ulong %536 %347
       %1752 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1751
       %1753 = OpBitcast %_ptr_Workgroup_v8half %1752
       %1754 = OpLoad %v8half %1753 Aligned 2
       %1755 = OpIAdd %ulong %536 %349
       %1756 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1755
       %1757 = OpBitcast %_ptr_Workgroup_v8half %1756
       %1758 = OpLoad %v8half %1757 Aligned 2
       %1759 = OpIAdd %ulong %536 %351
       %1760 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1759
       %1761 = OpBitcast %_ptr_Workgroup_v8half %1760
       %1762 = OpLoad %v8half %1761 Aligned 2
       %1763 = OpIAdd %ulong %537 %135
       %1764 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1763
       %1765 = OpBitcast %_ptr_Workgroup_v8half %1764
       %1766 = OpLoad %v8half %1765 Aligned 2
       %1767 = OpIAdd %ulong %537 %347
       %1768 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1767
       %1769 = OpBitcast %_ptr_Workgroup_v8half %1768
       %1770 = OpLoad %v8half %1769 Aligned 2
       %1771 = OpIAdd %ulong %537 %349
       %1772 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1771
       %1773 = OpBitcast %_ptr_Workgroup_v8half %1772
       %1774 = OpLoad %v8half %1773 Aligned 2
       %1775 = OpIAdd %ulong %537 %351
       %1776 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1775
       %1777 = OpBitcast %_ptr_Workgroup_v8half %1776
       %1778 = OpLoad %v8half %1777 Aligned 2
       %1779 = OpIAdd %ulong %538 %135
       %1780 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1779
       %1781 = OpBitcast %_ptr_Workgroup_v8half %1780
       %1782 = OpLoad %v8half %1781 Aligned 2
       %1783 = OpIAdd %ulong %538 %347
       %1784 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1783
       %1785 = OpBitcast %_ptr_Workgroup_v8half %1784
       %1786 = OpLoad %v8half %1785 Aligned 2
       %1787 = OpIAdd %ulong %538 %349
       %1788 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1787
       %1789 = OpBitcast %_ptr_Workgroup_v8half %1788
       %1790 = OpLoad %v8half %1789 Aligned 2
       %1791 = OpIAdd %ulong %538 %351
       %1792 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1791
       %1793 = OpBitcast %_ptr_Workgroup_v8half %1792
       %1794 = OpLoad %v8half %1793 Aligned 2
       %1795 = OpIAdd %ulong %539 %135
       %1796 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1795
       %1797 = OpBitcast %_ptr_Workgroup_v8half %1796
       %1798 = OpLoad %v8half %1797 Aligned 2
       %1799 = OpIAdd %ulong %539 %347
       %1800 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1799
       %1801 = OpBitcast %_ptr_Workgroup_v8half %1800
       %1802 = OpLoad %v8half %1801 Aligned 2
       %1803 = OpIAdd %ulong %539 %349
       %1804 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1803
       %1805 = OpBitcast %_ptr_Workgroup_v8half %1804
       %1806 = OpLoad %v8half %1805 Aligned 2
       %1807 = OpIAdd %ulong %539 %351
       %1808 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %1807
       %1809 = OpBitcast %_ptr_Workgroup_v8half %1808
       %1810 = OpLoad %v8half %1809 Aligned 2
       %1811 = OpIAdd %ulong %540 %361
       %1812 = OpBitcast %_ptr_Workgroup_half %__shared_memory___0
       %1813 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1811
       %1814 = OpLoad %half %1813 Aligned 2
       %1815 = OpIAdd %ulong %540 %363
       %1816 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1815
       %1817 = OpLoad %half %1816 Aligned 2
       %1818 = OpIAdd %ulong %540 %365
       %1819 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1818
       %1820 = OpLoad %half %1819 Aligned 2
       %1821 = OpIAdd %ulong %540 %367
       %1822 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1821
       %1823 = OpLoad %half %1822 Aligned 2
       %1824 = OpIAdd %ulong %541 %361
       %1825 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1824
       %1826 = OpLoad %half %1825 Aligned 2
       %1827 = OpIAdd %ulong %541 %363
       %1828 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1827
       %1829 = OpLoad %half %1828 Aligned 2
       %1830 = OpIAdd %ulong %541 %365
       %1831 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1830
       %1832 = OpLoad %half %1831 Aligned 2
       %1833 = OpIAdd %ulong %541 %367
       %1834 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1833
       %1835 = OpLoad %half %1834 Aligned 2
       %1836 = OpIAdd %ulong %542 %361
       %1837 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1836
       %1838 = OpLoad %half %1837 Aligned 2
       %1839 = OpIAdd %ulong %542 %363
       %1840 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1839
       %1841 = OpLoad %half %1840 Aligned 2
       %1842 = OpIAdd %ulong %542 %365
       %1843 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1842
       %1844 = OpLoad %half %1843 Aligned 2
       %1845 = OpIAdd %ulong %542 %367
       %1846 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1845
       %1847 = OpLoad %half %1846 Aligned 2
       %1848 = OpIAdd %ulong %543 %361
       %1849 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1848
       %1850 = OpLoad %half %1849 Aligned 2
       %1851 = OpIAdd %ulong %543 %363
       %1852 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1851
       %1853 = OpLoad %half %1852 Aligned 2
       %1854 = OpIAdd %ulong %543 %365
       %1855 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1854
       %1856 = OpLoad %half %1855 Aligned 2
       %1857 = OpIAdd %ulong %543 %367
       %1858 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1857
       %1859 = OpLoad %half %1858 Aligned 2
       %1860 = OpIAdd %ulong %544 %361
       %1861 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1860
       %1862 = OpLoad %half %1861 Aligned 2
       %1863 = OpIAdd %ulong %544 %363
       %1864 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1863
       %1865 = OpLoad %half %1864 Aligned 2
       %1866 = OpIAdd %ulong %544 %365
       %1867 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1866
       %1868 = OpLoad %half %1867 Aligned 2
       %1869 = OpIAdd %ulong %544 %367
       %1870 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1869
       %1871 = OpLoad %half %1870 Aligned 2
       %1872 = OpIAdd %ulong %545 %361
       %1873 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1872
       %1874 = OpLoad %half %1873 Aligned 2
       %1875 = OpIAdd %ulong %545 %363
       %1876 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1875
       %1877 = OpLoad %half %1876 Aligned 2
       %1878 = OpIAdd %ulong %545 %365
       %1879 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1878
       %1880 = OpLoad %half %1879 Aligned 2
       %1881 = OpIAdd %ulong %545 %367
       %1882 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1881
       %1883 = OpLoad %half %1882 Aligned 2
       %1884 = OpIAdd %ulong %546 %361
       %1885 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1884
       %1886 = OpLoad %half %1885 Aligned 2
       %1887 = OpIAdd %ulong %546 %363
       %1888 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1887
       %1889 = OpLoad %half %1888 Aligned 2
       %1890 = OpIAdd %ulong %546 %365
       %1891 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1890
       %1892 = OpLoad %half %1891 Aligned 2
       %1893 = OpIAdd %ulong %546 %367
       %1894 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1893
       %1895 = OpLoad %half %1894 Aligned 2
       %1896 = OpIAdd %ulong %547 %361
       %1897 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1896
       %1898 = OpLoad %half %1897 Aligned 2
       %1899 = OpIAdd %ulong %547 %363
       %1900 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1899
       %1901 = OpLoad %half %1900 Aligned 2
       %1902 = OpIAdd %ulong %547 %365
       %1903 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1902
       %1904 = OpLoad %half %1903 Aligned 2
       %1905 = OpIAdd %ulong %547 %367
       %1906 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1905
       %1907 = OpLoad %half %1906 Aligned 2
       %1908 = OpIAdd %ulong %548 %361
       %1909 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1908
       %1910 = OpLoad %half %1909 Aligned 2
       %1911 = OpIAdd %ulong %548 %363
       %1912 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1911
       %1913 = OpLoad %half %1912 Aligned 2
       %1914 = OpIAdd %ulong %548 %365
       %1915 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1914
       %1916 = OpLoad %half %1915 Aligned 2
       %1917 = OpIAdd %ulong %548 %367
       %1918 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1917
       %1919 = OpLoad %half %1918 Aligned 2
       %1920 = OpIAdd %ulong %549 %361
       %1921 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1920
       %1922 = OpLoad %half %1921 Aligned 2
       %1923 = OpIAdd %ulong %549 %363
       %1924 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1923
       %1925 = OpLoad %half %1924 Aligned 2
       %1926 = OpIAdd %ulong %549 %365
       %1927 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1926
       %1928 = OpLoad %half %1927 Aligned 2
       %1929 = OpIAdd %ulong %549 %367
       %1930 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1929
       %1931 = OpLoad %half %1930 Aligned 2
       %1932 = OpIAdd %ulong %550 %361
       %1933 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1932
       %1934 = OpLoad %half %1933 Aligned 2
       %1935 = OpIAdd %ulong %550 %363
       %1936 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1935
       %1937 = OpLoad %half %1936 Aligned 2
       %1938 = OpIAdd %ulong %550 %365
       %1939 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1938
       %1940 = OpLoad %half %1939 Aligned 2
       %1941 = OpIAdd %ulong %550 %367
       %1942 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1941
       %1943 = OpLoad %half %1942 Aligned 2
       %1944 = OpIAdd %ulong %551 %361
       %1945 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1944
       %1946 = OpLoad %half %1945 Aligned 2
       %1947 = OpIAdd %ulong %551 %363
       %1948 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1947
       %1949 = OpLoad %half %1948 Aligned 2
       %1950 = OpIAdd %ulong %551 %365
       %1951 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1950
       %1952 = OpLoad %half %1951 Aligned 2
       %1953 = OpIAdd %ulong %551 %367
       %1954 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1953
       %1955 = OpLoad %half %1954 Aligned 2
       %1956 = OpIAdd %ulong %552 %361
       %1957 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1956
       %1958 = OpLoad %half %1957 Aligned 2
       %1959 = OpIAdd %ulong %552 %363
       %1960 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1959
       %1961 = OpLoad %half %1960 Aligned 2
       %1962 = OpIAdd %ulong %552 %365
       %1963 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1962
       %1964 = OpLoad %half %1963 Aligned 2
       %1965 = OpIAdd %ulong %552 %367
       %1966 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1965
       %1967 = OpLoad %half %1966 Aligned 2
       %1968 = OpIAdd %ulong %553 %361
       %1969 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1968
       %1970 = OpLoad %half %1969 Aligned 2
       %1971 = OpIAdd %ulong %553 %363
       %1972 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1971
       %1973 = OpLoad %half %1972 Aligned 2
       %1974 = OpIAdd %ulong %553 %365
       %1975 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1974
       %1976 = OpLoad %half %1975 Aligned 2
       %1977 = OpIAdd %ulong %553 %367
       %1978 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1977
       %1979 = OpLoad %half %1978 Aligned 2
       %1980 = OpIAdd %ulong %554 %361
       %1981 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1980
       %1982 = OpLoad %half %1981 Aligned 2
       %1983 = OpIAdd %ulong %554 %363
       %1984 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1983
       %1985 = OpLoad %half %1984 Aligned 2
       %1986 = OpIAdd %ulong %554 %365
       %1987 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1986
       %1988 = OpLoad %half %1987 Aligned 2
       %1989 = OpIAdd %ulong %554 %367
       %1990 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1989
       %1991 = OpLoad %half %1990 Aligned 2
       %1992 = OpIAdd %ulong %555 %361
       %1993 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1992
       %1994 = OpLoad %half %1993 Aligned 2
       %1995 = OpIAdd %ulong %555 %363
       %1996 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1995
       %1997 = OpLoad %half %1996 Aligned 2
       %1998 = OpIAdd %ulong %555 %365
       %1999 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %1998
       %2000 = OpLoad %half %1999 Aligned 2
       %2001 = OpIAdd %ulong %555 %367
       %2002 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2001
       %2003 = OpLoad %half %2002 Aligned 2
       %2004 = OpIAdd %ulong %556 %361
       %2005 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2004
       %2006 = OpLoad %half %2005 Aligned 2
       %2007 = OpIAdd %ulong %556 %363
       %2008 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2007
       %2009 = OpLoad %half %2008 Aligned 2
       %2010 = OpIAdd %ulong %556 %365
       %2011 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2010
       %2012 = OpLoad %half %2011 Aligned 2
       %2013 = OpIAdd %ulong %556 %367
       %2014 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2013
       %2015 = OpLoad %half %2014 Aligned 2
       %2016 = OpIAdd %ulong %557 %361
       %2017 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2016
       %2018 = OpLoad %half %2017 Aligned 2
       %2019 = OpIAdd %ulong %557 %363
       %2020 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2019
       %2021 = OpLoad %half %2020 Aligned 2
       %2022 = OpIAdd %ulong %557 %365
       %2023 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2022
       %2024 = OpLoad %half %2023 Aligned 2
       %2025 = OpIAdd %ulong %557 %367
       %2026 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2025
       %2027 = OpLoad %half %2026 Aligned 2
       %2028 = OpIAdd %ulong %558 %361
       %2029 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2028
       %2030 = OpLoad %half %2029 Aligned 2
       %2031 = OpIAdd %ulong %558 %363
       %2032 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2031
       %2033 = OpLoad %half %2032 Aligned 2
       %2034 = OpIAdd %ulong %558 %365
       %2035 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2034
       %2036 = OpLoad %half %2035 Aligned 2
       %2037 = OpIAdd %ulong %558 %367
       %2038 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2037
       %2039 = OpLoad %half %2038 Aligned 2
       %2040 = OpIAdd %ulong %559 %361
       %2041 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2040
       %2042 = OpLoad %half %2041 Aligned 2
       %2043 = OpIAdd %ulong %559 %363
       %2044 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2043
       %2045 = OpLoad %half %2044 Aligned 2
       %2046 = OpIAdd %ulong %559 %365
       %2047 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2046
       %2048 = OpLoad %half %2047 Aligned 2
       %2049 = OpIAdd %ulong %559 %367
       %2050 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2049
       %2051 = OpLoad %half %2050 Aligned 2
       %2052 = OpIAdd %ulong %560 %361
       %2053 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2052
       %2054 = OpLoad %half %2053 Aligned 2
       %2055 = OpIAdd %ulong %560 %363
       %2056 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2055
       %2057 = OpLoad %half %2056 Aligned 2
       %2058 = OpIAdd %ulong %560 %365
       %2059 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2058
       %2060 = OpLoad %half %2059 Aligned 2
       %2061 = OpIAdd %ulong %560 %367
       %2062 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2061
       %2063 = OpLoad %half %2062 Aligned 2
       %2064 = OpIAdd %ulong %561 %361
       %2065 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2064
       %2066 = OpLoad %half %2065 Aligned 2
       %2067 = OpIAdd %ulong %561 %363
       %2068 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2067
       %2069 = OpLoad %half %2068 Aligned 2
       %2070 = OpIAdd %ulong %561 %365
       %2071 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2070
       %2072 = OpLoad %half %2071 Aligned 2
       %2073 = OpIAdd %ulong %561 %367
       %2074 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2073
       %2075 = OpLoad %half %2074 Aligned 2
       %2076 = OpIAdd %ulong %562 %361
       %2077 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2076
       %2078 = OpLoad %half %2077 Aligned 2
       %2079 = OpIAdd %ulong %562 %363
       %2080 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2079
       %2081 = OpLoad %half %2080 Aligned 2
       %2082 = OpIAdd %ulong %562 %365
       %2083 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2082
       %2084 = OpLoad %half %2083 Aligned 2
       %2085 = OpIAdd %ulong %562 %367
       %2086 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2085
       %2087 = OpLoad %half %2086 Aligned 2
       %2088 = OpIAdd %ulong %563 %361
       %2089 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2088
       %2090 = OpLoad %half %2089 Aligned 2
       %2091 = OpIAdd %ulong %563 %363
       %2092 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2091
       %2093 = OpLoad %half %2092 Aligned 2
       %2094 = OpIAdd %ulong %563 %365
       %2095 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2094
       %2096 = OpLoad %half %2095 Aligned 2
       %2097 = OpIAdd %ulong %563 %367
       %2098 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2097
       %2099 = OpLoad %half %2098 Aligned 2
       %2100 = OpIAdd %ulong %564 %361
       %2101 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2100
       %2102 = OpLoad %half %2101 Aligned 2
       %2103 = OpIAdd %ulong %564 %363
       %2104 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2103
       %2105 = OpLoad %half %2104 Aligned 2
       %2106 = OpIAdd %ulong %564 %365
       %2107 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2106
       %2108 = OpLoad %half %2107 Aligned 2
       %2109 = OpIAdd %ulong %564 %367
       %2110 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2109
       %2111 = OpLoad %half %2110 Aligned 2
       %2112 = OpIAdd %ulong %565 %361
       %2113 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2112
       %2114 = OpLoad %half %2113 Aligned 2
       %2115 = OpIAdd %ulong %565 %363
       %2116 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2115
       %2117 = OpLoad %half %2116 Aligned 2
       %2118 = OpIAdd %ulong %565 %365
       %2119 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2118
       %2120 = OpLoad %half %2119 Aligned 2
       %2121 = OpIAdd %ulong %565 %367
       %2122 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2121
       %2123 = OpLoad %half %2122 Aligned 2
       %2124 = OpIAdd %ulong %566 %361
       %2125 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2124
       %2126 = OpLoad %half %2125 Aligned 2
       %2127 = OpIAdd %ulong %566 %363
       %2128 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2127
       %2129 = OpLoad %half %2128 Aligned 2
       %2130 = OpIAdd %ulong %566 %365
       %2131 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2130
       %2132 = OpLoad %half %2131 Aligned 2
       %2133 = OpIAdd %ulong %566 %367
       %2134 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2133
       %2135 = OpLoad %half %2134 Aligned 2
       %2136 = OpIAdd %ulong %567 %361
       %2137 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2136
       %2138 = OpLoad %half %2137 Aligned 2
       %2139 = OpIAdd %ulong %567 %363
       %2140 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2139
       %2141 = OpLoad %half %2140 Aligned 2
       %2142 = OpIAdd %ulong %567 %365
       %2143 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2142
       %2144 = OpLoad %half %2143 Aligned 2
       %2145 = OpIAdd %ulong %567 %367
       %2146 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2145
       %2147 = OpLoad %half %2146 Aligned 2
       %2148 = OpIAdd %ulong %568 %361
       %2149 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2148
       %2150 = OpLoad %half %2149 Aligned 2
       %2151 = OpIAdd %ulong %568 %363
       %2152 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2151
       %2153 = OpLoad %half %2152 Aligned 2
       %2154 = OpIAdd %ulong %568 %365
       %2155 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2154
       %2156 = OpLoad %half %2155 Aligned 2
       %2157 = OpIAdd %ulong %568 %367
       %2158 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2157
       %2159 = OpLoad %half %2158 Aligned 2
       %2160 = OpIAdd %ulong %569 %361
       %2161 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2160
       %2162 = OpLoad %half %2161 Aligned 2
       %2163 = OpIAdd %ulong %569 %363
       %2164 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2163
       %2165 = OpLoad %half %2164 Aligned 2
       %2166 = OpIAdd %ulong %569 %365
       %2167 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2166
       %2168 = OpLoad %half %2167 Aligned 2
       %2169 = OpIAdd %ulong %569 %367
       %2170 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2169
       %2171 = OpLoad %half %2170 Aligned 2
       %2172 = OpIAdd %ulong %570 %361
       %2173 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2172
       %2174 = OpLoad %half %2173 Aligned 2
       %2175 = OpIAdd %ulong %570 %363
       %2176 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2175
       %2177 = OpLoad %half %2176 Aligned 2
       %2178 = OpIAdd %ulong %570 %365
       %2179 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2178
       %2180 = OpLoad %half %2179 Aligned 2
       %2181 = OpIAdd %ulong %570 %367
       %2182 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2181
       %2183 = OpLoad %half %2182 Aligned 2
       %2184 = OpIAdd %ulong %571 %361
       %2185 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2184
       %2186 = OpLoad %half %2185 Aligned 2
       %2187 = OpIAdd %ulong %571 %363
       %2188 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2187
       %2189 = OpLoad %half %2188 Aligned 2
       %2190 = OpIAdd %ulong %571 %365
       %2191 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2190
       %2192 = OpLoad %half %2191 Aligned 2
       %2193 = OpIAdd %ulong %571 %367
       %2194 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %2193
       %2195 = OpLoad %half %2194 Aligned 2
       %2196 = OpCompositeInsert %v8half %1814 %106 0
       %2197 = OpCompositeInsert %v8half %1826 %2196 1
       %2198 = OpCompositeInsert %v8half %1838 %2197 2
       %2199 = OpCompositeInsert %v8half %1850 %2198 3
       %2200 = OpCompositeInsert %v8half %1862 %2199 4
       %2201 = OpCompositeInsert %v8half %1874 %2200 5
       %2202 = OpCompositeInsert %v8half %1886 %2201 6
       %2203 = OpCompositeInsert %v8half %1898 %2202 7
       %2204 = OpCompositeInsert %v8half %1817 %106 0
       %2205 = OpCompositeInsert %v8half %1829 %2204 1
       %2206 = OpCompositeInsert %v8half %1841 %2205 2
       %2207 = OpCompositeInsert %v8half %1853 %2206 3
       %2208 = OpCompositeInsert %v8half %1865 %2207 4
       %2209 = OpCompositeInsert %v8half %1877 %2208 5
       %2210 = OpCompositeInsert %v8half %1889 %2209 6
       %2211 = OpCompositeInsert %v8half %1901 %2210 7
       %2212 = OpCompositeInsert %v8half %1820 %106 0
       %2213 = OpCompositeInsert %v8half %1832 %2212 1
       %2214 = OpCompositeInsert %v8half %1844 %2213 2
       %2215 = OpCompositeInsert %v8half %1856 %2214 3
       %2216 = OpCompositeInsert %v8half %1868 %2215 4
       %2217 = OpCompositeInsert %v8half %1880 %2216 5
       %2218 = OpCompositeInsert %v8half %1892 %2217 6
       %2219 = OpCompositeInsert %v8half %1904 %2218 7
       %2220 = OpCompositeInsert %v8half %1823 %106 0
       %2221 = OpCompositeInsert %v8half %1835 %2220 1
       %2222 = OpCompositeInsert %v8half %1847 %2221 2
       %2223 = OpCompositeInsert %v8half %1859 %2222 3
       %2224 = OpCompositeInsert %v8half %1871 %2223 4
       %2225 = OpCompositeInsert %v8half %1883 %2224 5
       %2226 = OpCompositeInsert %v8half %1895 %2225 6
       %2227 = OpCompositeInsert %v8half %1907 %2226 7
       %2228 = OpCompositeInsert %v8half %1910 %106 0
       %2229 = OpCompositeInsert %v8half %1922 %2228 1
       %2230 = OpCompositeInsert %v8half %1934 %2229 2
       %2231 = OpCompositeInsert %v8half %1946 %2230 3
       %2232 = OpCompositeInsert %v8half %1958 %2231 4
       %2233 = OpCompositeInsert %v8half %1970 %2232 5
       %2234 = OpCompositeInsert %v8half %1982 %2233 6
       %2235 = OpCompositeInsert %v8half %1994 %2234 7
       %2236 = OpCompositeInsert %v8half %1913 %106 0
       %2237 = OpCompositeInsert %v8half %1925 %2236 1
       %2238 = OpCompositeInsert %v8half %1937 %2237 2
       %2239 = OpCompositeInsert %v8half %1949 %2238 3
       %2240 = OpCompositeInsert %v8half %1961 %2239 4
       %2241 = OpCompositeInsert %v8half %1973 %2240 5
       %2242 = OpCompositeInsert %v8half %1985 %2241 6
       %2243 = OpCompositeInsert %v8half %1997 %2242 7
       %2244 = OpCompositeInsert %v8half %1916 %106 0
       %2245 = OpCompositeInsert %v8half %1928 %2244 1
       %2246 = OpCompositeInsert %v8half %1940 %2245 2
       %2247 = OpCompositeInsert %v8half %1952 %2246 3
       %2248 = OpCompositeInsert %v8half %1964 %2247 4
       %2249 = OpCompositeInsert %v8half %1976 %2248 5
       %2250 = OpCompositeInsert %v8half %1988 %2249 6
       %2251 = OpCompositeInsert %v8half %2000 %2250 7
       %2252 = OpCompositeInsert %v8half %1919 %106 0
       %2253 = OpCompositeInsert %v8half %1931 %2252 1
       %2254 = OpCompositeInsert %v8half %1943 %2253 2
       %2255 = OpCompositeInsert %v8half %1955 %2254 3
       %2256 = OpCompositeInsert %v8half %1967 %2255 4
       %2257 = OpCompositeInsert %v8half %1979 %2256 5
       %2258 = OpCompositeInsert %v8half %1991 %2257 6
       %2259 = OpCompositeInsert %v8half %2003 %2258 7
       %2260 = OpCompositeInsert %v8half %2006 %106 0
       %2261 = OpCompositeInsert %v8half %2018 %2260 1
       %2262 = OpCompositeInsert %v8half %2030 %2261 2
       %2263 = OpCompositeInsert %v8half %2042 %2262 3
       %2264 = OpCompositeInsert %v8half %2054 %2263 4
       %2265 = OpCompositeInsert %v8half %2066 %2264 5
       %2266 = OpCompositeInsert %v8half %2078 %2265 6
       %2267 = OpCompositeInsert %v8half %2090 %2266 7
       %2268 = OpCompositeInsert %v8half %2009 %106 0
       %2269 = OpCompositeInsert %v8half %2021 %2268 1
       %2270 = OpCompositeInsert %v8half %2033 %2269 2
       %2271 = OpCompositeInsert %v8half %2045 %2270 3
       %2272 = OpCompositeInsert %v8half %2057 %2271 4
       %2273 = OpCompositeInsert %v8half %2069 %2272 5
       %2274 = OpCompositeInsert %v8half %2081 %2273 6
       %2275 = OpCompositeInsert %v8half %2093 %2274 7
       %2276 = OpCompositeInsert %v8half %2012 %106 0
       %2277 = OpCompositeInsert %v8half %2024 %2276 1
       %2278 = OpCompositeInsert %v8half %2036 %2277 2
       %2279 = OpCompositeInsert %v8half %2048 %2278 3
       %2280 = OpCompositeInsert %v8half %2060 %2279 4
       %2281 = OpCompositeInsert %v8half %2072 %2280 5
       %2282 = OpCompositeInsert %v8half %2084 %2281 6
       %2283 = OpCompositeInsert %v8half %2096 %2282 7
       %2284 = OpCompositeInsert %v8half %2015 %106 0
       %2285 = OpCompositeInsert %v8half %2027 %2284 1
       %2286 = OpCompositeInsert %v8half %2039 %2285 2
       %2287 = OpCompositeInsert %v8half %2051 %2286 3
       %2288 = OpCompositeInsert %v8half %2063 %2287 4
       %2289 = OpCompositeInsert %v8half %2075 %2288 5
       %2290 = OpCompositeInsert %v8half %2087 %2289 6
       %2291 = OpCompositeInsert %v8half %2099 %2290 7
       %2292 = OpCompositeInsert %v8half %2102 %106 0
       %2293 = OpCompositeInsert %v8half %2114 %2292 1
       %2294 = OpCompositeInsert %v8half %2126 %2293 2
       %2295 = OpCompositeInsert %v8half %2138 %2294 3
       %2296 = OpCompositeInsert %v8half %2150 %2295 4
       %2297 = OpCompositeInsert %v8half %2162 %2296 5
       %2298 = OpCompositeInsert %v8half %2174 %2297 6
       %2299 = OpCompositeInsert %v8half %2186 %2298 7
       %2300 = OpCompositeInsert %v8half %2105 %106 0
       %2301 = OpCompositeInsert %v8half %2117 %2300 1
       %2302 = OpCompositeInsert %v8half %2129 %2301 2
       %2303 = OpCompositeInsert %v8half %2141 %2302 3
       %2304 = OpCompositeInsert %v8half %2153 %2303 4
       %2305 = OpCompositeInsert %v8half %2165 %2304 5
       %2306 = OpCompositeInsert %v8half %2177 %2305 6
       %2307 = OpCompositeInsert %v8half %2189 %2306 7
       %2308 = OpCompositeInsert %v8half %2108 %106 0
       %2309 = OpCompositeInsert %v8half %2120 %2308 1
       %2310 = OpCompositeInsert %v8half %2132 %2309 2
       %2311 = OpCompositeInsert %v8half %2144 %2310 3
       %2312 = OpCompositeInsert %v8half %2156 %2311 4
       %2313 = OpCompositeInsert %v8half %2168 %2312 5
       %2314 = OpCompositeInsert %v8half %2180 %2313 6
       %2315 = OpCompositeInsert %v8half %2192 %2314 7
       %2316 = OpCompositeInsert %v8half %2111 %106 0
       %2317 = OpCompositeInsert %v8half %2123 %2316 1
       %2318 = OpCompositeInsert %v8half %2135 %2317 2
       %2319 = OpCompositeInsert %v8half %2147 %2318 3
       %2320 = OpCompositeInsert %v8half %2159 %2319 4
       %2321 = OpCompositeInsert %v8half %2171 %2320 5
       %2322 = OpCompositeInsert %v8half %2183 %2321 6
       %2323 = OpCompositeInsert %v8half %2195 %2322 7
       %2324 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1750 %2203 %533
       %2325 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1750 %2211 %531
       %2326 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1750 %2219 %529
       %2327 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1750 %2227 %527
       %2328 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1766 %2203 %525
       %2329 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1766 %2211 %523
       %2330 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1766 %2219 %521
       %2331 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1766 %2227 %519
       %2332 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1782 %2203 %517
       %2333 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1782 %2211 %515
       %2334 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1782 %2219 %513
       %2335 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1782 %2227 %511
       %2336 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1798 %2203 %509
       %2337 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1798 %2211 %507
       %2338 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1798 %2219 %505
       %2339 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1798 %2227 %503
       %2340 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1754 %2235 %2324
       %2341 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1754 %2243 %2325
       %2342 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1754 %2251 %2326
       %2343 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1754 %2259 %2327
       %2344 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1770 %2235 %2328
       %2345 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1770 %2243 %2329
       %2346 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1770 %2251 %2330
       %2347 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1770 %2259 %2331
       %2348 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1786 %2235 %2332
       %2349 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1786 %2243 %2333
       %2350 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1786 %2251 %2334
       %2351 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1786 %2259 %2335
       %2352 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1802 %2235 %2336
       %2353 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1802 %2243 %2337
       %2354 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1802 %2251 %2338
       %2355 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1802 %2259 %2339
       %2356 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1758 %2267 %2340
       %2357 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1758 %2275 %2341
       %2358 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1758 %2283 %2342
       %2359 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1758 %2291 %2343
       %2360 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1774 %2267 %2344
       %2361 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1774 %2275 %2345
       %2362 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1774 %2283 %2346
       %2363 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1774 %2291 %2347
       %2364 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1790 %2267 %2348
       %2365 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1790 %2275 %2349
       %2366 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1790 %2283 %2350
       %2367 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1790 %2291 %2351
       %2368 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1806 %2267 %2352
       %2369 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1806 %2275 %2353
       %2370 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1806 %2283 %2354
       %2371 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1806 %2291 %2355
        %534 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1762 %2299 %2356
        %532 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1762 %2307 %2357
        %530 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1762 %2315 %2358
        %528 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1762 %2323 %2359
        %526 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1778 %2299 %2360
        %524 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1778 %2307 %2361
        %522 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1778 %2315 %2362
        %520 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1778 %2323 %2363
        %518 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1794 %2299 %2364
        %516 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1794 %2307 %2365
        %514 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1794 %2315 %2366
        %512 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1794 %2323 %2367
        %510 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1810 %2299 %2368
        %508 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1810 %2307 %2369
        %506 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1810 %2315 %2370
        %504 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %1810 %2323 %2371
               OpMemoryBarrier %uint_2 %uint_4
       %2372 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_signal %uint_4294967295
       %2373 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_wait %ushort_65535
               OpMemoryBarrier %uint_2 %uint_2
       %2374 = OpFunctionCall %void %spirv_llvm_amdgcn_sched_barrier %90
       %2375 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %293
       %2376 = OpBitcast %_ptr_Workgroup_v8half %2375
               OpStore %2376 %1714 Aligned 2
       %2377 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %298
       %2378 = OpBitcast %_ptr_Workgroup_v8half %2377
               OpStore %2378 %1717 Aligned 2
       %2379 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %302
       %2380 = OpBitcast %_ptr_Workgroup_v8half %2379
               OpStore %2380 %1720 Aligned 2
       %2381 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %306
       %2382 = OpBitcast %_ptr_Workgroup_v8half %2381
               OpStore %2382 %1723 Aligned 2
       %2383 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %310
       %2384 = OpBitcast %_ptr_Workgroup_v8half %2383
               OpStore %2384 %1726 Aligned 2
       %2385 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %314
       %2386 = OpBitcast %_ptr_Workgroup_v8half %2385
               OpStore %2386 %1729 Aligned 2
       %2387 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %318
       %2388 = OpBitcast %_ptr_Workgroup_v8half %2387
               OpStore %2388 %1732 Aligned 2
       %2389 = OpPtrAccessChain %_ptr_Workgroup_half %1747 %322
       %2390 = OpBitcast %_ptr_Workgroup_v8half %2389
               OpStore %2390 %1735 Aligned 2
       %2391 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %326
       %2392 = OpBitcast %_ptr_Workgroup_v8half %2391
               OpStore %2392 %1737 Aligned 2
       %2393 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %331
       %2394 = OpBitcast %_ptr_Workgroup_v8half %2393
               OpStore %2394 %1739 Aligned 2
       %2395 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %335
       %2396 = OpBitcast %_ptr_Workgroup_v8half %2395
               OpStore %2396 %1741 Aligned 2
       %2397 = OpPtrAccessChain %_ptr_Workgroup_half %1812 %339
       %2398 = OpBitcast %_ptr_Workgroup_v8half %2397
               OpStore %2398 %1743 Aligned 2
        %500 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %499 %ulong_262144
        %498 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %497 %ulong_262144
        %496 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %495 %ulong_262144
        %494 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %493 %ulong_262144
        %492 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %491 %ulong_128
        %490 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %489 %ulong_128
        %488 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %487 %ulong_128
        %486 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %485 %ulong_128
        %484 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %483 %ulong_128
        %482 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %481 %ulong_128
        %480 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %479 %ulong_128
        %478 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %477 %ulong_128
               OpBranch %6954
               OpFunctionEnd
%matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32 = OpFunction %void Inline %7
       %2399 = OpFunctionParameter %_ptr_CrossWorkgroup_half
       %2400 = OpFunctionParameter %_ptr_CrossWorkgroup_half
       %2401 = OpFunctionParameter %_ptr_CrossWorkgroup_float
       %6957 = OpLabel
       %2403 = OpFunctionCall %uint %spirv_llvm_amdgcn_workitem_id_x
       %2404 = OpSConvert %ulong %2403
       %2405 = OpUDiv %uint %2403 %uint_64
       %2406 = OpUMod %uint %2403 %uint_64
       %2407 = OpUDiv %uint %2406 %uint_32
       %2408 = OpUMod %uint %2403 %uint_32
       %2409 = OpUDiv %uint %2408 %uint_16
       %2410 = OpUMod %uint %2408 %uint_16
       %2411 = OpIMul %uint %2409 %uint_8
       %2412 = OpUConvert %ulong %2411
       %2413 = OpUDiv %uint %2403 %uint_8
       %2414 = OpUConvert %ulong %2413
       %2415 = OpUMod %uint %2403 %uint_8
       %2416 = OpIAdd %uint %2403 %uint_256
       %2417 = OpUDiv %uint %2416 %uint_8
       %2418 = OpUConvert %ulong %2417
       %2419 = OpUMod %uint %2416 %uint_8
       %2420 = OpIAdd %uint %2403 %uint_512
       %2421 = OpUDiv %uint %2420 %uint_8
       %2422 = OpUConvert %ulong %2421
       %2423 = OpUMod %uint %2420 %uint_8
       %2424 = OpIAdd %uint %2403 %uint_768
       %2425 = OpUDiv %uint %2424 %uint_8
       %2426 = OpUConvert %ulong %2425
       %2427 = OpUMod %uint %2424 %uint_8
       %2428 = OpIAdd %uint %2403 %uint_1024
       %2429 = OpUDiv %uint %2428 %uint_8
       %2430 = OpUConvert %ulong %2429
       %2431 = OpUMod %uint %2428 %uint_8
       %2432 = OpIAdd %uint %2403 %uint_1280
       %2433 = OpUDiv %uint %2432 %uint_8
       %2434 = OpUConvert %ulong %2433
       %2435 = OpUMod %uint %2432 %uint_8
       %2436 = OpIAdd %uint %2403 %uint_1536
       %2437 = OpUDiv %uint %2436 %uint_8
       %2438 = OpUConvert %ulong %2437
       %2439 = OpUMod %uint %2436 %uint_8
       %2440 = OpIAdd %uint %2403 %uint_1792
       %2441 = OpUDiv %uint %2440 %uint_8
       %2442 = OpUConvert %ulong %2441
       %2443 = OpUMod %uint %2440 %uint_8
       %2444 = OpUDiv %uint %2403 %uint_16
       %2445 = OpUConvert %ulong %2444
       %2446 = OpUMod %uint %2403 %uint_16
       %2447 = OpUDiv %uint %2416 %uint_16
       %2448 = OpUConvert %ulong %2447
       %2449 = OpUMod %uint %2416 %uint_16
       %2450 = OpUDiv %uint %2420 %uint_16
       %2451 = OpUConvert %ulong %2450
       %2452 = OpUMod %uint %2420 %uint_16
       %2453 = OpUDiv %uint %2424 %uint_16
       %2454 = OpUConvert %ulong %2453
       %2455 = OpUMod %uint %2424 %uint_16
       %2456 = OpIMul %uint %2415 %uint_8
       %2457 = OpUConvert %ulong %2456
       %2458 = OpIMul %uint %2419 %uint_8
       %2459 = OpUConvert %ulong %2458
       %2460 = OpIMul %uint %2423 %uint_8
       %2461 = OpUConvert %ulong %2460
       %2462 = OpIMul %uint %2427 %uint_8
       %2463 = OpUConvert %ulong %2462
       %2464 = OpIMul %uint %2431 %uint_8
       %2465 = OpUConvert %ulong %2464
       %2466 = OpIMul %uint %2435 %uint_8
       %2467 = OpUConvert %ulong %2466
       %2468 = OpIMul %uint %2439 %uint_8
       %2469 = OpUConvert %ulong %2468
       %2470 = OpIMul %uint %2443 %uint_8
       %2471 = OpUConvert %ulong %2470
       %2472 = OpIMul %uint %2446 %uint_8
       %2473 = OpUConvert %ulong %2472
       %2474 = OpIMul %uint %2449 %uint_8
       %2475 = OpUConvert %ulong %2474
       %2476 = OpIMul %uint %2452 %uint_8
       %2477 = OpUConvert %ulong %2476
       %2478 = OpIMul %uint %2455 %uint_8
       %2479 = OpUConvert %ulong %2478
       %2480 = OpFunctionCall %uint %spirv_llvm_amdgcn_workgroup_id_x
       %2481 = OpUDiv %uint %2480 %uint_8
       %2482 = OpUMod %uint %2480 %uint_8
       %2483 = OpIMul %uint %2481 %uint_256
       %2484 = OpIAdd %uint %2413 %2483
       %2485 = OpUConvert %ulong %2484
       %2486 = OpIAdd %uint %2417 %2483
       %2487 = OpUConvert %ulong %2486
       %2488 = OpIMul %ulong %2485 %ulong_4096
       %2489 = OpIAdd %ulong %2488 %2457
       %2490 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %2399 %2489
       %2491 = OpBitcast %_ptr_CrossWorkgroup_v8half %2490
       %2492 = OpLoad %v8half %2491 Aligned 2
       %2493 = OpIAdd %uint %2421 %2483
       %2494 = OpUConvert %ulong %2493
       %2495 = OpIMul %ulong %2487 %ulong_4096
       %2496 = OpIAdd %ulong %2495 %2459
       %2497 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %2399 %2496
       %2498 = OpBitcast %_ptr_CrossWorkgroup_v8half %2497
       %2499 = OpLoad %v8half %2498 Aligned 2
       %2500 = OpIAdd %uint %2425 %2483
       %2501 = OpUConvert %ulong %2500
       %2502 = OpIMul %ulong %2494 %ulong_4096
       %2503 = OpIAdd %ulong %2502 %2461
       %2504 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %2399 %2503
       %2505 = OpBitcast %_ptr_CrossWorkgroup_v8half %2504
       %2506 = OpLoad %v8half %2505 Aligned 2
       %2507 = OpIAdd %uint %2429 %2483
       %2508 = OpUConvert %ulong %2507
       %2509 = OpIMul %ulong %2501 %ulong_4096
       %2510 = OpIAdd %ulong %2509 %2463
       %2511 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %2399 %2510
       %2512 = OpBitcast %_ptr_CrossWorkgroup_v8half %2511
       %2513 = OpLoad %v8half %2512 Aligned 2
       %2514 = OpIAdd %uint %2433 %2483
       %2515 = OpUConvert %ulong %2514
       %2516 = OpIMul %ulong %2508 %ulong_4096
       %2517 = OpIAdd %ulong %2516 %2465
       %2518 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %2399 %2517
       %2519 = OpBitcast %_ptr_CrossWorkgroup_v8half %2518
       %2520 = OpLoad %v8half %2519 Aligned 2
       %2521 = OpIAdd %uint %2437 %2483
       %2522 = OpUConvert %ulong %2521
       %2523 = OpIMul %ulong %2515 %ulong_4096
       %2524 = OpIAdd %ulong %2523 %2467
       %2525 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %2399 %2524
       %2526 = OpBitcast %_ptr_CrossWorkgroup_v8half %2525
       %2527 = OpLoad %v8half %2526 Aligned 2
       %2528 = OpIAdd %uint %2441 %2483
       %2529 = OpUConvert %ulong %2528
       %2530 = OpIMul %ulong %2522 %ulong_4096
       %2531 = OpIAdd %ulong %2530 %2469
       %2532 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %2399 %2531
       %2533 = OpBitcast %_ptr_CrossWorkgroup_v8half %2532
       %2534 = OpLoad %v8half %2533 Aligned 2
       %2535 = OpIMul %uint %2482 %uint_128
       %2536 = OpIAdd %uint %2472 %2535
       %2537 = OpUConvert %ulong %2536
       %2538 = OpIMul %ulong %2529 %ulong_4096
       %2539 = OpIAdd %ulong %2538 %2471
       %2540 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %2399 %2539
       %2541 = OpBitcast %_ptr_CrossWorkgroup_v8half %2540
       %2542 = OpLoad %v8half %2541 Aligned 2
       %2543 = OpIAdd %uint %2474 %2535
       %2544 = OpUConvert %ulong %2543
       %2545 = OpIMul %ulong %2445 %ulong_1024
       %2546 = OpIAdd %ulong %2545 %2537
       %2547 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %2400 %2546
       %2548 = OpBitcast %_ptr_CrossWorkgroup_v8half %2547
       %2549 = OpLoad %v8half %2548 Aligned 2
       %2550 = OpIAdd %uint %2476 %2535
       %2551 = OpUConvert %ulong %2550
       %2552 = OpIMul %ulong %2448 %ulong_1024
       %2553 = OpIAdd %ulong %2552 %2544
       %2554 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %2400 %2553
       %2555 = OpBitcast %_ptr_CrossWorkgroup_v8half %2554
       %2556 = OpLoad %v8half %2555 Aligned 2
       %2557 = OpIAdd %uint %2478 %2535
       %2558 = OpUConvert %ulong %2557
       %2559 = OpIMul %ulong %2451 %ulong_1024
       %2560 = OpIAdd %ulong %2559 %2551
       %2561 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %2400 %2560
       %2562 = OpBitcast %_ptr_CrossWorkgroup_v8half %2561
       %2563 = OpLoad %v8half %2562 Aligned 2
       %2564 = OpIMul %ulong %2454 %ulong_1024
       %2565 = OpIAdd %ulong %2564 %2558
       %2566 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %2400 %2565
       %2567 = OpBitcast %_ptr_CrossWorkgroup_v8half %2566
       %2568 = OpLoad %v8half %2567 Aligned 2
       %2569 = OpIMul %ulong %2414 %ulong_68
       %2570 = OpIAdd %ulong %2569 %2457
       %2571 = OpBitcast %_ptr_Workgroup_half %__shared_memory__
       %2572 = OpPtrAccessChain %_ptr_Workgroup_half %2571 %2570
       %2573 = OpBitcast %_ptr_Workgroup_v8half %2572
               OpStore %2573 %2492 Aligned 2
       %2574 = OpIMul %ulong %2418 %ulong_68
       %2575 = OpIAdd %ulong %2574 %2459
       %2576 = OpPtrAccessChain %_ptr_Workgroup_half %2571 %2575
       %2577 = OpBitcast %_ptr_Workgroup_v8half %2576
               OpStore %2577 %2499 Aligned 2
       %2578 = OpIMul %ulong %2422 %ulong_68
       %2579 = OpIAdd %ulong %2578 %2461
       %2580 = OpPtrAccessChain %_ptr_Workgroup_half %2571 %2579
       %2581 = OpBitcast %_ptr_Workgroup_v8half %2580
               OpStore %2581 %2506 Aligned 2
       %2582 = OpIMul %ulong %2426 %ulong_68
       %2583 = OpIAdd %ulong %2582 %2463
       %2584 = OpPtrAccessChain %_ptr_Workgroup_half %2571 %2583
       %2585 = OpBitcast %_ptr_Workgroup_v8half %2584
               OpStore %2585 %2513 Aligned 2
       %2586 = OpIMul %ulong %2430 %ulong_68
       %2587 = OpIAdd %ulong %2586 %2465
       %2588 = OpPtrAccessChain %_ptr_Workgroup_half %2571 %2587
       %2589 = OpBitcast %_ptr_Workgroup_v8half %2588
               OpStore %2589 %2520 Aligned 2
       %2590 = OpIMul %ulong %2434 %ulong_68
       %2591 = OpIAdd %ulong %2590 %2467
       %2592 = OpPtrAccessChain %_ptr_Workgroup_half %2571 %2591
       %2593 = OpBitcast %_ptr_Workgroup_v8half %2592
               OpStore %2593 %2527 Aligned 2
       %2594 = OpIMul %ulong %2438 %ulong_68
       %2595 = OpIAdd %ulong %2594 %2469
       %2596 = OpPtrAccessChain %_ptr_Workgroup_half %2571 %2595
       %2597 = OpBitcast %_ptr_Workgroup_v8half %2596
               OpStore %2597 %2534 Aligned 2
       %2598 = OpIMul %ulong %2442 %ulong_68
       %2599 = OpIAdd %ulong %2598 %2471
       %2600 = OpPtrAccessChain %_ptr_Workgroup_half %2571 %2599
       %2601 = OpBitcast %_ptr_Workgroup_v8half %2600
               OpStore %2601 %2542 Aligned 2
       %2602 = OpIMul %ulong %2445 %ulong_132
       %2603 = OpIAdd %ulong %2602 %2473
       %2604 = OpBitcast %_ptr_Workgroup_half %__shared_memory___0
       %2605 = OpPtrAccessChain %_ptr_Workgroup_half %2604 %2603
       %2606 = OpBitcast %_ptr_Workgroup_v8half %2605
               OpStore %2606 %2549 Aligned 2
       %2607 = OpIMul %ulong %2448 %ulong_132
       %2608 = OpIAdd %ulong %2607 %2475
       %2609 = OpPtrAccessChain %_ptr_Workgroup_half %2604 %2608
       %2610 = OpBitcast %_ptr_Workgroup_v8half %2609
               OpStore %2610 %2556 Aligned 2
       %2611 = OpIMul %ulong %2451 %ulong_132
       %2612 = OpIAdd %ulong %2611 %2477
       %2613 = OpPtrAccessChain %_ptr_Workgroup_half %2604 %2612
       %2614 = OpBitcast %_ptr_Workgroup_v8half %2613
               OpStore %2614 %2563 Aligned 2
       %2615 = OpIMul %ulong %2454 %ulong_132
       %2616 = OpIAdd %ulong %2615 %2479
       %2617 = OpPtrAccessChain %_ptr_Workgroup_half %2604 %2616
       %2618 = OpBitcast %_ptr_Workgroup_v8half %2617
               OpStore %2618 %2568 Aligned 2
       %2619 = OpIMul %uint %2405 %uint_4
       %2620 = OpIMul %uint %2405 %uint_64
       %2621 = OpIAdd %uint %2620 %2410
       %2622 = OpUConvert %ulong %2621
       %2623 = OpIAdd %uint %2411 %uint_16
       %2624 = OpUConvert %ulong %2623
       %2625 = OpIAdd %uint %2411 %uint_32
       %2626 = OpUConvert %ulong %2625
       %2627 = OpIAdd %uint %2411 %uint_48
       %2628 = OpUConvert %ulong %2627
       %2629 = OpIAdd %uint %2621 %uint_16
       %2630 = OpUConvert %ulong %2629
       %2631 = OpIAdd %uint %2621 %uint_32
       %2632 = OpUConvert %ulong %2631
       %2633 = OpIAdd %uint %2621 %uint_48
       %2634 = OpUConvert %ulong %2633
       %2635 = OpIMul %uint %2407 %uint_4
       %2636 = OpIMul %uint %2407 %uint_64
       %2637 = OpIAdd %uint %2636 %2410
       %2638 = OpUConvert %ulong %2637
       %2639 = OpIAdd %uint %2637 %uint_16
       %2640 = OpUConvert %ulong %2639
       %2641 = OpIAdd %uint %2637 %uint_32
       %2642 = OpUConvert %ulong %2641
       %2643 = OpIAdd %uint %2637 %uint_48
       %2644 = OpUConvert %ulong %2643
       %2645 = OpIAdd %uint %2411 %uint_1
       %2646 = OpUConvert %ulong %2645
       %2647 = OpIAdd %uint %2411 %uint_2
       %2648 = OpUConvert %ulong %2647
       %2649 = OpIAdd %uint %2411 %uint_3
       %2650 = OpUConvert %ulong %2649
       %2651 = OpIAdd %uint %2411 %uint_4
       %2652 = OpUConvert %ulong %2651
       %2653 = OpIAdd %uint %2411 %uint_5
       %2654 = OpUConvert %ulong %2653
       %2655 = OpIAdd %uint %2411 %uint_6
       %2656 = OpUConvert %ulong %2655
       %2657 = OpIAdd %uint %2411 %uint_7
       %2658 = OpUConvert %ulong %2657
       %2659 = OpIAdd %uint %2411 %uint_17
       %2660 = OpUConvert %ulong %2659
       %2661 = OpIAdd %uint %2411 %uint_18
       %2662 = OpUConvert %ulong %2661
       %2663 = OpIAdd %uint %2411 %uint_19
       %2664 = OpUConvert %ulong %2663
       %2665 = OpIAdd %uint %2411 %uint_20
       %2666 = OpUConvert %ulong %2665
       %2667 = OpIAdd %uint %2411 %uint_21
       %2668 = OpUConvert %ulong %2667
       %2669 = OpIAdd %uint %2411 %uint_22
       %2670 = OpUConvert %ulong %2669
       %2671 = OpIAdd %uint %2411 %uint_23
       %2672 = OpUConvert %ulong %2671
       %2673 = OpIAdd %uint %2411 %uint_33
       %2674 = OpUConvert %ulong %2673
       %2675 = OpIAdd %uint %2411 %uint_34
       %2676 = OpUConvert %ulong %2675
       %2677 = OpIAdd %uint %2411 %uint_35
       %2678 = OpUConvert %ulong %2677
       %2679 = OpIAdd %uint %2411 %uint_36
       %2680 = OpUConvert %ulong %2679
       %2681 = OpIAdd %uint %2411 %uint_37
       %2682 = OpUConvert %ulong %2681
       %2683 = OpIAdd %uint %2411 %uint_38
       %2684 = OpUConvert %ulong %2683
       %2685 = OpIAdd %uint %2411 %uint_39
       %2686 = OpUConvert %ulong %2685
       %2687 = OpIAdd %uint %2411 %uint_49
       %2688 = OpUConvert %ulong %2687
       %2689 = OpIAdd %uint %2411 %uint_50
       %2690 = OpUConvert %ulong %2689
       %2691 = OpIAdd %uint %2411 %uint_51
       %2692 = OpUConvert %ulong %2691
       %2693 = OpIAdd %uint %2411 %uint_52
       %2694 = OpUConvert %ulong %2693
       %2695 = OpIAdd %uint %2411 %uint_53
       %2696 = OpUConvert %ulong %2695
       %2697 = OpIAdd %uint %2411 %uint_54
       %2698 = OpUConvert %ulong %2697
       %2699 = OpIAdd %uint %2411 %uint_55
       %2700 = OpUConvert %ulong %2699
       %2701 = OpIAdd %ulong %2404 %ulong_768
       %2702 = OpShiftRightLogical %ulong %2701 %ulong_4
       %2703 = OpShiftLeftLogical %ulong %2702 %ulong_11
       %2704 = OpShiftLeftLogical %ulong %2558 %ulong_1
       %2705 = OpIAdd %ulong %2703 %2704
       %2706 = OpIAdd %ulong %2705 %ulong_131072
       %2707 = OpBitcast %_ptr_CrossWorkgroup_uchar %2400
       %2708 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2707 %2706
       %2709 = OpIAdd %ulong %2404 %ulong_512
       %2710 = OpShiftRightLogical %ulong %2709 %ulong_4
       %2711 = OpShiftLeftLogical %ulong %2710 %ulong_11
       %2712 = OpIAdd %ulong %2711 %2704
       %2713 = OpIAdd %ulong %2712 %ulong_131072
       %2714 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2707 %2713
       %2715 = OpIAdd %ulong %2404 %ulong_256
       %2716 = OpShiftRightLogical %ulong %2715 %ulong_4
       %2717 = OpShiftLeftLogical %ulong %2716 %ulong_11
       %2718 = OpIAdd %ulong %2717 %2704
       %2719 = OpIAdd %ulong %2718 %ulong_131072
       %2720 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2707 %2719
       %2721 = OpShiftRightLogical %ulong %2404 %ulong_4
       %2722 = OpShiftLeftLogical %ulong %2721 %ulong_11
       %2723 = OpIAdd %ulong %2722 %2704
       %2724 = OpIAdd %ulong %2723 %ulong_131072
       %2725 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2707 %2724
       %2726 = OpUConvert %ulong %2403
       %2727 = OpBitwiseAnd %ulong %2726 %ulong_7
       %2728 = OpShiftLeftLogical %ulong %2727 %ulong_4
       %2729 = OpShiftLeftLogical %ulong %2529 %ulong_13
       %2730 = OpIAdd %ulong %2729 %ulong_128
       %2731 = OpBitcast %_ptr_CrossWorkgroup_uchar %2399
       %2732 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2731 %2730
       %2733 = OpShiftLeftLogical %ulong %2522 %ulong_13
       %2734 = OpIAdd %ulong %2733 %ulong_128
       %2735 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2731 %2734
       %2736 = OpShiftLeftLogical %ulong %2515 %ulong_13
       %2737 = OpIAdd %ulong %2736 %ulong_128
       %2738 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2731 %2737
       %2739 = OpShiftLeftLogical %ulong %2508 %ulong_13
       %2740 = OpIAdd %ulong %2739 %ulong_128
       %2741 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2731 %2740
       %2742 = OpShiftLeftLogical %ulong %2501 %ulong_13
       %2743 = OpIAdd %ulong %2742 %ulong_128
       %2744 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2731 %2743
       %2745 = OpShiftLeftLogical %ulong %2494 %ulong_13
       %2746 = OpIAdd %ulong %2745 %ulong_128
       %2747 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2731 %2746
       %2748 = OpShiftLeftLogical %ulong %2487 %ulong_13
       %2749 = OpIAdd %ulong %2748 %ulong_128
       %2750 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2731 %2749
       %2751 = OpShiftLeftLogical %ulong %2485 %ulong_13
       %2752 = OpIAdd %ulong %2751 %ulong_128
       %2753 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2731 %2752
               OpBranch %6958
       %6958 = OpLabel
       %2754 = OpPhi %_ptr_CrossWorkgroup_uchar %2755 %6959 %2753 %6957
       %2756 = OpPhi %_ptr_CrossWorkgroup_uchar %2757 %6959 %2750 %6957
       %2758 = OpPhi %_ptr_CrossWorkgroup_uchar %2759 %6959 %2747 %6957
       %2760 = OpPhi %_ptr_CrossWorkgroup_uchar %2761 %6959 %2744 %6957
       %2762 = OpPhi %_ptr_CrossWorkgroup_uchar %2763 %6959 %2741 %6957
       %2764 = OpPhi %_ptr_CrossWorkgroup_uchar %2765 %6959 %2738 %6957
       %2766 = OpPhi %_ptr_CrossWorkgroup_uchar %2767 %6959 %2735 %6957
       %2768 = OpPhi %_ptr_CrossWorkgroup_uchar %2769 %6959 %2732 %6957
       %2770 = OpPhi %_ptr_CrossWorkgroup_uchar %2771 %6959 %2725 %6957
       %2772 = OpPhi %_ptr_CrossWorkgroup_uchar %2773 %6959 %2720 %6957
       %2774 = OpPhi %_ptr_CrossWorkgroup_uchar %2775 %6959 %2714 %6957
       %2776 = OpPhi %_ptr_CrossWorkgroup_uchar %2777 %6959 %2708 %6957
       %2778 = OpPhi %uint %2779 %6959 %90 %6957
       %2780 = OpPhi %v8float %2781 %6959 %113 %6957
       %2782 = OpPhi %v8float %2783 %6959 %113 %6957
       %2784 = OpPhi %v8float %2785 %6959 %113 %6957
       %2786 = OpPhi %v8float %2787 %6959 %113 %6957
       %2788 = OpPhi %v8float %2789 %6959 %113 %6957
       %2790 = OpPhi %v8float %2791 %6959 %113 %6957
       %2792 = OpPhi %v8float %2793 %6959 %113 %6957
       %2794 = OpPhi %v8float %2795 %6959 %113 %6957
       %2796 = OpPhi %v8float %2797 %6959 %113 %6957
       %2798 = OpPhi %v8float %2799 %6959 %113 %6957
       %2800 = OpPhi %v8float %2801 %6959 %113 %6957
       %2802 = OpPhi %v8float %2803 %6959 %113 %6957
       %2804 = OpPhi %v8float %2805 %6959 %113 %6957
       %2806 = OpPhi %v8float %2807 %6959 %113 %6957
       %2808 = OpPhi %v8float %2809 %6959 %113 %6957
       %2810 = OpPhi %v8float %2811 %6959 %113 %6957
       %2812 = OpSLessThan %bool %2778 %uint_252
       %2813 = OpIMul %ulong %2622 %ulong_68
       %2814 = OpIMul %ulong %2630 %ulong_68
       %2815 = OpIMul %ulong %2632 %ulong_68
       %2816 = OpIMul %ulong %2634 %ulong_68
       %2817 = OpIMul %ulong %2412 %ulong_132
       %2818 = OpIMul %ulong %2646 %ulong_132
       %2819 = OpIMul %ulong %2648 %ulong_132
       %2820 = OpIMul %ulong %2650 %ulong_132
       %2821 = OpIMul %ulong %2652 %ulong_132
       %2822 = OpIMul %ulong %2654 %ulong_132
       %2823 = OpIMul %ulong %2656 %ulong_132
       %2824 = OpIMul %ulong %2658 %ulong_132
       %2825 = OpIMul %ulong %2624 %ulong_132
       %2826 = OpIMul %ulong %2660 %ulong_132
       %2827 = OpIMul %ulong %2662 %ulong_132
       %2828 = OpIMul %ulong %2664 %ulong_132
       %2829 = OpIMul %ulong %2666 %ulong_132
       %2830 = OpIMul %ulong %2668 %ulong_132
       %2831 = OpIMul %ulong %2670 %ulong_132
       %2832 = OpIMul %ulong %2672 %ulong_132
       %2833 = OpIMul %ulong %2626 %ulong_132
       %2834 = OpIMul %ulong %2674 %ulong_132
       %2835 = OpIMul %ulong %2676 %ulong_132
       %2836 = OpIMul %ulong %2678 %ulong_132
       %2837 = OpIMul %ulong %2680 %ulong_132
       %2838 = OpIMul %ulong %2682 %ulong_132
       %2839 = OpIMul %ulong %2684 %ulong_132
       %2840 = OpIMul %ulong %2686 %ulong_132
       %2841 = OpIMul %ulong %2628 %ulong_132
       %2842 = OpIMul %ulong %2688 %ulong_132
       %2843 = OpIMul %ulong %2690 %ulong_132
       %2844 = OpIMul %ulong %2692 %ulong_132
       %2845 = OpIMul %ulong %2694 %ulong_132
       %2846 = OpIMul %ulong %2696 %ulong_132
       %2847 = OpIMul %ulong %2698 %ulong_132
       %2848 = OpIMul %ulong %2700 %ulong_132
               OpBranchConditional %2812 %6959 %6960
       %6960 = OpLabel
               OpMemoryBarrier %uint_2 %uint_4
       %2849 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_signal %uint_4294967295
       %2850 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_wait %ushort_65535
               OpMemoryBarrier %uint_2 %uint_2
       %2851 = OpIAdd %ulong %2813 %2412
       %2852 = OpBitcast %_ptr_Workgroup_half %__shared_memory__
       %2853 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2851
       %2854 = OpBitcast %_ptr_Workgroup_v8half %2853
       %2855 = OpLoad %v8half %2854 Aligned 2
       %2856 = OpIAdd %ulong %2813 %2624
       %2857 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2856
       %2858 = OpBitcast %_ptr_Workgroup_v8half %2857
       %2859 = OpLoad %v8half %2858 Aligned 2
       %2860 = OpIAdd %ulong %2813 %2626
       %2861 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2860
       %2862 = OpBitcast %_ptr_Workgroup_v8half %2861
       %2863 = OpLoad %v8half %2862 Aligned 2
       %2864 = OpIAdd %ulong %2813 %2628
       %2865 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2864
       %2866 = OpBitcast %_ptr_Workgroup_v8half %2865
       %2867 = OpLoad %v8half %2866 Aligned 2
       %2868 = OpIAdd %ulong %2814 %2412
       %2869 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2868
       %2870 = OpBitcast %_ptr_Workgroup_v8half %2869
       %2871 = OpLoad %v8half %2870 Aligned 2
       %2872 = OpIAdd %ulong %2814 %2624
       %2873 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2872
       %2874 = OpBitcast %_ptr_Workgroup_v8half %2873
       %2875 = OpLoad %v8half %2874 Aligned 2
       %2876 = OpIAdd %ulong %2814 %2626
       %2877 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2876
       %2878 = OpBitcast %_ptr_Workgroup_v8half %2877
       %2879 = OpLoad %v8half %2878 Aligned 2
       %2880 = OpIAdd %ulong %2814 %2628
       %2881 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2880
       %2882 = OpBitcast %_ptr_Workgroup_v8half %2881
       %2883 = OpLoad %v8half %2882 Aligned 2
       %2884 = OpIAdd %ulong %2815 %2412
       %2885 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2884
       %2886 = OpBitcast %_ptr_Workgroup_v8half %2885
       %2887 = OpLoad %v8half %2886 Aligned 2
       %2888 = OpIAdd %ulong %2815 %2624
       %2889 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2888
       %2890 = OpBitcast %_ptr_Workgroup_v8half %2889
       %2891 = OpLoad %v8half %2890 Aligned 2
       %2892 = OpIAdd %ulong %2815 %2626
       %2893 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2892
       %2894 = OpBitcast %_ptr_Workgroup_v8half %2893
       %2895 = OpLoad %v8half %2894 Aligned 2
       %2896 = OpIAdd %ulong %2815 %2628
       %2897 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2896
       %2898 = OpBitcast %_ptr_Workgroup_v8half %2897
       %2899 = OpLoad %v8half %2898 Aligned 2
       %2900 = OpIAdd %ulong %2816 %2412
       %2901 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2900
       %2902 = OpBitcast %_ptr_Workgroup_v8half %2901
       %2903 = OpLoad %v8half %2902 Aligned 2
       %2904 = OpIAdd %ulong %2816 %2624
       %2905 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2904
       %2906 = OpBitcast %_ptr_Workgroup_v8half %2905
       %2907 = OpLoad %v8half %2906 Aligned 2
       %2908 = OpIAdd %ulong %2816 %2626
       %2909 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2908
       %2910 = OpBitcast %_ptr_Workgroup_v8half %2909
       %2911 = OpLoad %v8half %2910 Aligned 2
       %2912 = OpIAdd %ulong %2816 %2628
       %2913 = OpPtrAccessChain %_ptr_Workgroup_half %2852 %2912
       %2914 = OpBitcast %_ptr_Workgroup_v8half %2913
       %2915 = OpLoad %v8half %2914 Aligned 2
       %2916 = OpIAdd %ulong %2817 %2638
       %2917 = OpBitcast %_ptr_Workgroup_half %__shared_memory___0
       %2918 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2916
       %2919 = OpLoad %half %2918 Aligned 2
       %2920 = OpIAdd %ulong %2817 %2640
       %2921 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2920
       %2922 = OpLoad %half %2921 Aligned 2
       %2923 = OpIAdd %ulong %2817 %2642
       %2924 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2923
       %2925 = OpLoad %half %2924 Aligned 2
       %2926 = OpIAdd %ulong %2817 %2644
       %2927 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2926
       %2928 = OpLoad %half %2927 Aligned 2
       %2929 = OpIAdd %ulong %2818 %2638
       %2930 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2929
       %2931 = OpLoad %half %2930 Aligned 2
       %2932 = OpIAdd %ulong %2818 %2640
       %2933 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2932
       %2934 = OpLoad %half %2933 Aligned 2
       %2935 = OpIAdd %ulong %2818 %2642
       %2936 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2935
       %2937 = OpLoad %half %2936 Aligned 2
       %2938 = OpIAdd %ulong %2818 %2644
       %2939 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2938
       %2940 = OpLoad %half %2939 Aligned 2
       %2941 = OpIAdd %ulong %2819 %2638
       %2942 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2941
       %2943 = OpLoad %half %2942 Aligned 2
       %2944 = OpIAdd %ulong %2819 %2640
       %2945 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2944
       %2946 = OpLoad %half %2945 Aligned 2
       %2947 = OpIAdd %ulong %2819 %2642
       %2948 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2947
       %2949 = OpLoad %half %2948 Aligned 2
       %2950 = OpIAdd %ulong %2819 %2644
       %2951 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2950
       %2952 = OpLoad %half %2951 Aligned 2
       %2953 = OpIAdd %ulong %2820 %2638
       %2954 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2953
       %2955 = OpLoad %half %2954 Aligned 2
       %2956 = OpIAdd %ulong %2820 %2640
       %2957 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2956
       %2958 = OpLoad %half %2957 Aligned 2
       %2959 = OpIAdd %ulong %2820 %2642
       %2960 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2959
       %2961 = OpLoad %half %2960 Aligned 2
       %2962 = OpIAdd %ulong %2820 %2644
       %2963 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2962
       %2964 = OpLoad %half %2963 Aligned 2
       %2965 = OpIAdd %ulong %2821 %2638
       %2966 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2965
       %2967 = OpLoad %half %2966 Aligned 2
       %2968 = OpIAdd %ulong %2821 %2640
       %2969 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2968
       %2970 = OpLoad %half %2969 Aligned 2
       %2971 = OpIAdd %ulong %2821 %2642
       %2972 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2971
       %2973 = OpLoad %half %2972 Aligned 2
       %2974 = OpIAdd %ulong %2821 %2644
       %2975 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2974
       %2976 = OpLoad %half %2975 Aligned 2
       %2977 = OpIAdd %ulong %2822 %2638
       %2978 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2977
       %2979 = OpLoad %half %2978 Aligned 2
       %2980 = OpIAdd %ulong %2822 %2640
       %2981 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2980
       %2982 = OpLoad %half %2981 Aligned 2
       %2983 = OpIAdd %ulong %2822 %2642
       %2984 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2983
       %2985 = OpLoad %half %2984 Aligned 2
       %2986 = OpIAdd %ulong %2822 %2644
       %2987 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2986
       %2988 = OpLoad %half %2987 Aligned 2
       %2989 = OpIAdd %ulong %2823 %2638
       %2990 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2989
       %2991 = OpLoad %half %2990 Aligned 2
       %2992 = OpIAdd %ulong %2823 %2640
       %2993 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2992
       %2994 = OpLoad %half %2993 Aligned 2
       %2995 = OpIAdd %ulong %2823 %2642
       %2996 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2995
       %2997 = OpLoad %half %2996 Aligned 2
       %2998 = OpIAdd %ulong %2823 %2644
       %2999 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %2998
       %3000 = OpLoad %half %2999 Aligned 2
       %3001 = OpIAdd %ulong %2824 %2638
       %3002 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3001
       %3003 = OpLoad %half %3002 Aligned 2
       %3004 = OpIAdd %ulong %2824 %2640
       %3005 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3004
       %3006 = OpLoad %half %3005 Aligned 2
       %3007 = OpIAdd %ulong %2824 %2642
       %3008 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3007
       %3009 = OpLoad %half %3008 Aligned 2
       %3010 = OpIAdd %ulong %2824 %2644
       %3011 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3010
       %3012 = OpLoad %half %3011 Aligned 2
       %3013 = OpIAdd %ulong %2825 %2638
       %3014 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3013
       %3015 = OpLoad %half %3014 Aligned 2
       %3016 = OpIAdd %ulong %2825 %2640
       %3017 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3016
       %3018 = OpLoad %half %3017 Aligned 2
       %3019 = OpIAdd %ulong %2825 %2642
       %3020 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3019
       %3021 = OpLoad %half %3020 Aligned 2
       %3022 = OpIAdd %ulong %2825 %2644
       %3023 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3022
       %3024 = OpLoad %half %3023 Aligned 2
       %3025 = OpIAdd %ulong %2826 %2638
       %3026 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3025
       %3027 = OpLoad %half %3026 Aligned 2
       %3028 = OpIAdd %ulong %2826 %2640
       %3029 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3028
       %3030 = OpLoad %half %3029 Aligned 2
       %3031 = OpIAdd %ulong %2826 %2642
       %3032 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3031
       %3033 = OpLoad %half %3032 Aligned 2
       %3034 = OpIAdd %ulong %2826 %2644
       %3035 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3034
       %3036 = OpLoad %half %3035 Aligned 2
       %3037 = OpIAdd %ulong %2827 %2638
       %3038 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3037
       %3039 = OpLoad %half %3038 Aligned 2
       %3040 = OpIAdd %ulong %2827 %2640
       %3041 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3040
       %3042 = OpLoad %half %3041 Aligned 2
       %3043 = OpIAdd %ulong %2827 %2642
       %3044 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3043
       %3045 = OpLoad %half %3044 Aligned 2
       %3046 = OpIAdd %ulong %2827 %2644
       %3047 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3046
       %3048 = OpLoad %half %3047 Aligned 2
       %3049 = OpIAdd %ulong %2828 %2638
       %3050 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3049
       %3051 = OpLoad %half %3050 Aligned 2
       %3052 = OpIAdd %ulong %2828 %2640
       %3053 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3052
       %3054 = OpLoad %half %3053 Aligned 2
       %3055 = OpIAdd %ulong %2828 %2642
       %3056 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3055
       %3057 = OpLoad %half %3056 Aligned 2
       %3058 = OpIAdd %ulong %2828 %2644
       %3059 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3058
       %3060 = OpLoad %half %3059 Aligned 2
       %3061 = OpIAdd %ulong %2829 %2638
       %3062 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3061
       %3063 = OpLoad %half %3062 Aligned 2
       %3064 = OpIAdd %ulong %2829 %2640
       %3065 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3064
       %3066 = OpLoad %half %3065 Aligned 2
       %3067 = OpIAdd %ulong %2829 %2642
       %3068 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3067
       %3069 = OpLoad %half %3068 Aligned 2
       %3070 = OpIAdd %ulong %2829 %2644
       %3071 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3070
       %3072 = OpLoad %half %3071 Aligned 2
       %3073 = OpIAdd %ulong %2830 %2638
       %3074 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3073
       %3075 = OpLoad %half %3074 Aligned 2
       %3076 = OpIAdd %ulong %2830 %2640
       %3077 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3076
       %3078 = OpLoad %half %3077 Aligned 2
       %3079 = OpIAdd %ulong %2830 %2642
       %3080 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3079
       %3081 = OpLoad %half %3080 Aligned 2
       %3082 = OpIAdd %ulong %2830 %2644
       %3083 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3082
       %3084 = OpLoad %half %3083 Aligned 2
       %3085 = OpIAdd %ulong %2831 %2638
       %3086 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3085
       %3087 = OpLoad %half %3086 Aligned 2
       %3088 = OpIAdd %ulong %2831 %2640
       %3089 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3088
       %3090 = OpLoad %half %3089 Aligned 2
       %3091 = OpIAdd %ulong %2831 %2642
       %3092 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3091
       %3093 = OpLoad %half %3092 Aligned 2
       %3094 = OpIAdd %ulong %2831 %2644
       %3095 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3094
       %3096 = OpLoad %half %3095 Aligned 2
       %3097 = OpIAdd %ulong %2832 %2638
       %3098 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3097
       %3099 = OpLoad %half %3098 Aligned 2
       %3100 = OpIAdd %ulong %2832 %2640
       %3101 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3100
       %3102 = OpLoad %half %3101 Aligned 2
       %3103 = OpIAdd %ulong %2832 %2642
       %3104 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3103
       %3105 = OpLoad %half %3104 Aligned 2
       %3106 = OpIAdd %ulong %2832 %2644
       %3107 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3106
       %3108 = OpLoad %half %3107 Aligned 2
       %3109 = OpIAdd %ulong %2833 %2638
       %3110 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3109
       %3111 = OpLoad %half %3110 Aligned 2
       %3112 = OpIAdd %ulong %2833 %2640
       %3113 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3112
       %3114 = OpLoad %half %3113 Aligned 2
       %3115 = OpIAdd %ulong %2833 %2642
       %3116 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3115
       %3117 = OpLoad %half %3116 Aligned 2
       %3118 = OpIAdd %ulong %2833 %2644
       %3119 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3118
       %3120 = OpLoad %half %3119 Aligned 2
       %3121 = OpIAdd %ulong %2834 %2638
       %3122 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3121
       %3123 = OpLoad %half %3122 Aligned 2
       %3124 = OpIAdd %ulong %2834 %2640
       %3125 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3124
       %3126 = OpLoad %half %3125 Aligned 2
       %3127 = OpIAdd %ulong %2834 %2642
       %3128 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3127
       %3129 = OpLoad %half %3128 Aligned 2
       %3130 = OpIAdd %ulong %2834 %2644
       %3131 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3130
       %3132 = OpLoad %half %3131 Aligned 2
       %3133 = OpIAdd %ulong %2835 %2638
       %3134 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3133
       %3135 = OpLoad %half %3134 Aligned 2
       %3136 = OpIAdd %ulong %2835 %2640
       %3137 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3136
       %3138 = OpLoad %half %3137 Aligned 2
       %3139 = OpIAdd %ulong %2835 %2642
       %3140 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3139
       %3141 = OpLoad %half %3140 Aligned 2
       %3142 = OpIAdd %ulong %2835 %2644
       %3143 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3142
       %3144 = OpLoad %half %3143 Aligned 2
       %3145 = OpIAdd %ulong %2836 %2638
       %3146 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3145
       %3147 = OpLoad %half %3146 Aligned 2
       %3148 = OpIAdd %ulong %2836 %2640
       %3149 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3148
       %3150 = OpLoad %half %3149 Aligned 2
       %3151 = OpIAdd %ulong %2836 %2642
       %3152 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3151
       %3153 = OpLoad %half %3152 Aligned 2
       %3154 = OpIAdd %ulong %2836 %2644
       %3155 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3154
       %3156 = OpLoad %half %3155 Aligned 2
       %3157 = OpIAdd %ulong %2837 %2638
       %3158 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3157
       %3159 = OpLoad %half %3158 Aligned 2
       %3160 = OpIAdd %ulong %2837 %2640
       %3161 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3160
       %3162 = OpLoad %half %3161 Aligned 2
       %3163 = OpIAdd %ulong %2837 %2642
       %3164 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3163
       %3165 = OpLoad %half %3164 Aligned 2
       %3166 = OpIAdd %ulong %2837 %2644
       %3167 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3166
       %3168 = OpLoad %half %3167 Aligned 2
       %3169 = OpIAdd %ulong %2838 %2638
       %3170 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3169
       %3171 = OpLoad %half %3170 Aligned 2
       %3172 = OpIAdd %ulong %2838 %2640
       %3173 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3172
       %3174 = OpLoad %half %3173 Aligned 2
       %3175 = OpIAdd %ulong %2838 %2642
       %3176 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3175
       %3177 = OpLoad %half %3176 Aligned 2
       %3178 = OpIAdd %ulong %2838 %2644
       %3179 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3178
       %3180 = OpLoad %half %3179 Aligned 2
       %3181 = OpIAdd %ulong %2839 %2638
       %3182 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3181
       %3183 = OpLoad %half %3182 Aligned 2
       %3184 = OpIAdd %ulong %2839 %2640
       %3185 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3184
       %3186 = OpLoad %half %3185 Aligned 2
       %3187 = OpIAdd %ulong %2839 %2642
       %3188 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3187
       %3189 = OpLoad %half %3188 Aligned 2
       %3190 = OpIAdd %ulong %2839 %2644
       %3191 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3190
       %3192 = OpLoad %half %3191 Aligned 2
       %3193 = OpIAdd %ulong %2840 %2638
       %3194 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3193
       %3195 = OpLoad %half %3194 Aligned 2
       %3196 = OpIAdd %ulong %2840 %2640
       %3197 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3196
       %3198 = OpLoad %half %3197 Aligned 2
       %3199 = OpIAdd %ulong %2840 %2642
       %3200 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3199
       %3201 = OpLoad %half %3200 Aligned 2
       %3202 = OpIAdd %ulong %2840 %2644
       %3203 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3202
       %3204 = OpLoad %half %3203 Aligned 2
       %3205 = OpIAdd %ulong %2841 %2638
       %3206 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3205
       %3207 = OpLoad %half %3206 Aligned 2
       %3208 = OpIAdd %ulong %2841 %2640
       %3209 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3208
       %3210 = OpLoad %half %3209 Aligned 2
       %3211 = OpIAdd %ulong %2841 %2642
       %3212 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3211
       %3213 = OpLoad %half %3212 Aligned 2
       %3214 = OpIAdd %ulong %2841 %2644
       %3215 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3214
       %3216 = OpLoad %half %3215 Aligned 2
       %3217 = OpIAdd %ulong %2842 %2638
       %3218 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3217
       %3219 = OpLoad %half %3218 Aligned 2
       %3220 = OpIAdd %ulong %2842 %2640
       %3221 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3220
       %3222 = OpLoad %half %3221 Aligned 2
       %3223 = OpIAdd %ulong %2842 %2642
       %3224 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3223
       %3225 = OpLoad %half %3224 Aligned 2
       %3226 = OpIAdd %ulong %2842 %2644
       %3227 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3226
       %3228 = OpLoad %half %3227 Aligned 2
       %3229 = OpIAdd %ulong %2843 %2638
       %3230 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3229
       %3231 = OpLoad %half %3230 Aligned 2
       %3232 = OpIAdd %ulong %2843 %2640
       %3233 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3232
       %3234 = OpLoad %half %3233 Aligned 2
       %3235 = OpIAdd %ulong %2843 %2642
       %3236 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3235
       %3237 = OpLoad %half %3236 Aligned 2
       %3238 = OpIAdd %ulong %2843 %2644
       %3239 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3238
       %3240 = OpLoad %half %3239 Aligned 2
       %3241 = OpIAdd %ulong %2844 %2638
       %3242 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3241
       %3243 = OpLoad %half %3242 Aligned 2
       %3244 = OpIAdd %ulong %2844 %2640
       %3245 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3244
       %3246 = OpLoad %half %3245 Aligned 2
       %3247 = OpIAdd %ulong %2844 %2642
       %3248 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3247
       %3249 = OpLoad %half %3248 Aligned 2
       %3250 = OpIAdd %ulong %2844 %2644
       %3251 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3250
       %3252 = OpLoad %half %3251 Aligned 2
       %3253 = OpIAdd %ulong %2845 %2638
       %3254 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3253
       %3255 = OpLoad %half %3254 Aligned 2
       %3256 = OpIAdd %ulong %2845 %2640
       %3257 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3256
       %3258 = OpLoad %half %3257 Aligned 2
       %3259 = OpIAdd %ulong %2845 %2642
       %3260 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3259
       %3261 = OpLoad %half %3260 Aligned 2
       %3262 = OpIAdd %ulong %2845 %2644
       %3263 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3262
       %3264 = OpLoad %half %3263 Aligned 2
       %3265 = OpIAdd %ulong %2846 %2638
       %3266 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3265
       %3267 = OpLoad %half %3266 Aligned 2
       %3268 = OpIAdd %ulong %2846 %2640
       %3269 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3268
       %3270 = OpLoad %half %3269 Aligned 2
       %3271 = OpIAdd %ulong %2846 %2642
       %3272 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3271
       %3273 = OpLoad %half %3272 Aligned 2
       %3274 = OpIAdd %ulong %2846 %2644
       %3275 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3274
       %3276 = OpLoad %half %3275 Aligned 2
       %3277 = OpIAdd %ulong %2847 %2638
       %3278 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3277
       %3279 = OpLoad %half %3278 Aligned 2
       %3280 = OpIAdd %ulong %2847 %2640
       %3281 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3280
       %3282 = OpLoad %half %3281 Aligned 2
       %3283 = OpIAdd %ulong %2847 %2642
       %3284 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3283
       %3285 = OpLoad %half %3284 Aligned 2
       %3286 = OpIAdd %ulong %2847 %2644
       %3287 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3286
       %3288 = OpLoad %half %3287 Aligned 2
       %3289 = OpIAdd %ulong %2848 %2638
       %3290 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3289
       %3291 = OpLoad %half %3290 Aligned 2
       %3292 = OpIAdd %ulong %2848 %2640
       %3293 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3292
       %3294 = OpLoad %half %3293 Aligned 2
       %3295 = OpIAdd %ulong %2848 %2642
       %3296 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3295
       %3297 = OpLoad %half %3296 Aligned 2
       %3298 = OpIAdd %ulong %2848 %2644
       %3299 = OpPtrAccessChain %_ptr_Workgroup_half %2917 %3298
       %3300 = OpLoad %half %3299 Aligned 2
       %3301 = OpCompositeInsert %v8half %2919 %106 0
       %3302 = OpCompositeInsert %v8half %2931 %3301 1
       %3303 = OpCompositeInsert %v8half %2943 %3302 2
       %3304 = OpCompositeInsert %v8half %2955 %3303 3
       %3305 = OpCompositeInsert %v8half %2967 %3304 4
       %3306 = OpCompositeInsert %v8half %2979 %3305 5
       %3307 = OpCompositeInsert %v8half %2991 %3306 6
       %3308 = OpCompositeInsert %v8half %3003 %3307 7
       %3309 = OpCompositeInsert %v8half %2922 %106 0
       %3310 = OpCompositeInsert %v8half %2934 %3309 1
       %3311 = OpCompositeInsert %v8half %2946 %3310 2
       %3312 = OpCompositeInsert %v8half %2958 %3311 3
       %3313 = OpCompositeInsert %v8half %2970 %3312 4
       %3314 = OpCompositeInsert %v8half %2982 %3313 5
       %3315 = OpCompositeInsert %v8half %2994 %3314 6
       %3316 = OpCompositeInsert %v8half %3006 %3315 7
       %3317 = OpCompositeInsert %v8half %2925 %106 0
       %3318 = OpCompositeInsert %v8half %2937 %3317 1
       %3319 = OpCompositeInsert %v8half %2949 %3318 2
       %3320 = OpCompositeInsert %v8half %2961 %3319 3
       %3321 = OpCompositeInsert %v8half %2973 %3320 4
       %3322 = OpCompositeInsert %v8half %2985 %3321 5
       %3323 = OpCompositeInsert %v8half %2997 %3322 6
       %3324 = OpCompositeInsert %v8half %3009 %3323 7
       %3325 = OpCompositeInsert %v8half %2928 %106 0
       %3326 = OpCompositeInsert %v8half %2940 %3325 1
       %3327 = OpCompositeInsert %v8half %2952 %3326 2
       %3328 = OpCompositeInsert %v8half %2964 %3327 3
       %3329 = OpCompositeInsert %v8half %2976 %3328 4
       %3330 = OpCompositeInsert %v8half %2988 %3329 5
       %3331 = OpCompositeInsert %v8half %3000 %3330 6
       %3332 = OpCompositeInsert %v8half %3012 %3331 7
       %3333 = OpCompositeInsert %v8half %3015 %106 0
       %3334 = OpCompositeInsert %v8half %3027 %3333 1
       %3335 = OpCompositeInsert %v8half %3039 %3334 2
       %3336 = OpCompositeInsert %v8half %3051 %3335 3
       %3337 = OpCompositeInsert %v8half %3063 %3336 4
       %3338 = OpCompositeInsert %v8half %3075 %3337 5
       %3339 = OpCompositeInsert %v8half %3087 %3338 6
       %3340 = OpCompositeInsert %v8half %3099 %3339 7
       %3341 = OpCompositeInsert %v8half %3018 %106 0
       %3342 = OpCompositeInsert %v8half %3030 %3341 1
       %3343 = OpCompositeInsert %v8half %3042 %3342 2
       %3344 = OpCompositeInsert %v8half %3054 %3343 3
       %3345 = OpCompositeInsert %v8half %3066 %3344 4
       %3346 = OpCompositeInsert %v8half %3078 %3345 5
       %3347 = OpCompositeInsert %v8half %3090 %3346 6
       %3348 = OpCompositeInsert %v8half %3102 %3347 7
       %3349 = OpCompositeInsert %v8half %3021 %106 0
       %3350 = OpCompositeInsert %v8half %3033 %3349 1
       %3351 = OpCompositeInsert %v8half %3045 %3350 2
       %3352 = OpCompositeInsert %v8half %3057 %3351 3
       %3353 = OpCompositeInsert %v8half %3069 %3352 4
       %3354 = OpCompositeInsert %v8half %3081 %3353 5
       %3355 = OpCompositeInsert %v8half %3093 %3354 6
       %3356 = OpCompositeInsert %v8half %3105 %3355 7
       %3357 = OpCompositeInsert %v8half %3024 %106 0
       %3358 = OpCompositeInsert %v8half %3036 %3357 1
       %3359 = OpCompositeInsert %v8half %3048 %3358 2
       %3360 = OpCompositeInsert %v8half %3060 %3359 3
       %3361 = OpCompositeInsert %v8half %3072 %3360 4
       %3362 = OpCompositeInsert %v8half %3084 %3361 5
       %3363 = OpCompositeInsert %v8half %3096 %3362 6
       %3364 = OpCompositeInsert %v8half %3108 %3363 7
       %3365 = OpCompositeInsert %v8half %3111 %106 0
       %3366 = OpCompositeInsert %v8half %3123 %3365 1
       %3367 = OpCompositeInsert %v8half %3135 %3366 2
       %3368 = OpCompositeInsert %v8half %3147 %3367 3
       %3369 = OpCompositeInsert %v8half %3159 %3368 4
       %3370 = OpCompositeInsert %v8half %3171 %3369 5
       %3371 = OpCompositeInsert %v8half %3183 %3370 6
       %3372 = OpCompositeInsert %v8half %3195 %3371 7
       %3373 = OpCompositeInsert %v8half %3114 %106 0
       %3374 = OpCompositeInsert %v8half %3126 %3373 1
       %3375 = OpCompositeInsert %v8half %3138 %3374 2
       %3376 = OpCompositeInsert %v8half %3150 %3375 3
       %3377 = OpCompositeInsert %v8half %3162 %3376 4
       %3378 = OpCompositeInsert %v8half %3174 %3377 5
       %3379 = OpCompositeInsert %v8half %3186 %3378 6
       %3380 = OpCompositeInsert %v8half %3198 %3379 7
       %3381 = OpCompositeInsert %v8half %3117 %106 0
       %3382 = OpCompositeInsert %v8half %3129 %3381 1
       %3383 = OpCompositeInsert %v8half %3141 %3382 2
       %3384 = OpCompositeInsert %v8half %3153 %3383 3
       %3385 = OpCompositeInsert %v8half %3165 %3384 4
       %3386 = OpCompositeInsert %v8half %3177 %3385 5
       %3387 = OpCompositeInsert %v8half %3189 %3386 6
       %3388 = OpCompositeInsert %v8half %3201 %3387 7
       %3389 = OpCompositeInsert %v8half %3120 %106 0
       %3390 = OpCompositeInsert %v8half %3132 %3389 1
       %3391 = OpCompositeInsert %v8half %3144 %3390 2
       %3392 = OpCompositeInsert %v8half %3156 %3391 3
       %3393 = OpCompositeInsert %v8half %3168 %3392 4
       %3394 = OpCompositeInsert %v8half %3180 %3393 5
       %3395 = OpCompositeInsert %v8half %3192 %3394 6
       %3396 = OpCompositeInsert %v8half %3204 %3395 7
       %3397 = OpCompositeInsert %v8half %3207 %106 0
       %3398 = OpCompositeInsert %v8half %3219 %3397 1
       %3399 = OpCompositeInsert %v8half %3231 %3398 2
       %3400 = OpCompositeInsert %v8half %3243 %3399 3
       %3401 = OpCompositeInsert %v8half %3255 %3400 4
       %3402 = OpCompositeInsert %v8half %3267 %3401 5
       %3403 = OpCompositeInsert %v8half %3279 %3402 6
       %3404 = OpCompositeInsert %v8half %3291 %3403 7
       %3405 = OpCompositeInsert %v8half %3210 %106 0
       %3406 = OpCompositeInsert %v8half %3222 %3405 1
       %3407 = OpCompositeInsert %v8half %3234 %3406 2
       %3408 = OpCompositeInsert %v8half %3246 %3407 3
       %3409 = OpCompositeInsert %v8half %3258 %3408 4
       %3410 = OpCompositeInsert %v8half %3270 %3409 5
       %3411 = OpCompositeInsert %v8half %3282 %3410 6
       %3412 = OpCompositeInsert %v8half %3294 %3411 7
       %3413 = OpCompositeInsert %v8half %3213 %106 0
       %3414 = OpCompositeInsert %v8half %3225 %3413 1
       %3415 = OpCompositeInsert %v8half %3237 %3414 2
       %3416 = OpCompositeInsert %v8half %3249 %3415 3
       %3417 = OpCompositeInsert %v8half %3261 %3416 4
       %3418 = OpCompositeInsert %v8half %3273 %3417 5
       %3419 = OpCompositeInsert %v8half %3285 %3418 6
       %3420 = OpCompositeInsert %v8half %3297 %3419 7
       %3421 = OpCompositeInsert %v8half %3216 %106 0
       %3422 = OpCompositeInsert %v8half %3228 %3421 1
       %3423 = OpCompositeInsert %v8half %3240 %3422 2
       %3424 = OpCompositeInsert %v8half %3252 %3423 3
       %3425 = OpCompositeInsert %v8half %3264 %3424 4
       %3426 = OpCompositeInsert %v8half %3276 %3425 5
       %3427 = OpCompositeInsert %v8half %3288 %3426 6
       %3428 = OpCompositeInsert %v8half %3300 %3427 7
       %3429 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2855 %3308 %2810
       %3430 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2855 %3316 %2808
       %3431 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2855 %3324 %2806
       %3432 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2855 %3332 %2804
       %3433 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2871 %3308 %2802
       %3434 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2871 %3316 %2800
       %3435 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2871 %3324 %2798
       %3436 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2871 %3332 %2796
       %3437 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2887 %3308 %2794
       %3438 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2887 %3316 %2792
       %3439 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2887 %3324 %2790
       %3440 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2887 %3332 %2788
       %3441 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2903 %3308 %2786
       %3442 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2903 %3316 %2784
       %3443 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2903 %3324 %2782
       %3444 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2903 %3332 %2780
       %3445 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2859 %3340 %3429
       %3446 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2859 %3348 %3430
       %3447 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2859 %3356 %3431
       %3448 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2859 %3364 %3432
       %3449 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2875 %3340 %3433
       %3450 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2875 %3348 %3434
       %3451 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2875 %3356 %3435
       %3452 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2875 %3364 %3436
       %3453 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2891 %3340 %3437
       %3454 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2891 %3348 %3438
       %3455 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2891 %3356 %3439
       %3456 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2891 %3364 %3440
       %3457 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2907 %3340 %3441
       %3458 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2907 %3348 %3442
       %3459 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2907 %3356 %3443
       %3460 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2907 %3364 %3444
       %3461 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2863 %3372 %3445
       %3462 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2863 %3380 %3446
       %3463 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2863 %3388 %3447
       %3464 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2863 %3396 %3448
       %3465 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2879 %3372 %3449
       %3466 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2879 %3380 %3450
       %3467 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2879 %3388 %3451
       %3468 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2879 %3396 %3452
       %3469 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2895 %3372 %3453
       %3470 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2895 %3380 %3454
       %3471 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2895 %3388 %3455
       %3472 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2895 %3396 %3456
       %3473 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2911 %3372 %3457
       %3474 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2911 %3380 %3458
       %3475 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2911 %3388 %3459
       %3476 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2911 %3396 %3460
       %3477 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2867 %3404 %3461
       %3478 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2867 %3412 %3462
       %3479 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2867 %3420 %3463
       %3480 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2867 %3428 %3464
       %3481 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2883 %3404 %3465
       %3482 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2883 %3412 %3466
       %3483 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2883 %3420 %3467
       %3484 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2883 %3428 %3468
       %3485 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2899 %3404 %3469
       %3486 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2899 %3412 %3470
       %3487 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2899 %3420 %3471
       %3488 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2899 %3428 %3472
       %3489 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2915 %3404 %3473
       %3490 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2915 %3412 %3474
       %3491 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2915 %3420 %3475
       %3492 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %2915 %3428 %3476
       %3493 = OpCompositeExtract %float %3477 0
       %3494 = OpCompositeExtract %float %3477 1
       %3495 = OpCompositeExtract %float %3477 2
       %3496 = OpCompositeExtract %float %3477 3
       %3497 = OpCompositeExtract %float %3477 4
       %3498 = OpCompositeExtract %float %3477 5
       %3499 = OpCompositeExtract %float %3477 6
       %3500 = OpCompositeExtract %float %3477 7
       %3501 = OpCompositeExtract %float %3478 0
       %3502 = OpCompositeExtract %float %3478 1
       %3503 = OpCompositeExtract %float %3478 2
       %3504 = OpCompositeExtract %float %3478 3
       %3505 = OpCompositeExtract %float %3478 4
       %3506 = OpCompositeExtract %float %3478 5
       %3507 = OpCompositeExtract %float %3478 6
       %3508 = OpCompositeExtract %float %3478 7
       %3509 = OpCompositeExtract %float %3479 0
       %3510 = OpCompositeExtract %float %3479 1
       %3511 = OpCompositeExtract %float %3479 2
       %3512 = OpCompositeExtract %float %3479 3
       %3513 = OpCompositeExtract %float %3479 4
       %3514 = OpCompositeExtract %float %3479 5
       %3515 = OpCompositeExtract %float %3479 6
       %3516 = OpCompositeExtract %float %3479 7
       %3517 = OpCompositeExtract %float %3480 0
       %3518 = OpCompositeExtract %float %3480 1
       %3519 = OpCompositeExtract %float %3480 2
       %3520 = OpCompositeExtract %float %3480 3
       %3521 = OpCompositeExtract %float %3480 4
       %3522 = OpCompositeExtract %float %3480 5
       %3523 = OpCompositeExtract %float %3480 6
       %3524 = OpCompositeExtract %float %3480 7
       %3525 = OpCompositeExtract %float %3481 0
       %3526 = OpCompositeExtract %float %3481 1
       %3527 = OpCompositeExtract %float %3481 2
       %3528 = OpCompositeExtract %float %3481 3
       %3529 = OpCompositeExtract %float %3481 4
       %3530 = OpCompositeExtract %float %3481 5
       %3531 = OpCompositeExtract %float %3481 6
       %3532 = OpCompositeExtract %float %3481 7
       %3533 = OpCompositeExtract %float %3482 0
       %3534 = OpCompositeExtract %float %3482 1
       %3535 = OpCompositeExtract %float %3482 2
       %3536 = OpCompositeExtract %float %3482 3
       %3537 = OpCompositeExtract %float %3482 4
       %3538 = OpCompositeExtract %float %3482 5
       %3539 = OpCompositeExtract %float %3482 6
       %3540 = OpCompositeExtract %float %3482 7
       %3541 = OpCompositeExtract %float %3483 0
       %3542 = OpCompositeExtract %float %3483 1
       %3543 = OpCompositeExtract %float %3483 2
       %3544 = OpCompositeExtract %float %3483 3
       %3545 = OpCompositeExtract %float %3483 4
       %3546 = OpCompositeExtract %float %3483 5
       %3547 = OpCompositeExtract %float %3483 6
       %3548 = OpCompositeExtract %float %3483 7
       %3549 = OpCompositeExtract %float %3484 0
       %3550 = OpCompositeExtract %float %3484 1
       %3551 = OpCompositeExtract %float %3484 2
       %3552 = OpCompositeExtract %float %3484 3
       %3553 = OpCompositeExtract %float %3484 4
       %3554 = OpCompositeExtract %float %3484 5
       %3555 = OpCompositeExtract %float %3484 6
       %3556 = OpCompositeExtract %float %3484 7
       %3557 = OpCompositeExtract %float %3485 0
       %3558 = OpCompositeExtract %float %3485 1
       %3559 = OpCompositeExtract %float %3485 2
       %3560 = OpCompositeExtract %float %3485 3
       %3561 = OpCompositeExtract %float %3485 4
       %3562 = OpCompositeExtract %float %3485 5
       %3563 = OpCompositeExtract %float %3485 6
       %3564 = OpCompositeExtract %float %3485 7
       %3565 = OpCompositeExtract %float %3486 0
       %3566 = OpCompositeExtract %float %3486 1
       %3567 = OpCompositeExtract %float %3486 2
       %3568 = OpCompositeExtract %float %3486 3
       %3569 = OpCompositeExtract %float %3486 4
       %3570 = OpCompositeExtract %float %3486 5
       %3571 = OpCompositeExtract %float %3486 6
       %3572 = OpCompositeExtract %float %3486 7
       %3573 = OpCompositeExtract %float %3487 0
       %3574 = OpCompositeExtract %float %3487 1
       %3575 = OpCompositeExtract %float %3487 2
       %3576 = OpCompositeExtract %float %3487 3
       %3577 = OpCompositeExtract %float %3487 4
       %3578 = OpCompositeExtract %float %3487 5
       %3579 = OpCompositeExtract %float %3487 6
       %3580 = OpCompositeExtract %float %3487 7
       %3581 = OpCompositeExtract %float %3488 0
       %3582 = OpCompositeExtract %float %3488 1
       %3583 = OpCompositeExtract %float %3488 2
       %3584 = OpCompositeExtract %float %3488 3
       %3585 = OpCompositeExtract %float %3488 4
       %3586 = OpCompositeExtract %float %3488 5
       %3587 = OpCompositeExtract %float %3488 6
       %3588 = OpCompositeExtract %float %3488 7
       %3589 = OpCompositeExtract %float %3489 0
       %3590 = OpCompositeExtract %float %3489 1
       %3591 = OpCompositeExtract %float %3489 2
       %3592 = OpCompositeExtract %float %3489 3
       %3593 = OpCompositeExtract %float %3489 4
       %3594 = OpCompositeExtract %float %3489 5
       %3595 = OpCompositeExtract %float %3489 6
       %3596 = OpCompositeExtract %float %3489 7
       %3597 = OpCompositeExtract %float %3490 0
       %3598 = OpCompositeExtract %float %3490 1
       %3599 = OpCompositeExtract %float %3490 2
       %3600 = OpCompositeExtract %float %3490 3
       %3601 = OpCompositeExtract %float %3490 4
       %3602 = OpCompositeExtract %float %3490 5
       %3603 = OpCompositeExtract %float %3490 6
       %3604 = OpCompositeExtract %float %3490 7
       %3605 = OpCompositeExtract %float %3491 0
       %3606 = OpCompositeExtract %float %3491 1
       %3607 = OpCompositeExtract %float %3491 2
       %3608 = OpCompositeExtract %float %3491 3
       %3609 = OpCompositeExtract %float %3491 4
       %3610 = OpCompositeExtract %float %3491 5
       %3611 = OpCompositeExtract %float %3491 6
       %3612 = OpCompositeExtract %float %3491 7
       %3613 = OpCompositeExtract %float %3492 0
       %3614 = OpCompositeExtract %float %3492 1
       %3615 = OpCompositeExtract %float %3492 2
       %3616 = OpCompositeExtract %float %3492 3
       %3617 = OpCompositeExtract %float %3492 4
       %3618 = OpCompositeExtract %float %3492 5
       %3619 = OpCompositeExtract %float %3492 6
       %3620 = OpCompositeExtract %float %3492 7
       %3621 = OpIMul %uint %2481 %uint_16
       %3622 = OpIAdd %uint %2619 %3621
       %3623 = OpIMul %uint %2482 %uint_8
       %3624 = OpIAdd %uint %2635 %3623
       %3625 = OpIMul %uint %3622 %uint_16
       %3626 = OpIAdd %uint %3625 %2411
       %3627 = OpUConvert %ulong %3626
       %3628 = OpIMul %uint %3624 %uint_16
       %3629 = OpIAdd %uint %3628 %2410
       %3630 = OpUConvert %ulong %3629
       %3631 = OpIMul %ulong %3627 %ulong_1024
       %3632 = OpIAdd %ulong %3631 %3630
       %3633 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3632
               OpStore %3633 %3493 Aligned 4
       %3634 = OpIAdd %uint %3629 %uint_16
       %3635 = OpUConvert %ulong %3634
       %3636 = OpIAdd %ulong %3631 %3635
       %3637 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3636
               OpStore %3637 %3501 Aligned 4
       %3638 = OpIAdd %uint %3629 %uint_32
       %3639 = OpUConvert %ulong %3638
       %3640 = OpIAdd %ulong %3631 %3639
       %3641 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3640
               OpStore %3641 %3509 Aligned 4
       %3642 = OpIAdd %uint %3629 %uint_48
       %3643 = OpUConvert %ulong %3642
       %3644 = OpIAdd %ulong %3631 %3643
       %3645 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3644
               OpStore %3645 %3517 Aligned 4
       %3646 = OpIAdd %uint %3626 %uint_1
       %3647 = OpUConvert %ulong %3646
       %3648 = OpIMul %ulong %3647 %ulong_1024
       %3649 = OpIAdd %ulong %3648 %3630
       %3650 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3649
               OpStore %3650 %3494 Aligned 4
       %3651 = OpIAdd %ulong %3648 %3635
       %3652 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3651
               OpStore %3652 %3502 Aligned 4
       %3653 = OpIAdd %ulong %3648 %3639
       %3654 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3653
               OpStore %3654 %3510 Aligned 4
       %3655 = OpIAdd %ulong %3648 %3643
       %3656 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3655
               OpStore %3656 %3518 Aligned 4
       %3657 = OpIAdd %uint %3626 %uint_2
       %3658 = OpUConvert %ulong %3657
       %3659 = OpIMul %ulong %3658 %ulong_1024
       %3660 = OpIAdd %ulong %3659 %3630
       %3661 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3660
               OpStore %3661 %3495 Aligned 4
       %3662 = OpIAdd %ulong %3659 %3635
       %3663 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3662
               OpStore %3663 %3503 Aligned 4
       %3664 = OpIAdd %ulong %3659 %3639
       %3665 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3664
               OpStore %3665 %3511 Aligned 4
       %3666 = OpIAdd %ulong %3659 %3643
       %3667 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3666
               OpStore %3667 %3519 Aligned 4
       %3668 = OpIAdd %uint %3626 %uint_3
       %3669 = OpUConvert %ulong %3668
       %3670 = OpIMul %ulong %3669 %ulong_1024
       %3671 = OpIAdd %ulong %3670 %3630
       %3672 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3671
               OpStore %3672 %3496 Aligned 4
       %3673 = OpIAdd %ulong %3670 %3635
       %3674 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3673
               OpStore %3674 %3504 Aligned 4
       %3675 = OpIAdd %ulong %3670 %3639
       %3676 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3675
               OpStore %3676 %3512 Aligned 4
       %3677 = OpIAdd %ulong %3670 %3643
       %3678 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3677
               OpStore %3678 %3520 Aligned 4
       %3679 = OpIAdd %uint %3626 %uint_4
       %3680 = OpUConvert %ulong %3679
       %3681 = OpIMul %ulong %3680 %ulong_1024
       %3682 = OpIAdd %ulong %3681 %3630
       %3683 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3682
               OpStore %3683 %3497 Aligned 4
       %3684 = OpIAdd %ulong %3681 %3635
       %3685 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3684
               OpStore %3685 %3505 Aligned 4
       %3686 = OpIAdd %ulong %3681 %3639
       %3687 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3686
               OpStore %3687 %3513 Aligned 4
       %3688 = OpIAdd %ulong %3681 %3643
       %3689 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3688
               OpStore %3689 %3521 Aligned 4
       %3690 = OpIAdd %uint %3626 %uint_5
       %3691 = OpUConvert %ulong %3690
       %3692 = OpIMul %ulong %3691 %ulong_1024
       %3693 = OpIAdd %ulong %3692 %3630
       %3694 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3693
               OpStore %3694 %3498 Aligned 4
       %3695 = OpIAdd %ulong %3692 %3635
       %3696 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3695
               OpStore %3696 %3506 Aligned 4
       %3697 = OpIAdd %ulong %3692 %3639
       %3698 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3697
               OpStore %3698 %3514 Aligned 4
       %3699 = OpIAdd %ulong %3692 %3643
       %3700 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3699
               OpStore %3700 %3522 Aligned 4
       %3701 = OpIAdd %uint %3626 %uint_6
       %3702 = OpUConvert %ulong %3701
       %3703 = OpIMul %ulong %3702 %ulong_1024
       %3704 = OpIAdd %ulong %3703 %3630
       %3705 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3704
               OpStore %3705 %3499 Aligned 4
       %3706 = OpIAdd %ulong %3703 %3635
       %3707 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3706
               OpStore %3707 %3507 Aligned 4
       %3708 = OpIAdd %ulong %3703 %3639
       %3709 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3708
               OpStore %3709 %3515 Aligned 4
       %3710 = OpIAdd %ulong %3703 %3643
       %3711 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3710
               OpStore %3711 %3523 Aligned 4
       %3712 = OpIAdd %uint %3626 %uint_7
       %3713 = OpUConvert %ulong %3712
       %3714 = OpIMul %ulong %3713 %ulong_1024
       %3715 = OpIAdd %ulong %3714 %3630
       %3716 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3715
               OpStore %3716 %3500 Aligned 4
       %3717 = OpIAdd %ulong %3714 %3635
       %3718 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3717
               OpStore %3718 %3508 Aligned 4
       %3719 = OpIAdd %ulong %3714 %3639
       %3720 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3719
               OpStore %3720 %3516 Aligned 4
       %3721 = OpIAdd %ulong %3714 %3643
       %3722 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3721
               OpStore %3722 %3524 Aligned 4
       %3723 = OpIAdd %uint %3626 %uint_16
       %3724 = OpUConvert %ulong %3723
       %3725 = OpIMul %ulong %3724 %ulong_1024
       %3726 = OpIAdd %ulong %3725 %3630
       %3727 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3726
               OpStore %3727 %3525 Aligned 4
       %3728 = OpIAdd %ulong %3725 %3635
       %3729 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3728
               OpStore %3729 %3533 Aligned 4
       %3730 = OpIAdd %ulong %3725 %3639
       %3731 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3730
               OpStore %3731 %3541 Aligned 4
       %3732 = OpIAdd %ulong %3725 %3643
       %3733 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3732
               OpStore %3733 %3549 Aligned 4
       %3734 = OpIAdd %uint %3626 %uint_17
       %3735 = OpUConvert %ulong %3734
       %3736 = OpIMul %ulong %3735 %ulong_1024
       %3737 = OpIAdd %ulong %3736 %3630
       %3738 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3737
               OpStore %3738 %3526 Aligned 4
       %3739 = OpIAdd %ulong %3736 %3635
       %3740 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3739
               OpStore %3740 %3534 Aligned 4
       %3741 = OpIAdd %ulong %3736 %3639
       %3742 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3741
               OpStore %3742 %3542 Aligned 4
       %3743 = OpIAdd %ulong %3736 %3643
       %3744 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3743
               OpStore %3744 %3550 Aligned 4
       %3745 = OpIAdd %uint %3626 %uint_18
       %3746 = OpUConvert %ulong %3745
       %3747 = OpIMul %ulong %3746 %ulong_1024
       %3748 = OpIAdd %ulong %3747 %3630
       %3749 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3748
               OpStore %3749 %3527 Aligned 4
       %3750 = OpIAdd %ulong %3747 %3635
       %3751 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3750
               OpStore %3751 %3535 Aligned 4
       %3752 = OpIAdd %ulong %3747 %3639
       %3753 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3752
               OpStore %3753 %3543 Aligned 4
       %3754 = OpIAdd %ulong %3747 %3643
       %3755 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3754
               OpStore %3755 %3551 Aligned 4
       %3756 = OpIAdd %uint %3626 %uint_19
       %3757 = OpUConvert %ulong %3756
       %3758 = OpIMul %ulong %3757 %ulong_1024
       %3759 = OpIAdd %ulong %3758 %3630
       %3760 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3759
               OpStore %3760 %3528 Aligned 4
       %3761 = OpIAdd %ulong %3758 %3635
       %3762 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3761
               OpStore %3762 %3536 Aligned 4
       %3763 = OpIAdd %ulong %3758 %3639
       %3764 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3763
               OpStore %3764 %3544 Aligned 4
       %3765 = OpIAdd %ulong %3758 %3643
       %3766 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3765
               OpStore %3766 %3552 Aligned 4
       %3767 = OpIAdd %uint %3626 %uint_20
       %3768 = OpUConvert %ulong %3767
       %3769 = OpIMul %ulong %3768 %ulong_1024
       %3770 = OpIAdd %ulong %3769 %3630
       %3771 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3770
               OpStore %3771 %3529 Aligned 4
       %3772 = OpIAdd %ulong %3769 %3635
       %3773 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3772
               OpStore %3773 %3537 Aligned 4
       %3774 = OpIAdd %ulong %3769 %3639
       %3775 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3774
               OpStore %3775 %3545 Aligned 4
       %3776 = OpIAdd %ulong %3769 %3643
       %3777 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3776
               OpStore %3777 %3553 Aligned 4
       %3778 = OpIAdd %uint %3626 %uint_21
       %3779 = OpUConvert %ulong %3778
       %3780 = OpIMul %ulong %3779 %ulong_1024
       %3781 = OpIAdd %ulong %3780 %3630
       %3782 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3781
               OpStore %3782 %3530 Aligned 4
       %3783 = OpIAdd %ulong %3780 %3635
       %3784 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3783
               OpStore %3784 %3538 Aligned 4
       %3785 = OpIAdd %ulong %3780 %3639
       %3786 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3785
               OpStore %3786 %3546 Aligned 4
       %3787 = OpIAdd %ulong %3780 %3643
       %3788 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3787
               OpStore %3788 %3554 Aligned 4
       %3789 = OpIAdd %uint %3626 %uint_22
       %3790 = OpUConvert %ulong %3789
       %3791 = OpIMul %ulong %3790 %ulong_1024
       %3792 = OpIAdd %ulong %3791 %3630
       %3793 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3792
               OpStore %3793 %3531 Aligned 4
       %3794 = OpIAdd %ulong %3791 %3635
       %3795 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3794
               OpStore %3795 %3539 Aligned 4
       %3796 = OpIAdd %ulong %3791 %3639
       %3797 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3796
               OpStore %3797 %3547 Aligned 4
       %3798 = OpIAdd %ulong %3791 %3643
       %3799 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3798
               OpStore %3799 %3555 Aligned 4
       %3800 = OpIAdd %uint %3626 %uint_23
       %3801 = OpUConvert %ulong %3800
       %3802 = OpIMul %ulong %3801 %ulong_1024
       %3803 = OpIAdd %ulong %3802 %3630
       %3804 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3803
               OpStore %3804 %3532 Aligned 4
       %3805 = OpIAdd %ulong %3802 %3635
       %3806 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3805
               OpStore %3806 %3540 Aligned 4
       %3807 = OpIAdd %ulong %3802 %3639
       %3808 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3807
               OpStore %3808 %3548 Aligned 4
       %3809 = OpIAdd %ulong %3802 %3643
       %3810 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3809
               OpStore %3810 %3556 Aligned 4
       %3811 = OpIAdd %uint %3626 %uint_32
       %3812 = OpUConvert %ulong %3811
       %3813 = OpIMul %ulong %3812 %ulong_1024
       %3814 = OpIAdd %ulong %3813 %3630
       %3815 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3814
               OpStore %3815 %3557 Aligned 4
       %3816 = OpIAdd %ulong %3813 %3635
       %3817 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3816
               OpStore %3817 %3565 Aligned 4
       %3818 = OpIAdd %ulong %3813 %3639
       %3819 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3818
               OpStore %3819 %3573 Aligned 4
       %3820 = OpIAdd %ulong %3813 %3643
       %3821 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3820
               OpStore %3821 %3581 Aligned 4
       %3822 = OpIAdd %uint %3626 %uint_33
       %3823 = OpUConvert %ulong %3822
       %3824 = OpIMul %ulong %3823 %ulong_1024
       %3825 = OpIAdd %ulong %3824 %3630
       %3826 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3825
               OpStore %3826 %3558 Aligned 4
       %3827 = OpIAdd %ulong %3824 %3635
       %3828 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3827
               OpStore %3828 %3566 Aligned 4
       %3829 = OpIAdd %ulong %3824 %3639
       %3830 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3829
               OpStore %3830 %3574 Aligned 4
       %3831 = OpIAdd %ulong %3824 %3643
       %3832 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3831
               OpStore %3832 %3582 Aligned 4
       %3833 = OpIAdd %uint %3626 %uint_34
       %3834 = OpUConvert %ulong %3833
       %3835 = OpIMul %ulong %3834 %ulong_1024
       %3836 = OpIAdd %ulong %3835 %3630
       %3837 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3836
               OpStore %3837 %3559 Aligned 4
       %3838 = OpIAdd %ulong %3835 %3635
       %3839 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3838
               OpStore %3839 %3567 Aligned 4
       %3840 = OpIAdd %ulong %3835 %3639
       %3841 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3840
               OpStore %3841 %3575 Aligned 4
       %3842 = OpIAdd %ulong %3835 %3643
       %3843 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3842
               OpStore %3843 %3583 Aligned 4
       %3844 = OpIAdd %uint %3626 %uint_35
       %3845 = OpUConvert %ulong %3844
       %3846 = OpIMul %ulong %3845 %ulong_1024
       %3847 = OpIAdd %ulong %3846 %3630
       %3848 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3847
               OpStore %3848 %3560 Aligned 4
       %3849 = OpIAdd %ulong %3846 %3635
       %3850 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3849
               OpStore %3850 %3568 Aligned 4
       %3851 = OpIAdd %ulong %3846 %3639
       %3852 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3851
               OpStore %3852 %3576 Aligned 4
       %3853 = OpIAdd %ulong %3846 %3643
       %3854 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3853
               OpStore %3854 %3584 Aligned 4
       %3855 = OpIAdd %uint %3626 %uint_36
       %3856 = OpUConvert %ulong %3855
       %3857 = OpIMul %ulong %3856 %ulong_1024
       %3858 = OpIAdd %ulong %3857 %3630
       %3859 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3858
               OpStore %3859 %3561 Aligned 4
       %3860 = OpIAdd %ulong %3857 %3635
       %3861 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3860
               OpStore %3861 %3569 Aligned 4
       %3862 = OpIAdd %ulong %3857 %3639
       %3863 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3862
               OpStore %3863 %3577 Aligned 4
       %3864 = OpIAdd %ulong %3857 %3643
       %3865 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3864
               OpStore %3865 %3585 Aligned 4
       %3866 = OpIAdd %uint %3626 %uint_37
       %3867 = OpUConvert %ulong %3866
       %3868 = OpIMul %ulong %3867 %ulong_1024
       %3869 = OpIAdd %ulong %3868 %3630
       %3870 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3869
               OpStore %3870 %3562 Aligned 4
       %3871 = OpIAdd %ulong %3868 %3635
       %3872 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3871
               OpStore %3872 %3570 Aligned 4
       %3873 = OpIAdd %ulong %3868 %3639
       %3874 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3873
               OpStore %3874 %3578 Aligned 4
       %3875 = OpIAdd %ulong %3868 %3643
       %3876 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3875
               OpStore %3876 %3586 Aligned 4
       %3877 = OpIAdd %uint %3626 %uint_38
       %3878 = OpUConvert %ulong %3877
       %3879 = OpIMul %ulong %3878 %ulong_1024
       %3880 = OpIAdd %ulong %3879 %3630
       %3881 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3880
               OpStore %3881 %3563 Aligned 4
       %3882 = OpIAdd %ulong %3879 %3635
       %3883 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3882
               OpStore %3883 %3571 Aligned 4
       %3884 = OpIAdd %ulong %3879 %3639
       %3885 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3884
               OpStore %3885 %3579 Aligned 4
       %3886 = OpIAdd %ulong %3879 %3643
       %3887 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3886
               OpStore %3887 %3587 Aligned 4
       %3888 = OpIAdd %uint %3626 %uint_39
       %3889 = OpUConvert %ulong %3888
       %3890 = OpIMul %ulong %3889 %ulong_1024
       %3891 = OpIAdd %ulong %3890 %3630
       %3892 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3891
               OpStore %3892 %3564 Aligned 4
       %3893 = OpIAdd %ulong %3890 %3635
       %3894 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3893
               OpStore %3894 %3572 Aligned 4
       %3895 = OpIAdd %ulong %3890 %3639
       %3896 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3895
               OpStore %3896 %3580 Aligned 4
       %3897 = OpIAdd %ulong %3890 %3643
       %3898 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3897
               OpStore %3898 %3588 Aligned 4
       %3899 = OpIAdd %uint %3626 %uint_48
       %3900 = OpUConvert %ulong %3899
       %3901 = OpIMul %ulong %3900 %ulong_1024
       %3902 = OpIAdd %ulong %3901 %3630
       %3903 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3902
               OpStore %3903 %3589 Aligned 4
       %3904 = OpIAdd %ulong %3901 %3635
       %3905 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3904
               OpStore %3905 %3597 Aligned 4
       %3906 = OpIAdd %ulong %3901 %3639
       %3907 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3906
               OpStore %3907 %3605 Aligned 4
       %3908 = OpIAdd %ulong %3901 %3643
       %3909 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3908
               OpStore %3909 %3613 Aligned 4
       %3910 = OpIAdd %uint %3626 %uint_49
       %3911 = OpUConvert %ulong %3910
       %3912 = OpIMul %ulong %3911 %ulong_1024
       %3913 = OpIAdd %ulong %3912 %3630
       %3914 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3913
               OpStore %3914 %3590 Aligned 4
       %3915 = OpIAdd %ulong %3912 %3635
       %3916 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3915
               OpStore %3916 %3598 Aligned 4
       %3917 = OpIAdd %ulong %3912 %3639
       %3918 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3917
               OpStore %3918 %3606 Aligned 4
       %3919 = OpIAdd %ulong %3912 %3643
       %3920 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3919
               OpStore %3920 %3614 Aligned 4
       %3921 = OpIAdd %uint %3626 %uint_50
       %3922 = OpUConvert %ulong %3921
       %3923 = OpIMul %ulong %3922 %ulong_1024
       %3924 = OpIAdd %ulong %3923 %3630
       %3925 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3924
               OpStore %3925 %3591 Aligned 4
       %3926 = OpIAdd %ulong %3923 %3635
       %3927 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3926
               OpStore %3927 %3599 Aligned 4
       %3928 = OpIAdd %ulong %3923 %3639
       %3929 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3928
               OpStore %3929 %3607 Aligned 4
       %3930 = OpIAdd %ulong %3923 %3643
       %3931 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3930
               OpStore %3931 %3615 Aligned 4
       %3932 = OpIAdd %uint %3626 %uint_51
       %3933 = OpUConvert %ulong %3932
       %3934 = OpIMul %ulong %3933 %ulong_1024
       %3935 = OpIAdd %ulong %3934 %3630
       %3936 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3935
               OpStore %3936 %3592 Aligned 4
       %3937 = OpIAdd %ulong %3934 %3635
       %3938 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3937
               OpStore %3938 %3600 Aligned 4
       %3939 = OpIAdd %ulong %3934 %3639
       %3940 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3939
               OpStore %3940 %3608 Aligned 4
       %3941 = OpIAdd %ulong %3934 %3643
       %3942 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3941
               OpStore %3942 %3616 Aligned 4
       %3943 = OpIAdd %uint %3626 %uint_52
       %3944 = OpUConvert %ulong %3943
       %3945 = OpIMul %ulong %3944 %ulong_1024
       %3946 = OpIAdd %ulong %3945 %3630
       %3947 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3946
               OpStore %3947 %3593 Aligned 4
       %3948 = OpIAdd %ulong %3945 %3635
       %3949 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3948
               OpStore %3949 %3601 Aligned 4
       %3950 = OpIAdd %ulong %3945 %3639
       %3951 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3950
               OpStore %3951 %3609 Aligned 4
       %3952 = OpIAdd %ulong %3945 %3643
       %3953 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3952
               OpStore %3953 %3617 Aligned 4
       %3954 = OpIAdd %uint %3626 %uint_53
       %3955 = OpUConvert %ulong %3954
       %3956 = OpIMul %ulong %3955 %ulong_1024
       %3957 = OpIAdd %ulong %3956 %3630
       %3958 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3957
               OpStore %3958 %3594 Aligned 4
       %3959 = OpIAdd %ulong %3956 %3635
       %3960 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3959
               OpStore %3960 %3602 Aligned 4
       %3961 = OpIAdd %ulong %3956 %3639
       %3962 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3961
               OpStore %3962 %3610 Aligned 4
       %3963 = OpIAdd %ulong %3956 %3643
       %3964 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3963
               OpStore %3964 %3618 Aligned 4
       %3965 = OpIAdd %uint %3626 %uint_54
       %3966 = OpUConvert %ulong %3965
       %3967 = OpIMul %ulong %3966 %ulong_1024
       %3968 = OpIAdd %ulong %3967 %3630
       %3969 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3968
               OpStore %3969 %3595 Aligned 4
       %3970 = OpIAdd %ulong %3967 %3635
       %3971 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3970
               OpStore %3971 %3603 Aligned 4
       %3972 = OpIAdd %ulong %3967 %3639
       %3973 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3972
               OpStore %3973 %3611 Aligned 4
       %3974 = OpIAdd %ulong %3967 %3643
       %3975 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3974
               OpStore %3975 %3619 Aligned 4
       %3976 = OpIAdd %uint %3626 %uint_55
       %3977 = OpUConvert %ulong %3976
       %3978 = OpIMul %ulong %3977 %ulong_1024
       %3979 = OpIAdd %ulong %3978 %3630
       %3980 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3979
               OpStore %3980 %3596 Aligned 4
       %3981 = OpIAdd %ulong %3978 %3635
       %3982 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3981
               OpStore %3982 %3604 Aligned 4
       %3983 = OpIAdd %ulong %3978 %3639
       %3984 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3983
               OpStore %3984 %3612 Aligned 4
       %3985 = OpIAdd %ulong %3978 %3643
       %3986 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %2401 %3985
               OpStore %3986 %3620 Aligned 4
               OpMemoryBarrier %uint_2 %uint_4
       %3987 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_signal %uint_4294967295
       %3988 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_wait %ushort_65535
               OpMemoryBarrier %uint_2 %uint_2
               OpReturn
       %6959 = OpLabel
       %2779 = OpIAdd %uint %2778 %uint_4
       %3989 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2754 %2728
       %3990 = OpBitcast %_ptr_CrossWorkgroup_v8half %3989
       %3991 = OpLoad %v8half %3990 Aligned 2
       %3992 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2756 %2728
       %3993 = OpBitcast %_ptr_CrossWorkgroup_v8half %3992
       %3994 = OpLoad %v8half %3993 Aligned 2
       %3995 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2758 %2728
       %3996 = OpBitcast %_ptr_CrossWorkgroup_v8half %3995
       %3997 = OpLoad %v8half %3996 Aligned 2
       %3998 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2760 %2728
       %3999 = OpBitcast %_ptr_CrossWorkgroup_v8half %3998
       %4000 = OpLoad %v8half %3999 Aligned 2
       %4001 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2762 %2728
       %4002 = OpBitcast %_ptr_CrossWorkgroup_v8half %4001
       %4003 = OpLoad %v8half %4002 Aligned 2
       %4004 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2764 %2728
       %4005 = OpBitcast %_ptr_CrossWorkgroup_v8half %4004
       %4006 = OpLoad %v8half %4005 Aligned 2
       %4007 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2766 %2728
       %4008 = OpBitcast %_ptr_CrossWorkgroup_v8half %4007
       %4009 = OpLoad %v8half %4008 Aligned 2
       %4010 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2768 %2728
       %4011 = OpBitcast %_ptr_CrossWorkgroup_v8half %4010
       %4012 = OpLoad %v8half %4011 Aligned 2
       %4013 = OpBitcast %_ptr_CrossWorkgroup_v8half %2770
       %4014 = OpLoad %v8half %4013 Aligned 2
       %4015 = OpBitcast %_ptr_CrossWorkgroup_v8half %2772
       %4016 = OpLoad %v8half %4015 Aligned 2
       %4017 = OpBitcast %_ptr_CrossWorkgroup_v8half %2774
       %4018 = OpLoad %v8half %4017 Aligned 2
       %4019 = OpBitcast %_ptr_CrossWorkgroup_v8half %2776
       %4020 = OpLoad %v8half %4019 Aligned 2
               OpMemoryBarrier %uint_2 %uint_4
       %4021 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_signal %uint_4294967295
       %4022 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_wait %ushort_65535
               OpMemoryBarrier %uint_2 %uint_2
       %4023 = OpIAdd %ulong %2813 %2412
       %4024 = OpBitcast %_ptr_Workgroup_half %__shared_memory__
       %4025 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4023
       %4026 = OpBitcast %_ptr_Workgroup_v8half %4025
       %4027 = OpLoad %v8half %4026 Aligned 2
       %4028 = OpIAdd %ulong %2813 %2624
       %4029 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4028
       %4030 = OpBitcast %_ptr_Workgroup_v8half %4029
       %4031 = OpLoad %v8half %4030 Aligned 2
       %4032 = OpIAdd %ulong %2813 %2626
       %4033 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4032
       %4034 = OpBitcast %_ptr_Workgroup_v8half %4033
       %4035 = OpLoad %v8half %4034 Aligned 2
       %4036 = OpIAdd %ulong %2813 %2628
       %4037 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4036
       %4038 = OpBitcast %_ptr_Workgroup_v8half %4037
       %4039 = OpLoad %v8half %4038 Aligned 2
       %4040 = OpIAdd %ulong %2814 %2412
       %4041 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4040
       %4042 = OpBitcast %_ptr_Workgroup_v8half %4041
       %4043 = OpLoad %v8half %4042 Aligned 2
       %4044 = OpIAdd %ulong %2814 %2624
       %4045 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4044
       %4046 = OpBitcast %_ptr_Workgroup_v8half %4045
       %4047 = OpLoad %v8half %4046 Aligned 2
       %4048 = OpIAdd %ulong %2814 %2626
       %4049 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4048
       %4050 = OpBitcast %_ptr_Workgroup_v8half %4049
       %4051 = OpLoad %v8half %4050 Aligned 2
       %4052 = OpIAdd %ulong %2814 %2628
       %4053 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4052
       %4054 = OpBitcast %_ptr_Workgroup_v8half %4053
       %4055 = OpLoad %v8half %4054 Aligned 2
       %4056 = OpIAdd %ulong %2815 %2412
       %4057 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4056
       %4058 = OpBitcast %_ptr_Workgroup_v8half %4057
       %4059 = OpLoad %v8half %4058 Aligned 2
       %4060 = OpIAdd %ulong %2815 %2624
       %4061 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4060
       %4062 = OpBitcast %_ptr_Workgroup_v8half %4061
       %4063 = OpLoad %v8half %4062 Aligned 2
       %4064 = OpIAdd %ulong %2815 %2626
       %4065 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4064
       %4066 = OpBitcast %_ptr_Workgroup_v8half %4065
       %4067 = OpLoad %v8half %4066 Aligned 2
       %4068 = OpIAdd %ulong %2815 %2628
       %4069 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4068
       %4070 = OpBitcast %_ptr_Workgroup_v8half %4069
       %4071 = OpLoad %v8half %4070 Aligned 2
       %4072 = OpIAdd %ulong %2816 %2412
       %4073 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4072
       %4074 = OpBitcast %_ptr_Workgroup_v8half %4073
       %4075 = OpLoad %v8half %4074 Aligned 2
       %4076 = OpIAdd %ulong %2816 %2624
       %4077 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4076
       %4078 = OpBitcast %_ptr_Workgroup_v8half %4077
       %4079 = OpLoad %v8half %4078 Aligned 2
       %4080 = OpIAdd %ulong %2816 %2626
       %4081 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4080
       %4082 = OpBitcast %_ptr_Workgroup_v8half %4081
       %4083 = OpLoad %v8half %4082 Aligned 2
       %4084 = OpIAdd %ulong %2816 %2628
       %4085 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %4084
       %4086 = OpBitcast %_ptr_Workgroup_v8half %4085
       %4087 = OpLoad %v8half %4086 Aligned 2
       %4088 = OpIAdd %ulong %2817 %2638
       %4089 = OpBitcast %_ptr_Workgroup_half %__shared_memory___0
       %4090 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4088
       %4091 = OpLoad %half %4090 Aligned 2
       %4092 = OpIAdd %ulong %2817 %2640
       %4093 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4092
       %4094 = OpLoad %half %4093 Aligned 2
       %4095 = OpIAdd %ulong %2817 %2642
       %4096 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4095
       %4097 = OpLoad %half %4096 Aligned 2
       %4098 = OpIAdd %ulong %2817 %2644
       %4099 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4098
       %4100 = OpLoad %half %4099 Aligned 2
       %4101 = OpIAdd %ulong %2818 %2638
       %4102 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4101
       %4103 = OpLoad %half %4102 Aligned 2
       %4104 = OpIAdd %ulong %2818 %2640
       %4105 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4104
       %4106 = OpLoad %half %4105 Aligned 2
       %4107 = OpIAdd %ulong %2818 %2642
       %4108 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4107
       %4109 = OpLoad %half %4108 Aligned 2
       %4110 = OpIAdd %ulong %2818 %2644
       %4111 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4110
       %4112 = OpLoad %half %4111 Aligned 2
       %4113 = OpIAdd %ulong %2819 %2638
       %4114 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4113
       %4115 = OpLoad %half %4114 Aligned 2
       %4116 = OpIAdd %ulong %2819 %2640
       %4117 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4116
       %4118 = OpLoad %half %4117 Aligned 2
       %4119 = OpIAdd %ulong %2819 %2642
       %4120 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4119
       %4121 = OpLoad %half %4120 Aligned 2
       %4122 = OpIAdd %ulong %2819 %2644
       %4123 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4122
       %4124 = OpLoad %half %4123 Aligned 2
       %4125 = OpIAdd %ulong %2820 %2638
       %4126 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4125
       %4127 = OpLoad %half %4126 Aligned 2
       %4128 = OpIAdd %ulong %2820 %2640
       %4129 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4128
       %4130 = OpLoad %half %4129 Aligned 2
       %4131 = OpIAdd %ulong %2820 %2642
       %4132 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4131
       %4133 = OpLoad %half %4132 Aligned 2
       %4134 = OpIAdd %ulong %2820 %2644
       %4135 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4134
       %4136 = OpLoad %half %4135 Aligned 2
       %4137 = OpIAdd %ulong %2821 %2638
       %4138 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4137
       %4139 = OpLoad %half %4138 Aligned 2
       %4140 = OpIAdd %ulong %2821 %2640
       %4141 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4140
       %4142 = OpLoad %half %4141 Aligned 2
       %4143 = OpIAdd %ulong %2821 %2642
       %4144 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4143
       %4145 = OpLoad %half %4144 Aligned 2
       %4146 = OpIAdd %ulong %2821 %2644
       %4147 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4146
       %4148 = OpLoad %half %4147 Aligned 2
       %4149 = OpIAdd %ulong %2822 %2638
       %4150 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4149
       %4151 = OpLoad %half %4150 Aligned 2
       %4152 = OpIAdd %ulong %2822 %2640
       %4153 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4152
       %4154 = OpLoad %half %4153 Aligned 2
       %4155 = OpIAdd %ulong %2822 %2642
       %4156 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4155
       %4157 = OpLoad %half %4156 Aligned 2
       %4158 = OpIAdd %ulong %2822 %2644
       %4159 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4158
       %4160 = OpLoad %half %4159 Aligned 2
       %4161 = OpIAdd %ulong %2823 %2638
       %4162 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4161
       %4163 = OpLoad %half %4162 Aligned 2
       %4164 = OpIAdd %ulong %2823 %2640
       %4165 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4164
       %4166 = OpLoad %half %4165 Aligned 2
       %4167 = OpIAdd %ulong %2823 %2642
       %4168 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4167
       %4169 = OpLoad %half %4168 Aligned 2
       %4170 = OpIAdd %ulong %2823 %2644
       %4171 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4170
       %4172 = OpLoad %half %4171 Aligned 2
       %4173 = OpIAdd %ulong %2824 %2638
       %4174 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4173
       %4175 = OpLoad %half %4174 Aligned 2
       %4176 = OpIAdd %ulong %2824 %2640
       %4177 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4176
       %4178 = OpLoad %half %4177 Aligned 2
       %4179 = OpIAdd %ulong %2824 %2642
       %4180 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4179
       %4181 = OpLoad %half %4180 Aligned 2
       %4182 = OpIAdd %ulong %2824 %2644
       %4183 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4182
       %4184 = OpLoad %half %4183 Aligned 2
       %4185 = OpIAdd %ulong %2825 %2638
       %4186 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4185
       %4187 = OpLoad %half %4186 Aligned 2
       %4188 = OpIAdd %ulong %2825 %2640
       %4189 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4188
       %4190 = OpLoad %half %4189 Aligned 2
       %4191 = OpIAdd %ulong %2825 %2642
       %4192 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4191
       %4193 = OpLoad %half %4192 Aligned 2
       %4194 = OpIAdd %ulong %2825 %2644
       %4195 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4194
       %4196 = OpLoad %half %4195 Aligned 2
       %4197 = OpIAdd %ulong %2826 %2638
       %4198 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4197
       %4199 = OpLoad %half %4198 Aligned 2
       %4200 = OpIAdd %ulong %2826 %2640
       %4201 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4200
       %4202 = OpLoad %half %4201 Aligned 2
       %4203 = OpIAdd %ulong %2826 %2642
       %4204 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4203
       %4205 = OpLoad %half %4204 Aligned 2
       %4206 = OpIAdd %ulong %2826 %2644
       %4207 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4206
       %4208 = OpLoad %half %4207 Aligned 2
       %4209 = OpIAdd %ulong %2827 %2638
       %4210 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4209
       %4211 = OpLoad %half %4210 Aligned 2
       %4212 = OpIAdd %ulong %2827 %2640
       %4213 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4212
       %4214 = OpLoad %half %4213 Aligned 2
       %4215 = OpIAdd %ulong %2827 %2642
       %4216 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4215
       %4217 = OpLoad %half %4216 Aligned 2
       %4218 = OpIAdd %ulong %2827 %2644
       %4219 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4218
       %4220 = OpLoad %half %4219 Aligned 2
       %4221 = OpIAdd %ulong %2828 %2638
       %4222 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4221
       %4223 = OpLoad %half %4222 Aligned 2
       %4224 = OpIAdd %ulong %2828 %2640
       %4225 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4224
       %4226 = OpLoad %half %4225 Aligned 2
       %4227 = OpIAdd %ulong %2828 %2642
       %4228 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4227
       %4229 = OpLoad %half %4228 Aligned 2
       %4230 = OpIAdd %ulong %2828 %2644
       %4231 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4230
       %4232 = OpLoad %half %4231 Aligned 2
       %4233 = OpIAdd %ulong %2829 %2638
       %4234 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4233
       %4235 = OpLoad %half %4234 Aligned 2
       %4236 = OpIAdd %ulong %2829 %2640
       %4237 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4236
       %4238 = OpLoad %half %4237 Aligned 2
       %4239 = OpIAdd %ulong %2829 %2642
       %4240 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4239
       %4241 = OpLoad %half %4240 Aligned 2
       %4242 = OpIAdd %ulong %2829 %2644
       %4243 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4242
       %4244 = OpLoad %half %4243 Aligned 2
       %4245 = OpIAdd %ulong %2830 %2638
       %4246 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4245
       %4247 = OpLoad %half %4246 Aligned 2
       %4248 = OpIAdd %ulong %2830 %2640
       %4249 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4248
       %4250 = OpLoad %half %4249 Aligned 2
       %4251 = OpIAdd %ulong %2830 %2642
       %4252 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4251
       %4253 = OpLoad %half %4252 Aligned 2
       %4254 = OpIAdd %ulong %2830 %2644
       %4255 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4254
       %4256 = OpLoad %half %4255 Aligned 2
       %4257 = OpIAdd %ulong %2831 %2638
       %4258 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4257
       %4259 = OpLoad %half %4258 Aligned 2
       %4260 = OpIAdd %ulong %2831 %2640
       %4261 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4260
       %4262 = OpLoad %half %4261 Aligned 2
       %4263 = OpIAdd %ulong %2831 %2642
       %4264 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4263
       %4265 = OpLoad %half %4264 Aligned 2
       %4266 = OpIAdd %ulong %2831 %2644
       %4267 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4266
       %4268 = OpLoad %half %4267 Aligned 2
       %4269 = OpIAdd %ulong %2832 %2638
       %4270 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4269
       %4271 = OpLoad %half %4270 Aligned 2
       %4272 = OpIAdd %ulong %2832 %2640
       %4273 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4272
       %4274 = OpLoad %half %4273 Aligned 2
       %4275 = OpIAdd %ulong %2832 %2642
       %4276 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4275
       %4277 = OpLoad %half %4276 Aligned 2
       %4278 = OpIAdd %ulong %2832 %2644
       %4279 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4278
       %4280 = OpLoad %half %4279 Aligned 2
       %4281 = OpIAdd %ulong %2833 %2638
       %4282 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4281
       %4283 = OpLoad %half %4282 Aligned 2
       %4284 = OpIAdd %ulong %2833 %2640
       %4285 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4284
       %4286 = OpLoad %half %4285 Aligned 2
       %4287 = OpIAdd %ulong %2833 %2642
       %4288 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4287
       %4289 = OpLoad %half %4288 Aligned 2
       %4290 = OpIAdd %ulong %2833 %2644
       %4291 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4290
       %4292 = OpLoad %half %4291 Aligned 2
       %4293 = OpIAdd %ulong %2834 %2638
       %4294 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4293
       %4295 = OpLoad %half %4294 Aligned 2
       %4296 = OpIAdd %ulong %2834 %2640
       %4297 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4296
       %4298 = OpLoad %half %4297 Aligned 2
       %4299 = OpIAdd %ulong %2834 %2642
       %4300 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4299
       %4301 = OpLoad %half %4300 Aligned 2
       %4302 = OpIAdd %ulong %2834 %2644
       %4303 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4302
       %4304 = OpLoad %half %4303 Aligned 2
       %4305 = OpIAdd %ulong %2835 %2638
       %4306 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4305
       %4307 = OpLoad %half %4306 Aligned 2
       %4308 = OpIAdd %ulong %2835 %2640
       %4309 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4308
       %4310 = OpLoad %half %4309 Aligned 2
       %4311 = OpIAdd %ulong %2835 %2642
       %4312 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4311
       %4313 = OpLoad %half %4312 Aligned 2
       %4314 = OpIAdd %ulong %2835 %2644
       %4315 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4314
       %4316 = OpLoad %half %4315 Aligned 2
       %4317 = OpIAdd %ulong %2836 %2638
       %4318 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4317
       %4319 = OpLoad %half %4318 Aligned 2
       %4320 = OpIAdd %ulong %2836 %2640
       %4321 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4320
       %4322 = OpLoad %half %4321 Aligned 2
       %4323 = OpIAdd %ulong %2836 %2642
       %4324 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4323
       %4325 = OpLoad %half %4324 Aligned 2
       %4326 = OpIAdd %ulong %2836 %2644
       %4327 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4326
       %4328 = OpLoad %half %4327 Aligned 2
       %4329 = OpIAdd %ulong %2837 %2638
       %4330 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4329
       %4331 = OpLoad %half %4330 Aligned 2
       %4332 = OpIAdd %ulong %2837 %2640
       %4333 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4332
       %4334 = OpLoad %half %4333 Aligned 2
       %4335 = OpIAdd %ulong %2837 %2642
       %4336 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4335
       %4337 = OpLoad %half %4336 Aligned 2
       %4338 = OpIAdd %ulong %2837 %2644
       %4339 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4338
       %4340 = OpLoad %half %4339 Aligned 2
       %4341 = OpIAdd %ulong %2838 %2638
       %4342 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4341
       %4343 = OpLoad %half %4342 Aligned 2
       %4344 = OpIAdd %ulong %2838 %2640
       %4345 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4344
       %4346 = OpLoad %half %4345 Aligned 2
       %4347 = OpIAdd %ulong %2838 %2642
       %4348 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4347
       %4349 = OpLoad %half %4348 Aligned 2
       %4350 = OpIAdd %ulong %2838 %2644
       %4351 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4350
       %4352 = OpLoad %half %4351 Aligned 2
       %4353 = OpIAdd %ulong %2839 %2638
       %4354 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4353
       %4355 = OpLoad %half %4354 Aligned 2
       %4356 = OpIAdd %ulong %2839 %2640
       %4357 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4356
       %4358 = OpLoad %half %4357 Aligned 2
       %4359 = OpIAdd %ulong %2839 %2642
       %4360 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4359
       %4361 = OpLoad %half %4360 Aligned 2
       %4362 = OpIAdd %ulong %2839 %2644
       %4363 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4362
       %4364 = OpLoad %half %4363 Aligned 2
       %4365 = OpIAdd %ulong %2840 %2638
       %4366 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4365
       %4367 = OpLoad %half %4366 Aligned 2
       %4368 = OpIAdd %ulong %2840 %2640
       %4369 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4368
       %4370 = OpLoad %half %4369 Aligned 2
       %4371 = OpIAdd %ulong %2840 %2642
       %4372 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4371
       %4373 = OpLoad %half %4372 Aligned 2
       %4374 = OpIAdd %ulong %2840 %2644
       %4375 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4374
       %4376 = OpLoad %half %4375 Aligned 2
       %4377 = OpIAdd %ulong %2841 %2638
       %4378 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4377
       %4379 = OpLoad %half %4378 Aligned 2
       %4380 = OpIAdd %ulong %2841 %2640
       %4381 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4380
       %4382 = OpLoad %half %4381 Aligned 2
       %4383 = OpIAdd %ulong %2841 %2642
       %4384 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4383
       %4385 = OpLoad %half %4384 Aligned 2
       %4386 = OpIAdd %ulong %2841 %2644
       %4387 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4386
       %4388 = OpLoad %half %4387 Aligned 2
       %4389 = OpIAdd %ulong %2842 %2638
       %4390 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4389
       %4391 = OpLoad %half %4390 Aligned 2
       %4392 = OpIAdd %ulong %2842 %2640
       %4393 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4392
       %4394 = OpLoad %half %4393 Aligned 2
       %4395 = OpIAdd %ulong %2842 %2642
       %4396 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4395
       %4397 = OpLoad %half %4396 Aligned 2
       %4398 = OpIAdd %ulong %2842 %2644
       %4399 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4398
       %4400 = OpLoad %half %4399 Aligned 2
       %4401 = OpIAdd %ulong %2843 %2638
       %4402 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4401
       %4403 = OpLoad %half %4402 Aligned 2
       %4404 = OpIAdd %ulong %2843 %2640
       %4405 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4404
       %4406 = OpLoad %half %4405 Aligned 2
       %4407 = OpIAdd %ulong %2843 %2642
       %4408 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4407
       %4409 = OpLoad %half %4408 Aligned 2
       %4410 = OpIAdd %ulong %2843 %2644
       %4411 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4410
       %4412 = OpLoad %half %4411 Aligned 2
       %4413 = OpIAdd %ulong %2844 %2638
       %4414 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4413
       %4415 = OpLoad %half %4414 Aligned 2
       %4416 = OpIAdd %ulong %2844 %2640
       %4417 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4416
       %4418 = OpLoad %half %4417 Aligned 2
       %4419 = OpIAdd %ulong %2844 %2642
       %4420 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4419
       %4421 = OpLoad %half %4420 Aligned 2
       %4422 = OpIAdd %ulong %2844 %2644
       %4423 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4422
       %4424 = OpLoad %half %4423 Aligned 2
       %4425 = OpIAdd %ulong %2845 %2638
       %4426 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4425
       %4427 = OpLoad %half %4426 Aligned 2
       %4428 = OpIAdd %ulong %2845 %2640
       %4429 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4428
       %4430 = OpLoad %half %4429 Aligned 2
       %4431 = OpIAdd %ulong %2845 %2642
       %4432 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4431
       %4433 = OpLoad %half %4432 Aligned 2
       %4434 = OpIAdd %ulong %2845 %2644
       %4435 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4434
       %4436 = OpLoad %half %4435 Aligned 2
       %4437 = OpIAdd %ulong %2846 %2638
       %4438 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4437
       %4439 = OpLoad %half %4438 Aligned 2
       %4440 = OpIAdd %ulong %2846 %2640
       %4441 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4440
       %4442 = OpLoad %half %4441 Aligned 2
       %4443 = OpIAdd %ulong %2846 %2642
       %4444 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4443
       %4445 = OpLoad %half %4444 Aligned 2
       %4446 = OpIAdd %ulong %2846 %2644
       %4447 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4446
       %4448 = OpLoad %half %4447 Aligned 2
       %4449 = OpIAdd %ulong %2847 %2638
       %4450 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4449
       %4451 = OpLoad %half %4450 Aligned 2
       %4452 = OpIAdd %ulong %2847 %2640
       %4453 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4452
       %4454 = OpLoad %half %4453 Aligned 2
       %4455 = OpIAdd %ulong %2847 %2642
       %4456 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4455
       %4457 = OpLoad %half %4456 Aligned 2
       %4458 = OpIAdd %ulong %2847 %2644
       %4459 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4458
       %4460 = OpLoad %half %4459 Aligned 2
       %4461 = OpIAdd %ulong %2848 %2638
       %4462 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4461
       %4463 = OpLoad %half %4462 Aligned 2
       %4464 = OpIAdd %ulong %2848 %2640
       %4465 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4464
       %4466 = OpLoad %half %4465 Aligned 2
       %4467 = OpIAdd %ulong %2848 %2642
       %4468 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4467
       %4469 = OpLoad %half %4468 Aligned 2
       %4470 = OpIAdd %ulong %2848 %2644
       %4471 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %4470
       %4472 = OpLoad %half %4471 Aligned 2
       %4473 = OpCompositeInsert %v8half %4091 %106 0
       %4474 = OpCompositeInsert %v8half %4103 %4473 1
       %4475 = OpCompositeInsert %v8half %4115 %4474 2
       %4476 = OpCompositeInsert %v8half %4127 %4475 3
       %4477 = OpCompositeInsert %v8half %4139 %4476 4
       %4478 = OpCompositeInsert %v8half %4151 %4477 5
       %4479 = OpCompositeInsert %v8half %4163 %4478 6
       %4480 = OpCompositeInsert %v8half %4175 %4479 7
       %4481 = OpCompositeInsert %v8half %4094 %106 0
       %4482 = OpCompositeInsert %v8half %4106 %4481 1
       %4483 = OpCompositeInsert %v8half %4118 %4482 2
       %4484 = OpCompositeInsert %v8half %4130 %4483 3
       %4485 = OpCompositeInsert %v8half %4142 %4484 4
       %4486 = OpCompositeInsert %v8half %4154 %4485 5
       %4487 = OpCompositeInsert %v8half %4166 %4486 6
       %4488 = OpCompositeInsert %v8half %4178 %4487 7
       %4489 = OpCompositeInsert %v8half %4097 %106 0
       %4490 = OpCompositeInsert %v8half %4109 %4489 1
       %4491 = OpCompositeInsert %v8half %4121 %4490 2
       %4492 = OpCompositeInsert %v8half %4133 %4491 3
       %4493 = OpCompositeInsert %v8half %4145 %4492 4
       %4494 = OpCompositeInsert %v8half %4157 %4493 5
       %4495 = OpCompositeInsert %v8half %4169 %4494 6
       %4496 = OpCompositeInsert %v8half %4181 %4495 7
       %4497 = OpCompositeInsert %v8half %4100 %106 0
       %4498 = OpCompositeInsert %v8half %4112 %4497 1
       %4499 = OpCompositeInsert %v8half %4124 %4498 2
       %4500 = OpCompositeInsert %v8half %4136 %4499 3
       %4501 = OpCompositeInsert %v8half %4148 %4500 4
       %4502 = OpCompositeInsert %v8half %4160 %4501 5
       %4503 = OpCompositeInsert %v8half %4172 %4502 6
       %4504 = OpCompositeInsert %v8half %4184 %4503 7
       %4505 = OpCompositeInsert %v8half %4187 %106 0
       %4506 = OpCompositeInsert %v8half %4199 %4505 1
       %4507 = OpCompositeInsert %v8half %4211 %4506 2
       %4508 = OpCompositeInsert %v8half %4223 %4507 3
       %4509 = OpCompositeInsert %v8half %4235 %4508 4
       %4510 = OpCompositeInsert %v8half %4247 %4509 5
       %4511 = OpCompositeInsert %v8half %4259 %4510 6
       %4512 = OpCompositeInsert %v8half %4271 %4511 7
       %4513 = OpCompositeInsert %v8half %4190 %106 0
       %4514 = OpCompositeInsert %v8half %4202 %4513 1
       %4515 = OpCompositeInsert %v8half %4214 %4514 2
       %4516 = OpCompositeInsert %v8half %4226 %4515 3
       %4517 = OpCompositeInsert %v8half %4238 %4516 4
       %4518 = OpCompositeInsert %v8half %4250 %4517 5
       %4519 = OpCompositeInsert %v8half %4262 %4518 6
       %4520 = OpCompositeInsert %v8half %4274 %4519 7
       %4521 = OpCompositeInsert %v8half %4193 %106 0
       %4522 = OpCompositeInsert %v8half %4205 %4521 1
       %4523 = OpCompositeInsert %v8half %4217 %4522 2
       %4524 = OpCompositeInsert %v8half %4229 %4523 3
       %4525 = OpCompositeInsert %v8half %4241 %4524 4
       %4526 = OpCompositeInsert %v8half %4253 %4525 5
       %4527 = OpCompositeInsert %v8half %4265 %4526 6
       %4528 = OpCompositeInsert %v8half %4277 %4527 7
       %4529 = OpCompositeInsert %v8half %4196 %106 0
       %4530 = OpCompositeInsert %v8half %4208 %4529 1
       %4531 = OpCompositeInsert %v8half %4220 %4530 2
       %4532 = OpCompositeInsert %v8half %4232 %4531 3
       %4533 = OpCompositeInsert %v8half %4244 %4532 4
       %4534 = OpCompositeInsert %v8half %4256 %4533 5
       %4535 = OpCompositeInsert %v8half %4268 %4534 6
       %4536 = OpCompositeInsert %v8half %4280 %4535 7
       %4537 = OpCompositeInsert %v8half %4283 %106 0
       %4538 = OpCompositeInsert %v8half %4295 %4537 1
       %4539 = OpCompositeInsert %v8half %4307 %4538 2
       %4540 = OpCompositeInsert %v8half %4319 %4539 3
       %4541 = OpCompositeInsert %v8half %4331 %4540 4
       %4542 = OpCompositeInsert %v8half %4343 %4541 5
       %4543 = OpCompositeInsert %v8half %4355 %4542 6
       %4544 = OpCompositeInsert %v8half %4367 %4543 7
       %4545 = OpCompositeInsert %v8half %4286 %106 0
       %4546 = OpCompositeInsert %v8half %4298 %4545 1
       %4547 = OpCompositeInsert %v8half %4310 %4546 2
       %4548 = OpCompositeInsert %v8half %4322 %4547 3
       %4549 = OpCompositeInsert %v8half %4334 %4548 4
       %4550 = OpCompositeInsert %v8half %4346 %4549 5
       %4551 = OpCompositeInsert %v8half %4358 %4550 6
       %4552 = OpCompositeInsert %v8half %4370 %4551 7
       %4553 = OpCompositeInsert %v8half %4289 %106 0
       %4554 = OpCompositeInsert %v8half %4301 %4553 1
       %4555 = OpCompositeInsert %v8half %4313 %4554 2
       %4556 = OpCompositeInsert %v8half %4325 %4555 3
       %4557 = OpCompositeInsert %v8half %4337 %4556 4
       %4558 = OpCompositeInsert %v8half %4349 %4557 5
       %4559 = OpCompositeInsert %v8half %4361 %4558 6
       %4560 = OpCompositeInsert %v8half %4373 %4559 7
       %4561 = OpCompositeInsert %v8half %4292 %106 0
       %4562 = OpCompositeInsert %v8half %4304 %4561 1
       %4563 = OpCompositeInsert %v8half %4316 %4562 2
       %4564 = OpCompositeInsert %v8half %4328 %4563 3
       %4565 = OpCompositeInsert %v8half %4340 %4564 4
       %4566 = OpCompositeInsert %v8half %4352 %4565 5
       %4567 = OpCompositeInsert %v8half %4364 %4566 6
       %4568 = OpCompositeInsert %v8half %4376 %4567 7
       %4569 = OpCompositeInsert %v8half %4379 %106 0
       %4570 = OpCompositeInsert %v8half %4391 %4569 1
       %4571 = OpCompositeInsert %v8half %4403 %4570 2
       %4572 = OpCompositeInsert %v8half %4415 %4571 3
       %4573 = OpCompositeInsert %v8half %4427 %4572 4
       %4574 = OpCompositeInsert %v8half %4439 %4573 5
       %4575 = OpCompositeInsert %v8half %4451 %4574 6
       %4576 = OpCompositeInsert %v8half %4463 %4575 7
       %4577 = OpCompositeInsert %v8half %4382 %106 0
       %4578 = OpCompositeInsert %v8half %4394 %4577 1
       %4579 = OpCompositeInsert %v8half %4406 %4578 2
       %4580 = OpCompositeInsert %v8half %4418 %4579 3
       %4581 = OpCompositeInsert %v8half %4430 %4580 4
       %4582 = OpCompositeInsert %v8half %4442 %4581 5
       %4583 = OpCompositeInsert %v8half %4454 %4582 6
       %4584 = OpCompositeInsert %v8half %4466 %4583 7
       %4585 = OpCompositeInsert %v8half %4385 %106 0
       %4586 = OpCompositeInsert %v8half %4397 %4585 1
       %4587 = OpCompositeInsert %v8half %4409 %4586 2
       %4588 = OpCompositeInsert %v8half %4421 %4587 3
       %4589 = OpCompositeInsert %v8half %4433 %4588 4
       %4590 = OpCompositeInsert %v8half %4445 %4589 5
       %4591 = OpCompositeInsert %v8half %4457 %4590 6
       %4592 = OpCompositeInsert %v8half %4469 %4591 7
       %4593 = OpCompositeInsert %v8half %4388 %106 0
       %4594 = OpCompositeInsert %v8half %4400 %4593 1
       %4595 = OpCompositeInsert %v8half %4412 %4594 2
       %4596 = OpCompositeInsert %v8half %4424 %4595 3
       %4597 = OpCompositeInsert %v8half %4436 %4596 4
       %4598 = OpCompositeInsert %v8half %4448 %4597 5
       %4599 = OpCompositeInsert %v8half %4460 %4598 6
       %4600 = OpCompositeInsert %v8half %4472 %4599 7
       %4601 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4027 %4480 %2810
       %4602 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4027 %4488 %2808
       %4603 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4027 %4496 %2806
       %4604 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4027 %4504 %2804
       %4605 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4043 %4480 %2802
       %4606 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4043 %4488 %2800
       %4607 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4043 %4496 %2798
       %4608 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4043 %4504 %2796
       %4609 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4059 %4480 %2794
       %4610 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4059 %4488 %2792
       %4611 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4059 %4496 %2790
       %4612 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4059 %4504 %2788
       %4613 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4075 %4480 %2786
       %4614 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4075 %4488 %2784
       %4615 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4075 %4496 %2782
       %4616 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4075 %4504 %2780
       %4617 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4031 %4512 %4601
       %4618 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4031 %4520 %4602
       %4619 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4031 %4528 %4603
       %4620 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4031 %4536 %4604
       %4621 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4047 %4512 %4605
       %4622 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4047 %4520 %4606
       %4623 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4047 %4528 %4607
       %4624 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4047 %4536 %4608
       %4625 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4063 %4512 %4609
       %4626 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4063 %4520 %4610
       %4627 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4063 %4528 %4611
       %4628 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4063 %4536 %4612
       %4629 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4079 %4512 %4613
       %4630 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4079 %4520 %4614
       %4631 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4079 %4528 %4615
       %4632 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4079 %4536 %4616
       %4633 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4035 %4544 %4617
       %4634 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4035 %4552 %4618
       %4635 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4035 %4560 %4619
       %4636 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4035 %4568 %4620
       %4637 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4051 %4544 %4621
       %4638 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4051 %4552 %4622
       %4639 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4051 %4560 %4623
       %4640 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4051 %4568 %4624
       %4641 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4067 %4544 %4625
       %4642 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4067 %4552 %4626
       %4643 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4067 %4560 %4627
       %4644 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4067 %4568 %4628
       %4645 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4083 %4544 %4629
       %4646 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4083 %4552 %4630
       %4647 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4083 %4560 %4631
       %4648 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4083 %4568 %4632
       %2811 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4039 %4576 %4633
       %2809 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4039 %4584 %4634
       %2807 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4039 %4592 %4635
       %2805 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4039 %4600 %4636
       %2803 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4055 %4576 %4637
       %2801 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4055 %4584 %4638
       %2799 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4055 %4592 %4639
       %2797 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4055 %4600 %4640
       %2795 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4071 %4576 %4641
       %2793 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4071 %4584 %4642
       %2791 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4071 %4592 %4643
       %2789 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4071 %4600 %4644
       %2787 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4087 %4576 %4645
       %2785 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4087 %4584 %4646
       %2783 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4087 %4592 %4647
       %2781 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %4087 %4600 %4648
               OpMemoryBarrier %uint_2 %uint_4
       %4649 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_signal %uint_4294967295
       %4650 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_wait %ushort_65535
               OpMemoryBarrier %uint_2 %uint_2
       %4651 = OpFunctionCall %void %spirv_llvm_amdgcn_sched_barrier %90
       %4652 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %2570
       %4653 = OpBitcast %_ptr_Workgroup_v8half %4652
               OpStore %4653 %3991 Aligned 2
       %4654 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %2575
       %4655 = OpBitcast %_ptr_Workgroup_v8half %4654
               OpStore %4655 %3994 Aligned 2
       %4656 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %2579
       %4657 = OpBitcast %_ptr_Workgroup_v8half %4656
               OpStore %4657 %3997 Aligned 2
       %4658 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %2583
       %4659 = OpBitcast %_ptr_Workgroup_v8half %4658
               OpStore %4659 %4000 Aligned 2
       %4660 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %2587
       %4661 = OpBitcast %_ptr_Workgroup_v8half %4660
               OpStore %4661 %4003 Aligned 2
       %4662 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %2591
       %4663 = OpBitcast %_ptr_Workgroup_v8half %4662
               OpStore %4663 %4006 Aligned 2
       %4664 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %2595
       %4665 = OpBitcast %_ptr_Workgroup_v8half %4664
               OpStore %4665 %4009 Aligned 2
       %4666 = OpPtrAccessChain %_ptr_Workgroup_half %4024 %2599
       %4667 = OpBitcast %_ptr_Workgroup_v8half %4666
               OpStore %4667 %4012 Aligned 2
       %4668 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %2603
       %4669 = OpBitcast %_ptr_Workgroup_v8half %4668
               OpStore %4669 %4014 Aligned 2
       %4670 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %2608
       %4671 = OpBitcast %_ptr_Workgroup_v8half %4670
               OpStore %4671 %4016 Aligned 2
       %4672 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %2612
       %4673 = OpBitcast %_ptr_Workgroup_v8half %4672
               OpStore %4673 %4018 Aligned 2
       %4674 = OpPtrAccessChain %_ptr_Workgroup_half %4089 %2616
       %4675 = OpBitcast %_ptr_Workgroup_v8half %4674
               OpStore %4675 %4020 Aligned 2
       %2777 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2776 %ulong_131072
       %2775 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2774 %ulong_131072
       %2773 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2772 %ulong_131072
       %2771 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2770 %ulong_131072
       %2769 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2768 %ulong_128
       %2767 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2766 %ulong_128
       %2765 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2764 %ulong_128
       %2763 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2762 %ulong_128
       %2761 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2760 %ulong_128
       %2759 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2758 %ulong_128
       %2757 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2756 %ulong_128
       %2755 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %2754 %ulong_128
               OpBranch %6958
               OpFunctionEnd
%matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32 = OpFunction %void Inline %7
       %4676 = OpFunctionParameter %_ptr_CrossWorkgroup_half
       %4677 = OpFunctionParameter %_ptr_CrossWorkgroup_half
       %4678 = OpFunctionParameter %_ptr_CrossWorkgroup_float
       %6961 = OpLabel
       %4680 = OpFunctionCall %uint %spirv_llvm_amdgcn_workitem_id_x
       %4681 = OpSConvert %ulong %4680
       %4682 = OpUDiv %uint %4680 %uint_64
       %4683 = OpUMod %uint %4680 %uint_64
       %4684 = OpUDiv %uint %4683 %uint_32
       %4685 = OpUMod %uint %4680 %uint_32
       %4686 = OpUDiv %uint %4685 %uint_16
       %4687 = OpUMod %uint %4685 %uint_16
       %4688 = OpIMul %uint %4686 %uint_8
       %4689 = OpUConvert %ulong %4688
       %4690 = OpUDiv %uint %4680 %uint_8
       %4691 = OpUConvert %ulong %4690
       %4692 = OpUMod %uint %4680 %uint_8
       %4693 = OpIAdd %uint %4680 %uint_256
       %4694 = OpUDiv %uint %4693 %uint_8
       %4695 = OpUConvert %ulong %4694
       %4696 = OpUMod %uint %4693 %uint_8
       %4697 = OpIAdd %uint %4680 %uint_512
       %4698 = OpUDiv %uint %4697 %uint_8
       %4699 = OpUConvert %ulong %4698
       %4700 = OpUMod %uint %4697 %uint_8
       %4701 = OpIAdd %uint %4680 %uint_768
       %4702 = OpUDiv %uint %4701 %uint_8
       %4703 = OpUConvert %ulong %4702
       %4704 = OpUMod %uint %4701 %uint_8
       %4705 = OpIAdd %uint %4680 %uint_1024
       %4706 = OpUDiv %uint %4705 %uint_8
       %4707 = OpUConvert %ulong %4706
       %4708 = OpUMod %uint %4705 %uint_8
       %4709 = OpIAdd %uint %4680 %uint_1280
       %4710 = OpUDiv %uint %4709 %uint_8
       %4711 = OpUConvert %ulong %4710
       %4712 = OpUMod %uint %4709 %uint_8
       %4713 = OpIAdd %uint %4680 %uint_1536
       %4714 = OpUDiv %uint %4713 %uint_8
       %4715 = OpUConvert %ulong %4714
       %4716 = OpUMod %uint %4713 %uint_8
       %4717 = OpIAdd %uint %4680 %uint_1792
       %4718 = OpUDiv %uint %4717 %uint_8
       %4719 = OpUConvert %ulong %4718
       %4720 = OpUMod %uint %4717 %uint_8
       %4721 = OpUDiv %uint %4680 %uint_16
       %4722 = OpUConvert %ulong %4721
       %4723 = OpUMod %uint %4680 %uint_16
       %4724 = OpUDiv %uint %4693 %uint_16
       %4725 = OpUConvert %ulong %4724
       %4726 = OpUMod %uint %4693 %uint_16
       %4727 = OpUDiv %uint %4697 %uint_16
       %4728 = OpUConvert %ulong %4727
       %4729 = OpUMod %uint %4697 %uint_16
       %4730 = OpUDiv %uint %4701 %uint_16
       %4731 = OpUConvert %ulong %4730
       %4732 = OpUMod %uint %4701 %uint_16
       %4733 = OpIMul %uint %4692 %uint_8
       %4734 = OpUConvert %ulong %4733
       %4735 = OpIMul %uint %4696 %uint_8
       %4736 = OpUConvert %ulong %4735
       %4737 = OpIMul %uint %4700 %uint_8
       %4738 = OpUConvert %ulong %4737
       %4739 = OpIMul %uint %4704 %uint_8
       %4740 = OpUConvert %ulong %4739
       %4741 = OpIMul %uint %4708 %uint_8
       %4742 = OpUConvert %ulong %4741
       %4743 = OpIMul %uint %4712 %uint_8
       %4744 = OpUConvert %ulong %4743
       %4745 = OpIMul %uint %4716 %uint_8
       %4746 = OpUConvert %ulong %4745
       %4747 = OpIMul %uint %4720 %uint_8
       %4748 = OpUConvert %ulong %4747
       %4749 = OpIMul %uint %4723 %uint_8
       %4750 = OpUConvert %ulong %4749
       %4751 = OpIMul %uint %4726 %uint_8
       %4752 = OpUConvert %ulong %4751
       %4753 = OpIMul %uint %4729 %uint_8
       %4754 = OpUConvert %ulong %4753
       %4755 = OpIMul %uint %4732 %uint_8
       %4756 = OpUConvert %ulong %4755
       %4757 = OpFunctionCall %uint %spirv_llvm_amdgcn_workgroup_id_x
       %4758 = OpUDiv %uint %4757 %uint_32
       %4759 = OpUMod %uint %4757 %uint_32
       %4760 = OpIMul %uint %4758 %uint_256
       %4761 = OpIAdd %uint %4690 %4760
       %4762 = OpUConvert %ulong %4761
       %4763 = OpIAdd %uint %4694 %4760
       %4764 = OpUConvert %ulong %4763
       %4765 = OpIMul %ulong %4762 %ulong_4096
       %4766 = OpIAdd %ulong %4765 %4734
       %4767 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %4676 %4766
       %4768 = OpBitcast %_ptr_CrossWorkgroup_v8half %4767
       %4769 = OpLoad %v8half %4768 Aligned 2
       %4770 = OpIAdd %uint %4698 %4760
       %4771 = OpUConvert %ulong %4770
       %4772 = OpIMul %ulong %4764 %ulong_4096
       %4773 = OpIAdd %ulong %4772 %4736
       %4774 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %4676 %4773
       %4775 = OpBitcast %_ptr_CrossWorkgroup_v8half %4774
       %4776 = OpLoad %v8half %4775 Aligned 2
       %4777 = OpIAdd %uint %4702 %4760
       %4778 = OpUConvert %ulong %4777
       %4779 = OpIMul %ulong %4771 %ulong_4096
       %4780 = OpIAdd %ulong %4779 %4738
       %4781 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %4676 %4780
       %4782 = OpBitcast %_ptr_CrossWorkgroup_v8half %4781
       %4783 = OpLoad %v8half %4782 Aligned 2
       %4784 = OpIAdd %uint %4706 %4760
       %4785 = OpUConvert %ulong %4784
       %4786 = OpIMul %ulong %4778 %ulong_4096
       %4787 = OpIAdd %ulong %4786 %4740
       %4788 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %4676 %4787
       %4789 = OpBitcast %_ptr_CrossWorkgroup_v8half %4788
       %4790 = OpLoad %v8half %4789 Aligned 2
       %4791 = OpIAdd %uint %4710 %4760
       %4792 = OpUConvert %ulong %4791
       %4793 = OpIMul %ulong %4785 %ulong_4096
       %4794 = OpIAdd %ulong %4793 %4742
       %4795 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %4676 %4794
       %4796 = OpBitcast %_ptr_CrossWorkgroup_v8half %4795
       %4797 = OpLoad %v8half %4796 Aligned 2
       %4798 = OpIAdd %uint %4714 %4760
       %4799 = OpUConvert %ulong %4798
       %4800 = OpIMul %ulong %4792 %ulong_4096
       %4801 = OpIAdd %ulong %4800 %4744
       %4802 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %4676 %4801
       %4803 = OpBitcast %_ptr_CrossWorkgroup_v8half %4802
       %4804 = OpLoad %v8half %4803 Aligned 2
       %4805 = OpIAdd %uint %4718 %4760
       %4806 = OpUConvert %ulong %4805
       %4807 = OpIMul %ulong %4799 %ulong_4096
       %4808 = OpIAdd %ulong %4807 %4746
       %4809 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %4676 %4808
       %4810 = OpBitcast %_ptr_CrossWorkgroup_v8half %4809
       %4811 = OpLoad %v8half %4810 Aligned 2
       %4812 = OpIMul %uint %4759 %uint_128
       %4813 = OpIAdd %uint %4749 %4812
       %4814 = OpUConvert %ulong %4813
       %4815 = OpIMul %ulong %4806 %ulong_4096
       %4816 = OpIAdd %ulong %4815 %4748
       %4817 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %4676 %4816
       %4818 = OpBitcast %_ptr_CrossWorkgroup_v8half %4817
       %4819 = OpLoad %v8half %4818 Aligned 2
       %4820 = OpIAdd %uint %4751 %4812
       %4821 = OpUConvert %ulong %4820
       %4822 = OpIMul %ulong %4722 %ulong_4096
       %4823 = OpIAdd %ulong %4822 %4814
       %4824 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %4677 %4823
       %4825 = OpBitcast %_ptr_CrossWorkgroup_v8half %4824
       %4826 = OpLoad %v8half %4825 Aligned 2
       %4827 = OpIAdd %uint %4753 %4812
       %4828 = OpUConvert %ulong %4827
       %4829 = OpIMul %ulong %4725 %ulong_4096
       %4830 = OpIAdd %ulong %4829 %4821
       %4831 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %4677 %4830
       %4832 = OpBitcast %_ptr_CrossWorkgroup_v8half %4831
       %4833 = OpLoad %v8half %4832 Aligned 2
       %4834 = OpIAdd %uint %4755 %4812
       %4835 = OpUConvert %ulong %4834
       %4836 = OpIMul %ulong %4728 %ulong_4096
       %4837 = OpIAdd %ulong %4836 %4828
       %4838 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %4677 %4837
       %4839 = OpBitcast %_ptr_CrossWorkgroup_v8half %4838
       %4840 = OpLoad %v8half %4839 Aligned 2
       %4841 = OpIMul %ulong %4731 %ulong_4096
       %4842 = OpIAdd %ulong %4841 %4835
       %4843 = OpPtrAccessChain %_ptr_CrossWorkgroup_half %4677 %4842
       %4844 = OpBitcast %_ptr_CrossWorkgroup_v8half %4843
       %4845 = OpLoad %v8half %4844 Aligned 2
       %4846 = OpIMul %ulong %4691 %ulong_68
       %4847 = OpIAdd %ulong %4846 %4734
       %4848 = OpBitcast %_ptr_Workgroup_half %__shared_memory__
       %4849 = OpPtrAccessChain %_ptr_Workgroup_half %4848 %4847
       %4850 = OpBitcast %_ptr_Workgroup_v8half %4849
               OpStore %4850 %4769 Aligned 2
       %4851 = OpIMul %ulong %4695 %ulong_68
       %4852 = OpIAdd %ulong %4851 %4736
       %4853 = OpPtrAccessChain %_ptr_Workgroup_half %4848 %4852
       %4854 = OpBitcast %_ptr_Workgroup_v8half %4853
               OpStore %4854 %4776 Aligned 2
       %4855 = OpIMul %ulong %4699 %ulong_68
       %4856 = OpIAdd %ulong %4855 %4738
       %4857 = OpPtrAccessChain %_ptr_Workgroup_half %4848 %4856
       %4858 = OpBitcast %_ptr_Workgroup_v8half %4857
               OpStore %4858 %4783 Aligned 2
       %4859 = OpIMul %ulong %4703 %ulong_68
       %4860 = OpIAdd %ulong %4859 %4740
       %4861 = OpPtrAccessChain %_ptr_Workgroup_half %4848 %4860
       %4862 = OpBitcast %_ptr_Workgroup_v8half %4861
               OpStore %4862 %4790 Aligned 2
       %4863 = OpIMul %ulong %4707 %ulong_68
       %4864 = OpIAdd %ulong %4863 %4742
       %4865 = OpPtrAccessChain %_ptr_Workgroup_half %4848 %4864
       %4866 = OpBitcast %_ptr_Workgroup_v8half %4865
               OpStore %4866 %4797 Aligned 2
       %4867 = OpIMul %ulong %4711 %ulong_68
       %4868 = OpIAdd %ulong %4867 %4744
       %4869 = OpPtrAccessChain %_ptr_Workgroup_half %4848 %4868
       %4870 = OpBitcast %_ptr_Workgroup_v8half %4869
               OpStore %4870 %4804 Aligned 2
       %4871 = OpIMul %ulong %4715 %ulong_68
       %4872 = OpIAdd %ulong %4871 %4746
       %4873 = OpPtrAccessChain %_ptr_Workgroup_half %4848 %4872
       %4874 = OpBitcast %_ptr_Workgroup_v8half %4873
               OpStore %4874 %4811 Aligned 2
       %4875 = OpIMul %ulong %4719 %ulong_68
       %4876 = OpIAdd %ulong %4875 %4748
       %4877 = OpPtrAccessChain %_ptr_Workgroup_half %4848 %4876
       %4878 = OpBitcast %_ptr_Workgroup_v8half %4877
               OpStore %4878 %4819 Aligned 2
       %4879 = OpIMul %ulong %4722 %ulong_132
       %4880 = OpIAdd %ulong %4879 %4750
       %4881 = OpBitcast %_ptr_Workgroup_half %__shared_memory___0
       %4882 = OpPtrAccessChain %_ptr_Workgroup_half %4881 %4880
       %4883 = OpBitcast %_ptr_Workgroup_v8half %4882
               OpStore %4883 %4826 Aligned 2
       %4884 = OpIMul %ulong %4725 %ulong_132
       %4885 = OpIAdd %ulong %4884 %4752
       %4886 = OpPtrAccessChain %_ptr_Workgroup_half %4881 %4885
       %4887 = OpBitcast %_ptr_Workgroup_v8half %4886
               OpStore %4887 %4833 Aligned 2
       %4888 = OpIMul %ulong %4728 %ulong_132
       %4889 = OpIAdd %ulong %4888 %4754
       %4890 = OpPtrAccessChain %_ptr_Workgroup_half %4881 %4889
       %4891 = OpBitcast %_ptr_Workgroup_v8half %4890
               OpStore %4891 %4840 Aligned 2
       %4892 = OpIMul %ulong %4731 %ulong_132
       %4893 = OpIAdd %ulong %4892 %4756
       %4894 = OpPtrAccessChain %_ptr_Workgroup_half %4881 %4893
       %4895 = OpBitcast %_ptr_Workgroup_v8half %4894
               OpStore %4895 %4845 Aligned 2
       %4896 = OpIMul %uint %4682 %uint_4
       %4897 = OpIMul %uint %4682 %uint_64
       %4898 = OpIAdd %uint %4897 %4687
       %4899 = OpUConvert %ulong %4898
       %4900 = OpIAdd %uint %4688 %uint_16
       %4901 = OpUConvert %ulong %4900
       %4902 = OpIAdd %uint %4688 %uint_32
       %4903 = OpUConvert %ulong %4902
       %4904 = OpIAdd %uint %4688 %uint_48
       %4905 = OpUConvert %ulong %4904
       %4906 = OpIAdd %uint %4898 %uint_16
       %4907 = OpUConvert %ulong %4906
       %4908 = OpIAdd %uint %4898 %uint_32
       %4909 = OpUConvert %ulong %4908
       %4910 = OpIAdd %uint %4898 %uint_48
       %4911 = OpUConvert %ulong %4910
       %4912 = OpIMul %uint %4684 %uint_4
       %4913 = OpIMul %uint %4684 %uint_64
       %4914 = OpIAdd %uint %4913 %4687
       %4915 = OpUConvert %ulong %4914
       %4916 = OpIAdd %uint %4914 %uint_16
       %4917 = OpUConvert %ulong %4916
       %4918 = OpIAdd %uint %4914 %uint_32
       %4919 = OpUConvert %ulong %4918
       %4920 = OpIAdd %uint %4914 %uint_48
       %4921 = OpUConvert %ulong %4920
       %4922 = OpIAdd %uint %4688 %uint_1
       %4923 = OpUConvert %ulong %4922
       %4924 = OpIAdd %uint %4688 %uint_2
       %4925 = OpUConvert %ulong %4924
       %4926 = OpIAdd %uint %4688 %uint_3
       %4927 = OpUConvert %ulong %4926
       %4928 = OpIAdd %uint %4688 %uint_4
       %4929 = OpUConvert %ulong %4928
       %4930 = OpIAdd %uint %4688 %uint_5
       %4931 = OpUConvert %ulong %4930
       %4932 = OpIAdd %uint %4688 %uint_6
       %4933 = OpUConvert %ulong %4932
       %4934 = OpIAdd %uint %4688 %uint_7
       %4935 = OpUConvert %ulong %4934
       %4936 = OpIAdd %uint %4688 %uint_17
       %4937 = OpUConvert %ulong %4936
       %4938 = OpIAdd %uint %4688 %uint_18
       %4939 = OpUConvert %ulong %4938
       %4940 = OpIAdd %uint %4688 %uint_19
       %4941 = OpUConvert %ulong %4940
       %4942 = OpIAdd %uint %4688 %uint_20
       %4943 = OpUConvert %ulong %4942
       %4944 = OpIAdd %uint %4688 %uint_21
       %4945 = OpUConvert %ulong %4944
       %4946 = OpIAdd %uint %4688 %uint_22
       %4947 = OpUConvert %ulong %4946
       %4948 = OpIAdd %uint %4688 %uint_23
       %4949 = OpUConvert %ulong %4948
       %4950 = OpIAdd %uint %4688 %uint_33
       %4951 = OpUConvert %ulong %4950
       %4952 = OpIAdd %uint %4688 %uint_34
       %4953 = OpUConvert %ulong %4952
       %4954 = OpIAdd %uint %4688 %uint_35
       %4955 = OpUConvert %ulong %4954
       %4956 = OpIAdd %uint %4688 %uint_36
       %4957 = OpUConvert %ulong %4956
       %4958 = OpIAdd %uint %4688 %uint_37
       %4959 = OpUConvert %ulong %4958
       %4960 = OpIAdd %uint %4688 %uint_38
       %4961 = OpUConvert %ulong %4960
       %4962 = OpIAdd %uint %4688 %uint_39
       %4963 = OpUConvert %ulong %4962
       %4964 = OpIAdd %uint %4688 %uint_49
       %4965 = OpUConvert %ulong %4964
       %4966 = OpIAdd %uint %4688 %uint_50
       %4967 = OpUConvert %ulong %4966
       %4968 = OpIAdd %uint %4688 %uint_51
       %4969 = OpUConvert %ulong %4968
       %4970 = OpIAdd %uint %4688 %uint_52
       %4971 = OpUConvert %ulong %4970
       %4972 = OpIAdd %uint %4688 %uint_53
       %4973 = OpUConvert %ulong %4972
       %4974 = OpIAdd %uint %4688 %uint_54
       %4975 = OpUConvert %ulong %4974
       %4976 = OpIAdd %uint %4688 %uint_55
       %4977 = OpUConvert %ulong %4976
       %4978 = OpIAdd %ulong %4681 %ulong_768
       %4979 = OpShiftRightLogical %ulong %4978 %ulong_4
       %4980 = OpShiftLeftLogical %ulong %4979 %ulong_13
       %4981 = OpShiftLeftLogical %ulong %4835 %ulong_1
       %4982 = OpIAdd %ulong %4980 %4981
       %4983 = OpIAdd %ulong %4982 %ulong_524288
       %4984 = OpBitcast %_ptr_CrossWorkgroup_uchar %4677
       %4985 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %4984 %4983
       %4986 = OpIAdd %ulong %4681 %ulong_512
       %4987 = OpShiftRightLogical %ulong %4986 %ulong_4
       %4988 = OpShiftLeftLogical %ulong %4987 %ulong_13
       %4989 = OpIAdd %ulong %4988 %4981
       %4990 = OpIAdd %ulong %4989 %ulong_524288
       %4991 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %4984 %4990
       %4992 = OpIAdd %ulong %4681 %ulong_256
       %4993 = OpShiftRightLogical %ulong %4992 %ulong_4
       %4994 = OpShiftLeftLogical %ulong %4993 %ulong_13
       %4995 = OpIAdd %ulong %4994 %4981
       %4996 = OpIAdd %ulong %4995 %ulong_524288
       %4997 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %4984 %4996
       %4998 = OpShiftRightLogical %ulong %4681 %ulong_4
       %4999 = OpShiftLeftLogical %ulong %4998 %ulong_13
       %5000 = OpIAdd %ulong %4999 %4981
       %5001 = OpIAdd %ulong %5000 %ulong_524288
       %5002 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %4984 %5001
       %5003 = OpUConvert %ulong %4680
       %5004 = OpBitwiseAnd %ulong %5003 %ulong_7
       %5005 = OpShiftLeftLogical %ulong %5004 %ulong_4
       %5006 = OpShiftLeftLogical %ulong %4806 %ulong_13
       %5007 = OpIAdd %ulong %5006 %ulong_128
       %5008 = OpBitcast %_ptr_CrossWorkgroup_uchar %4676
       %5009 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5008 %5007
       %5010 = OpShiftLeftLogical %ulong %4799 %ulong_13
       %5011 = OpIAdd %ulong %5010 %ulong_128
       %5012 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5008 %5011
       %5013 = OpShiftLeftLogical %ulong %4792 %ulong_13
       %5014 = OpIAdd %ulong %5013 %ulong_128
       %5015 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5008 %5014
       %5016 = OpShiftLeftLogical %ulong %4785 %ulong_13
       %5017 = OpIAdd %ulong %5016 %ulong_128
       %5018 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5008 %5017
       %5019 = OpShiftLeftLogical %ulong %4778 %ulong_13
       %5020 = OpIAdd %ulong %5019 %ulong_128
       %5021 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5008 %5020
       %5022 = OpShiftLeftLogical %ulong %4771 %ulong_13
       %5023 = OpIAdd %ulong %5022 %ulong_128
       %5024 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5008 %5023
       %5025 = OpShiftLeftLogical %ulong %4764 %ulong_13
       %5026 = OpIAdd %ulong %5025 %ulong_128
       %5027 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5008 %5026
       %5028 = OpShiftLeftLogical %ulong %4762 %ulong_13
       %5029 = OpIAdd %ulong %5028 %ulong_128
       %5030 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5008 %5029
               OpBranch %6962
       %6962 = OpLabel
       %5031 = OpPhi %_ptr_CrossWorkgroup_uchar %5032 %6963 %5030 %6961
       %5033 = OpPhi %_ptr_CrossWorkgroup_uchar %5034 %6963 %5027 %6961
       %5035 = OpPhi %_ptr_CrossWorkgroup_uchar %5036 %6963 %5024 %6961
       %5037 = OpPhi %_ptr_CrossWorkgroup_uchar %5038 %6963 %5021 %6961
       %5039 = OpPhi %_ptr_CrossWorkgroup_uchar %5040 %6963 %5018 %6961
       %5041 = OpPhi %_ptr_CrossWorkgroup_uchar %5042 %6963 %5015 %6961
       %5043 = OpPhi %_ptr_CrossWorkgroup_uchar %5044 %6963 %5012 %6961
       %5045 = OpPhi %_ptr_CrossWorkgroup_uchar %5046 %6963 %5009 %6961
       %5047 = OpPhi %_ptr_CrossWorkgroup_uchar %5048 %6963 %5002 %6961
       %5049 = OpPhi %_ptr_CrossWorkgroup_uchar %5050 %6963 %4997 %6961
       %5051 = OpPhi %_ptr_CrossWorkgroup_uchar %5052 %6963 %4991 %6961
       %5053 = OpPhi %_ptr_CrossWorkgroup_uchar %5054 %6963 %4985 %6961
       %5055 = OpPhi %uint %5056 %6963 %90 %6961
       %5057 = OpPhi %v8float %5058 %6963 %113 %6961
       %5059 = OpPhi %v8float %5060 %6963 %113 %6961
       %5061 = OpPhi %v8float %5062 %6963 %113 %6961
       %5063 = OpPhi %v8float %5064 %6963 %113 %6961
       %5065 = OpPhi %v8float %5066 %6963 %113 %6961
       %5067 = OpPhi %v8float %5068 %6963 %113 %6961
       %5069 = OpPhi %v8float %5070 %6963 %113 %6961
       %5071 = OpPhi %v8float %5072 %6963 %113 %6961
       %5073 = OpPhi %v8float %5074 %6963 %113 %6961
       %5075 = OpPhi %v8float %5076 %6963 %113 %6961
       %5077 = OpPhi %v8float %5078 %6963 %113 %6961
       %5079 = OpPhi %v8float %5080 %6963 %113 %6961
       %5081 = OpPhi %v8float %5082 %6963 %113 %6961
       %5083 = OpPhi %v8float %5084 %6963 %113 %6961
       %5085 = OpPhi %v8float %5086 %6963 %113 %6961
       %5087 = OpPhi %v8float %5088 %6963 %113 %6961
       %5089 = OpSLessThan %bool %5055 %uint_252
       %5090 = OpIMul %ulong %4899 %ulong_68
       %5091 = OpIMul %ulong %4907 %ulong_68
       %5092 = OpIMul %ulong %4909 %ulong_68
       %5093 = OpIMul %ulong %4911 %ulong_68
       %5094 = OpIMul %ulong %4689 %ulong_132
       %5095 = OpIMul %ulong %4923 %ulong_132
       %5096 = OpIMul %ulong %4925 %ulong_132
       %5097 = OpIMul %ulong %4927 %ulong_132
       %5098 = OpIMul %ulong %4929 %ulong_132
       %5099 = OpIMul %ulong %4931 %ulong_132
       %5100 = OpIMul %ulong %4933 %ulong_132
       %5101 = OpIMul %ulong %4935 %ulong_132
       %5102 = OpIMul %ulong %4901 %ulong_132
       %5103 = OpIMul %ulong %4937 %ulong_132
       %5104 = OpIMul %ulong %4939 %ulong_132
       %5105 = OpIMul %ulong %4941 %ulong_132
       %5106 = OpIMul %ulong %4943 %ulong_132
       %5107 = OpIMul %ulong %4945 %ulong_132
       %5108 = OpIMul %ulong %4947 %ulong_132
       %5109 = OpIMul %ulong %4949 %ulong_132
       %5110 = OpIMul %ulong %4903 %ulong_132
       %5111 = OpIMul %ulong %4951 %ulong_132
       %5112 = OpIMul %ulong %4953 %ulong_132
       %5113 = OpIMul %ulong %4955 %ulong_132
       %5114 = OpIMul %ulong %4957 %ulong_132
       %5115 = OpIMul %ulong %4959 %ulong_132
       %5116 = OpIMul %ulong %4961 %ulong_132
       %5117 = OpIMul %ulong %4963 %ulong_132
       %5118 = OpIMul %ulong %4905 %ulong_132
       %5119 = OpIMul %ulong %4965 %ulong_132
       %5120 = OpIMul %ulong %4967 %ulong_132
       %5121 = OpIMul %ulong %4969 %ulong_132
       %5122 = OpIMul %ulong %4971 %ulong_132
       %5123 = OpIMul %ulong %4973 %ulong_132
       %5124 = OpIMul %ulong %4975 %ulong_132
       %5125 = OpIMul %ulong %4977 %ulong_132
               OpBranchConditional %5089 %6963 %6964
       %6964 = OpLabel
               OpMemoryBarrier %uint_2 %uint_4
       %5126 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_signal %uint_4294967295
       %5127 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_wait %ushort_65535
               OpMemoryBarrier %uint_2 %uint_2
       %5128 = OpIAdd %ulong %5090 %4689
       %5129 = OpBitcast %_ptr_Workgroup_half %__shared_memory__
       %5130 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5128
       %5131 = OpBitcast %_ptr_Workgroup_v8half %5130
       %5132 = OpLoad %v8half %5131 Aligned 2
       %5133 = OpIAdd %ulong %5090 %4901
       %5134 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5133
       %5135 = OpBitcast %_ptr_Workgroup_v8half %5134
       %5136 = OpLoad %v8half %5135 Aligned 2
       %5137 = OpIAdd %ulong %5090 %4903
       %5138 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5137
       %5139 = OpBitcast %_ptr_Workgroup_v8half %5138
       %5140 = OpLoad %v8half %5139 Aligned 2
       %5141 = OpIAdd %ulong %5090 %4905
       %5142 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5141
       %5143 = OpBitcast %_ptr_Workgroup_v8half %5142
       %5144 = OpLoad %v8half %5143 Aligned 2
       %5145 = OpIAdd %ulong %5091 %4689
       %5146 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5145
       %5147 = OpBitcast %_ptr_Workgroup_v8half %5146
       %5148 = OpLoad %v8half %5147 Aligned 2
       %5149 = OpIAdd %ulong %5091 %4901
       %5150 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5149
       %5151 = OpBitcast %_ptr_Workgroup_v8half %5150
       %5152 = OpLoad %v8half %5151 Aligned 2
       %5153 = OpIAdd %ulong %5091 %4903
       %5154 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5153
       %5155 = OpBitcast %_ptr_Workgroup_v8half %5154
       %5156 = OpLoad %v8half %5155 Aligned 2
       %5157 = OpIAdd %ulong %5091 %4905
       %5158 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5157
       %5159 = OpBitcast %_ptr_Workgroup_v8half %5158
       %5160 = OpLoad %v8half %5159 Aligned 2
       %5161 = OpIAdd %ulong %5092 %4689
       %5162 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5161
       %5163 = OpBitcast %_ptr_Workgroup_v8half %5162
       %5164 = OpLoad %v8half %5163 Aligned 2
       %5165 = OpIAdd %ulong %5092 %4901
       %5166 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5165
       %5167 = OpBitcast %_ptr_Workgroup_v8half %5166
       %5168 = OpLoad %v8half %5167 Aligned 2
       %5169 = OpIAdd %ulong %5092 %4903
       %5170 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5169
       %5171 = OpBitcast %_ptr_Workgroup_v8half %5170
       %5172 = OpLoad %v8half %5171 Aligned 2
       %5173 = OpIAdd %ulong %5092 %4905
       %5174 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5173
       %5175 = OpBitcast %_ptr_Workgroup_v8half %5174
       %5176 = OpLoad %v8half %5175 Aligned 2
       %5177 = OpIAdd %ulong %5093 %4689
       %5178 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5177
       %5179 = OpBitcast %_ptr_Workgroup_v8half %5178
       %5180 = OpLoad %v8half %5179 Aligned 2
       %5181 = OpIAdd %ulong %5093 %4901
       %5182 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5181
       %5183 = OpBitcast %_ptr_Workgroup_v8half %5182
       %5184 = OpLoad %v8half %5183 Aligned 2
       %5185 = OpIAdd %ulong %5093 %4903
       %5186 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5185
       %5187 = OpBitcast %_ptr_Workgroup_v8half %5186
       %5188 = OpLoad %v8half %5187 Aligned 2
       %5189 = OpIAdd %ulong %5093 %4905
       %5190 = OpPtrAccessChain %_ptr_Workgroup_half %5129 %5189
       %5191 = OpBitcast %_ptr_Workgroup_v8half %5190
       %5192 = OpLoad %v8half %5191 Aligned 2
       %5193 = OpIAdd %ulong %5094 %4915
       %5194 = OpBitcast %_ptr_Workgroup_half %__shared_memory___0
       %5195 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5193
       %5196 = OpLoad %half %5195 Aligned 2
       %5197 = OpIAdd %ulong %5094 %4917
       %5198 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5197
       %5199 = OpLoad %half %5198 Aligned 2
       %5200 = OpIAdd %ulong %5094 %4919
       %5201 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5200
       %5202 = OpLoad %half %5201 Aligned 2
       %5203 = OpIAdd %ulong %5094 %4921
       %5204 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5203
       %5205 = OpLoad %half %5204 Aligned 2
       %5206 = OpIAdd %ulong %5095 %4915
       %5207 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5206
       %5208 = OpLoad %half %5207 Aligned 2
       %5209 = OpIAdd %ulong %5095 %4917
       %5210 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5209
       %5211 = OpLoad %half %5210 Aligned 2
       %5212 = OpIAdd %ulong %5095 %4919
       %5213 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5212
       %5214 = OpLoad %half %5213 Aligned 2
       %5215 = OpIAdd %ulong %5095 %4921
       %5216 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5215
       %5217 = OpLoad %half %5216 Aligned 2
       %5218 = OpIAdd %ulong %5096 %4915
       %5219 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5218
       %5220 = OpLoad %half %5219 Aligned 2
       %5221 = OpIAdd %ulong %5096 %4917
       %5222 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5221
       %5223 = OpLoad %half %5222 Aligned 2
       %5224 = OpIAdd %ulong %5096 %4919
       %5225 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5224
       %5226 = OpLoad %half %5225 Aligned 2
       %5227 = OpIAdd %ulong %5096 %4921
       %5228 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5227
       %5229 = OpLoad %half %5228 Aligned 2
       %5230 = OpIAdd %ulong %5097 %4915
       %5231 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5230
       %5232 = OpLoad %half %5231 Aligned 2
       %5233 = OpIAdd %ulong %5097 %4917
       %5234 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5233
       %5235 = OpLoad %half %5234 Aligned 2
       %5236 = OpIAdd %ulong %5097 %4919
       %5237 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5236
       %5238 = OpLoad %half %5237 Aligned 2
       %5239 = OpIAdd %ulong %5097 %4921
       %5240 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5239
       %5241 = OpLoad %half %5240 Aligned 2
       %5242 = OpIAdd %ulong %5098 %4915
       %5243 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5242
       %5244 = OpLoad %half %5243 Aligned 2
       %5245 = OpIAdd %ulong %5098 %4917
       %5246 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5245
       %5247 = OpLoad %half %5246 Aligned 2
       %5248 = OpIAdd %ulong %5098 %4919
       %5249 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5248
       %5250 = OpLoad %half %5249 Aligned 2
       %5251 = OpIAdd %ulong %5098 %4921
       %5252 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5251
       %5253 = OpLoad %half %5252 Aligned 2
       %5254 = OpIAdd %ulong %5099 %4915
       %5255 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5254
       %5256 = OpLoad %half %5255 Aligned 2
       %5257 = OpIAdd %ulong %5099 %4917
       %5258 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5257
       %5259 = OpLoad %half %5258 Aligned 2
       %5260 = OpIAdd %ulong %5099 %4919
       %5261 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5260
       %5262 = OpLoad %half %5261 Aligned 2
       %5263 = OpIAdd %ulong %5099 %4921
       %5264 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5263
       %5265 = OpLoad %half %5264 Aligned 2
       %5266 = OpIAdd %ulong %5100 %4915
       %5267 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5266
       %5268 = OpLoad %half %5267 Aligned 2
       %5269 = OpIAdd %ulong %5100 %4917
       %5270 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5269
       %5271 = OpLoad %half %5270 Aligned 2
       %5272 = OpIAdd %ulong %5100 %4919
       %5273 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5272
       %5274 = OpLoad %half %5273 Aligned 2
       %5275 = OpIAdd %ulong %5100 %4921
       %5276 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5275
       %5277 = OpLoad %half %5276 Aligned 2
       %5278 = OpIAdd %ulong %5101 %4915
       %5279 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5278
       %5280 = OpLoad %half %5279 Aligned 2
       %5281 = OpIAdd %ulong %5101 %4917
       %5282 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5281
       %5283 = OpLoad %half %5282 Aligned 2
       %5284 = OpIAdd %ulong %5101 %4919
       %5285 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5284
       %5286 = OpLoad %half %5285 Aligned 2
       %5287 = OpIAdd %ulong %5101 %4921
       %5288 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5287
       %5289 = OpLoad %half %5288 Aligned 2
       %5290 = OpIAdd %ulong %5102 %4915
       %5291 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5290
       %5292 = OpLoad %half %5291 Aligned 2
       %5293 = OpIAdd %ulong %5102 %4917
       %5294 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5293
       %5295 = OpLoad %half %5294 Aligned 2
       %5296 = OpIAdd %ulong %5102 %4919
       %5297 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5296
       %5298 = OpLoad %half %5297 Aligned 2
       %5299 = OpIAdd %ulong %5102 %4921
       %5300 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5299
       %5301 = OpLoad %half %5300 Aligned 2
       %5302 = OpIAdd %ulong %5103 %4915
       %5303 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5302
       %5304 = OpLoad %half %5303 Aligned 2
       %5305 = OpIAdd %ulong %5103 %4917
       %5306 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5305
       %5307 = OpLoad %half %5306 Aligned 2
       %5308 = OpIAdd %ulong %5103 %4919
       %5309 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5308
       %5310 = OpLoad %half %5309 Aligned 2
       %5311 = OpIAdd %ulong %5103 %4921
       %5312 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5311
       %5313 = OpLoad %half %5312 Aligned 2
       %5314 = OpIAdd %ulong %5104 %4915
       %5315 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5314
       %5316 = OpLoad %half %5315 Aligned 2
       %5317 = OpIAdd %ulong %5104 %4917
       %5318 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5317
       %5319 = OpLoad %half %5318 Aligned 2
       %5320 = OpIAdd %ulong %5104 %4919
       %5321 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5320
       %5322 = OpLoad %half %5321 Aligned 2
       %5323 = OpIAdd %ulong %5104 %4921
       %5324 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5323
       %5325 = OpLoad %half %5324 Aligned 2
       %5326 = OpIAdd %ulong %5105 %4915
       %5327 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5326
       %5328 = OpLoad %half %5327 Aligned 2
       %5329 = OpIAdd %ulong %5105 %4917
       %5330 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5329
       %5331 = OpLoad %half %5330 Aligned 2
       %5332 = OpIAdd %ulong %5105 %4919
       %5333 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5332
       %5334 = OpLoad %half %5333 Aligned 2
       %5335 = OpIAdd %ulong %5105 %4921
       %5336 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5335
       %5337 = OpLoad %half %5336 Aligned 2
       %5338 = OpIAdd %ulong %5106 %4915
       %5339 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5338
       %5340 = OpLoad %half %5339 Aligned 2
       %5341 = OpIAdd %ulong %5106 %4917
       %5342 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5341
       %5343 = OpLoad %half %5342 Aligned 2
       %5344 = OpIAdd %ulong %5106 %4919
       %5345 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5344
       %5346 = OpLoad %half %5345 Aligned 2
       %5347 = OpIAdd %ulong %5106 %4921
       %5348 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5347
       %5349 = OpLoad %half %5348 Aligned 2
       %5350 = OpIAdd %ulong %5107 %4915
       %5351 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5350
       %5352 = OpLoad %half %5351 Aligned 2
       %5353 = OpIAdd %ulong %5107 %4917
       %5354 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5353
       %5355 = OpLoad %half %5354 Aligned 2
       %5356 = OpIAdd %ulong %5107 %4919
       %5357 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5356
       %5358 = OpLoad %half %5357 Aligned 2
       %5359 = OpIAdd %ulong %5107 %4921
       %5360 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5359
       %5361 = OpLoad %half %5360 Aligned 2
       %5362 = OpIAdd %ulong %5108 %4915
       %5363 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5362
       %5364 = OpLoad %half %5363 Aligned 2
       %5365 = OpIAdd %ulong %5108 %4917
       %5366 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5365
       %5367 = OpLoad %half %5366 Aligned 2
       %5368 = OpIAdd %ulong %5108 %4919
       %5369 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5368
       %5370 = OpLoad %half %5369 Aligned 2
       %5371 = OpIAdd %ulong %5108 %4921
       %5372 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5371
       %5373 = OpLoad %half %5372 Aligned 2
       %5374 = OpIAdd %ulong %5109 %4915
       %5375 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5374
       %5376 = OpLoad %half %5375 Aligned 2
       %5377 = OpIAdd %ulong %5109 %4917
       %5378 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5377
       %5379 = OpLoad %half %5378 Aligned 2
       %5380 = OpIAdd %ulong %5109 %4919
       %5381 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5380
       %5382 = OpLoad %half %5381 Aligned 2
       %5383 = OpIAdd %ulong %5109 %4921
       %5384 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5383
       %5385 = OpLoad %half %5384 Aligned 2
       %5386 = OpIAdd %ulong %5110 %4915
       %5387 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5386
       %5388 = OpLoad %half %5387 Aligned 2
       %5389 = OpIAdd %ulong %5110 %4917
       %5390 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5389
       %5391 = OpLoad %half %5390 Aligned 2
       %5392 = OpIAdd %ulong %5110 %4919
       %5393 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5392
       %5394 = OpLoad %half %5393 Aligned 2
       %5395 = OpIAdd %ulong %5110 %4921
       %5396 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5395
       %5397 = OpLoad %half %5396 Aligned 2
       %5398 = OpIAdd %ulong %5111 %4915
       %5399 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5398
       %5400 = OpLoad %half %5399 Aligned 2
       %5401 = OpIAdd %ulong %5111 %4917
       %5402 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5401
       %5403 = OpLoad %half %5402 Aligned 2
       %5404 = OpIAdd %ulong %5111 %4919
       %5405 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5404
       %5406 = OpLoad %half %5405 Aligned 2
       %5407 = OpIAdd %ulong %5111 %4921
       %5408 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5407
       %5409 = OpLoad %half %5408 Aligned 2
       %5410 = OpIAdd %ulong %5112 %4915
       %5411 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5410
       %5412 = OpLoad %half %5411 Aligned 2
       %5413 = OpIAdd %ulong %5112 %4917
       %5414 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5413
       %5415 = OpLoad %half %5414 Aligned 2
       %5416 = OpIAdd %ulong %5112 %4919
       %5417 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5416
       %5418 = OpLoad %half %5417 Aligned 2
       %5419 = OpIAdd %ulong %5112 %4921
       %5420 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5419
       %5421 = OpLoad %half %5420 Aligned 2
       %5422 = OpIAdd %ulong %5113 %4915
       %5423 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5422
       %5424 = OpLoad %half %5423 Aligned 2
       %5425 = OpIAdd %ulong %5113 %4917
       %5426 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5425
       %5427 = OpLoad %half %5426 Aligned 2
       %5428 = OpIAdd %ulong %5113 %4919
       %5429 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5428
       %5430 = OpLoad %half %5429 Aligned 2
       %5431 = OpIAdd %ulong %5113 %4921
       %5432 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5431
       %5433 = OpLoad %half %5432 Aligned 2
       %5434 = OpIAdd %ulong %5114 %4915
       %5435 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5434
       %5436 = OpLoad %half %5435 Aligned 2
       %5437 = OpIAdd %ulong %5114 %4917
       %5438 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5437
       %5439 = OpLoad %half %5438 Aligned 2
       %5440 = OpIAdd %ulong %5114 %4919
       %5441 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5440
       %5442 = OpLoad %half %5441 Aligned 2
       %5443 = OpIAdd %ulong %5114 %4921
       %5444 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5443
       %5445 = OpLoad %half %5444 Aligned 2
       %5446 = OpIAdd %ulong %5115 %4915
       %5447 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5446
       %5448 = OpLoad %half %5447 Aligned 2
       %5449 = OpIAdd %ulong %5115 %4917
       %5450 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5449
       %5451 = OpLoad %half %5450 Aligned 2
       %5452 = OpIAdd %ulong %5115 %4919
       %5453 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5452
       %5454 = OpLoad %half %5453 Aligned 2
       %5455 = OpIAdd %ulong %5115 %4921
       %5456 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5455
       %5457 = OpLoad %half %5456 Aligned 2
       %5458 = OpIAdd %ulong %5116 %4915
       %5459 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5458
       %5460 = OpLoad %half %5459 Aligned 2
       %5461 = OpIAdd %ulong %5116 %4917
       %5462 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5461
       %5463 = OpLoad %half %5462 Aligned 2
       %5464 = OpIAdd %ulong %5116 %4919
       %5465 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5464
       %5466 = OpLoad %half %5465 Aligned 2
       %5467 = OpIAdd %ulong %5116 %4921
       %5468 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5467
       %5469 = OpLoad %half %5468 Aligned 2
       %5470 = OpIAdd %ulong %5117 %4915
       %5471 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5470
       %5472 = OpLoad %half %5471 Aligned 2
       %5473 = OpIAdd %ulong %5117 %4917
       %5474 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5473
       %5475 = OpLoad %half %5474 Aligned 2
       %5476 = OpIAdd %ulong %5117 %4919
       %5477 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5476
       %5478 = OpLoad %half %5477 Aligned 2
       %5479 = OpIAdd %ulong %5117 %4921
       %5480 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5479
       %5481 = OpLoad %half %5480 Aligned 2
       %5482 = OpIAdd %ulong %5118 %4915
       %5483 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5482
       %5484 = OpLoad %half %5483 Aligned 2
       %5485 = OpIAdd %ulong %5118 %4917
       %5486 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5485
       %5487 = OpLoad %half %5486 Aligned 2
       %5488 = OpIAdd %ulong %5118 %4919
       %5489 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5488
       %5490 = OpLoad %half %5489 Aligned 2
       %5491 = OpIAdd %ulong %5118 %4921
       %5492 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5491
       %5493 = OpLoad %half %5492 Aligned 2
       %5494 = OpIAdd %ulong %5119 %4915
       %5495 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5494
       %5496 = OpLoad %half %5495 Aligned 2
       %5497 = OpIAdd %ulong %5119 %4917
       %5498 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5497
       %5499 = OpLoad %half %5498 Aligned 2
       %5500 = OpIAdd %ulong %5119 %4919
       %5501 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5500
       %5502 = OpLoad %half %5501 Aligned 2
       %5503 = OpIAdd %ulong %5119 %4921
       %5504 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5503
       %5505 = OpLoad %half %5504 Aligned 2
       %5506 = OpIAdd %ulong %5120 %4915
       %5507 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5506
       %5508 = OpLoad %half %5507 Aligned 2
       %5509 = OpIAdd %ulong %5120 %4917
       %5510 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5509
       %5511 = OpLoad %half %5510 Aligned 2
       %5512 = OpIAdd %ulong %5120 %4919
       %5513 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5512
       %5514 = OpLoad %half %5513 Aligned 2
       %5515 = OpIAdd %ulong %5120 %4921
       %5516 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5515
       %5517 = OpLoad %half %5516 Aligned 2
       %5518 = OpIAdd %ulong %5121 %4915
       %5519 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5518
       %5520 = OpLoad %half %5519 Aligned 2
       %5521 = OpIAdd %ulong %5121 %4917
       %5522 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5521
       %5523 = OpLoad %half %5522 Aligned 2
       %5524 = OpIAdd %ulong %5121 %4919
       %5525 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5524
       %5526 = OpLoad %half %5525 Aligned 2
       %5527 = OpIAdd %ulong %5121 %4921
       %5528 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5527
       %5529 = OpLoad %half %5528 Aligned 2
       %5530 = OpIAdd %ulong %5122 %4915
       %5531 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5530
       %5532 = OpLoad %half %5531 Aligned 2
       %5533 = OpIAdd %ulong %5122 %4917
       %5534 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5533
       %5535 = OpLoad %half %5534 Aligned 2
       %5536 = OpIAdd %ulong %5122 %4919
       %5537 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5536
       %5538 = OpLoad %half %5537 Aligned 2
       %5539 = OpIAdd %ulong %5122 %4921
       %5540 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5539
       %5541 = OpLoad %half %5540 Aligned 2
       %5542 = OpIAdd %ulong %5123 %4915
       %5543 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5542
       %5544 = OpLoad %half %5543 Aligned 2
       %5545 = OpIAdd %ulong %5123 %4917
       %5546 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5545
       %5547 = OpLoad %half %5546 Aligned 2
       %5548 = OpIAdd %ulong %5123 %4919
       %5549 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5548
       %5550 = OpLoad %half %5549 Aligned 2
       %5551 = OpIAdd %ulong %5123 %4921
       %5552 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5551
       %5553 = OpLoad %half %5552 Aligned 2
       %5554 = OpIAdd %ulong %5124 %4915
       %5555 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5554
       %5556 = OpLoad %half %5555 Aligned 2
       %5557 = OpIAdd %ulong %5124 %4917
       %5558 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5557
       %5559 = OpLoad %half %5558 Aligned 2
       %5560 = OpIAdd %ulong %5124 %4919
       %5561 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5560
       %5562 = OpLoad %half %5561 Aligned 2
       %5563 = OpIAdd %ulong %5124 %4921
       %5564 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5563
       %5565 = OpLoad %half %5564 Aligned 2
       %5566 = OpIAdd %ulong %5125 %4915
       %5567 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5566
       %5568 = OpLoad %half %5567 Aligned 2
       %5569 = OpIAdd %ulong %5125 %4917
       %5570 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5569
       %5571 = OpLoad %half %5570 Aligned 2
       %5572 = OpIAdd %ulong %5125 %4919
       %5573 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5572
       %5574 = OpLoad %half %5573 Aligned 2
       %5575 = OpIAdd %ulong %5125 %4921
       %5576 = OpPtrAccessChain %_ptr_Workgroup_half %5194 %5575
       %5577 = OpLoad %half %5576 Aligned 2
       %5578 = OpCompositeInsert %v8half %5196 %106 0
       %5579 = OpCompositeInsert %v8half %5208 %5578 1
       %5580 = OpCompositeInsert %v8half %5220 %5579 2
       %5581 = OpCompositeInsert %v8half %5232 %5580 3
       %5582 = OpCompositeInsert %v8half %5244 %5581 4
       %5583 = OpCompositeInsert %v8half %5256 %5582 5
       %5584 = OpCompositeInsert %v8half %5268 %5583 6
       %5585 = OpCompositeInsert %v8half %5280 %5584 7
       %5586 = OpCompositeInsert %v8half %5199 %106 0
       %5587 = OpCompositeInsert %v8half %5211 %5586 1
       %5588 = OpCompositeInsert %v8half %5223 %5587 2
       %5589 = OpCompositeInsert %v8half %5235 %5588 3
       %5590 = OpCompositeInsert %v8half %5247 %5589 4
       %5591 = OpCompositeInsert %v8half %5259 %5590 5
       %5592 = OpCompositeInsert %v8half %5271 %5591 6
       %5593 = OpCompositeInsert %v8half %5283 %5592 7
       %5594 = OpCompositeInsert %v8half %5202 %106 0
       %5595 = OpCompositeInsert %v8half %5214 %5594 1
       %5596 = OpCompositeInsert %v8half %5226 %5595 2
       %5597 = OpCompositeInsert %v8half %5238 %5596 3
       %5598 = OpCompositeInsert %v8half %5250 %5597 4
       %5599 = OpCompositeInsert %v8half %5262 %5598 5
       %5600 = OpCompositeInsert %v8half %5274 %5599 6
       %5601 = OpCompositeInsert %v8half %5286 %5600 7
       %5602 = OpCompositeInsert %v8half %5205 %106 0
       %5603 = OpCompositeInsert %v8half %5217 %5602 1
       %5604 = OpCompositeInsert %v8half %5229 %5603 2
       %5605 = OpCompositeInsert %v8half %5241 %5604 3
       %5606 = OpCompositeInsert %v8half %5253 %5605 4
       %5607 = OpCompositeInsert %v8half %5265 %5606 5
       %5608 = OpCompositeInsert %v8half %5277 %5607 6
       %5609 = OpCompositeInsert %v8half %5289 %5608 7
       %5610 = OpCompositeInsert %v8half %5292 %106 0
       %5611 = OpCompositeInsert %v8half %5304 %5610 1
       %5612 = OpCompositeInsert %v8half %5316 %5611 2
       %5613 = OpCompositeInsert %v8half %5328 %5612 3
       %5614 = OpCompositeInsert %v8half %5340 %5613 4
       %5615 = OpCompositeInsert %v8half %5352 %5614 5
       %5616 = OpCompositeInsert %v8half %5364 %5615 6
       %5617 = OpCompositeInsert %v8half %5376 %5616 7
       %5618 = OpCompositeInsert %v8half %5295 %106 0
       %5619 = OpCompositeInsert %v8half %5307 %5618 1
       %5620 = OpCompositeInsert %v8half %5319 %5619 2
       %5621 = OpCompositeInsert %v8half %5331 %5620 3
       %5622 = OpCompositeInsert %v8half %5343 %5621 4
       %5623 = OpCompositeInsert %v8half %5355 %5622 5
       %5624 = OpCompositeInsert %v8half %5367 %5623 6
       %5625 = OpCompositeInsert %v8half %5379 %5624 7
       %5626 = OpCompositeInsert %v8half %5298 %106 0
       %5627 = OpCompositeInsert %v8half %5310 %5626 1
       %5628 = OpCompositeInsert %v8half %5322 %5627 2
       %5629 = OpCompositeInsert %v8half %5334 %5628 3
       %5630 = OpCompositeInsert %v8half %5346 %5629 4
       %5631 = OpCompositeInsert %v8half %5358 %5630 5
       %5632 = OpCompositeInsert %v8half %5370 %5631 6
       %5633 = OpCompositeInsert %v8half %5382 %5632 7
       %5634 = OpCompositeInsert %v8half %5301 %106 0
       %5635 = OpCompositeInsert %v8half %5313 %5634 1
       %5636 = OpCompositeInsert %v8half %5325 %5635 2
       %5637 = OpCompositeInsert %v8half %5337 %5636 3
       %5638 = OpCompositeInsert %v8half %5349 %5637 4
       %5639 = OpCompositeInsert %v8half %5361 %5638 5
       %5640 = OpCompositeInsert %v8half %5373 %5639 6
       %5641 = OpCompositeInsert %v8half %5385 %5640 7
       %5642 = OpCompositeInsert %v8half %5388 %106 0
       %5643 = OpCompositeInsert %v8half %5400 %5642 1
       %5644 = OpCompositeInsert %v8half %5412 %5643 2
       %5645 = OpCompositeInsert %v8half %5424 %5644 3
       %5646 = OpCompositeInsert %v8half %5436 %5645 4
       %5647 = OpCompositeInsert %v8half %5448 %5646 5
       %5648 = OpCompositeInsert %v8half %5460 %5647 6
       %5649 = OpCompositeInsert %v8half %5472 %5648 7
       %5650 = OpCompositeInsert %v8half %5391 %106 0
       %5651 = OpCompositeInsert %v8half %5403 %5650 1
       %5652 = OpCompositeInsert %v8half %5415 %5651 2
       %5653 = OpCompositeInsert %v8half %5427 %5652 3
       %5654 = OpCompositeInsert %v8half %5439 %5653 4
       %5655 = OpCompositeInsert %v8half %5451 %5654 5
       %5656 = OpCompositeInsert %v8half %5463 %5655 6
       %5657 = OpCompositeInsert %v8half %5475 %5656 7
       %5658 = OpCompositeInsert %v8half %5394 %106 0
       %5659 = OpCompositeInsert %v8half %5406 %5658 1
       %5660 = OpCompositeInsert %v8half %5418 %5659 2
       %5661 = OpCompositeInsert %v8half %5430 %5660 3
       %5662 = OpCompositeInsert %v8half %5442 %5661 4
       %5663 = OpCompositeInsert %v8half %5454 %5662 5
       %5664 = OpCompositeInsert %v8half %5466 %5663 6
       %5665 = OpCompositeInsert %v8half %5478 %5664 7
       %5666 = OpCompositeInsert %v8half %5397 %106 0
       %5667 = OpCompositeInsert %v8half %5409 %5666 1
       %5668 = OpCompositeInsert %v8half %5421 %5667 2
       %5669 = OpCompositeInsert %v8half %5433 %5668 3
       %5670 = OpCompositeInsert %v8half %5445 %5669 4
       %5671 = OpCompositeInsert %v8half %5457 %5670 5
       %5672 = OpCompositeInsert %v8half %5469 %5671 6
       %5673 = OpCompositeInsert %v8half %5481 %5672 7
       %5674 = OpCompositeInsert %v8half %5484 %106 0
       %5675 = OpCompositeInsert %v8half %5496 %5674 1
       %5676 = OpCompositeInsert %v8half %5508 %5675 2
       %5677 = OpCompositeInsert %v8half %5520 %5676 3
       %5678 = OpCompositeInsert %v8half %5532 %5677 4
       %5679 = OpCompositeInsert %v8half %5544 %5678 5
       %5680 = OpCompositeInsert %v8half %5556 %5679 6
       %5681 = OpCompositeInsert %v8half %5568 %5680 7
       %5682 = OpCompositeInsert %v8half %5487 %106 0
       %5683 = OpCompositeInsert %v8half %5499 %5682 1
       %5684 = OpCompositeInsert %v8half %5511 %5683 2
       %5685 = OpCompositeInsert %v8half %5523 %5684 3
       %5686 = OpCompositeInsert %v8half %5535 %5685 4
       %5687 = OpCompositeInsert %v8half %5547 %5686 5
       %5688 = OpCompositeInsert %v8half %5559 %5687 6
       %5689 = OpCompositeInsert %v8half %5571 %5688 7
       %5690 = OpCompositeInsert %v8half %5490 %106 0
       %5691 = OpCompositeInsert %v8half %5502 %5690 1
       %5692 = OpCompositeInsert %v8half %5514 %5691 2
       %5693 = OpCompositeInsert %v8half %5526 %5692 3
       %5694 = OpCompositeInsert %v8half %5538 %5693 4
       %5695 = OpCompositeInsert %v8half %5550 %5694 5
       %5696 = OpCompositeInsert %v8half %5562 %5695 6
       %5697 = OpCompositeInsert %v8half %5574 %5696 7
       %5698 = OpCompositeInsert %v8half %5493 %106 0
       %5699 = OpCompositeInsert %v8half %5505 %5698 1
       %5700 = OpCompositeInsert %v8half %5517 %5699 2
       %5701 = OpCompositeInsert %v8half %5529 %5700 3
       %5702 = OpCompositeInsert %v8half %5541 %5701 4
       %5703 = OpCompositeInsert %v8half %5553 %5702 5
       %5704 = OpCompositeInsert %v8half %5565 %5703 6
       %5705 = OpCompositeInsert %v8half %5577 %5704 7
       %5706 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5132 %5585 %5087
       %5707 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5132 %5593 %5085
       %5708 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5132 %5601 %5083
       %5709 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5132 %5609 %5081
       %5710 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5148 %5585 %5079
       %5711 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5148 %5593 %5077
       %5712 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5148 %5601 %5075
       %5713 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5148 %5609 %5073
       %5714 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5164 %5585 %5071
       %5715 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5164 %5593 %5069
       %5716 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5164 %5601 %5067
       %5717 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5164 %5609 %5065
       %5718 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5180 %5585 %5063
       %5719 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5180 %5593 %5061
       %5720 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5180 %5601 %5059
       %5721 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5180 %5609 %5057
       %5722 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5136 %5617 %5706
       %5723 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5136 %5625 %5707
       %5724 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5136 %5633 %5708
       %5725 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5136 %5641 %5709
       %5726 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5152 %5617 %5710
       %5727 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5152 %5625 %5711
       %5728 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5152 %5633 %5712
       %5729 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5152 %5641 %5713
       %5730 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5168 %5617 %5714
       %5731 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5168 %5625 %5715
       %5732 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5168 %5633 %5716
       %5733 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5168 %5641 %5717
       %5734 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5184 %5617 %5718
       %5735 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5184 %5625 %5719
       %5736 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5184 %5633 %5720
       %5737 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5184 %5641 %5721
       %5738 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5140 %5649 %5722
       %5739 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5140 %5657 %5723
       %5740 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5140 %5665 %5724
       %5741 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5140 %5673 %5725
       %5742 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5156 %5649 %5726
       %5743 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5156 %5657 %5727
       %5744 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5156 %5665 %5728
       %5745 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5156 %5673 %5729
       %5746 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5172 %5649 %5730
       %5747 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5172 %5657 %5731
       %5748 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5172 %5665 %5732
       %5749 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5172 %5673 %5733
       %5750 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5188 %5649 %5734
       %5751 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5188 %5657 %5735
       %5752 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5188 %5665 %5736
       %5753 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5188 %5673 %5737
       %5754 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5144 %5681 %5738
       %5755 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5144 %5689 %5739
       %5756 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5144 %5697 %5740
       %5757 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5144 %5705 %5741
       %5758 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5160 %5681 %5742
       %5759 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5160 %5689 %5743
       %5760 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5160 %5697 %5744
       %5761 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5160 %5705 %5745
       %5762 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5176 %5681 %5746
       %5763 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5176 %5689 %5747
       %5764 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5176 %5697 %5748
       %5765 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5176 %5705 %5749
       %5766 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5192 %5681 %5750
       %5767 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5192 %5689 %5751
       %5768 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5192 %5697 %5752
       %5769 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %5192 %5705 %5753
       %5770 = OpCompositeExtract %float %5754 0
       %5771 = OpCompositeExtract %float %5754 1
       %5772 = OpCompositeExtract %float %5754 2
       %5773 = OpCompositeExtract %float %5754 3
       %5774 = OpCompositeExtract %float %5754 4
       %5775 = OpCompositeExtract %float %5754 5
       %5776 = OpCompositeExtract %float %5754 6
       %5777 = OpCompositeExtract %float %5754 7
       %5778 = OpCompositeExtract %float %5755 0
       %5779 = OpCompositeExtract %float %5755 1
       %5780 = OpCompositeExtract %float %5755 2
       %5781 = OpCompositeExtract %float %5755 3
       %5782 = OpCompositeExtract %float %5755 4
       %5783 = OpCompositeExtract %float %5755 5
       %5784 = OpCompositeExtract %float %5755 6
       %5785 = OpCompositeExtract %float %5755 7
       %5786 = OpCompositeExtract %float %5756 0
       %5787 = OpCompositeExtract %float %5756 1
       %5788 = OpCompositeExtract %float %5756 2
       %5789 = OpCompositeExtract %float %5756 3
       %5790 = OpCompositeExtract %float %5756 4
       %5791 = OpCompositeExtract %float %5756 5
       %5792 = OpCompositeExtract %float %5756 6
       %5793 = OpCompositeExtract %float %5756 7
       %5794 = OpCompositeExtract %float %5757 0
       %5795 = OpCompositeExtract %float %5757 1
       %5796 = OpCompositeExtract %float %5757 2
       %5797 = OpCompositeExtract %float %5757 3
       %5798 = OpCompositeExtract %float %5757 4
       %5799 = OpCompositeExtract %float %5757 5
       %5800 = OpCompositeExtract %float %5757 6
       %5801 = OpCompositeExtract %float %5757 7
       %5802 = OpCompositeExtract %float %5758 0
       %5803 = OpCompositeExtract %float %5758 1
       %5804 = OpCompositeExtract %float %5758 2
       %5805 = OpCompositeExtract %float %5758 3
       %5806 = OpCompositeExtract %float %5758 4
       %5807 = OpCompositeExtract %float %5758 5
       %5808 = OpCompositeExtract %float %5758 6
       %5809 = OpCompositeExtract %float %5758 7
       %5810 = OpCompositeExtract %float %5759 0
       %5811 = OpCompositeExtract %float %5759 1
       %5812 = OpCompositeExtract %float %5759 2
       %5813 = OpCompositeExtract %float %5759 3
       %5814 = OpCompositeExtract %float %5759 4
       %5815 = OpCompositeExtract %float %5759 5
       %5816 = OpCompositeExtract %float %5759 6
       %5817 = OpCompositeExtract %float %5759 7
       %5818 = OpCompositeExtract %float %5760 0
       %5819 = OpCompositeExtract %float %5760 1
       %5820 = OpCompositeExtract %float %5760 2
       %5821 = OpCompositeExtract %float %5760 3
       %5822 = OpCompositeExtract %float %5760 4
       %5823 = OpCompositeExtract %float %5760 5
       %5824 = OpCompositeExtract %float %5760 6
       %5825 = OpCompositeExtract %float %5760 7
       %5826 = OpCompositeExtract %float %5761 0
       %5827 = OpCompositeExtract %float %5761 1
       %5828 = OpCompositeExtract %float %5761 2
       %5829 = OpCompositeExtract %float %5761 3
       %5830 = OpCompositeExtract %float %5761 4
       %5831 = OpCompositeExtract %float %5761 5
       %5832 = OpCompositeExtract %float %5761 6
       %5833 = OpCompositeExtract %float %5761 7
       %5834 = OpCompositeExtract %float %5762 0
       %5835 = OpCompositeExtract %float %5762 1
       %5836 = OpCompositeExtract %float %5762 2
       %5837 = OpCompositeExtract %float %5762 3
       %5838 = OpCompositeExtract %float %5762 4
       %5839 = OpCompositeExtract %float %5762 5
       %5840 = OpCompositeExtract %float %5762 6
       %5841 = OpCompositeExtract %float %5762 7
       %5842 = OpCompositeExtract %float %5763 0
       %5843 = OpCompositeExtract %float %5763 1
       %5844 = OpCompositeExtract %float %5763 2
       %5845 = OpCompositeExtract %float %5763 3
       %5846 = OpCompositeExtract %float %5763 4
       %5847 = OpCompositeExtract %float %5763 5
       %5848 = OpCompositeExtract %float %5763 6
       %5849 = OpCompositeExtract %float %5763 7
       %5850 = OpCompositeExtract %float %5764 0
       %5851 = OpCompositeExtract %float %5764 1
       %5852 = OpCompositeExtract %float %5764 2
       %5853 = OpCompositeExtract %float %5764 3
       %5854 = OpCompositeExtract %float %5764 4
       %5855 = OpCompositeExtract %float %5764 5
       %5856 = OpCompositeExtract %float %5764 6
       %5857 = OpCompositeExtract %float %5764 7
       %5858 = OpCompositeExtract %float %5765 0
       %5859 = OpCompositeExtract %float %5765 1
       %5860 = OpCompositeExtract %float %5765 2
       %5861 = OpCompositeExtract %float %5765 3
       %5862 = OpCompositeExtract %float %5765 4
       %5863 = OpCompositeExtract %float %5765 5
       %5864 = OpCompositeExtract %float %5765 6
       %5865 = OpCompositeExtract %float %5765 7
       %5866 = OpCompositeExtract %float %5766 0
       %5867 = OpCompositeExtract %float %5766 1
       %5868 = OpCompositeExtract %float %5766 2
       %5869 = OpCompositeExtract %float %5766 3
       %5870 = OpCompositeExtract %float %5766 4
       %5871 = OpCompositeExtract %float %5766 5
       %5872 = OpCompositeExtract %float %5766 6
       %5873 = OpCompositeExtract %float %5766 7
       %5874 = OpCompositeExtract %float %5767 0
       %5875 = OpCompositeExtract %float %5767 1
       %5876 = OpCompositeExtract %float %5767 2
       %5877 = OpCompositeExtract %float %5767 3
       %5878 = OpCompositeExtract %float %5767 4
       %5879 = OpCompositeExtract %float %5767 5
       %5880 = OpCompositeExtract %float %5767 6
       %5881 = OpCompositeExtract %float %5767 7
       %5882 = OpCompositeExtract %float %5768 0
       %5883 = OpCompositeExtract %float %5768 1
       %5884 = OpCompositeExtract %float %5768 2
       %5885 = OpCompositeExtract %float %5768 3
       %5886 = OpCompositeExtract %float %5768 4
       %5887 = OpCompositeExtract %float %5768 5
       %5888 = OpCompositeExtract %float %5768 6
       %5889 = OpCompositeExtract %float %5768 7
       %5890 = OpCompositeExtract %float %5769 0
       %5891 = OpCompositeExtract %float %5769 1
       %5892 = OpCompositeExtract %float %5769 2
       %5893 = OpCompositeExtract %float %5769 3
       %5894 = OpCompositeExtract %float %5769 4
       %5895 = OpCompositeExtract %float %5769 5
       %5896 = OpCompositeExtract %float %5769 6
       %5897 = OpCompositeExtract %float %5769 7
       %5898 = OpIMul %uint %4758 %uint_16
       %5899 = OpIAdd %uint %4896 %5898
       %5900 = OpIMul %uint %4759 %uint_8
       %5901 = OpIAdd %uint %4912 %5900
       %5902 = OpIMul %uint %5899 %uint_16
       %5903 = OpIAdd %uint %5902 %4688
       %5904 = OpUConvert %ulong %5903
       %5905 = OpIMul %uint %5901 %uint_16
       %5906 = OpIAdd %uint %5905 %4687
       %5907 = OpUConvert %ulong %5906
       %5908 = OpIMul %ulong %5904 %ulong_4096
       %5909 = OpIAdd %ulong %5908 %5907
       %5910 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5909
               OpStore %5910 %5770 Aligned 4
       %5911 = OpIAdd %uint %5906 %uint_16
       %5912 = OpUConvert %ulong %5911
       %5913 = OpIAdd %ulong %5908 %5912
       %5914 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5913
               OpStore %5914 %5778 Aligned 4
       %5915 = OpIAdd %uint %5906 %uint_32
       %5916 = OpUConvert %ulong %5915
       %5917 = OpIAdd %ulong %5908 %5916
       %5918 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5917
               OpStore %5918 %5786 Aligned 4
       %5919 = OpIAdd %uint %5906 %uint_48
       %5920 = OpUConvert %ulong %5919
       %5921 = OpIAdd %ulong %5908 %5920
       %5922 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5921
               OpStore %5922 %5794 Aligned 4
       %5923 = OpIAdd %uint %5903 %uint_1
       %5924 = OpUConvert %ulong %5923
       %5925 = OpIMul %ulong %5924 %ulong_4096
       %5926 = OpIAdd %ulong %5925 %5907
       %5927 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5926
               OpStore %5927 %5771 Aligned 4
       %5928 = OpIAdd %ulong %5925 %5912
       %5929 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5928
               OpStore %5929 %5779 Aligned 4
       %5930 = OpIAdd %ulong %5925 %5916
       %5931 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5930
               OpStore %5931 %5787 Aligned 4
       %5932 = OpIAdd %ulong %5925 %5920
       %5933 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5932
               OpStore %5933 %5795 Aligned 4
       %5934 = OpIAdd %uint %5903 %uint_2
       %5935 = OpUConvert %ulong %5934
       %5936 = OpIMul %ulong %5935 %ulong_4096
       %5937 = OpIAdd %ulong %5936 %5907
       %5938 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5937
               OpStore %5938 %5772 Aligned 4
       %5939 = OpIAdd %ulong %5936 %5912
       %5940 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5939
               OpStore %5940 %5780 Aligned 4
       %5941 = OpIAdd %ulong %5936 %5916
       %5942 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5941
               OpStore %5942 %5788 Aligned 4
       %5943 = OpIAdd %ulong %5936 %5920
       %5944 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5943
               OpStore %5944 %5796 Aligned 4
       %5945 = OpIAdd %uint %5903 %uint_3
       %5946 = OpUConvert %ulong %5945
       %5947 = OpIMul %ulong %5946 %ulong_4096
       %5948 = OpIAdd %ulong %5947 %5907
       %5949 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5948
               OpStore %5949 %5773 Aligned 4
       %5950 = OpIAdd %ulong %5947 %5912
       %5951 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5950
               OpStore %5951 %5781 Aligned 4
       %5952 = OpIAdd %ulong %5947 %5916
       %5953 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5952
               OpStore %5953 %5789 Aligned 4
       %5954 = OpIAdd %ulong %5947 %5920
       %5955 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5954
               OpStore %5955 %5797 Aligned 4
       %5956 = OpIAdd %uint %5903 %uint_4
       %5957 = OpUConvert %ulong %5956
       %5958 = OpIMul %ulong %5957 %ulong_4096
       %5959 = OpIAdd %ulong %5958 %5907
       %5960 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5959
               OpStore %5960 %5774 Aligned 4
       %5961 = OpIAdd %ulong %5958 %5912
       %5962 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5961
               OpStore %5962 %5782 Aligned 4
       %5963 = OpIAdd %ulong %5958 %5916
       %5964 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5963
               OpStore %5964 %5790 Aligned 4
       %5965 = OpIAdd %ulong %5958 %5920
       %5966 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5965
               OpStore %5966 %5798 Aligned 4
       %5967 = OpIAdd %uint %5903 %uint_5
       %5968 = OpUConvert %ulong %5967
       %5969 = OpIMul %ulong %5968 %ulong_4096
       %5970 = OpIAdd %ulong %5969 %5907
       %5971 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5970
               OpStore %5971 %5775 Aligned 4
       %5972 = OpIAdd %ulong %5969 %5912
       %5973 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5972
               OpStore %5973 %5783 Aligned 4
       %5974 = OpIAdd %ulong %5969 %5916
       %5975 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5974
               OpStore %5975 %5791 Aligned 4
       %5976 = OpIAdd %ulong %5969 %5920
       %5977 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5976
               OpStore %5977 %5799 Aligned 4
       %5978 = OpIAdd %uint %5903 %uint_6
       %5979 = OpUConvert %ulong %5978
       %5980 = OpIMul %ulong %5979 %ulong_4096
       %5981 = OpIAdd %ulong %5980 %5907
       %5982 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5981
               OpStore %5982 %5776 Aligned 4
       %5983 = OpIAdd %ulong %5980 %5912
       %5984 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5983
               OpStore %5984 %5784 Aligned 4
       %5985 = OpIAdd %ulong %5980 %5916
       %5986 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5985
               OpStore %5986 %5792 Aligned 4
       %5987 = OpIAdd %ulong %5980 %5920
       %5988 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5987
               OpStore %5988 %5800 Aligned 4
       %5989 = OpIAdd %uint %5903 %uint_7
       %5990 = OpUConvert %ulong %5989
       %5991 = OpIMul %ulong %5990 %ulong_4096
       %5992 = OpIAdd %ulong %5991 %5907
       %5993 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5992
               OpStore %5993 %5777 Aligned 4
       %5994 = OpIAdd %ulong %5991 %5912
       %5995 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5994
               OpStore %5995 %5785 Aligned 4
       %5996 = OpIAdd %ulong %5991 %5916
       %5997 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5996
               OpStore %5997 %5793 Aligned 4
       %5998 = OpIAdd %ulong %5991 %5920
       %5999 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %5998
               OpStore %5999 %5801 Aligned 4
       %6000 = OpIAdd %uint %5903 %uint_16
       %6001 = OpUConvert %ulong %6000
       %6002 = OpIMul %ulong %6001 %ulong_4096
       %6003 = OpIAdd %ulong %6002 %5907
       %6004 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6003
               OpStore %6004 %5802 Aligned 4
       %6005 = OpIAdd %ulong %6002 %5912
       %6006 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6005
               OpStore %6006 %5810 Aligned 4
       %6007 = OpIAdd %ulong %6002 %5916
       %6008 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6007
               OpStore %6008 %5818 Aligned 4
       %6009 = OpIAdd %ulong %6002 %5920
       %6010 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6009
               OpStore %6010 %5826 Aligned 4
       %6011 = OpIAdd %uint %5903 %uint_17
       %6012 = OpUConvert %ulong %6011
       %6013 = OpIMul %ulong %6012 %ulong_4096
       %6014 = OpIAdd %ulong %6013 %5907
       %6015 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6014
               OpStore %6015 %5803 Aligned 4
       %6016 = OpIAdd %ulong %6013 %5912
       %6017 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6016
               OpStore %6017 %5811 Aligned 4
       %6018 = OpIAdd %ulong %6013 %5916
       %6019 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6018
               OpStore %6019 %5819 Aligned 4
       %6020 = OpIAdd %ulong %6013 %5920
       %6021 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6020
               OpStore %6021 %5827 Aligned 4
       %6022 = OpIAdd %uint %5903 %uint_18
       %6023 = OpUConvert %ulong %6022
       %6024 = OpIMul %ulong %6023 %ulong_4096
       %6025 = OpIAdd %ulong %6024 %5907
       %6026 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6025
               OpStore %6026 %5804 Aligned 4
       %6027 = OpIAdd %ulong %6024 %5912
       %6028 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6027
               OpStore %6028 %5812 Aligned 4
       %6029 = OpIAdd %ulong %6024 %5916
       %6030 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6029
               OpStore %6030 %5820 Aligned 4
       %6031 = OpIAdd %ulong %6024 %5920
       %6032 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6031
               OpStore %6032 %5828 Aligned 4
       %6033 = OpIAdd %uint %5903 %uint_19
       %6034 = OpUConvert %ulong %6033
       %6035 = OpIMul %ulong %6034 %ulong_4096
       %6036 = OpIAdd %ulong %6035 %5907
       %6037 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6036
               OpStore %6037 %5805 Aligned 4
       %6038 = OpIAdd %ulong %6035 %5912
       %6039 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6038
               OpStore %6039 %5813 Aligned 4
       %6040 = OpIAdd %ulong %6035 %5916
       %6041 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6040
               OpStore %6041 %5821 Aligned 4
       %6042 = OpIAdd %ulong %6035 %5920
       %6043 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6042
               OpStore %6043 %5829 Aligned 4
       %6044 = OpIAdd %uint %5903 %uint_20
       %6045 = OpUConvert %ulong %6044
       %6046 = OpIMul %ulong %6045 %ulong_4096
       %6047 = OpIAdd %ulong %6046 %5907
       %6048 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6047
               OpStore %6048 %5806 Aligned 4
       %6049 = OpIAdd %ulong %6046 %5912
       %6050 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6049
               OpStore %6050 %5814 Aligned 4
       %6051 = OpIAdd %ulong %6046 %5916
       %6052 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6051
               OpStore %6052 %5822 Aligned 4
       %6053 = OpIAdd %ulong %6046 %5920
       %6054 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6053
               OpStore %6054 %5830 Aligned 4
       %6055 = OpIAdd %uint %5903 %uint_21
       %6056 = OpUConvert %ulong %6055
       %6057 = OpIMul %ulong %6056 %ulong_4096
       %6058 = OpIAdd %ulong %6057 %5907
       %6059 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6058
               OpStore %6059 %5807 Aligned 4
       %6060 = OpIAdd %ulong %6057 %5912
       %6061 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6060
               OpStore %6061 %5815 Aligned 4
       %6062 = OpIAdd %ulong %6057 %5916
       %6063 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6062
               OpStore %6063 %5823 Aligned 4
       %6064 = OpIAdd %ulong %6057 %5920
       %6065 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6064
               OpStore %6065 %5831 Aligned 4
       %6066 = OpIAdd %uint %5903 %uint_22
       %6067 = OpUConvert %ulong %6066
       %6068 = OpIMul %ulong %6067 %ulong_4096
       %6069 = OpIAdd %ulong %6068 %5907
       %6070 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6069
               OpStore %6070 %5808 Aligned 4
       %6071 = OpIAdd %ulong %6068 %5912
       %6072 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6071
               OpStore %6072 %5816 Aligned 4
       %6073 = OpIAdd %ulong %6068 %5916
       %6074 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6073
               OpStore %6074 %5824 Aligned 4
       %6075 = OpIAdd %ulong %6068 %5920
       %6076 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6075
               OpStore %6076 %5832 Aligned 4
       %6077 = OpIAdd %uint %5903 %uint_23
       %6078 = OpUConvert %ulong %6077
       %6079 = OpIMul %ulong %6078 %ulong_4096
       %6080 = OpIAdd %ulong %6079 %5907
       %6081 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6080
               OpStore %6081 %5809 Aligned 4
       %6082 = OpIAdd %ulong %6079 %5912
       %6083 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6082
               OpStore %6083 %5817 Aligned 4
       %6084 = OpIAdd %ulong %6079 %5916
       %6085 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6084
               OpStore %6085 %5825 Aligned 4
       %6086 = OpIAdd %ulong %6079 %5920
       %6087 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6086
               OpStore %6087 %5833 Aligned 4
       %6088 = OpIAdd %uint %5903 %uint_32
       %6089 = OpUConvert %ulong %6088
       %6090 = OpIMul %ulong %6089 %ulong_4096
       %6091 = OpIAdd %ulong %6090 %5907
       %6092 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6091
               OpStore %6092 %5834 Aligned 4
       %6093 = OpIAdd %ulong %6090 %5912
       %6094 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6093
               OpStore %6094 %5842 Aligned 4
       %6095 = OpIAdd %ulong %6090 %5916
       %6096 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6095
               OpStore %6096 %5850 Aligned 4
       %6097 = OpIAdd %ulong %6090 %5920
       %6098 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6097
               OpStore %6098 %5858 Aligned 4
       %6099 = OpIAdd %uint %5903 %uint_33
       %6100 = OpUConvert %ulong %6099
       %6101 = OpIMul %ulong %6100 %ulong_4096
       %6102 = OpIAdd %ulong %6101 %5907
       %6103 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6102
               OpStore %6103 %5835 Aligned 4
       %6104 = OpIAdd %ulong %6101 %5912
       %6105 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6104
               OpStore %6105 %5843 Aligned 4
       %6106 = OpIAdd %ulong %6101 %5916
       %6107 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6106
               OpStore %6107 %5851 Aligned 4
       %6108 = OpIAdd %ulong %6101 %5920
       %6109 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6108
               OpStore %6109 %5859 Aligned 4
       %6110 = OpIAdd %uint %5903 %uint_34
       %6111 = OpUConvert %ulong %6110
       %6112 = OpIMul %ulong %6111 %ulong_4096
       %6113 = OpIAdd %ulong %6112 %5907
       %6114 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6113
               OpStore %6114 %5836 Aligned 4
       %6115 = OpIAdd %ulong %6112 %5912
       %6116 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6115
               OpStore %6116 %5844 Aligned 4
       %6117 = OpIAdd %ulong %6112 %5916
       %6118 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6117
               OpStore %6118 %5852 Aligned 4
       %6119 = OpIAdd %ulong %6112 %5920
       %6120 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6119
               OpStore %6120 %5860 Aligned 4
       %6121 = OpIAdd %uint %5903 %uint_35
       %6122 = OpUConvert %ulong %6121
       %6123 = OpIMul %ulong %6122 %ulong_4096
       %6124 = OpIAdd %ulong %6123 %5907
       %6125 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6124
               OpStore %6125 %5837 Aligned 4
       %6126 = OpIAdd %ulong %6123 %5912
       %6127 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6126
               OpStore %6127 %5845 Aligned 4
       %6128 = OpIAdd %ulong %6123 %5916
       %6129 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6128
               OpStore %6129 %5853 Aligned 4
       %6130 = OpIAdd %ulong %6123 %5920
       %6131 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6130
               OpStore %6131 %5861 Aligned 4
       %6132 = OpIAdd %uint %5903 %uint_36
       %6133 = OpUConvert %ulong %6132
       %6134 = OpIMul %ulong %6133 %ulong_4096
       %6135 = OpIAdd %ulong %6134 %5907
       %6136 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6135
               OpStore %6136 %5838 Aligned 4
       %6137 = OpIAdd %ulong %6134 %5912
       %6138 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6137
               OpStore %6138 %5846 Aligned 4
       %6139 = OpIAdd %ulong %6134 %5916
       %6140 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6139
               OpStore %6140 %5854 Aligned 4
       %6141 = OpIAdd %ulong %6134 %5920
       %6142 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6141
               OpStore %6142 %5862 Aligned 4
       %6143 = OpIAdd %uint %5903 %uint_37
       %6144 = OpUConvert %ulong %6143
       %6145 = OpIMul %ulong %6144 %ulong_4096
       %6146 = OpIAdd %ulong %6145 %5907
       %6147 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6146
               OpStore %6147 %5839 Aligned 4
       %6148 = OpIAdd %ulong %6145 %5912
       %6149 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6148
               OpStore %6149 %5847 Aligned 4
       %6150 = OpIAdd %ulong %6145 %5916
       %6151 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6150
               OpStore %6151 %5855 Aligned 4
       %6152 = OpIAdd %ulong %6145 %5920
       %6153 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6152
               OpStore %6153 %5863 Aligned 4
       %6154 = OpIAdd %uint %5903 %uint_38
       %6155 = OpUConvert %ulong %6154
       %6156 = OpIMul %ulong %6155 %ulong_4096
       %6157 = OpIAdd %ulong %6156 %5907
       %6158 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6157
               OpStore %6158 %5840 Aligned 4
       %6159 = OpIAdd %ulong %6156 %5912
       %6160 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6159
               OpStore %6160 %5848 Aligned 4
       %6161 = OpIAdd %ulong %6156 %5916
       %6162 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6161
               OpStore %6162 %5856 Aligned 4
       %6163 = OpIAdd %ulong %6156 %5920
       %6164 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6163
               OpStore %6164 %5864 Aligned 4
       %6165 = OpIAdd %uint %5903 %uint_39
       %6166 = OpUConvert %ulong %6165
       %6167 = OpIMul %ulong %6166 %ulong_4096
       %6168 = OpIAdd %ulong %6167 %5907
       %6169 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6168
               OpStore %6169 %5841 Aligned 4
       %6170 = OpIAdd %ulong %6167 %5912
       %6171 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6170
               OpStore %6171 %5849 Aligned 4
       %6172 = OpIAdd %ulong %6167 %5916
       %6173 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6172
               OpStore %6173 %5857 Aligned 4
       %6174 = OpIAdd %ulong %6167 %5920
       %6175 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6174
               OpStore %6175 %5865 Aligned 4
       %6176 = OpIAdd %uint %5903 %uint_48
       %6177 = OpUConvert %ulong %6176
       %6178 = OpIMul %ulong %6177 %ulong_4096
       %6179 = OpIAdd %ulong %6178 %5907
       %6180 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6179
               OpStore %6180 %5866 Aligned 4
       %6181 = OpIAdd %ulong %6178 %5912
       %6182 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6181
               OpStore %6182 %5874 Aligned 4
       %6183 = OpIAdd %ulong %6178 %5916
       %6184 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6183
               OpStore %6184 %5882 Aligned 4
       %6185 = OpIAdd %ulong %6178 %5920
       %6186 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6185
               OpStore %6186 %5890 Aligned 4
       %6187 = OpIAdd %uint %5903 %uint_49
       %6188 = OpUConvert %ulong %6187
       %6189 = OpIMul %ulong %6188 %ulong_4096
       %6190 = OpIAdd %ulong %6189 %5907
       %6191 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6190
               OpStore %6191 %5867 Aligned 4
       %6192 = OpIAdd %ulong %6189 %5912
       %6193 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6192
               OpStore %6193 %5875 Aligned 4
       %6194 = OpIAdd %ulong %6189 %5916
       %6195 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6194
               OpStore %6195 %5883 Aligned 4
       %6196 = OpIAdd %ulong %6189 %5920
       %6197 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6196
               OpStore %6197 %5891 Aligned 4
       %6198 = OpIAdd %uint %5903 %uint_50
       %6199 = OpUConvert %ulong %6198
       %6200 = OpIMul %ulong %6199 %ulong_4096
       %6201 = OpIAdd %ulong %6200 %5907
       %6202 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6201
               OpStore %6202 %5868 Aligned 4
       %6203 = OpIAdd %ulong %6200 %5912
       %6204 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6203
               OpStore %6204 %5876 Aligned 4
       %6205 = OpIAdd %ulong %6200 %5916
       %6206 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6205
               OpStore %6206 %5884 Aligned 4
       %6207 = OpIAdd %ulong %6200 %5920
       %6208 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6207
               OpStore %6208 %5892 Aligned 4
       %6209 = OpIAdd %uint %5903 %uint_51
       %6210 = OpUConvert %ulong %6209
       %6211 = OpIMul %ulong %6210 %ulong_4096
       %6212 = OpIAdd %ulong %6211 %5907
       %6213 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6212
               OpStore %6213 %5869 Aligned 4
       %6214 = OpIAdd %ulong %6211 %5912
       %6215 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6214
               OpStore %6215 %5877 Aligned 4
       %6216 = OpIAdd %ulong %6211 %5916
       %6217 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6216
               OpStore %6217 %5885 Aligned 4
       %6218 = OpIAdd %ulong %6211 %5920
       %6219 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6218
               OpStore %6219 %5893 Aligned 4
       %6220 = OpIAdd %uint %5903 %uint_52
       %6221 = OpUConvert %ulong %6220
       %6222 = OpIMul %ulong %6221 %ulong_4096
       %6223 = OpIAdd %ulong %6222 %5907
       %6224 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6223
               OpStore %6224 %5870 Aligned 4
       %6225 = OpIAdd %ulong %6222 %5912
       %6226 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6225
               OpStore %6226 %5878 Aligned 4
       %6227 = OpIAdd %ulong %6222 %5916
       %6228 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6227
               OpStore %6228 %5886 Aligned 4
       %6229 = OpIAdd %ulong %6222 %5920
       %6230 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6229
               OpStore %6230 %5894 Aligned 4
       %6231 = OpIAdd %uint %5903 %uint_53
       %6232 = OpUConvert %ulong %6231
       %6233 = OpIMul %ulong %6232 %ulong_4096
       %6234 = OpIAdd %ulong %6233 %5907
       %6235 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6234
               OpStore %6235 %5871 Aligned 4
       %6236 = OpIAdd %ulong %6233 %5912
       %6237 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6236
               OpStore %6237 %5879 Aligned 4
       %6238 = OpIAdd %ulong %6233 %5916
       %6239 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6238
               OpStore %6239 %5887 Aligned 4
       %6240 = OpIAdd %ulong %6233 %5920
       %6241 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6240
               OpStore %6241 %5895 Aligned 4
       %6242 = OpIAdd %uint %5903 %uint_54
       %6243 = OpUConvert %ulong %6242
       %6244 = OpIMul %ulong %6243 %ulong_4096
       %6245 = OpIAdd %ulong %6244 %5907
       %6246 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6245
               OpStore %6246 %5872 Aligned 4
       %6247 = OpIAdd %ulong %6244 %5912
       %6248 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6247
               OpStore %6248 %5880 Aligned 4
       %6249 = OpIAdd %ulong %6244 %5916
       %6250 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6249
               OpStore %6250 %5888 Aligned 4
       %6251 = OpIAdd %ulong %6244 %5920
       %6252 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6251
               OpStore %6252 %5896 Aligned 4
       %6253 = OpIAdd %uint %5903 %uint_55
       %6254 = OpUConvert %ulong %6253
       %6255 = OpIMul %ulong %6254 %ulong_4096
       %6256 = OpIAdd %ulong %6255 %5907
       %6257 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6256
               OpStore %6257 %5873 Aligned 4
       %6258 = OpIAdd %ulong %6255 %5912
       %6259 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6258
               OpStore %6259 %5881 Aligned 4
       %6260 = OpIAdd %ulong %6255 %5916
       %6261 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6260
               OpStore %6261 %5889 Aligned 4
       %6262 = OpIAdd %ulong %6255 %5920
       %6263 = OpPtrAccessChain %_ptr_CrossWorkgroup_float %4678 %6262
               OpStore %6263 %5897 Aligned 4
               OpMemoryBarrier %uint_2 %uint_4
       %6264 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_signal %uint_4294967295
       %6265 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_wait %ushort_65535
               OpMemoryBarrier %uint_2 %uint_2
               OpReturn
       %6963 = OpLabel
       %5056 = OpIAdd %uint %5055 %uint_4
       %6266 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5031 %5005
       %6267 = OpBitcast %_ptr_CrossWorkgroup_v8half %6266
       %6268 = OpLoad %v8half %6267 Aligned 2
       %6269 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5033 %5005
       %6270 = OpBitcast %_ptr_CrossWorkgroup_v8half %6269
       %6271 = OpLoad %v8half %6270 Aligned 2
       %6272 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5035 %5005
       %6273 = OpBitcast %_ptr_CrossWorkgroup_v8half %6272
       %6274 = OpLoad %v8half %6273 Aligned 2
       %6275 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5037 %5005
       %6276 = OpBitcast %_ptr_CrossWorkgroup_v8half %6275
       %6277 = OpLoad %v8half %6276 Aligned 2
       %6278 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5039 %5005
       %6279 = OpBitcast %_ptr_CrossWorkgroup_v8half %6278
       %6280 = OpLoad %v8half %6279 Aligned 2
       %6281 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5041 %5005
       %6282 = OpBitcast %_ptr_CrossWorkgroup_v8half %6281
       %6283 = OpLoad %v8half %6282 Aligned 2
       %6284 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5043 %5005
       %6285 = OpBitcast %_ptr_CrossWorkgroup_v8half %6284
       %6286 = OpLoad %v8half %6285 Aligned 2
       %6287 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5045 %5005
       %6288 = OpBitcast %_ptr_CrossWorkgroup_v8half %6287
       %6289 = OpLoad %v8half %6288 Aligned 2
       %6290 = OpBitcast %_ptr_CrossWorkgroup_v8half %5047
       %6291 = OpLoad %v8half %6290 Aligned 2
       %6292 = OpBitcast %_ptr_CrossWorkgroup_v8half %5049
       %6293 = OpLoad %v8half %6292 Aligned 2
       %6294 = OpBitcast %_ptr_CrossWorkgroup_v8half %5051
       %6295 = OpLoad %v8half %6294 Aligned 2
       %6296 = OpBitcast %_ptr_CrossWorkgroup_v8half %5053
       %6297 = OpLoad %v8half %6296 Aligned 2
               OpMemoryBarrier %uint_2 %uint_4
       %6298 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_signal %uint_4294967295
       %6299 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_wait %ushort_65535
               OpMemoryBarrier %uint_2 %uint_2
       %6300 = OpIAdd %ulong %5090 %4689
       %6301 = OpBitcast %_ptr_Workgroup_half %__shared_memory__
       %6302 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6300
       %6303 = OpBitcast %_ptr_Workgroup_v8half %6302
       %6304 = OpLoad %v8half %6303 Aligned 2
       %6305 = OpIAdd %ulong %5090 %4901
       %6306 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6305
       %6307 = OpBitcast %_ptr_Workgroup_v8half %6306
       %6308 = OpLoad %v8half %6307 Aligned 2
       %6309 = OpIAdd %ulong %5090 %4903
       %6310 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6309
       %6311 = OpBitcast %_ptr_Workgroup_v8half %6310
       %6312 = OpLoad %v8half %6311 Aligned 2
       %6313 = OpIAdd %ulong %5090 %4905
       %6314 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6313
       %6315 = OpBitcast %_ptr_Workgroup_v8half %6314
       %6316 = OpLoad %v8half %6315 Aligned 2
       %6317 = OpIAdd %ulong %5091 %4689
       %6318 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6317
       %6319 = OpBitcast %_ptr_Workgroup_v8half %6318
       %6320 = OpLoad %v8half %6319 Aligned 2
       %6321 = OpIAdd %ulong %5091 %4901
       %6322 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6321
       %6323 = OpBitcast %_ptr_Workgroup_v8half %6322
       %6324 = OpLoad %v8half %6323 Aligned 2
       %6325 = OpIAdd %ulong %5091 %4903
       %6326 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6325
       %6327 = OpBitcast %_ptr_Workgroup_v8half %6326
       %6328 = OpLoad %v8half %6327 Aligned 2
       %6329 = OpIAdd %ulong %5091 %4905
       %6330 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6329
       %6331 = OpBitcast %_ptr_Workgroup_v8half %6330
       %6332 = OpLoad %v8half %6331 Aligned 2
       %6333 = OpIAdd %ulong %5092 %4689
       %6334 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6333
       %6335 = OpBitcast %_ptr_Workgroup_v8half %6334
       %6336 = OpLoad %v8half %6335 Aligned 2
       %6337 = OpIAdd %ulong %5092 %4901
       %6338 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6337
       %6339 = OpBitcast %_ptr_Workgroup_v8half %6338
       %6340 = OpLoad %v8half %6339 Aligned 2
       %6341 = OpIAdd %ulong %5092 %4903
       %6342 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6341
       %6343 = OpBitcast %_ptr_Workgroup_v8half %6342
       %6344 = OpLoad %v8half %6343 Aligned 2
       %6345 = OpIAdd %ulong %5092 %4905
       %6346 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6345
       %6347 = OpBitcast %_ptr_Workgroup_v8half %6346
       %6348 = OpLoad %v8half %6347 Aligned 2
       %6349 = OpIAdd %ulong %5093 %4689
       %6350 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6349
       %6351 = OpBitcast %_ptr_Workgroup_v8half %6350
       %6352 = OpLoad %v8half %6351 Aligned 2
       %6353 = OpIAdd %ulong %5093 %4901
       %6354 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6353
       %6355 = OpBitcast %_ptr_Workgroup_v8half %6354
       %6356 = OpLoad %v8half %6355 Aligned 2
       %6357 = OpIAdd %ulong %5093 %4903
       %6358 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6357
       %6359 = OpBitcast %_ptr_Workgroup_v8half %6358
       %6360 = OpLoad %v8half %6359 Aligned 2
       %6361 = OpIAdd %ulong %5093 %4905
       %6362 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %6361
       %6363 = OpBitcast %_ptr_Workgroup_v8half %6362
       %6364 = OpLoad %v8half %6363 Aligned 2
       %6365 = OpIAdd %ulong %5094 %4915
       %6366 = OpBitcast %_ptr_Workgroup_half %__shared_memory___0
       %6367 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6365
       %6368 = OpLoad %half %6367 Aligned 2
       %6369 = OpIAdd %ulong %5094 %4917
       %6370 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6369
       %6371 = OpLoad %half %6370 Aligned 2
       %6372 = OpIAdd %ulong %5094 %4919
       %6373 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6372
       %6374 = OpLoad %half %6373 Aligned 2
       %6375 = OpIAdd %ulong %5094 %4921
       %6376 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6375
       %6377 = OpLoad %half %6376 Aligned 2
       %6378 = OpIAdd %ulong %5095 %4915
       %6379 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6378
       %6380 = OpLoad %half %6379 Aligned 2
       %6381 = OpIAdd %ulong %5095 %4917
       %6382 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6381
       %6383 = OpLoad %half %6382 Aligned 2
       %6384 = OpIAdd %ulong %5095 %4919
       %6385 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6384
       %6386 = OpLoad %half %6385 Aligned 2
       %6387 = OpIAdd %ulong %5095 %4921
       %6388 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6387
       %6389 = OpLoad %half %6388 Aligned 2
       %6390 = OpIAdd %ulong %5096 %4915
       %6391 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6390
       %6392 = OpLoad %half %6391 Aligned 2
       %6393 = OpIAdd %ulong %5096 %4917
       %6394 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6393
       %6395 = OpLoad %half %6394 Aligned 2
       %6396 = OpIAdd %ulong %5096 %4919
       %6397 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6396
       %6398 = OpLoad %half %6397 Aligned 2
       %6399 = OpIAdd %ulong %5096 %4921
       %6400 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6399
       %6401 = OpLoad %half %6400 Aligned 2
       %6402 = OpIAdd %ulong %5097 %4915
       %6403 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6402
       %6404 = OpLoad %half %6403 Aligned 2
       %6405 = OpIAdd %ulong %5097 %4917
       %6406 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6405
       %6407 = OpLoad %half %6406 Aligned 2
       %6408 = OpIAdd %ulong %5097 %4919
       %6409 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6408
       %6410 = OpLoad %half %6409 Aligned 2
       %6411 = OpIAdd %ulong %5097 %4921
       %6412 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6411
       %6413 = OpLoad %half %6412 Aligned 2
       %6414 = OpIAdd %ulong %5098 %4915
       %6415 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6414
       %6416 = OpLoad %half %6415 Aligned 2
       %6417 = OpIAdd %ulong %5098 %4917
       %6418 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6417
       %6419 = OpLoad %half %6418 Aligned 2
       %6420 = OpIAdd %ulong %5098 %4919
       %6421 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6420
       %6422 = OpLoad %half %6421 Aligned 2
       %6423 = OpIAdd %ulong %5098 %4921
       %6424 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6423
       %6425 = OpLoad %half %6424 Aligned 2
       %6426 = OpIAdd %ulong %5099 %4915
       %6427 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6426
       %6428 = OpLoad %half %6427 Aligned 2
       %6429 = OpIAdd %ulong %5099 %4917
       %6430 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6429
       %6431 = OpLoad %half %6430 Aligned 2
       %6432 = OpIAdd %ulong %5099 %4919
       %6433 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6432
       %6434 = OpLoad %half %6433 Aligned 2
       %6435 = OpIAdd %ulong %5099 %4921
       %6436 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6435
       %6437 = OpLoad %half %6436 Aligned 2
       %6438 = OpIAdd %ulong %5100 %4915
       %6439 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6438
       %6440 = OpLoad %half %6439 Aligned 2
       %6441 = OpIAdd %ulong %5100 %4917
       %6442 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6441
       %6443 = OpLoad %half %6442 Aligned 2
       %6444 = OpIAdd %ulong %5100 %4919
       %6445 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6444
       %6446 = OpLoad %half %6445 Aligned 2
       %6447 = OpIAdd %ulong %5100 %4921
       %6448 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6447
       %6449 = OpLoad %half %6448 Aligned 2
       %6450 = OpIAdd %ulong %5101 %4915
       %6451 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6450
       %6452 = OpLoad %half %6451 Aligned 2
       %6453 = OpIAdd %ulong %5101 %4917
       %6454 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6453
       %6455 = OpLoad %half %6454 Aligned 2
       %6456 = OpIAdd %ulong %5101 %4919
       %6457 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6456
       %6458 = OpLoad %half %6457 Aligned 2
       %6459 = OpIAdd %ulong %5101 %4921
       %6460 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6459
       %6461 = OpLoad %half %6460 Aligned 2
       %6462 = OpIAdd %ulong %5102 %4915
       %6463 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6462
       %6464 = OpLoad %half %6463 Aligned 2
       %6465 = OpIAdd %ulong %5102 %4917
       %6466 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6465
       %6467 = OpLoad %half %6466 Aligned 2
       %6468 = OpIAdd %ulong %5102 %4919
       %6469 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6468
       %6470 = OpLoad %half %6469 Aligned 2
       %6471 = OpIAdd %ulong %5102 %4921
       %6472 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6471
       %6473 = OpLoad %half %6472 Aligned 2
       %6474 = OpIAdd %ulong %5103 %4915
       %6475 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6474
       %6476 = OpLoad %half %6475 Aligned 2
       %6477 = OpIAdd %ulong %5103 %4917
       %6478 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6477
       %6479 = OpLoad %half %6478 Aligned 2
       %6480 = OpIAdd %ulong %5103 %4919
       %6481 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6480
       %6482 = OpLoad %half %6481 Aligned 2
       %6483 = OpIAdd %ulong %5103 %4921
       %6484 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6483
       %6485 = OpLoad %half %6484 Aligned 2
       %6486 = OpIAdd %ulong %5104 %4915
       %6487 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6486
       %6488 = OpLoad %half %6487 Aligned 2
       %6489 = OpIAdd %ulong %5104 %4917
       %6490 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6489
       %6491 = OpLoad %half %6490 Aligned 2
       %6492 = OpIAdd %ulong %5104 %4919
       %6493 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6492
       %6494 = OpLoad %half %6493 Aligned 2
       %6495 = OpIAdd %ulong %5104 %4921
       %6496 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6495
       %6497 = OpLoad %half %6496 Aligned 2
       %6498 = OpIAdd %ulong %5105 %4915
       %6499 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6498
       %6500 = OpLoad %half %6499 Aligned 2
       %6501 = OpIAdd %ulong %5105 %4917
       %6502 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6501
       %6503 = OpLoad %half %6502 Aligned 2
       %6504 = OpIAdd %ulong %5105 %4919
       %6505 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6504
       %6506 = OpLoad %half %6505 Aligned 2
       %6507 = OpIAdd %ulong %5105 %4921
       %6508 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6507
       %6509 = OpLoad %half %6508 Aligned 2
       %6510 = OpIAdd %ulong %5106 %4915
       %6511 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6510
       %6512 = OpLoad %half %6511 Aligned 2
       %6513 = OpIAdd %ulong %5106 %4917
       %6514 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6513
       %6515 = OpLoad %half %6514 Aligned 2
       %6516 = OpIAdd %ulong %5106 %4919
       %6517 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6516
       %6518 = OpLoad %half %6517 Aligned 2
       %6519 = OpIAdd %ulong %5106 %4921
       %6520 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6519
       %6521 = OpLoad %half %6520 Aligned 2
       %6522 = OpIAdd %ulong %5107 %4915
       %6523 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6522
       %6524 = OpLoad %half %6523 Aligned 2
       %6525 = OpIAdd %ulong %5107 %4917
       %6526 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6525
       %6527 = OpLoad %half %6526 Aligned 2
       %6528 = OpIAdd %ulong %5107 %4919
       %6529 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6528
       %6530 = OpLoad %half %6529 Aligned 2
       %6531 = OpIAdd %ulong %5107 %4921
       %6532 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6531
       %6533 = OpLoad %half %6532 Aligned 2
       %6534 = OpIAdd %ulong %5108 %4915
       %6535 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6534
       %6536 = OpLoad %half %6535 Aligned 2
       %6537 = OpIAdd %ulong %5108 %4917
       %6538 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6537
       %6539 = OpLoad %half %6538 Aligned 2
       %6540 = OpIAdd %ulong %5108 %4919
       %6541 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6540
       %6542 = OpLoad %half %6541 Aligned 2
       %6543 = OpIAdd %ulong %5108 %4921
       %6544 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6543
       %6545 = OpLoad %half %6544 Aligned 2
       %6546 = OpIAdd %ulong %5109 %4915
       %6547 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6546
       %6548 = OpLoad %half %6547 Aligned 2
       %6549 = OpIAdd %ulong %5109 %4917
       %6550 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6549
       %6551 = OpLoad %half %6550 Aligned 2
       %6552 = OpIAdd %ulong %5109 %4919
       %6553 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6552
       %6554 = OpLoad %half %6553 Aligned 2
       %6555 = OpIAdd %ulong %5109 %4921
       %6556 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6555
       %6557 = OpLoad %half %6556 Aligned 2
       %6558 = OpIAdd %ulong %5110 %4915
       %6559 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6558
       %6560 = OpLoad %half %6559 Aligned 2
       %6561 = OpIAdd %ulong %5110 %4917
       %6562 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6561
       %6563 = OpLoad %half %6562 Aligned 2
       %6564 = OpIAdd %ulong %5110 %4919
       %6565 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6564
       %6566 = OpLoad %half %6565 Aligned 2
       %6567 = OpIAdd %ulong %5110 %4921
       %6568 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6567
       %6569 = OpLoad %half %6568 Aligned 2
       %6570 = OpIAdd %ulong %5111 %4915
       %6571 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6570
       %6572 = OpLoad %half %6571 Aligned 2
       %6573 = OpIAdd %ulong %5111 %4917
       %6574 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6573
       %6575 = OpLoad %half %6574 Aligned 2
       %6576 = OpIAdd %ulong %5111 %4919
       %6577 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6576
       %6578 = OpLoad %half %6577 Aligned 2
       %6579 = OpIAdd %ulong %5111 %4921
       %6580 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6579
       %6581 = OpLoad %half %6580 Aligned 2
       %6582 = OpIAdd %ulong %5112 %4915
       %6583 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6582
       %6584 = OpLoad %half %6583 Aligned 2
       %6585 = OpIAdd %ulong %5112 %4917
       %6586 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6585
       %6587 = OpLoad %half %6586 Aligned 2
       %6588 = OpIAdd %ulong %5112 %4919
       %6589 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6588
       %6590 = OpLoad %half %6589 Aligned 2
       %6591 = OpIAdd %ulong %5112 %4921
       %6592 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6591
       %6593 = OpLoad %half %6592 Aligned 2
       %6594 = OpIAdd %ulong %5113 %4915
       %6595 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6594
       %6596 = OpLoad %half %6595 Aligned 2
       %6597 = OpIAdd %ulong %5113 %4917
       %6598 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6597
       %6599 = OpLoad %half %6598 Aligned 2
       %6600 = OpIAdd %ulong %5113 %4919
       %6601 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6600
       %6602 = OpLoad %half %6601 Aligned 2
       %6603 = OpIAdd %ulong %5113 %4921
       %6604 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6603
       %6605 = OpLoad %half %6604 Aligned 2
       %6606 = OpIAdd %ulong %5114 %4915
       %6607 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6606
       %6608 = OpLoad %half %6607 Aligned 2
       %6609 = OpIAdd %ulong %5114 %4917
       %6610 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6609
       %6611 = OpLoad %half %6610 Aligned 2
       %6612 = OpIAdd %ulong %5114 %4919
       %6613 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6612
       %6614 = OpLoad %half %6613 Aligned 2
       %6615 = OpIAdd %ulong %5114 %4921
       %6616 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6615
       %6617 = OpLoad %half %6616 Aligned 2
       %6618 = OpIAdd %ulong %5115 %4915
       %6619 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6618
       %6620 = OpLoad %half %6619 Aligned 2
       %6621 = OpIAdd %ulong %5115 %4917
       %6622 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6621
       %6623 = OpLoad %half %6622 Aligned 2
       %6624 = OpIAdd %ulong %5115 %4919
       %6625 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6624
       %6626 = OpLoad %half %6625 Aligned 2
       %6627 = OpIAdd %ulong %5115 %4921
       %6628 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6627
       %6629 = OpLoad %half %6628 Aligned 2
       %6630 = OpIAdd %ulong %5116 %4915
       %6631 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6630
       %6632 = OpLoad %half %6631 Aligned 2
       %6633 = OpIAdd %ulong %5116 %4917
       %6634 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6633
       %6635 = OpLoad %half %6634 Aligned 2
       %6636 = OpIAdd %ulong %5116 %4919
       %6637 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6636
       %6638 = OpLoad %half %6637 Aligned 2
       %6639 = OpIAdd %ulong %5116 %4921
       %6640 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6639
       %6641 = OpLoad %half %6640 Aligned 2
       %6642 = OpIAdd %ulong %5117 %4915
       %6643 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6642
       %6644 = OpLoad %half %6643 Aligned 2
       %6645 = OpIAdd %ulong %5117 %4917
       %6646 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6645
       %6647 = OpLoad %half %6646 Aligned 2
       %6648 = OpIAdd %ulong %5117 %4919
       %6649 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6648
       %6650 = OpLoad %half %6649 Aligned 2
       %6651 = OpIAdd %ulong %5117 %4921
       %6652 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6651
       %6653 = OpLoad %half %6652 Aligned 2
       %6654 = OpIAdd %ulong %5118 %4915
       %6655 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6654
       %6656 = OpLoad %half %6655 Aligned 2
       %6657 = OpIAdd %ulong %5118 %4917
       %6658 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6657
       %6659 = OpLoad %half %6658 Aligned 2
       %6660 = OpIAdd %ulong %5118 %4919
       %6661 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6660
       %6662 = OpLoad %half %6661 Aligned 2
       %6663 = OpIAdd %ulong %5118 %4921
       %6664 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6663
       %6665 = OpLoad %half %6664 Aligned 2
       %6666 = OpIAdd %ulong %5119 %4915
       %6667 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6666
       %6668 = OpLoad %half %6667 Aligned 2
       %6669 = OpIAdd %ulong %5119 %4917
       %6670 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6669
       %6671 = OpLoad %half %6670 Aligned 2
       %6672 = OpIAdd %ulong %5119 %4919
       %6673 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6672
       %6674 = OpLoad %half %6673 Aligned 2
       %6675 = OpIAdd %ulong %5119 %4921
       %6676 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6675
       %6677 = OpLoad %half %6676 Aligned 2
       %6678 = OpIAdd %ulong %5120 %4915
       %6679 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6678
       %6680 = OpLoad %half %6679 Aligned 2
       %6681 = OpIAdd %ulong %5120 %4917
       %6682 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6681
       %6683 = OpLoad %half %6682 Aligned 2
       %6684 = OpIAdd %ulong %5120 %4919
       %6685 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6684
       %6686 = OpLoad %half %6685 Aligned 2
       %6687 = OpIAdd %ulong %5120 %4921
       %6688 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6687
       %6689 = OpLoad %half %6688 Aligned 2
       %6690 = OpIAdd %ulong %5121 %4915
       %6691 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6690
       %6692 = OpLoad %half %6691 Aligned 2
       %6693 = OpIAdd %ulong %5121 %4917
       %6694 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6693
       %6695 = OpLoad %half %6694 Aligned 2
       %6696 = OpIAdd %ulong %5121 %4919
       %6697 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6696
       %6698 = OpLoad %half %6697 Aligned 2
       %6699 = OpIAdd %ulong %5121 %4921
       %6700 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6699
       %6701 = OpLoad %half %6700 Aligned 2
       %6702 = OpIAdd %ulong %5122 %4915
       %6703 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6702
       %6704 = OpLoad %half %6703 Aligned 2
       %6705 = OpIAdd %ulong %5122 %4917
       %6706 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6705
       %6707 = OpLoad %half %6706 Aligned 2
       %6708 = OpIAdd %ulong %5122 %4919
       %6709 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6708
       %6710 = OpLoad %half %6709 Aligned 2
       %6711 = OpIAdd %ulong %5122 %4921
       %6712 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6711
       %6713 = OpLoad %half %6712 Aligned 2
       %6714 = OpIAdd %ulong %5123 %4915
       %6715 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6714
       %6716 = OpLoad %half %6715 Aligned 2
       %6717 = OpIAdd %ulong %5123 %4917
       %6718 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6717
       %6719 = OpLoad %half %6718 Aligned 2
       %6720 = OpIAdd %ulong %5123 %4919
       %6721 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6720
       %6722 = OpLoad %half %6721 Aligned 2
       %6723 = OpIAdd %ulong %5123 %4921
       %6724 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6723
       %6725 = OpLoad %half %6724 Aligned 2
       %6726 = OpIAdd %ulong %5124 %4915
       %6727 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6726
       %6728 = OpLoad %half %6727 Aligned 2
       %6729 = OpIAdd %ulong %5124 %4917
       %6730 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6729
       %6731 = OpLoad %half %6730 Aligned 2
       %6732 = OpIAdd %ulong %5124 %4919
       %6733 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6732
       %6734 = OpLoad %half %6733 Aligned 2
       %6735 = OpIAdd %ulong %5124 %4921
       %6736 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6735
       %6737 = OpLoad %half %6736 Aligned 2
       %6738 = OpIAdd %ulong %5125 %4915
       %6739 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6738
       %6740 = OpLoad %half %6739 Aligned 2
       %6741 = OpIAdd %ulong %5125 %4917
       %6742 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6741
       %6743 = OpLoad %half %6742 Aligned 2
       %6744 = OpIAdd %ulong %5125 %4919
       %6745 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6744
       %6746 = OpLoad %half %6745 Aligned 2
       %6747 = OpIAdd %ulong %5125 %4921
       %6748 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %6747
       %6749 = OpLoad %half %6748 Aligned 2
       %6750 = OpCompositeInsert %v8half %6368 %106 0
       %6751 = OpCompositeInsert %v8half %6380 %6750 1
       %6752 = OpCompositeInsert %v8half %6392 %6751 2
       %6753 = OpCompositeInsert %v8half %6404 %6752 3
       %6754 = OpCompositeInsert %v8half %6416 %6753 4
       %6755 = OpCompositeInsert %v8half %6428 %6754 5
       %6756 = OpCompositeInsert %v8half %6440 %6755 6
       %6757 = OpCompositeInsert %v8half %6452 %6756 7
       %6758 = OpCompositeInsert %v8half %6371 %106 0
       %6759 = OpCompositeInsert %v8half %6383 %6758 1
       %6760 = OpCompositeInsert %v8half %6395 %6759 2
       %6761 = OpCompositeInsert %v8half %6407 %6760 3
       %6762 = OpCompositeInsert %v8half %6419 %6761 4
       %6763 = OpCompositeInsert %v8half %6431 %6762 5
       %6764 = OpCompositeInsert %v8half %6443 %6763 6
       %6765 = OpCompositeInsert %v8half %6455 %6764 7
       %6766 = OpCompositeInsert %v8half %6374 %106 0
       %6767 = OpCompositeInsert %v8half %6386 %6766 1
       %6768 = OpCompositeInsert %v8half %6398 %6767 2
       %6769 = OpCompositeInsert %v8half %6410 %6768 3
       %6770 = OpCompositeInsert %v8half %6422 %6769 4
       %6771 = OpCompositeInsert %v8half %6434 %6770 5
       %6772 = OpCompositeInsert %v8half %6446 %6771 6
       %6773 = OpCompositeInsert %v8half %6458 %6772 7
       %6774 = OpCompositeInsert %v8half %6377 %106 0
       %6775 = OpCompositeInsert %v8half %6389 %6774 1
       %6776 = OpCompositeInsert %v8half %6401 %6775 2
       %6777 = OpCompositeInsert %v8half %6413 %6776 3
       %6778 = OpCompositeInsert %v8half %6425 %6777 4
       %6779 = OpCompositeInsert %v8half %6437 %6778 5
       %6780 = OpCompositeInsert %v8half %6449 %6779 6
       %6781 = OpCompositeInsert %v8half %6461 %6780 7
       %6782 = OpCompositeInsert %v8half %6464 %106 0
       %6783 = OpCompositeInsert %v8half %6476 %6782 1
       %6784 = OpCompositeInsert %v8half %6488 %6783 2
       %6785 = OpCompositeInsert %v8half %6500 %6784 3
       %6786 = OpCompositeInsert %v8half %6512 %6785 4
       %6787 = OpCompositeInsert %v8half %6524 %6786 5
       %6788 = OpCompositeInsert %v8half %6536 %6787 6
       %6789 = OpCompositeInsert %v8half %6548 %6788 7
       %6790 = OpCompositeInsert %v8half %6467 %106 0
       %6791 = OpCompositeInsert %v8half %6479 %6790 1
       %6792 = OpCompositeInsert %v8half %6491 %6791 2
       %6793 = OpCompositeInsert %v8half %6503 %6792 3
       %6794 = OpCompositeInsert %v8half %6515 %6793 4
       %6795 = OpCompositeInsert %v8half %6527 %6794 5
       %6796 = OpCompositeInsert %v8half %6539 %6795 6
       %6797 = OpCompositeInsert %v8half %6551 %6796 7
       %6798 = OpCompositeInsert %v8half %6470 %106 0
       %6799 = OpCompositeInsert %v8half %6482 %6798 1
       %6800 = OpCompositeInsert %v8half %6494 %6799 2
       %6801 = OpCompositeInsert %v8half %6506 %6800 3
       %6802 = OpCompositeInsert %v8half %6518 %6801 4
       %6803 = OpCompositeInsert %v8half %6530 %6802 5
       %6804 = OpCompositeInsert %v8half %6542 %6803 6
       %6805 = OpCompositeInsert %v8half %6554 %6804 7
       %6806 = OpCompositeInsert %v8half %6473 %106 0
       %6807 = OpCompositeInsert %v8half %6485 %6806 1
       %6808 = OpCompositeInsert %v8half %6497 %6807 2
       %6809 = OpCompositeInsert %v8half %6509 %6808 3
       %6810 = OpCompositeInsert %v8half %6521 %6809 4
       %6811 = OpCompositeInsert %v8half %6533 %6810 5
       %6812 = OpCompositeInsert %v8half %6545 %6811 6
       %6813 = OpCompositeInsert %v8half %6557 %6812 7
       %6814 = OpCompositeInsert %v8half %6560 %106 0
       %6815 = OpCompositeInsert %v8half %6572 %6814 1
       %6816 = OpCompositeInsert %v8half %6584 %6815 2
       %6817 = OpCompositeInsert %v8half %6596 %6816 3
       %6818 = OpCompositeInsert %v8half %6608 %6817 4
       %6819 = OpCompositeInsert %v8half %6620 %6818 5
       %6820 = OpCompositeInsert %v8half %6632 %6819 6
       %6821 = OpCompositeInsert %v8half %6644 %6820 7
       %6822 = OpCompositeInsert %v8half %6563 %106 0
       %6823 = OpCompositeInsert %v8half %6575 %6822 1
       %6824 = OpCompositeInsert %v8half %6587 %6823 2
       %6825 = OpCompositeInsert %v8half %6599 %6824 3
       %6826 = OpCompositeInsert %v8half %6611 %6825 4
       %6827 = OpCompositeInsert %v8half %6623 %6826 5
       %6828 = OpCompositeInsert %v8half %6635 %6827 6
       %6829 = OpCompositeInsert %v8half %6647 %6828 7
       %6830 = OpCompositeInsert %v8half %6566 %106 0
       %6831 = OpCompositeInsert %v8half %6578 %6830 1
       %6832 = OpCompositeInsert %v8half %6590 %6831 2
       %6833 = OpCompositeInsert %v8half %6602 %6832 3
       %6834 = OpCompositeInsert %v8half %6614 %6833 4
       %6835 = OpCompositeInsert %v8half %6626 %6834 5
       %6836 = OpCompositeInsert %v8half %6638 %6835 6
       %6837 = OpCompositeInsert %v8half %6650 %6836 7
       %6838 = OpCompositeInsert %v8half %6569 %106 0
       %6839 = OpCompositeInsert %v8half %6581 %6838 1
       %6840 = OpCompositeInsert %v8half %6593 %6839 2
       %6841 = OpCompositeInsert %v8half %6605 %6840 3
       %6842 = OpCompositeInsert %v8half %6617 %6841 4
       %6843 = OpCompositeInsert %v8half %6629 %6842 5
       %6844 = OpCompositeInsert %v8half %6641 %6843 6
       %6845 = OpCompositeInsert %v8half %6653 %6844 7
       %6846 = OpCompositeInsert %v8half %6656 %106 0
       %6847 = OpCompositeInsert %v8half %6668 %6846 1
       %6848 = OpCompositeInsert %v8half %6680 %6847 2
       %6849 = OpCompositeInsert %v8half %6692 %6848 3
       %6850 = OpCompositeInsert %v8half %6704 %6849 4
       %6851 = OpCompositeInsert %v8half %6716 %6850 5
       %6852 = OpCompositeInsert %v8half %6728 %6851 6
       %6853 = OpCompositeInsert %v8half %6740 %6852 7
       %6854 = OpCompositeInsert %v8half %6659 %106 0
       %6855 = OpCompositeInsert %v8half %6671 %6854 1
       %6856 = OpCompositeInsert %v8half %6683 %6855 2
       %6857 = OpCompositeInsert %v8half %6695 %6856 3
       %6858 = OpCompositeInsert %v8half %6707 %6857 4
       %6859 = OpCompositeInsert %v8half %6719 %6858 5
       %6860 = OpCompositeInsert %v8half %6731 %6859 6
       %6861 = OpCompositeInsert %v8half %6743 %6860 7
       %6862 = OpCompositeInsert %v8half %6662 %106 0
       %6863 = OpCompositeInsert %v8half %6674 %6862 1
       %6864 = OpCompositeInsert %v8half %6686 %6863 2
       %6865 = OpCompositeInsert %v8half %6698 %6864 3
       %6866 = OpCompositeInsert %v8half %6710 %6865 4
       %6867 = OpCompositeInsert %v8half %6722 %6866 5
       %6868 = OpCompositeInsert %v8half %6734 %6867 6
       %6869 = OpCompositeInsert %v8half %6746 %6868 7
       %6870 = OpCompositeInsert %v8half %6665 %106 0
       %6871 = OpCompositeInsert %v8half %6677 %6870 1
       %6872 = OpCompositeInsert %v8half %6689 %6871 2
       %6873 = OpCompositeInsert %v8half %6701 %6872 3
       %6874 = OpCompositeInsert %v8half %6713 %6873 4
       %6875 = OpCompositeInsert %v8half %6725 %6874 5
       %6876 = OpCompositeInsert %v8half %6737 %6875 6
       %6877 = OpCompositeInsert %v8half %6749 %6876 7
       %6878 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6304 %6757 %5087
       %6879 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6304 %6765 %5085
       %6880 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6304 %6773 %5083
       %6881 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6304 %6781 %5081
       %6882 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6320 %6757 %5079
       %6883 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6320 %6765 %5077
       %6884 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6320 %6773 %5075
       %6885 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6320 %6781 %5073
       %6886 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6336 %6757 %5071
       %6887 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6336 %6765 %5069
       %6888 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6336 %6773 %5067
       %6889 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6336 %6781 %5065
       %6890 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6352 %6757 %5063
       %6891 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6352 %6765 %5061
       %6892 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6352 %6773 %5059
       %6893 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6352 %6781 %5057
       %6894 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6308 %6789 %6878
       %6895 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6308 %6797 %6879
       %6896 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6308 %6805 %6880
       %6897 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6308 %6813 %6881
       %6898 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6324 %6789 %6882
       %6899 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6324 %6797 %6883
       %6900 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6324 %6805 %6884
       %6901 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6324 %6813 %6885
       %6902 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6340 %6789 %6886
       %6903 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6340 %6797 %6887
       %6904 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6340 %6805 %6888
       %6905 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6340 %6813 %6889
       %6906 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6356 %6789 %6890
       %6907 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6356 %6797 %6891
       %6908 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6356 %6805 %6892
       %6909 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6356 %6813 %6893
       %6910 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6312 %6821 %6894
       %6911 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6312 %6829 %6895
       %6912 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6312 %6837 %6896
       %6913 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6312 %6845 %6897
       %6914 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6328 %6821 %6898
       %6915 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6328 %6829 %6899
       %6916 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6328 %6837 %6900
       %6917 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6328 %6845 %6901
       %6918 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6344 %6821 %6902
       %6919 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6344 %6829 %6903
       %6920 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6344 %6837 %6904
       %6921 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6344 %6845 %6905
       %6922 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6360 %6821 %6906
       %6923 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6360 %6829 %6907
       %6924 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6360 %6837 %6908
       %6925 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6360 %6845 %6909
       %5088 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6316 %6853 %6910
       %5086 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6316 %6861 %6911
       %5084 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6316 %6869 %6912
       %5082 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6316 %6877 %6913
       %5080 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6332 %6853 %6914
       %5078 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6332 %6861 %6915
       %5076 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6332 %6869 %6916
       %5074 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6332 %6877 %6917
       %5072 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6348 %6853 %6918
       %5070 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6348 %6861 %6919
       %5068 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6348 %6869 %6920
       %5066 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6348 %6877 %6921
       %5064 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6364 %6853 %6922
       %5062 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6364 %6861 %6923
       %5060 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6364 %6869 %6924
       %5058 = OpFunctionCall %v8float %spirv_llvm_amdgcn_wmma_f32_16x16x16_f16_v8f32_v8f16 %6364 %6877 %6925
               OpMemoryBarrier %uint_2 %uint_4
       %6926 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_signal %uint_4294967295
       %6927 = OpFunctionCall %void %spirv_llvm_amdgcn_s_barrier_wait %ushort_65535
               OpMemoryBarrier %uint_2 %uint_2
       %6928 = OpFunctionCall %void %spirv_llvm_amdgcn_sched_barrier %90
       %6929 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %4847
       %6930 = OpBitcast %_ptr_Workgroup_v8half %6929
               OpStore %6930 %6268 Aligned 2
       %6931 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %4852
       %6932 = OpBitcast %_ptr_Workgroup_v8half %6931
               OpStore %6932 %6271 Aligned 2
       %6933 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %4856
       %6934 = OpBitcast %_ptr_Workgroup_v8half %6933
               OpStore %6934 %6274 Aligned 2
       %6935 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %4860
       %6936 = OpBitcast %_ptr_Workgroup_v8half %6935
               OpStore %6936 %6277 Aligned 2
       %6937 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %4864
       %6938 = OpBitcast %_ptr_Workgroup_v8half %6937
               OpStore %6938 %6280 Aligned 2
       %6939 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %4868
       %6940 = OpBitcast %_ptr_Workgroup_v8half %6939
               OpStore %6940 %6283 Aligned 2
       %6941 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %4872
       %6942 = OpBitcast %_ptr_Workgroup_v8half %6941
               OpStore %6942 %6286 Aligned 2
       %6943 = OpPtrAccessChain %_ptr_Workgroup_half %6301 %4876
       %6944 = OpBitcast %_ptr_Workgroup_v8half %6943
               OpStore %6944 %6289 Aligned 2
       %6945 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %4880
       %6946 = OpBitcast %_ptr_Workgroup_v8half %6945
               OpStore %6946 %6291 Aligned 2
       %6947 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %4885
       %6948 = OpBitcast %_ptr_Workgroup_v8half %6947
               OpStore %6948 %6293 Aligned 2
       %6949 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %4889
       %6950 = OpBitcast %_ptr_Workgroup_v8half %6949
               OpStore %6950 %6295 Aligned 2
       %6951 = OpPtrAccessChain %_ptr_Workgroup_half %6366 %4893
       %6952 = OpBitcast %_ptr_Workgroup_v8half %6951
               OpStore %6952 %6297 Aligned 2
       %5054 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5053 %ulong_524288
       %5052 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5051 %ulong_524288
       %5050 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5049 %ulong_524288
       %5048 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5047 %ulong_524288
       %5046 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5045 %ulong_128
       %5044 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5043 %ulong_128
       %5042 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5041 %ulong_128
       %5040 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5039 %ulong_128
       %5038 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5037 %ulong_128
       %5036 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5035 %ulong_128
       %5034 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5033 %ulong_128
       %5032 = OpPtrAccessChain %_ptr_CrossWorkgroup_uchar %5031 %ulong_128
               OpBranch %6962
               OpFunctionEnd
