// RUN: iree-opt --split-input-file --iree-rocdl-prepare-for-spirv %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-rocdl-prepare-for-spirv --iree-rocdl-prepare-for-spirv %s | FileCheck %s --check-prefix=IDEMPOTENT

// IDEMPOTENT: llvm.mlir.global private constant @llvm.cmdline

// Test triple, data layout, calling conventions, attributes, and address spaces.

// CHECK: module attributes
// CHECK-SAME: llvm.data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n32:64-S32-G1-P4-A0"
// CHECK-SAME: llvm.target_triple = "spirv64-amd-amdhsa"

// Shared memory (AS 3) stays AS 3.
// CHECK: llvm.mlir.global external @__dynamic_shared_memory__()
// CHECK-SAME: addr_space = 3

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {

  llvm.mlir.global external @__dynamic_shared_memory__() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func spir_kernelcc @simple_kernel
  llvm.func amdgpu_kernelcc @simple_kernel(%arg0: !llvm.ptr<1> {llvm.align = 16 : i64}) attributes {
    target_cpu = "gfx1201",
    target_features = "+wavefrontsize64"
  } {
    // Alloca in AS 5 (private) should become AS 0 (Function).
    // CHECK: llvm.alloca %{{.*}} x f32 {addr_space = 0 : i32} : (i64) -> !llvm.ptr
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %0 = llvm.alloca %c1 x f32 {addr_space = 5 : i32} : (i64) -> !llvm.ptr<5>
    // Flat/generic AS 0 -> SPIR-V Generic AS 4.
    // CHECK: llvm.addrspacecast %{{.*}} : !llvm.ptr to !llvm.ptr<4>
    %1 = llvm.addrspacecast %0 : !llvm.ptr<5> to !llvm.ptr
    llvm.return
  }

  // CHECK-LABEL: llvm.func spir_funccc @helper_func
  llvm.func @helper_func(%arg0: f32) -> f32 {
    llvm.return %arg0 : f32
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @kernel_with_shared_mem
  llvm.func amdgpu_kernelcc @kernel_with_shared_mem() {
    // CHECK: llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
    %0 = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
    llvm.return
  }

  // Intrinsic declarations should keep their calling convention unchanged.
  // CHECK: llvm.func @llvm.amdgcn.workitem.id.x() -> i32
  llvm.func @llvm.amdgcn.workitem.id.x() -> i32

  // External device library declarations get spir_func.
  // CHECK: llvm.func spir_funccc @__ocml_sin_f32(f32) -> f32
  llvm.func @__ocml_sin_f32(f32) -> f32
}

// -----

// Test that inreg attributes are removed from kernel arguments.

module {
  // CHECK-LABEL: llvm.func spir_kernelcc @kernel_with_inreg
  // CHECK-SAME:    (%{{.+}}: i32, %{{.+}}: !llvm.ptr<1>, %{{.+}}: i64)
  // CHECK-NOT:     llvm.inreg
  llvm.func amdgpu_kernelcc @kernel_with_inreg(
      %arg0: i32 {llvm.inreg},
      %arg1: !llvm.ptr<1> {llvm.inreg},
      %arg2: i64) {
    llvm.return
  }
}

// -----

// Test that llvm.intr.assume calls are stripped (SPIR-V backend workaround).
// The SPIR-V backend crashes on llvm.assume with operand bundles
// (SPIRVEmitIntrinsics::paramHasAttr out-of-bounds).

module {
  // CHECK-LABEL: llvm.func spir_kernelcc @kernel_with_assumes
  llvm.func amdgpu_kernelcc @kernel_with_assumes(%arg0: !llvm.ptr<1>) {
    %true = llvm.mlir.constant(true) : i1
    %c64 = llvm.mlir.constant(64 : i32) : i32
    // CHECK-NOT:   llvm.intr.assume
    // Simple assume (no operand bundles).
    llvm.intr.assume %true : i1
    // Assume with operand bundles — this is the case that crashes the SPIR-V
    // backend (paramHasAttr on bundle operands).
    llvm.intr.assume %true ["align"(%arg0, %c64 : !llvm.ptr<1>, i32)] : i1
    %c0 = llvm.mlir.constant(0 : i64) : i64
    // CHECK:       llvm.getelementptr
    %gep = llvm.getelementptr %arg0[%c0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    // CHECK:       llvm.load
    %val = llvm.load %gep : !llvm.ptr<1> -> f32
    llvm.return
  }
}

// -----

// Test that rocdl.kernel attribute triggers spir_kernel CC.
// After ROCDLAnnotateKernelForTranslation, kernels have rocdl.kernel
// instead of amdgpu_kernel CC.

module {
  // CHECK-LABEL: llvm.func spir_kernelcc @annotated_kernel
  // CHECK-NOT:     rocdl.kernel
  // CHECK-NOT:     rocdl.flat_work_group_size
  // CHECK-NOT:     rocdl.reqd_work_group_size
  // CHECK-NOT:     rocdl.max_flat_work_group_size
  llvm.func @annotated_kernel(%arg0: !llvm.ptr<1>)
      attributes {
        rocdl.kernel,
        rocdl.flat_work_group_size = "1,256",
        rocdl.reqd_work_group_size = dense<[64, 1, 1]> : vector<3xi32>,
        rocdl.max_flat_work_group_size = 256 : i32
      } {
    llvm.return
  }
}

// -----

// Test constant address space remapping (AS 4 -> AS 2).

module {
  // CHECK: llvm.mlir.global external @const_data() {addr_space = 2 : i32}
  llvm.mlir.global external @const_data() {addr_space = 4 : i32} : !llvm.array<16 x f32>
}

// -----

// Test that @llvm.cmdline is created for comgr JIT flags.

module {
  llvm.func @kernel() {
    llvm.return
  }
  // CHECK: llvm.mlir.global private constant @llvm.cmdline("-O3\00")
  // CHECK-SAME: addr_space = 1
  // CHECK-SAME: section = ".llvmcmd"
}

// -----

// Test that an existing @llvm.cmdline has its optimization flag replaced.

module {
  llvm.mlir.global private constant @llvm.cmdline("-cc1\00-O2\00foo\00")
      {addr_space = 1 : i32, alignment = 1 : i64, section = ".llvmcmd"} : !llvm.array<13 x i8>

  llvm.func @kernel_with_existing_cmdline() {
    llvm.return
  }

  // CHECK: llvm.mlir.global private constant @llvm.cmdline("-cc1\00-O3\00foo\00")
  // CHECK-SAME: addr_space = 1
  // CHECK-SAME: section = ".llvmcmd"
}

// -----

// Test that an existing @llvm.cmdline without an optimization flag has -O3
// appended.

module {
  llvm.mlir.global private constant @llvm.cmdline("-cc1\00")
      {addr_space = 1 : i32, alignment = 1 : i64, section = ".llvmcmd"} : !llvm.array<5 x i8>

  llvm.func @kernel_with_existing_cmdline_without_opt() {
    llvm.return
  }

  // CHECK: llvm.mlir.global private constant @llvm.cmdline("-cc1\00-O3\00")
  // CHECK-SAME: addr_space = 1
  // CHECK-SAME: section = ".llvmcmd"
}
