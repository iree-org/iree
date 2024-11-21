// RUN: iree-opt --split-input-file -pass-pipeline"=builtin.module(func.func(iree-codegen-hoist-statically-bound-allocations,iree-codegen-gpu-reuse-shared-memory-allocs))" %s | FileCheck %s

func.func @trivial_reuse() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %alloc0 = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
    %alloc1 = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
    %alloc2 = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
    %alloc3 = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>

    %r0 = vector.transfer_read %alloc0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
    vector.transfer_write %r0, %alloc1[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>
    %r1 = vector.transfer_read %alloc1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
    vector.transfer_write %r1, %alloc2[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>
    %r2 = vector.transfer_read %alloc2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
    vector.transfer_write %r2, %alloc3[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>
    return
}

// CHECK-LABEL: @trivial_reuse
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
// CHECK-COUNT-4: memref.view %[[ALLOC]][%c0{{.*}}]

// -----

func.func @trivial_non_reuse() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %alloc0 = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
    %alloc1 = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
    %alloc2 = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
    %alloc3 = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>

    %r0 = vector.transfer_read %alloc0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
    vector.transfer_write %r0, %alloc1[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>
    %r1 = vector.transfer_read %alloc1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
    vector.transfer_write %r1, %alloc2[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>
    %r2 = vector.transfer_read %alloc2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
    vector.transfer_write %r2, %alloc3[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>
    %r3 = vector.transfer_read %alloc0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
    vector.transfer_write %r3, %alloc1[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>
    %r4 = vector.transfer_read %alloc1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
    vector.transfer_write %r4, %alloc2[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>
    %r5 = vector.transfer_read %alloc2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
    vector.transfer_write %r5, %alloc3[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>
    return
}

// CHECK-LABEL: @trivial_non_reuse
// CHECK-COUNT-4: memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>

// -----

func.func @shared_reuse_scf_for(%in0: memref<128x128xf16>, %in1: memref<1x4096x128xf16>, %in2: memref<1x4096x128xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant dense<0.000000e+00> : vector<64x128xf16>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    gpu.barrier
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
    %5 = vector.transfer_read %in0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x128xf16>, vector<128x128xf16>
    %alloc = memref.alloc() : memref<128x128xf16, #gpu.address_space<workgroup>>
    vector.transfer_write %5, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<128x128xf16>, memref<128x128xf16, #gpu.address_space<workgroup>>
    gpu.barrier
    %8 = vector.transfer_read %alloc[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x128xf16, #gpu.address_space<workgroup>>, vector<128x128xf16>

    %11 = scf.for %arg0 = %c0 to %c4096 step %c64 iter_args(%arg1 = %cst_0) -> vector<64x128xf16> {
        %17 = vector.transfer_read %in1[%c0, %arg0, %c0], %cst {in_bounds = [true, true]} : memref<1x4096x128xf16>, vector<64x128xf16>
        %19 = vector.transfer_read %in2[%c0, %arg0, %c0], %cst {in_bounds = [true, true]} : memref<1x4096x128xf16>, vector<64x128xf16>
        %alloc_10 = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %17, %alloc_10[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>
        %alloc_11 = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %19, %alloc_11[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>

        %21 = vector.transfer_read %alloc_10[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
        %22 = arith.addf %arg1, %21 : vector<64x128xf16>
        %38 = vector.transfer_read %alloc_11[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
        %39 = arith.addf %22, %38 : vector<64x128xf16>
        scf.yield %39 : vector<64x128xf16>
    }
    return
}

// CHECK-LABEL: @shared_reuse_scf_for
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<32768xi8, #gpu.address_space<workgroup>>
// CHECK-DAG: memref.view %[[ALLOC]][%c0{{.*}}][] : memref<32768xi8, #gpu.address_space<workgroup>> to memref<128x128xf16, #gpu.address_space<workgroup>>
// CHECK-DAG: memref.view %[[ALLOC]][%c0{{.*}}][] : memref<32768xi8, #gpu.address_space<workgroup>> to memref<64x128xf16, #gpu.address_space<workgroup>>
// CHECK-DAG: memref.view %[[ALLOC]][%c16384][] : memref<32768xi8, #gpu.address_space<workgroup>> to memref<64x128xf16, #gpu.address_space<workgroup>>

// -----

func.func @shared_no_reuse_scf_for(%in0: memref<128x128xf16>, %in1: memref<1x4096x128xf16>, %in2: memref<1x4096x128xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant dense<0.000000e+00> : vector<64x128xf16>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    gpu.barrier
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
    %5 = vector.transfer_read %in0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x128xf16>, vector<128x128xf16>

    %11 = scf.for %arg0 = %c0 to %c4096 step %c64 iter_args(%arg1 = %cst_0) -> vector<64x128xf16> {
        %17 = vector.transfer_read %in1[%c0, %arg0, %c0], %cst {in_bounds = [true, true]} : memref<1x4096x128xf16>, vector<64x128xf16>
        %19 = vector.transfer_read %in2[%c0, %arg0, %c0], %cst {in_bounds = [true, true]} : memref<1x4096x128xf16>, vector<64x128xf16>
        %alloc_10 = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %17, %alloc_10[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>
        %alloc_11 = memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
        vector.transfer_write %19, %alloc_11[%c0, %c0] {in_bounds = [true, true]} : vector<64x128xf16>, memref<64x128xf16, #gpu.address_space<workgroup>>

        %21 = vector.transfer_read %alloc_10[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
        %22 = arith.addf %arg1, %21 : vector<64x128xf16>
        %38 = vector.transfer_read %alloc_11[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x128xf16, #gpu.address_space<workgroup>>, vector<64x128xf16>
        %39 = arith.addf %22, %38 : vector<64x128xf16>
        scf.yield %39 : vector<64x128xf16>
    }
    return
}

// The IR is expected not to be modified if there is no opportunities
// for reuse.
// CHECK-LABEL: @shared_no_reuse_scf_for
// CHECK-COUNT-2: memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
