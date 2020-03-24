// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK:spv.module Logical GLSL450
  // CHECK-DAG: spv.globalVariable [[NUMWORKGROUPSVAR:@.*]] built_in("NumWorkgroups") : !spv.ptr<vector<3xi32>, Input>
  // CHECK-DAG: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.func [[FN:@simple_load_store]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<504 x i32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: {{spirv|spv}}.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}
  func @simple_load_store(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = [32 : index, 1 : index, 1 : index], iree.ordinal = 0 : i32} {
    // CHECK: [[EXTENTX:%.*]] = spv.constant 42 : i32
    // CHECK: [[EXTENTY:%.*]] = spv.constant 12 : i32
    // CHECK: [[EXTENTZ:%.*]] = spv.constant 1 : i32

    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDZ:%.*]] = spv.CompositeExtract [[GLOBALID]][2 : i32]
    // CHECK: [[NUMWORKGROUPSPTR:%.*]] = spv._address_of [[NUMWORKGROUPSVAR]]
    // CHECK: [[NUMWORKGROUPS:%.*]] = spv.Load "Input" [[NUMWORKGROUPSPTR]]
    // CHECK: [[NUMWORKGROUPSZ:%.*]] = spv.CompositeExtract [[NUMWORKGROUPS]][2 : i32]
    // CHECK: [[WORKGROUPSIZEZ:%.*]] = spv.constant 1 : i32
    // CHECK: [[STEPZ:%.*]] = spv.IMul [[NUMWORKGROUPSZ]], [[WORKGROUPSIZEZ]] : i32
    // CHECK: spv.loop {
    // CHECK: spv.Branch ^[[HEADERZ:.*]]([[GLOBALIDZ]] : i32)
    // CHECK: ^[[HEADERZ]]([[IVZ:%.*]]: i32):
    // CHECK: [[CONDZ:%.*]] = spv.SLessThan [[IVZ]], [[EXTENTZ]] : i32
    // CHECK: spv.BranchConditional [[CONDZ]], ^[[BODYZ:.*]], ^[[MERGEZ:.*]]
    // CHECK: ^[[BODYZ]]:

    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[NUMWORKGROUPSPTR:%.*]] = spv._address_of [[NUMWORKGROUPSVAR]]
    // CHECK: [[NUMWORKGROUPS:%.*]] = spv.Load "Input" [[NUMWORKGROUPSPTR]]
    // CHECK: [[NUMWORKGROUPSY:%.*]] = spv.CompositeExtract [[NUMWORKGROUPS]][1 : i32]
    // CHECK: [[WORKGROUPSIZEY:%.*]] = spv.constant 1 : i32
    // CHECK: [[STEPY:%.*]] = spv.IMul [[NUMWORKGROUPSY]], [[WORKGROUPSIZEY]] : i32
    // CHECK: spv.loop {
    // CHECK: spv.Branch ^[[HEADERY:.*]]([[GLOBALIDY]] : i32)
    // CHECK: ^[[HEADERY]]([[IVY:%.*]]: i32):
    // CHECK: [[CONDY:%.*]] = spv.SLessThan [[IVY]], [[EXTENTY]] : i32
    // CHECK: spv.BranchConditional [[CONDY]], ^[[BODYY:.*]], ^[[MERGEY:.*]]
    // CHECK: ^[[BODYY]]:

    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[NUMWORKGROUPSPTR:%.*]] = spv._address_of [[NUMWORKGROUPSVAR]]
    // CHECK: [[NUMWORKGROUPS:%.*]] = spv.Load "Input" [[NUMWORKGROUPSPTR]]
    // CHECK: [[NUMWORKGROUPSX:%.*]] = spv.CompositeExtract [[NUMWORKGROUPS]][0 : i32]
    // CHECK: [[WORKGROUPSIZEX:%.*]] = spv.constant 32 : i32
    // CHECK: [[STEPX:%.*]] = spv.IMul [[NUMWORKGROUPSX]], [[WORKGROUPSIZEX]] : i32
    // CHECK: spv.loop {
    // CHECK: spv.Branch ^[[HEADERX:.*]]([[GLOBALIDX]] : i32)
    // CHECK: ^[[HEADERX]]([[IVX:%.*]]: i32):
    // CHECK: [[CONDX:%.*]] = spv.SLessThan [[IVX]], [[EXTENTX]] : i32
    // CHECK: spv.BranchConditional [[CONDX]], ^[[BODYX:.*]], ^[[MERGEX:.*]]
    // CHECK: ^[[BODYX]]:

    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[STRIDE1:%.*]] = spv.constant 42 : i32
    // CHECK: [[TEMP1:%.*]] = spv.IMul [[IVY]], [[STRIDE1]] : i32
    // CHECK: [[INDEX1:%.*]] = spv.IAdd [[TEMP1]], [[IVX]] : i32
    // CHECK: [[ARG1LOADPTR:%.*]] = spv.AccessChain [[ARG1]]{{\[}}[[ZERO1]], [[INDEX1]]{{\]}}
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG1LOADPTR]]

    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[STRIDE2:%.*]] = spv.constant 42 : i32
    // CHECK: [[TEMP2:%.*]] = spv.IMul [[IVY]], [[STRIDE2]] : i32
    // CHECK: [[INDEX2:%.*]] = spv.IAdd [[TEMP2]], [[IVX]] : i32
    // CHECK: [[ARG2STOREPTR:%.*]] = spv.AccessChain [[ARG2]]{{\[}}[[ZERO2]], [[INDEX2]]{{\]}}
    // CHECK: spv.Store "StorageBuffer" [[ARG2STOREPTR]], [[VAL]]

    // CHECK: [[NEXTIVX:%.*]] = spv.IAdd [[IVX]], [[STEPX]] : i32
    // CHECK: spv.Branch ^[[HEADERX]]([[NEXTIVX]] : i32)
    // CHECK: ^[[MERGEX]]:
    // CHECK: spv._merge
    // CHECK: }

    // CHECK: [[NEXTIVY:%.*]] = spv.IAdd [[IVY]], [[STEPY]] : i32
    // CHECK: spv.Branch ^[[HEADERY]]([[NEXTIVY]] : i32)
    // CHECK: ^[[MERGEY]]:
    // CHECK: spv._merge
    // CHECK: }

    // CHECK: [[NEXTIVZ:%.*]] = spv.IAdd [[IVZ]], [[STEPZ]] : i32
    // CHECK: spv.Branch ^[[HEADERZ]]([[NEXTIVZ]] : i32)
    // CHECK: ^[[MERGEZ]]:
    // CHECK: spv._merge
    // CHECK: }
    // CHECK: spv.Return
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = "xla_hlo.copy"(%0) : (tensor<12x42xi32>) -> tensor<12x42xi32>
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    return
  }
}
