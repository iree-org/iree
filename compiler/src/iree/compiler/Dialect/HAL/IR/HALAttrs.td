// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_IR_HAL_ATTRS
#define IREE_DIALECT_HAL_IR_HAL_ATTRS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Dialect/HAL/IR/HALInterfaces.td"
include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// General enums
//===----------------------------------------------------------------------===//

// Wrapper over base I32EnumAttr to set common fields for HAL enums.
class HAL_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}
class HAL_I32EnumAttr<string name, string description, string mnemonic,
                      list<I32EnumAttrCase> cases>
    : EnumAttr<HAL_Dialect, HAL_I32Enum<name, description, cases>, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def HAL_MemoryModel_Unified : I32EnumAttrCase<"Unified", 0>;
def HAL_MemoryModel_Discrete : I32EnumAttrCase<"Discrete", 1>;
def HAL_MemoryModelAttr :
    I32EnumAttr<"MemoryModel", "IREE HAL MemoryModel", [
      HAL_MemoryModel_Unified,
      HAL_MemoryModel_Discrete,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

def HAL_MemoryType_None             : I32BitEnumAttrCase<"None",             0x0000>;  // ?
def HAL_MemoryType_Optimal          : I32BitEnumAttrCase<"Optimal",          0x0001>;  // !
def HAL_MemoryType_HostVisible      : I32BitEnumAttrCase<"HostVisible",      0x0002>;  // h
def HAL_MemoryType_HostCoherent     : I32BitEnumAttrCase<"HostCoherent",     0x0004>;  // c
def HAL_MemoryType_HostCached       : I32BitEnumAttrCase<"HostCached",       0x0008>;  // C
def HAL_MemoryType_HostLocal        : I32BitEnumAttrCase<"HostLocal",        0x0046>;  // H
def HAL_MemoryType_DeviceVisible    : I32BitEnumAttrCase<"DeviceVisible",    0x0010>;  // d
def HAL_MemoryType_DeviceLocal      : I32BitEnumAttrCase<"DeviceLocal",      0x0030>;  // D
def HAL_MemoryTypeBitfieldAttr :
    I32BitEnumAttr<"MemoryTypeBitfield", "valid MemoryType", [
      HAL_MemoryType_None,
      HAL_MemoryType_Optimal,
      HAL_MemoryType_HostVisible,
      HAL_MemoryType_HostCoherent,
      HAL_MemoryType_HostCached,
      HAL_MemoryType_HostLocal,
      HAL_MemoryType_DeviceVisible,
      HAL_MemoryType_DeviceLocal,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_MemoryAccess_None                        : I32BitEnumAttrCase<"None",                         0x00000000>;
def HAL_MemoryAccess_Read                        : I32BitEnumAttrCase<"Read",                         0x00000001>;
def HAL_MemoryAccess_Write                       : I32BitEnumAttrCase<"Write",                        0x00000002>;
def HAL_MemoryAccess_Discard                     : I32BitEnumAttrCase<"Discard",                      0x00000004>;
def HAL_MemoryAccess_MayAlias                    : I32BitEnumAttrCase<"MayAlias",                     0x00000008>;
def HAL_MemoryAccess_Unaligned                   : I32BitEnumAttrCase<"Unaligned",                    0x00000010>;
def HAL_MemoryAccess_Any                         : I32BitEnumAttrCase<"Any",                          0x00000020>;
def HAL_MemoryAccessBitfieldAttr :
    I32BitEnumAttr<"MemoryAccessBitfield", "valid MemoryAccess", [
      HAL_MemoryAccess_None,
      HAL_MemoryAccess_Read,
      HAL_MemoryAccess_Write,
      HAL_MemoryAccess_Discard,
      HAL_MemoryAccess_MayAlias,
      HAL_MemoryAccess_Unaligned,
      HAL_MemoryAccess_Any,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_BufferUsage_None                         : I32BitEnumAttrCase<"None",                         0x00000000>;
def HAL_BufferUsage_TransferSource               : I32BitEnumAttrCase<"TransferSource",               0x00000001>;
def HAL_BufferUsage_TransferTarget               : I32BitEnumAttrCase<"TransferTarget",               0x00000002>;
def HAL_BufferUsage_Transfer                     : I32BitEnumAttrCase<"Transfer",                     0x00000003>;
def HAL_BufferUsage_DispatchIndirectParams       : I32BitEnumAttrCase<"DispatchIndirectParams",       0x00000100>;
def HAL_BufferUsage_DispatchUniformRead          : I32BitEnumAttrCase<"DispatchUniformRead",          0x00000200>;
def HAL_BufferUsage_DispatchStorageRead          : I32BitEnumAttrCase<"DispatchStorageRead",          0x00000400>;
def HAL_BufferUsage_DispatchStorageWrite         : I32BitEnumAttrCase<"DispatchStorageWrite",         0x00000800>;
def HAL_BufferUsage_DispatchStorage              : I32BitEnumAttrCase<"DispatchStorage",              0x00000C00>;
def HAL_BufferUsage_DispatchImageRead            : I32BitEnumAttrCase<"DispatchImageRead",            0x00001000>;
def HAL_BufferUsage_DispatchImageWrite           : I32BitEnumAttrCase<"DispatchImageWrite",           0x00002000>;
def HAL_BufferUsage_DispatchImage                : I32BitEnumAttrCase<"DispatchImage",                0x00003000>;
def HAL_BufferUsage_SharingExport                : I32BitEnumAttrCase<"SharingExport",                0x00010000>;
def HAL_BufferUsage_SharingReplicate             : I32BitEnumAttrCase<"SharingReplicate",             0x00020000>;
def HAL_BufferUsage_SharingConcurrent            : I32BitEnumAttrCase<"SharingConcurrent",            0x00040000>;
def HAL_BufferUsage_SharingImmutable             : I32BitEnumAttrCase<"SharingImmutable",             0x00080000>;
def HAL_BufferUsage_MappingScoped                : I32BitEnumAttrCase<"MappingScoped",                0x01000000>;
def HAL_BufferUsage_MappingPersistent            : I32BitEnumAttrCase<"MappingPersistent",            0x02000000>;
def HAL_BufferUsage_MappingOptional              : I32BitEnumAttrCase<"MappingOptional",              0x04000000>;
def HAL_BufferUsage_MappingAccessRandom          : I32BitEnumAttrCase<"MappingAccessRandom",          0x08000000>;
def HAL_BufferUsage_MappingAccessSequentialWrite : I32BitEnumAttrCase<"MappingAccessSequentialWrite", 0x10000000>;
def HAL_BufferUsage_Mapping                      : I32BitEnumAttrCase<"Mapping",                      0x09000000>;
def HAL_BufferUsageBitfieldAttr :
    I32BitEnumAttr<"BufferUsageBitfield", "valid BufferUsage", [
      HAL_BufferUsage_None,
      HAL_BufferUsage_TransferSource,
      HAL_BufferUsage_TransferTarget,
      HAL_BufferUsage_Transfer,
      HAL_BufferUsage_DispatchIndirectParams,
      HAL_BufferUsage_DispatchUniformRead,
      HAL_BufferUsage_DispatchStorageRead,
      HAL_BufferUsage_DispatchStorageWrite,
      HAL_BufferUsage_DispatchStorage,
      HAL_BufferUsage_DispatchImageRead,
      HAL_BufferUsage_DispatchImageWrite,
      HAL_BufferUsage_DispatchImage,
      HAL_BufferUsage_SharingExport,
      HAL_BufferUsage_SharingReplicate,
      HAL_BufferUsage_SharingConcurrent,
      HAL_BufferUsage_SharingImmutable,
      HAL_BufferUsage_MappingScoped,
      HAL_BufferUsage_MappingPersistent,
      HAL_BufferUsage_MappingOptional,
      HAL_BufferUsage_MappingAccessRandom,
      HAL_BufferUsage_MappingAccessSequentialWrite,
      HAL_BufferUsage_Mapping,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_CommandBufferMode_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_CommandBufferMode_OneShot : I32BitEnumAttrCase<"OneShot", 0x0001>;
def HAL_CommandBufferMode_Nested : I32BitEnumAttrCase<"Nested", 0x0002>;
def HAL_CommandBufferMode_AllowInlineExecution : I32BitEnumAttrCase<"AllowInlineExecution", 0x0010>;
def HAL_CommandBufferModeBitfieldAttr :
    I32BitEnumAttr<"CommandBufferModeBitfield", "valid CommandBufferMode", [
      HAL_CommandBufferMode_None,
      HAL_CommandBufferMode_OneShot,
      HAL_CommandBufferMode_Nested,
      HAL_CommandBufferMode_AllowInlineExecution,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_CommandCategory_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_CommandCategory_Transfer : I32BitEnumAttrCase<"Transfer", 0x0001>;
def HAL_CommandCategory_Dispatch : I32BitEnumAttrCase<"Dispatch", 0x0002>;
def HAL_CommandCategoryBitfieldAttr :
    I32BitEnumAttr<"CommandCategoryBitfield", "valid CommandCategory", [
      HAL_CommandCategory_None,
      HAL_CommandCategory_Transfer,
      HAL_CommandCategory_Dispatch,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_DescriptorType_UniformBuffer : I32EnumAttrCase<"UniformBuffer", 6, "uniform_buffer">;
def HAL_DescriptorType_StorageBuffer : I32EnumAttrCase<"StorageBuffer", 7, "storage_buffer">;
def HAL_DescriptorTypeAttr :
    HAL_I32EnumAttr<"DescriptorType", "valid DescriptorType", "descriptor_type", [
      HAL_DescriptorType_UniformBuffer,
      HAL_DescriptorType_StorageBuffer,
    ]>;

def HAL_DescriptorFlags_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_DescriptorFlags_ReadOnly : I32BitEnumAttrCase<"ReadOnly", 0x0001>;
def HAL_DescriptorFlagsAttr :
    I32BitEnumAttr<"DescriptorFlags", "valid Descriptor flags", [
      HAL_DescriptorFlags_None,
      HAL_DescriptorFlags_ReadOnly,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

def HAL_DescriptorSetLayoutFlags_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_DescriptorSetLayoutFlags_Indirect : I32BitEnumAttrCase<"Indirect", 0x0001>;
def HAL_DescriptorSetLayoutFlagsAttr :
    I32BitEnumAttr<"DescriptorSetLayoutFlags", "valid DescriptorSetLayout flags", [
      HAL_DescriptorSetLayoutFlags_None,
      HAL_DescriptorSetLayoutFlags_Indirect,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

def HAL_ExecutionStage_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_ExecutionStage_CommandIssue : I32BitEnumAttrCase<"CommandIssue", 0x0001>;
def HAL_ExecutionStage_CommandProcess : I32BitEnumAttrCase<"CommandProcess", 0x0002>;
def HAL_ExecutionStage_Dispatch : I32BitEnumAttrCase<"Dispatch", 0x0004>;
def HAL_ExecutionStage_Transfer : I32BitEnumAttrCase<"Transfer", 0x0008>;
def HAL_ExecutionStage_CommandRetire : I32BitEnumAttrCase<"CommandRetire", 0x0010>;
def HAL_ExecutionStage_Host : I32BitEnumAttrCase<"Host", 0x0020>;
def HAL_ExecutionStageBitfieldAttr :
    I32BitEnumAttr<"ExecutionStageBitfield", "valid ExecutionStage", [
      HAL_ExecutionStage_None,
      HAL_ExecutionStage_CommandIssue,
      HAL_ExecutionStage_CommandProcess,
      HAL_ExecutionStage_Dispatch,
      HAL_ExecutionStage_Transfer,
      HAL_ExecutionStage_CommandRetire,
      HAL_ExecutionStage_Host
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_ExecutionBarrierFlag_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_ExecutionBarrierFlag_Reserved : I32BitEnumAttrCase<"Reserved", 0x0001>;
def HAL_ExecutionBarrierFlagBitfieldAttr :
    I32BitEnumAttr<"ExecutionBarrierFlagBitfield", "valid ExecutionBarrierFlag", [
      HAL_ExecutionBarrierFlag_None,
      HAL_ExecutionBarrierFlag_Reserved,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_FenceFlag_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_FenceFlag_Reserved : I32BitEnumAttrCase<"Reserved", 0x0001>;
def HAL_FenceFlagBitfieldAttr :
    I32BitEnumAttr<"FenceFlagBitfield", "valid FenceFlag", [
      HAL_FenceFlag_None,
      HAL_FenceFlag_Reserved,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_AccessScope_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_AccessScope_IndirectCommandRead : I32BitEnumAttrCase<"IndirectCommandRead", 0x0001>;
def HAL_AccessScope_ConstantRead : I32BitEnumAttrCase<"ConstantRead", 0x0002>;
def HAL_AccessScope_DispatchRead : I32BitEnumAttrCase<"DispatchRead", 0x0004>;
def HAL_AccessScope_DispatchWrite : I32BitEnumAttrCase<"DispatchWrite", 0x0008>;
def HAL_AccessScope_TransferRead : I32BitEnumAttrCase<"TransferRead", 0x0010>;
def HAL_AccessScope_TransferWrite : I32BitEnumAttrCase<"TransferWrite", 0x0020>;
def HAL_AccessScope_HostRead : I32BitEnumAttrCase<"HostRead", 0x0040>;
def HAL_AccessScope_HostWrite : I32BitEnumAttrCase<"HostWrite", 0x0080>;
def HAL_AccessScope_MemoryRead : I32BitEnumAttrCase<"MemoryRead", 0x0100>;
def HAL_AccessScope_MemoryWrite : I32BitEnumAttrCase<"MemoryWrite", 0x0200>;
def HAL_AccessScopeBitfieldAttr :
    I32BitEnumAttr<"AccessScopeBitfield", "valid AccessScope", [
      HAL_AccessScope_None,
      HAL_AccessScope_IndirectCommandRead,
      HAL_AccessScope_ConstantRead,
      HAL_AccessScope_DispatchRead,
      HAL_AccessScope_DispatchWrite,
      HAL_AccessScope_TransferRead,
      HAL_AccessScope_TransferWrite,
      HAL_AccessScope_HostRead,
      HAL_AccessScope_HostWrite,
      HAL_AccessScope_MemoryRead,
      HAL_AccessScope_MemoryWrite
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_CallingConvention_Default : I32EnumAttrCase<"Default", 0>;
def HAL_CallingConvention_ParameterStruct : I32EnumAttrCase<"ParameterStruct", 1>;
def HAL_CallingConventionAttr :
    I32EnumAttr<
        "CallingConvention",
        "Calling conversions for linked functions",[
          HAL_CallingConvention_Default,
          HAL_CallingConvention_ParameterStruct,
        ]>{
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

//===----------------------------------------------------------------------===//
// #hal.collective<*>
//===----------------------------------------------------------------------===//

def HAL_CollectiveKind_AllGather : I32EnumAttrCase<"AllGather", 0, "all_gather">;
def HAL_CollectiveKind_AllReduce : I32EnumAttrCase<"AllReduce", 1, "all_reduce">;
def HAL_CollectiveKind_AllToAll : I32EnumAttrCase<"AllToAll", 2, "all_to_all">;
def HAL_CollectiveKind_Broadcast : I32EnumAttrCase<"Broadcast", 3, "broadcast">;
def HAL_CollectiveKind_Reduce : I32EnumAttrCase<"Reduce", 4, "reduce">;
def HAL_CollectiveKind_ReduceScatter : I32EnumAttrCase<"ReduceScatter", 5, "reduce_scatter">;
def HAL_CollectiveKind_Send : I32EnumAttrCase<"Send", 6, "send">;
def HAL_CollectiveKind_Recv : I32EnumAttrCase<"Recv", 7, "recv">;
def HAL_CollectiveKind_SendRecv: I32EnumAttrCase<"SendRecv", 8, "send_recv">;
def HAL_CollectiveKindAttr :
    I32EnumAttr<"CollectiveKind", "valid CollectiveKind", [
      HAL_CollectiveKind_AllGather,
      HAL_CollectiveKind_AllReduce,
      HAL_CollectiveKind_AllToAll,
      HAL_CollectiveKind_Broadcast,
      HAL_CollectiveKind_Reduce,
      HAL_CollectiveKind_ReduceScatter,
      HAL_CollectiveKind_Send,
      HAL_CollectiveKind_Recv,
      HAL_CollectiveKind_SendRecv,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

def HAL_CollectiveReductionOp_None : I32EnumAttrCase<"None", 0, "none">;
def HAL_CollectiveReductionOp_ReductionSum : I32EnumAttrCase<"ReductionSum", 1, "sum">;
def HAL_CollectiveReductionOp_ReductionProduct : I32EnumAttrCase<"ReductionProduct", 2, "product">;
def HAL_CollectiveReductionOp_ReductionMinimum : I32EnumAttrCase<"ReductionMinimum", 3, "minimum">;
def HAL_CollectiveReductionOp_ReductionMaximum : I32EnumAttrCase<"ReductionMaximum", 4, "maximum">;
def HAL_CollectiveReductionOp_ReductionAverage : I32EnumAttrCase<"ReductionAverage", 5, "average">;
def HAL_CollectiveReductionOpAttr :
    I32EnumAttr<"CollectiveReductionOp", "valid CollectiveReductionOp", [
      HAL_CollectiveReductionOp_None,
      HAL_CollectiveReductionOp_ReductionSum,
      HAL_CollectiveReductionOp_ReductionProduct,
      HAL_CollectiveReductionOp_ReductionMinimum,
      HAL_CollectiveReductionOp_ReductionMaximum,
      HAL_CollectiveReductionOp_ReductionAverage,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_CollectiveElementType_Sint8 : I32EnumAttrCase<"Sint8", 0, "si8">;
def HAL_CollectiveElementType_Uint8 : I32EnumAttrCase<"Uint8", 1, "ui8">;
def HAL_CollectiveElementType_Sint16 : I32EnumAttrCase<"Sint16", 2, "si16">;
def HAL_CollectiveElementType_Uint16 : I32EnumAttrCase<"Uint16", 3, "ui16">;
def HAL_CollectiveElementType_Sint32 : I32EnumAttrCase<"Sint32", 4, "si32">;
def HAL_CollectiveElementType_Uint32 : I32EnumAttrCase<"Uint32", 5, "ui32">;
def HAL_CollectiveElementType_Sint64 : I32EnumAttrCase<"Sint64", 6, "si64">;
def HAL_CollectiveElementType_Uint64 : I32EnumAttrCase<"Uint64", 7, "ui64">;
def HAL_CollectiveElementType_Float16 : I32EnumAttrCase<"Float16", 8, "f16">;
def HAL_CollectiveElementType_Float32 : I32EnumAttrCase<"Float32", 9, "f32">;
def HAL_CollectiveElementType_Float64 : I32EnumAttrCase<"Float64", 10, "f64">;
def HAL_CollectiveElementType_BFloat16 : I32EnumAttrCase<"BFloat16", 11, "bf16">;
def HAL_CollectiveElementTypeAttr :
    I32EnumAttr<"CollectiveElementType", "valid CollectiveElementType", [
      HAL_CollectiveElementType_Sint8,
      HAL_CollectiveElementType_Uint8,
      HAL_CollectiveElementType_Sint16,
      HAL_CollectiveElementType_Uint16,
      HAL_CollectiveElementType_Sint32,
      HAL_CollectiveElementType_Uint32,
      HAL_CollectiveElementType_Sint64,
      HAL_CollectiveElementType_Uint64,
      HAL_CollectiveElementType_Float16,
      HAL_CollectiveElementType_Float32,
      HAL_CollectiveElementType_Float64,
      HAL_CollectiveElementType_BFloat16,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

def HAL_CollectiveAttr :
    AttrDef<HAL_Dialect, "Collective", []> {
  let mnemonic = "collective";
  let summary = [{collective operation and specification}];
  let description = [{
    Specifies the collective operation to perform and any mode bits required.
  }];
  let parameters = (ins
    AttrParameter<"CollectiveKind", "">:$kind,
    OptionalParameter<"std::optional<CollectiveReductionOp>">:$reduction,
    AttrParameter<"CollectiveElementType", "">:$element_type
  );
  let assemblyFormat = [{
    `<` $kind (`with` $reduction^)? `:` $element_type `>`
  }];
  let extraClassDeclaration = [{
    // Returns the runtime encoding of the collective attribute.
    uint32_t getEncodedValue() const;
  }];
}

//===----------------------------------------------------------------------===//
// hal.descriptor_set.binding<*>
//===----------------------------------------------------------------------===//

def HAL_DescriptorSetBindingAttr :
    AttrDef<HAL_Dialect, "DescriptorSetBinding", []> {
  let mnemonic = "descriptor_set.binding";
  let summary = [{descriptor set binding specification}];
  let description = [{
    Specifies a single binding within a descriptor set layout.
  }];
  let parameters = (ins
    AttrParameter<"int64_t", "">:$ordinal,
    AttrParameter<"DescriptorType", "">:$type,
    OptionalParameter<"std::optional<DescriptorFlags>">:$flags
  );
  let assemblyFormat = [{
    `<` $ordinal `,` $type (`,` $flags^)? `>`
  }];
}

def HAL_DescriptorSetLayoutBindingArrayAttr :
    TypedArrayAttrBase<HAL_DescriptorSetBindingAttr,
                       "HAL descriptor set layout binding array attribute">;

//===----------------------------------------------------------------------===//
// hal.descriptor_set.layout<*>
//===----------------------------------------------------------------------===//

def HAL_DescriptorSetLayoutAttr :
    AttrDef<HAL_Dialect, "DescriptorSetLayout", []> {
  let mnemonic = "descriptor_set.layout";
  let summary = [{descriptor set layout specification}];
  let description = [{
    Specifies the layout information of a single set of descriptors used within
    an pipeline layout. Multiple of these sets may be used by a single entry
    point to allow for bindings with similar update frequencies to be grouped.
  }];
  let parameters = (ins
    AttrParameter<"int64_t", "">:$ordinal,
    ArrayRefParameter<"DescriptorSetBindingAttr", "">:$bindings,
    OptionalParameter<"std::optional<DescriptorSetLayoutFlags>">:$flags
  );
  let assemblyFormat = [{
    `<`
    $ordinal `,`
    `bindings` `=` `[` $bindings `]`
    (`,` `flags` `=` $flags^)?
    `>`
  }];
}

//===----------------------------------------------------------------------===//
// hal.pipeline.layout<*>
//===----------------------------------------------------------------------===//

def HAL_PipelineLayoutAttr :
    AttrDef<HAL_Dialect, "PipelineLayout", []> {
  let mnemonic = "pipeline.layout";
  let summary = [{executable entry point layout specification}];
  let description = [{
    Specifies the layout information used for interacting with executable
    functions. This allows host code to correctly map parameters to the
    lower-level target-specific argument passing behavior.
  }];
  let parameters = (ins
    AttrParameter<"int64_t", "">:$pushConstants,
    ArrayRefParameter<"DescriptorSetLayoutAttr", "">:$setLayouts
  );
  let assemblyFormat = [{
    `<`
    `push_constants` `=` $pushConstants `,`
    `sets` `=` `[` $setLayouts `]`
    `>`
  }];
}

//===----------------------------------------------------------------------===//
// hal.interface.binding<*>
//===----------------------------------------------------------------------===//

def HAL_InterfaceBindingAttr :
    AttrDef<HAL_Dialect, "InterfaceBinding", []> {
  let mnemonic = "interface.binding";
  let summary = [{interface binding specification}];
  let description = [{
    Specifies the descriptor set and binding ordinal of a particular layout
    binding.

    Example:
    ```mlir
    #hal.interface.binding<0, 1>
    ```
  }];
  let parameters = (ins
    AttrParameter<"int64_t", "">:$set,
    AttrParameter<"int64_t", "">:$binding
  );
  let assemblyFormat = [{
    `<` $set `,` $binding `>`
  }];
}

def HAL_InterfaceBindingArrayAttr :
    TypedArrayAttrBase<HAL_InterfaceBindingAttr,
                       "HAL binding array attribute">;

//===----------------------------------------------------------------------===//
// #hal.device.target<*>
//===----------------------------------------------------------------------===//

def HAL_DeviceTargetAttr :
    AttrDef<HAL_Dialect, "DeviceTarget"> {
  let mnemonic = "device.target";
  let summary = [{generic device target specification}];
  let description = [{
    Specifies the properties of a target runtime device.
    Target devices are specified with a canonical identifier matching those used
    by the runtime (such as `cpu`, `vulkan`, etc). Target devices may support
    several target executable formats specified with `#hal.executable.target`.
    An optional configuration dictionary allows for overriding backend defaults.

    Example:
    ```mlir
    #hal.device.target<"llvm-cpu", {
      executable_targets = [
        #hal.executable.target<"llvm-cpu", "embedded-elf-arm_32">,
        #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64">,
      ]
    }>
    ```
  }];
  let parameters = (ins
    AttrParameter<"StringAttr", "">:$deviceID,
    AttrParameter<"DictionaryAttr", "">:$configuration
  );
  let builders = [
    AttrBuilder<(ins "StringRef":$deviceID)>,
  ];

  let extraClassDeclaration = [{
    // Returns a symbol-compatible name that pseudo-uniquely identifies this
    // target. Callers must perform deduplication when required.
    std::string getSymbolNameFragment();

    // Returns true if there's an attribute with the given name in the
    // configuration dictionary.
    bool hasConfigurationAttr(StringRef name);

    // Returns zero or more executable targets that this device supports.
    SmallVector<IREE::HAL::ExecutableTargetAttr, 4> getExecutableTargets();

    // Returns a list of target devices that may be active for the given
    // operation. This will recursively walk parent operations until one with
    // the `hal.device.targets` attribute is found.
    static SmallVector<DeviceTargetAttr, 4> lookup(Operation *op);

    // Returns true if there is any UnitAttr with |name| in any device
    // configuration for the given |op|.
    static bool lookupConfigAttrAny(Operation *op, StringRef name);

    // Returns true if all device configurations found for the given |op| have
    // a UnitAttr with |name|.
    static bool lookupConfigAttrAll(Operation *op, StringRef name);

    // Returns the AND of boolean attributes of |name| in all device
    // configurations found for the given |op|.
    // Returns nullopt if any config does not have the key defined indicating
    // that it's not statically known/runtime dynamic.
    static std::optional<bool>
    lookupConfigAttrAnd(Operation *op, StringRef name);

    // Returns the OR of boolean attributes of |name| in all device
    // configurations found for the given |op|.
    // Returns nullopt if any config does not have the key defined indicating
    // that it's not statically known/runtime dynamic.
    static std::optional<bool>
    lookupConfigAttrOr(Operation *op, StringRef name);

    // Returns the range of integer attributes of |name| in all device
    // configurations found for the given |op|.
    // Returns nullopt if any config does not have the key defined indicating
    // that it's not statically known/runtime dynamic.
    static std::optional<StaticRange<APInt>>
    lookupConfigAttrRange(Operation *op, StringRef name);

    // Returns a list of all target executable configurations that may be
    // required for the given operation.
    static SmallVector<IREE::HAL::ExecutableTargetAttr, 4>
    lookupExecutableTargets(Operation *op);
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// #hal.executable.target<*>
//===----------------------------------------------------------------------===//

def HAL_ExecutableTargetAttr :
    AttrDef<HAL_Dialect, "ExecutableTarget"> {
  let mnemonic = "executable.target";
  let summary = [{generic executable target specification}];
  let description = [{
    Specifies how to compile an executable for a specific target backend.
    A backend is used to translate and serialize the executable into the final
    form passed to the runtime. The format of the executable is a
    target-specific value indicating the required runtime support to load the
    deployed artifact. An optionally provided configuration dictionary overrides
    backend-specific defaults.

    Example:
    ```mlir
      // Produce a system-native ELF for x86-64 systems using the LLVM backend:
      #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {
        triple = "x86_64-unknown-linux-elf",
        cpu = "host",
        cpu_features = "host",
        abi = "lp32",
        ...
      }>
    ```

    The same compilation backend may be used to translate executables for
    several different runtime devices. Likewise the same runtime device may use
    one of many different executable targets. Assume an N:M mapping between the
    two in all cases.
  }];

  let parameters = (ins
    AttrParameter<"StringAttr", "">:$backend,
    AttrParameter<"StringAttr", "">:$format,
    AttrParameter<"DictionaryAttr", "">:$configuration
  );

  let builders = [
    AttrBuilder<(ins "StringRef":$backend, "StringRef":$format)>,
  ];

  let extraClassDeclaration = [{
    // Returns a symbol-compatible name that pseudo-uniquely identifies this
    // target. Callers must perform deduplication when required.
    std::string getSymbolNameFragment() const;

    // Returns true if there's an attribute with the given name in the
    // configuration dictionary.
    bool hasConfigurationAttr(StringRef name);

    // Returns true if this attribute is a generic version of |specificAttr|.
    // A more generic version will match with many specific versions.
    bool isGenericOf(IREE::HAL::ExecutableTargetAttr specificAttr);

    // Returns the executable target configuration for the given operation.
    // This will recursively walk parent operations until one with the
    // `hal.executable.target` attribute is found or a `hal.executable.variant`
    // specifies a value. Returns nullptr if no target specification can be found.
    static IREE::HAL::ExecutableTargetAttr lookup(Operation *op);
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// #hal.executable.object<*>
//===----------------------------------------------------------------------===//

def HAL_ExecutableObjectAttr : AttrDef<HAL_Dialect, "ExecutableObject"> {
  let mnemonic = "executable.object";
  let summary = [{object file reference}];
  let description = [{
    Defines an object file that can be linked into executables.
    Today this is only supported for external file references with paths the
    compiler can successfully resolve from its current working directory.
    Inlined data can optionally be provided to avoid the need for file system
    access and ensure the data source is attached to the IR as it makes its way
    through multiple compiler stages or reproducers.

    Future revisions may change this to an interface that allows both internal
    and external resources to define the object contents. Linking needs to be
    updated to support various object compositions and certain backends may
    require additional infrastructure support.

    In the long term the goal is to allow combinations of declared objects and
    generated code in order to give control of linking behavior to frontends.
    Instead of needing global command line flags to link in additional blobs
    the frontend can emit executables with the dependencies already defined per
    variant without needing to reach into the IREE compiler code.

    Example:
    ```mlir
    #hal.executable.object<{path = "some/file.obj"}>
    #hal.executable.object<{
      path = "some/embedded/file.obj",
      data = dense<[...]> : vector<2048xi8>
    }>
    ```
  }];

  let parameters = (ins
    AttrParameter<"StringAttr", "">:$path,
    OptionalParameter<"DenseIntElementsAttr", "">:$data
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Returns a list of all objects with a path matching one of the provided
    // file extensions.
    static void filterObjects(
        ArrayAttr objectAttrs, ArrayRef<StringRef> extensions,
        SmallVectorImpl<ExecutableObjectAttr> &filteredAttrs);

    // Returns the absolute path of the referenced object file if it exists.
    FailureOr<std::string> getAbsolutePath();

    // Returns the contents of the object file or None if loading failed.
    // TODO(benvanik): better return type to support mapping/etc? eh
    std::optional<std::string> loadData();
  }];
}

def HAL_ExecutableObjectArrayAttr :
    TypedArrayAttrBase<HAL_ExecutableObjectAttr,
                       "HAL executable object references">;

//===----------------------------------------------------------------------===//
// #hal.executable.objects<*>
//===----------------------------------------------------------------------===//

def HAL_ExecutableObjectsAttr : AttrDef<HAL_Dialect, "ExecutableObjects"> {
  let mnemonic = "executable.objects";
  let summary = [{target-specific object file references}];
  let description = [{
    A dictionary mapping executable target specifications to a list of objects.
    This is used to allow layers of the stack that support multi-targeting to
    specify information used during lowering into each particular target.

    The key attributes are matched against each target variant based on the
    backend and format as well as any configuration data provided. When
    comparing the configuration only fields present in both the key and
    target variant will be checked and must match. This allows specification of
    generic sets ("all x86_64 targets get these objects") as well as specific
    ones ("only x86_64 targets with vector_size = 64 get these objects").

    Example:
    ```mlir
    #hal.executable.objects<{
      #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64"> = [
        #hal.executable.object<{path = "some/file_arm_64.obj"}>
      ],
      #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64"> = [
        #hal.executable.object<{path = "some/file_x86_64.obj"}>
      ]
    }>
    ```
  }];

  let parameters = (ins
    AttrParameter<"ArrayAttr", "">:$targets,
    AttrParameter<"ArrayAttr", "">:$targetObjects
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Returns the objects specified for the given generic target.
    std::optional<ArrayAttr> getApplicableObjects(
        IREE::HAL::ExecutableTargetAttr specificTargetAttr);
  }];
}

//===----------------------------------------------------------------------===//
// #hal.affinity.queue<*>
//===----------------------------------------------------------------------===//

def HAL_AffinityQueueAttr : AttrDef<HAL_Dialect, "AffinityQueue", [
  DeclareAttrInterfaceMethods<Stream_AffinityAttr, [
    "isExecutableWith",
    "joinOR",
    "joinAND",
  ]>,
]> {
  let mnemonic = "affinity.queue";
  let summary = [{specifies a set of allowed queues for an operation}];
  let description = [{
    WIP; see [#10765](https://github.com/openxla/iree/issues/10765).
    This may change in the future to either be a nested attribute on a larger
    affinity struct or be defined by an implementation of the affinity attr
    interface. For now this allows higher levels of the stack to specify
    queues such that the stream dialect can understand them and they can be
    lowered into the HAL dialect.

    Specifies that an annotated operation or scope is only allowed to execute on
    the set of queues (0-64) provided. Operations will not run on other queues.

    Example:
    ```mlir
    // any queue
    #hal.affinity.queue<*>
    // queues 4 and 5
    #hal.affinity.queue<[4, 5]>
    ```
  }];

  let parameters = (ins
    AttrParameter<"int64_t", "">:$mask
  );

  let hasCustomAssemblyFormat = 1;
}

#endif  // IREE_DIALECT_HAL_IR_HAL_ATTRS
