// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_STREAM_TYPES
#define IREE_DIALECT_STREAM_TYPES

include "iree/compiler/Dialect/Stream/IR/StreamAttrs.td"
include "iree/compiler/Dialect/Stream/IR/StreamBase.td"
include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"

//===----------------------------------------------------------------------===//
// Stream synchronization types
//===----------------------------------------------------------------------===//

def Stream_Timepoint : Stream_TypeDef<"Timepoint", [
  Util_GlobalType,
]> {
  let mnemonic = "timepoint";

  let summary = [{A timepoint indicating execution availability.}];
  let description = [{
    Represents a point in the execution timeline that when resolved indicates
    that all of the execution prior to this timepoint has completed and the
    results of the execution are available for use. This includes transitive
    dependencies as well; if timepoint B is dependent on timepoint A then when
    B is available so too must be A.
  }];

  // TODO(benvanik): track affinity so we know where timepoints come from.
  let parameters = (ins);
}

def Stream_TimepointAttr : Stream_AttrDef<"Timepoint",
                             [TypedAttrInterface]> {
  let mnemonic = "timepoint";
  let summary = [{An immediately-resolved timepoint.}];
  let description = [{}];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let valueType = Stream_Timepoint;
  let constBuilderCall = [{
    IREE::Stream::TimepointAttr::get(
        $_builder.getContext(),
        IREE::Stream::TimepointType::get($_builder.getContext()));
  }];
  let hasCustomAssemblyFormat = 1;
}

def Stream_TestFence : Stream_TypeDef<"TestFence", []> {
  let mnemonic = "test.fence";

  let summary = [{Test-only fence type for timeline-aware op testing.}];
  let description = [{
    A test-only type that behaves like hal.fence for testing TimelineAwareOpInterface
    without introducing HAL dependencies in Stream unit tests.
  }];

  let parameters = (ins);

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Stream resource value types
//===----------------------------------------------------------------------===//

// Indicates a value whose lifetime is not yet analyzed.
def Stream_Lifetime_Unknown : I32EnumAttrCase<"Unknown", 0, "*">;
// An externally managed value.
def Stream_Lifetime_External : I32EnumAttrCase<"External", 1, "external">;
// A staging buffer for uploads/downloads.
def Stream_Lifetime_Staging : I32EnumAttrCase<"Staging", 2, "staging">;
// A short-lived value that is used across streams.
def Stream_Lifetime_Transient : I32EnumAttrCase<"Transient", 3, "transient">;
// A long-lived value that persists across streams.
def Stream_Lifetime_Variable : I32EnumAttrCase<"Variable", 4, "variable">;
// An immutable value that persist for the duration of the program.
def Stream_Lifetime_Constant : I32EnumAttrCase<"Constant", 5, "constant">;
def Stream_LifetimeAttr :
    I32EnumAttr<"Lifetime", "IREE Stream value lifetime", [
      Stream_Lifetime_Unknown,
      Stream_Lifetime_External,
      Stream_Lifetime_Staging,
      Stream_Lifetime_Transient,
      Stream_Lifetime_Variable,
      Stream_Lifetime_Constant,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";
}

def Stream_AnyResource : Type<
    CPred<"::llvm::isa<::mlir::iree_compiler::IREE::Stream::ResourceType>($_self)">,
    "any stream-compatible type">;

// TODO(benvanik): support other types; the interface may be enough.
def Stream_TransferType : AnyTypeOf<[
  Stream_AnyResource,
  Stream_Tensor,
  Stream_PrimitiveType,
]>;

def Stream_Channel : Stream_TypeDef<"Channel", [
  Util_ReferenceType
]> {
  let mnemonic = "channel";

  let summary = [{A collective communication channel.}];
  let description = [{
    Represents a single participant in a collective clique. Multiple channels
    may exist within the same program to allow for partial operations or
    hierarchical operations.

    In programs that model SPMD behavior internally channels can be created or
    provided by hosting applications. For example, the program could expose a
    `@set_channels(!util.list<!stream.channel>)` method that stores the channels
    in globals for use throughout the program allowing for
    application-controlled channel configuration.
  }];
}

def Stream_Resource : Stream_TypeDef<"Resource", [
  Stream_AffinityType,
  Util_ReferenceType,
  Util_SizeAwareType,
  DeclareTypeInterfaceMethods<Util_GlobalType, [
    "isAccessStorageCompatible",
  ]>,
  DeclareTypeInterfaceMethods<Util_InferTypeSize, [
    "inferSizeFromValue",
  ]>,
  DeclareTypeInterfaceMethods<Util_SubrangeType, [
    "createSubrangeOp",
  ]>,
]> {
  let mnemonic = "resource";

  let summary = [{A managed resource.}];
  let description = [{
    Stream external values represent asynchronously-available and sequenced
    values that are owned and managed by external code - such as those passed in
    or out of the program entry points. Though external values are managed
    during an invocation the same as other stream values the visibility into
    them does not extend outside of the invocation they are provided to.

    Stream values are not usable directly outside of a stream execution or
    transfer operation. If the contents of the value are needed for host access
    or access from a device without memory compatibility they must first
    be transferred via a transfer operation (e.g. `stream.async.transfer`).
  }];

  let parameters = (ins
    "IREE::Stream::Lifetime":$lifetime
  );

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, IREE::Stream::Lifetime::Unknown);
    }]>,
    TypeBuilderWithInferredContext<(ins
      "IREE::Stream::LifetimeAttr":$lifetime
    ), [{
      return $_get(lifetime.getContext(), lifetime.getValue());
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;
}

def Stream_ResourceLifetimeUnknown : CPred<[{
  ::llvm::cast<::mlir::iree_compiler::IREE::Stream::ResourceType>($_self).getLifetime() ==
      ::mlir::iree_compiler::IREE::Stream::Lifetime::Unknown
}]>;
def Stream_ResourceLifetimeExternal : CPred<[{
  ::llvm::cast<::mlir::iree_compiler::IREE::Stream::ResourceType>($_self).getLifetime() ==
      ::mlir::iree_compiler::IREE::Stream::Lifetime::External
}]>;
def Stream_ResourceLifetimeStaging : CPred<[{
  ::llvm::cast<::mlir::iree_compiler::IREE::Stream::ResourceType>($_self).getLifetime() ==
      ::mlir::iree_compiler::IREE::Stream::Lifetime::Staging
}]>;
def Stream_ResourceLifetimeTransient : CPred<[{
  ::llvm::cast<::mlir::iree_compiler::IREE::Stream::ResourceType>($_self).getLifetime() ==
      ::mlir::iree_compiler::IREE::Stream::Lifetime::Transient
}]>;
def Stream_ResourceLifetimeVariable : CPred<[{
  ::llvm::cast<::mlir::iree_compiler::IREE::Stream::ResourceType>($_self).getLifetime() ==
      ::mlir::iree_compiler::IREE::Stream::Lifetime::Variable
}]>;
def Stream_ResourceLifetimeConstant : CPred<[{
  ::llvm::cast<::mlir::iree_compiler::IREE::Stream::ResourceType>($_self).getLifetime() ==
      ::mlir::iree_compiler::IREE::Stream::Lifetime::Constant
}]>;

def Stream_UnknownResource : DialectType<Stream_Dialect, And<[
  CPred<"::llvm::isa<::mlir::iree_compiler::IREE::Stream::ResourceType>($_self)">,
  Stream_ResourceLifetimeUnknown,
]>, "resource"> {
  let description = [{
    A stream resource that has not yet had its lifetime calculated.
  }];
}

def Stream_ExternalResource : DialectType<Stream_Dialect, And<[
  CPred<"::llvm::isa<::mlir::iree_compiler::IREE::Stream::ResourceType>($_self)">,
  Stream_ResourceLifetimeExternal,
]>, "external resource"> {
  let description = [{
    Stream external values represent asynchronously-available and sequenced
    values that are owned and managed by external code - such as those passed in
    or out of the program entry points. Though external values are managed
    during an invocation the same as other stream values the visibility into
    them does not extend outside of the invocation they are provided to.

    Stream values are not usable directly outside of a stream execution or
    transfer operation. If the contents of the value are needed they must first
    be transferred via `stream.transfer` - which may incur a copy.
  }];
}

def Stream_StagingResource : DialectType<Stream_Dialect, And<[
  CPred<"::llvm::isa<::mlir::iree_compiler::IREE::Stream::ResourceType>($_self)">,
  Stream_ResourceLifetimeStaging,
]>, "staging resource"> {
  let description = [{
    Stream upload/download staging resource. These are used outside of streams
    and then transferred to other stream resources such as variables or
    transients for use inside of streams. Dispatches and several other
    operations cannot directly operate on these resources.
  }];
}

def Stream_TransientResource : DialectType<Stream_Dialect, And<[
  CPred<"::llvm::isa<::mlir::iree_compiler::IREE::Stream::ResourceType>($_self)">,
  Stream_ResourceLifetimeTransient,
]>, "transient resource"> {
  let description = [{
    Stream transients represent asynchronously-available and sequenced values
    that have a short lifetime - often only passed between stream executions.
    It is expected that transient values are not stored in global state and have
    minimal lifetime as they may be heavily pooled or suballocated.

    Stream values are not usable directly outside of a stream execution or
    transfer operation. If the contents of the value are needed they must first
    be transferred via `stream.transfer` - which may incur a copy.
  }];
}

def Stream_VariableResource : DialectType<Stream_Dialect, And<[
  CPred<"::llvm::isa<::mlir::iree_compiler::IREE::Stream::ResourceType>($_self)">,
  Stream_ResourceLifetimeVariable,
]>, "variable resource"> {
  let description = [{
    Stream variables represent asynchronously-available and sequenced values
    that have a long lifetime relative to the work being performed on them.
    These variables are often stored in global state and may live for the entire
    duration of the program.

    Stream values are not usable directly outside of a stream execution or
    transfer operation. If the contents of the value are needed they must first
    be transferred via `stream.transfer` - which may incur a copy.
  }];
}

def Stream_ConstantResource : DialectType<Stream_Dialect, And<[
  CPred<"::llvm::isa<mlir::iree_compiler::IREE::Stream::ResourceType>($_self)">,
  Stream_ResourceLifetimeConstant,
]>, "constant resource"> {
  let description = [{
    Stream constants are immutable values that are available for the lifetime of
    the program once initialized.
  }];
}

def Stream_AnyStreamResource : AnyTypeOf<[
  Stream_UnknownResource,
  Stream_ExternalResource,
  Stream_TransientResource,
  Stream_VariableResource,
  Stream_ConstantResource,
]>;

//===----------------------------------------------------------------------===//
// File resources
//===----------------------------------------------------------------------===//

def Stream_File : Stream_TypeDef<"File", []> {
  let mnemonic = "file";

  let summary = [{A file handle used for I/O operations.}];
  let description = [{
    A file handle that can be asynchronously read and written into/from
    stream resources.
  }];

  let parameters = (ins);

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Executable bindings
//===----------------------------------------------------------------------===//

def Stream_Binding : Stream_TypeDef<"Binding", []> {
  let mnemonic = "binding";

  let summary = [{A managed resource binding into an executable scope.}];
  let description = [{
    A resource binding available within an executable dispatch function.
    The bindings map 1:1 with the resources bound during dispatch operations.
  }];

  // TODO(benvanik): carry lifetime like resources.
  let parameters = (ins);

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt);
    }]>,
  ];
}

def Stream_AnyBinding : AnyTypeOf<[
  Stream_Binding,
]>;

#endif  // IREE_DIALECT_STREAM_TYPES
