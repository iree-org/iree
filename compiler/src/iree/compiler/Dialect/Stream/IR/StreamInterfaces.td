// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_STREAM_INTERFACES
#define IREE_DIALECT_STREAM_INTERFACES

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

//===----------------------------------------------------------------------===//
// IREE::Stream::AffinityAttrInterface
//===----------------------------------------------------------------------===//

def Stream_AffinityAttr : AttrInterface<"AffinityAttr"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";

  let summary = [{defines execution context affinity}];
  let description = [{
    WIP; see https://github.com/openxla/iree/issues/10765.

    TBD. The intent is that this can specify host, device, and queue affinity.
    Scopes can be annotated with an affinity to ensure execution within happens
    in a particular location. Arrays of affinities or wildcard specifiers will
    allow for refinement ("do it on this device but auto select a queue"). It
    will also allow us to indicate host affinity such that device<->device and
    host<->device can be identified in the IR structure. Today all affinities
    are no-op'ed and assumed to be 'current device'.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the resource configuration specified by the placement.
        If the affinity is a composite of several placements the intersection
        will be taken. Returns an empty attribute if no resource configuration
        is specified.
      }],
      /*retTy=*/"IREE::Stream::ResourceConfigAttr",
      /*methodName=*/"getResourceConfigAttr",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if executable operations with this affinity specified may
        be executed together with executable operations with another affinity.
        When true operations will be submitted to the same execution queues.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isExecutableWith",
      /*args=*/(ins "IREE::Stream::AffinityAttr":$other),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return IREE::Stream::AffinityAttr::areCompatible($_attr, other);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns an affinity describing the union with |other| constraints.
        The returned affinity specifies that a particular operation may execute
        on _either_ of the source affinities.
      }],
      /*retTy=*/"IREE::Stream::AffinityAttr",
      /*methodName=*/"joinOR",
      /*args=*/(ins "IREE::Stream::AffinityAttr":$other),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(IREE::Stream::AffinityAttr::canExecuteTogether(*this, other) &&
               "divergent affinities not yet implemented");
        return *this;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns an affinity describing the intersection with |other|
        constraints. The returned affinity specifies that a particular operation
        may execute on _both_ of the source affinities.
      }],
      /*retTy=*/"IREE::Stream::AffinityAttr",
      /*methodName=*/"joinAND",
      /*args=*/(ins "IREE::Stream::AffinityAttr":$other),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(IREE::Stream::AffinityAttr::canExecuteTogether(*this, other) &&
               "divergent affinities not yet implemented");
        return *this;
      }]
    >,
  ];

  let extraClassDeclaration = [{
    // Returns an affinity active for the given operation.
    // This will recursively walk parent operations until one with the
    // `stream.affinity` attribute is found.
    static AffinityAttr lookup(Operation *op);

    // TODO(benvanik): replace with more fine-grained compatibility checks.
    // "Compatible" can mean a lot of things: are they cache-coherent, are they
    // a shared address space, are they able to perform collective operations,
    // etc. May be able to represent it with a bitfield or a dedicated
    // compatibility struct result.
    // Returns true if |desiredAffinity| (if any) is compatible with
    // |requiredAffinity|.
    static bool areCompatible(AffinityAttr desiredAffinity,
                              AffinityAttr requiredAffinity);

    // Returns true if |lhs| and |rhs| indicate that their operations can
    // execute together on the same execution queue.
    static bool canExecuteTogether(AffinityAttr lhs, AffinityAttr rhs);
  }];
}

//===----------------------------------------------------------------------===//
// IREE::Stream::AffinityOpInterface
//===----------------------------------------------------------------------===//

def Stream_AffinityOp : OpInterface<"AffinityOpInterface"> {
  let description = [{
    TBD. Used to denote a stream affinity for ops and specify the kind of
    environment the ops are expected run in.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the stream affinity for the op, indicating where it should run.
      }],
      /*retTy=*/"IREE::Stream::AffinityAttr",
      /*methodName=*/"getAffinity",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_self->getAttr("affinity").template dyn_cast_or_null<IREE::Stream::AffinityAttr>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets the stream affinity for the op, indicating where it should run.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setAffinity",
      /*args=*/(ins "IREE::Stream::AffinityAttr":$value),
      /*methodBody=*/[{
        if (value) $_self->setAttr("affinity", value);
        else $_self->removeAttr("affinity");
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Stream::StreamableOpInterface
//===----------------------------------------------------------------------===//

def Stream_StreamableOp : OpInterface<"StreamableOpInterface"> {
  let description = [{
    Interface for ops that can be asynchronous executed in a streaming context.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the op is slicing out/in memory instead of real work.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isMetadata",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the op should be cloned into consumer streams.
        These ops should be cheaper to recompute than to transfer their contents
        across streams (such as splats).
      }],
      /*retTy=*/"bool",
      /*methodName=*/"preferCloneToConsumers",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Stream::AsyncAccessOpInterface
//===----------------------------------------------------------------------===//

def Stream_AsyncAccessOp : OpInterface<"AsyncAccessOpInterface"> {
  let description = [{
    Interface for stream.async.* ops that access subviews of resources.
    This allows for some basic analysis and is only valid prior to allocation.
    Not all ops need to implement this as such analysis is conservative and will
    bail whenever resources are used by unknown ops.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Queries the resource ranges accessed by the operation.
        Each access should be represented: for example a read of an operand
        and a complete discard and write as the result should be two separate
        accesses indicating that the existing contents are read and the new
        contents are written.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getAsyncAccessRanges",
      /*args=*/(ins "SmallVectorImpl<AsyncAccessRange> &":$ranges)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Stream::SubviewEffectOpInterface
//===----------------------------------------------------------------------===//

def Stream_SubviewEffectOp : OpInterface<"SubviewEffectOpInterface"> {
  let description = [{
    Interface for ops that operate on subviews of resources used to query the
    memory effects for subviews on operands.
  }];

  let methods = [
    // TODO(benvanik): get memory effect + range of an operand. Unlike the
    // AsyncAccessOpInterface this will not be able to provide the access
    // range end as we don't track that (we assume it's been taken care of
    // earlier on in lowering).
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Stream::BuiltinOpInterface
//===----------------------------------------------------------------------===//

def Stream_BuiltinOp : OpInterface<"BuiltinOpInterface"> {
  let description = [{
    Interface for ops that can be asynchronous executed in a streaming context.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Merges the builtin module for the op into the target user module. By
        doing a top level module merge the builtin can provide host functions
        and multiple executables as needed.

        This should not specialize the executable yet as callers currently
        assume all builtin ops of the same type will have the same module IR.
        If we want to specialize the executables based on usage we'll need a
        more complex interface.

        Fails if there's a name conflict; we have a __ prefix and things outside
        the compiler shouldn't use it.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"mergeBuiltinModule",
      /*args=*/(ins "Operation *":$targetOp, "OpBuilder &":$targetBuilder)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Converts a stream.builtin.* op to its stream.async.* form. Any number of
        ops may be inserted such as transfers and resource operations.

        This is called in the stream.async.* phase of lowering and before
        resource refinement or copy-on-write materialization has happened; this
        enables the conversion to create new tensors as `!stream.resource<*>`.

        The original op must remain valid after the call completes; the caller
        of this needs to erase it themselves.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"convertBuiltinOp",
      /*args=*/(ins "OpBuilder &":$builder)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Stream::TimelineOpInterface
//===----------------------------------------------------------------------===//

def Stream_TimelineOp : OpInterface<"TimelineOpInterface"> {
  let description = [{
    Interface for ops that operate in an ordered sequence defined by timepoints.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns zero or more timepoints consumed by this timeline operation
        indicating the asynchronous operations that must complete before it can
        perform its operation.
      }],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"getAwaitTimepoints",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the timepoint produced by this timeline operation indicating the
        asynchronous completion of the operation.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getResultTimepoint",
      /*args=*/(ins)
    >,
  ];
}

#endif  // IREE_DIALECT_STREAM_INTERFACES
