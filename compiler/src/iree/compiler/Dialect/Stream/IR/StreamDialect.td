// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_STREAM_DIALECT
#define IREE_DIALECT_STREAM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE Asynchronous Streaming Execution Dialect
//===----------------------------------------------------------------------===//

def Stream_Dialect : Dialect {
  let name = "stream";
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";

  let summary = [{
    A dialect designed to model execution partitioning and scheduling.
  }];
  let description = [{
    The stream dialect is designed to take tensor programs and convert them to
    explicitly scheduled asynchronous programs. This includes placing ops on
    specific targets, partitioning the work between the targets, scheduling the
    work for concurrency, and encoding tensors into target-specific resources.

    ```
    +--------+    +----------+    +-------+
    | flow.* | -> | stream.* | -> | hal.* |
    +--------+    +----------+    +-------+
    ```

    This sits in-between the `flow` and `hal` dialects.

    * `flow` models tensor programs by separating work into dispatchable
      functions in order to isolate the main host program data flow and the
      dense tensor compute operations.

    * `stream` models explicitly scheduled asynchronous programs by partitioning
      the dispatchable work, specifying target affinities, encoding tensors into
      target-specific forms, and scheduling the work to run concurrently.

    * `hal` models a low-level hardware abstraction layer used to manage
      buffers and issue asynchronous work across a variety of device types. The
      dialect is largely 1:1 with the IREE HAL C API.

    Transforms in the dialect lower tensor values into opaque resources with the
    goal of ensuring no tensors survive in the IR. At entry `stream.tensor.*`
    ops are used to capture the source tensor encoding information (data type,
    shapes, etc) and then lowered into `stream.async.*` ops that model the
    asynchronous workloads on the opaque resources. The asynchronous operations
    are then partitioned, allocated, and scheduled for execution using the
    `stream.cmd.*` ops.

    It's intended that after transformation through the stream dialect the
    program is ready for execution on an abstract machine. At this level of
    representation buffers have still not been allocated and devices are not
    yet resolved, however the information captured in the `stream` IR allows
    such operations to be done trivially. To this end all ops carry the symbolic
    size of the resources on which they operate as well as the lifetime of the
    resources they are acting upon. This manifests in the usage of the
    `!stream.resource` type:

    ```mlir
    // Unresolved lifetime (resolved during the iree-stream-refine-usage pass):
    !stream.resource<*>
    // An externally managed value (passed in via the program API).
    !stream.resource<external>
    // A staging buffer for uploads/downloads.
    !stream.resource<staging>
    // A short-lived value that is used across streams.
    !stream.resource<transient>
    // A long-lived value that persists across streams in globals.
    !stream.resource<variable>
    // An immutable value that persists for the duration of the program.
    !stream.resource<constant>
    ```

    Operations using resources carry the size of all operand result resources:

    ```mlir
    // %update (40 bytes) is being inserted into %target (296 bytes).
    // Can be dynamic values such as those originating from dynamic dimensions.
    %13 = stream.async.update %update, %target[%c256 to %c296] :
        !stream.resource<transient>{%c40} ->
        %target as !stream.resource<transient>{%c296}
    ```

    Once all `stream.async.*` work is moved into executable regions (such as
    `stream.async.execute`) `!stream.timepoint` values are used to sequence
    the execution. These timepoints represent some point in time where all
    execution up to that timepoint has completed and any results that were
    produced by the execution are available for use. Attempting to use the
    resources before their corresponding timepoint has been reached will lead
    to undefined behavior. The benefit of this is that after timepoints are
    established in the IR it's possible to induce aliasing of resources without
    breaking execution correctness.
  }];

  let dependentDialects = [
    "mlir::complex::ComplexDialect",
    "IREE::Util::UtilDialect",
  ];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Stream Dialect TableGen File Structure and Dependencies
//===----------------------------------------------------------------------===//
//
// This dialect's definitions are partitioned into the following files
// to enhance modularity and manage dependencies:
//
// 1. StreamDialect.td: Defines the core 'stream' dialect.
//    (No direct stream-specific includes)
//
// 2. StreamBase.td: Defines fundamental base classes for ops, types,
//    attributes, and interfaces, along with common type aliases and op traits.
//    Includes: StreamDialect.td (for dialect definition).
//
// 3. StreamAttrs.td: Defines all dialect-specific attributes.
//    Includes: StreamBase.td (for Stream_AttrDef).
//
// 4. StreamInterfaces.td: Defines all dialect-specific interfaces.
//    Includes: StreamBase.td (for base interface classes),
//              StreamAttrs.td (for attributes used in interfaces).
//
// 5. StreamTypes.td: Defines all dialect-specific types.
//    Includes: StreamBase.td (for Stream_TypeDef),
//              StreamInterfaces.td (for types implementing interfaces).
//    Note: Stream_LifetimeAttr is defined here due to its tight coupling
//          with Stream_Resource type.
//
// 6. StreamOps.td: Defines all dialect-specific operations.
//    Includes: StreamBase.td (for Stream_Op and traits),
//              StreamInterfaces.td (for ops implementing interfaces),
//              StreamTypes.td (for types used in ops).
//
// This structure aims to prevent circular dependencies and improve
// maintainability.

#endif  // IREE_DIALECT_STREAM_DIALECT
