// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_DIALECT
#define IREE_DIALECT_HAL_DIALECT

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

//===----------------------------------------------------------------------===//
// IREE HAL (Hardware Abstraction Layer) dialect
//===----------------------------------------------------------------------===//

def HAL_Dialect : Dialect {
  let name = "hal";
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";

  let summary = [{
    A dialect representing operations against the IREE HAL.
  }];
  let description = [{
    This can be thought of as a Vulkan-like model with all of the graphics bits
    chopped out.

    The type set is limited to those that can be represented in the IREE HAL
    design: buffers and views, synchronization primitives like semaphores,
    and command buffers. The intent is that if a device could implement the HAL
    interface the sequencer ops could run on that device, such as being able to
    run on a GPU via indirect command buffers.

    Though this is mostly a 1:1 mapping to the iree::hal API there are some
    methods omitted as they are not likely to be needed in IR. It's assumed that
    either sequencer interfaces will encapsulate the logic (such as device
    resolution) or that certain features are unsafe to expose to user-defined
    input.
  }];

  let useDefaultAttributePrinterParser = 0;
}

//===----------------------------------------------------------------------===//
// HAL Dialect TableGen File Structure and Dependencies
//===----------------------------------------------------------------------===//
//
// This dialect's definitions are partitioned into the following files
// to enhance modularity and manage dependencies:
//
// 1. HALDialect.td: Defines the core 'hal' dialect.
//
// 2. HALBase.td: Defines fundamental base classes for ops, types,
//    attributes, and interfaces, along with common type aliases and op traits.
//    Includes: HALDialect.td (for dialect definition).
//
// 3. HALAttrs.td: Defines all dialect-specific attributes and enums.
//    Includes: HALBase.td (for HAL_AttrDef), HALTypes.td (for types used in
//              attributes).
//
// 4. HALInterfaces.td: Defines all dialect-specific interfaces.
//    Includes: UtilBase.td (for base interface classes).
//
// 5. HALTypes.td: Defines all dialect-specific types.
//    Includes: HALBase.td (for HAL_TypeDef),
//              HALInterfaces.td (for types implementing interfaces).
//
// 6. HALOps.td: Defines all dialect-specific operations.
//    Includes: HALBase.td (for HAL_Op and traits),
//              HALAttrs.td (for attributes used in ops),
//              HALTypes.td (for types used in ops).
//
// This structure aims to prevent circular dependencies and improve
// maintainability.

#endif // IREE_DIALECT_HAL_DIALECT
