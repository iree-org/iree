// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_IO_PARAMETERS_BASE
#define IREE_DIALECT_MODULES_IO_PARAMETERS_BASE

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

//===----------------------------------------------------------------------===//
// External parameter resource management
//===----------------------------------------------------------------------===//

def IOParameters_Dialect : Dialect {
  let name = "io_parameters";
  let cppNamespace = "::mlir::iree_compiler::IREE::IO::Parameters";

  let summary = [{
    External parameter resource management APIs.
  }];
  let description = [{
    Parameters are externalized storage for resources that are
    asynchronously accessible and device-aware. Parameters can be read or
    written on the same device timelines as the operations that consume or
    produce them and with locality pinning to ensure memory doesn't need to
    move. Parameters are referenced by a scope and a key, with the scope
    being optional but strongly recommended as a way to distinguish sets of
    parameters that may exist when multiple model parts are compiled together
    and would otherwise collide.

    Parameters are provided by a few operations implementing a virtual
    interface and can support shared parameters (same storage used
    in multiple contexts, or outliving a single instantiation in a context),
    in-memory caches, memory-mapped files (including directly using the
    mapped memory for execution when devices support it), `iree_hal_file_t`
    usage for device-supported I/O, and parameter subsetting for things
    like runtime sharding.

    Alongside read(+load) and write operations gather and scatter allow for
    batching of large numbers of reads and writes into/from single buffers.
    For parameter providers that can batch operations this allows for a
    handful (~1-4) of calls out to perform many more operations (~thousands).
    Modeling the gather/scatter also gives us a point where we could extract
    the mapping and use it to repack files/defrag memory in the future.

    See `io_parameters.imports.mlir` for the full list of exported functions.
  }];
}

//===----------------------------------------------------------------------===//
// IOParameters enums
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IOParameters types
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IOParameters op traits
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base IOParameters op classes
//===----------------------------------------------------------------------===//

class IOParameters_Op<string mnemonic, list<Trait> traits = []> :
    Op<IOParameters_Dialect, mnemonic, !listconcat(traits, [])> {}

#endif  // IREE_DIALECT_MODULES_IO_PARAMETERS_BASE
