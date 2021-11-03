// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_STREAM_PASSES
#define IREE_DIALECT_STREAM_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Optimizations
//===----------------------------------------------------------------------===//

def OutlineConstants :
    Pass<"iree-stream-outline-constants", "mlir::ModuleOp"> {
  let summary = "Outlines tensor constants into util.globals at the module level.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createOutlineConstantsPass()
  }];
}

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

def ConvertToStream :
    Pass<"iree-stream-conversion", "mlir::ModuleOp"> {
  let summary = "Converts from flow/std/etc dialects into the stream dialect.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createConvertToStreamPass()
  }];
}

//===----------------------------------------------------------------------===//
// Placement/affinity management
//===----------------------------------------------------------------------===//

def EncodeTensors :
    Pass<"iree-stream-encode-tensors", ""> {
  let summary = "Encodes tensors into binary formats based on affinity and target support.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createEncodeTensorsPass()
  }];
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

def VerifyInput :
    Pass<"iree-stream-verify-input", "mlir::ModuleOp"> {
  let summary = "Verifies that input dialects are supported by the streams dialect.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createVerifyInputPass()
  }];
}

def VerifyLoweringToTensors :
    Pass<"iree-stream-verify-lowering-to-tensors", "mlir::ModuleOp"> {
  let summary = "Verifies that input dialects are converted to stream.tensor.* ops.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createVerifyLoweringToTensorsPass()
  }];
}

def VerifyLoweringToAsync :
    Pass<"iree-stream-verify-lowering-to-async", "mlir::ModuleOp"> {
  let summary = "Verifies that all stream.tensor.* ops and types are fully lowered to stream.async.* ops.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createVerifyLoweringToAsyncPass()
  }];
}

def VerifyLoweringToCmd :
    Pass<"iree-stream-verify-lowering-to-cmd", "mlir::ModuleOp"> {
  let summary = "Verifies that all stream.async.* ops and types are fully lowered to stream.cmd.* ops.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createVerifyLoweringToCmdPass()
  }];
}

#endif  // IREE_DIALECT_STREAM_PASSES
