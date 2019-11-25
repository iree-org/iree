// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_DIALECT_HAL_OPS
#define IREE_DIALECT_HAL_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"

//===----------------------------------------------------------------------===//
// iree::hal::Executable
//===----------------------------------------------------------------------===//

// TODO(benvanik): executable runtime type.

def HAL_ExecutableOp : HAL_Op<"executable", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"IREE::HAL::ExecutableEndOp">,
    Symbol,
  ]> {
  let summary = [{target-specific executable module}];
  let description = [{
    An executable module representing a target-specific compiled
    kernel/shader/etc.
  }];

  let arguments = (ins
    StrAttr:$sym_name
    // TODO(benvanik): entry point types for verification.
  );

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, StringRef name
    }]>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }
  }];

  let verifier = [{ return verifyExecutableOp(*this); }];
}

def HAL_ExecutableEndOp : HAL_Op<"executable_end", [
    HasParent<"IREE::HAL::ExecutableOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable op}];
  let parser = [{ return parseRegionEndOp(parser, &result); }];
  let printer = [{ return printRegionEndOp(p, *this); }];
}

// TODO(benvanik): add HAL_ExecutableSourceOp for auto serialization.

def HAL_ExecutableBinaryOp : HAL_Op<"executable.binary", [
    IsolatedFromAbove,
    HasParent<"IREE::HAL::ExecutableOp">,
    SingleBlockImplicitTerminator<"IREE::HAL::ExecutableBinaryEndOp">,
  ]> {
  let summary = [{compiled executable binary data}];
  let description = [{
    A compiled executable binary with an optional nested module containing the
    IR prior to serialization (for debugging).
  }];

  let arguments = (ins
    HAL_ExecutableFormatAttr:$format,
    HAL_ExecutableDataAttr:$data
    // TODO(benvanik): add compatibility and versioning attributes.
  );

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, uint32_t format,
      std::vector<uint8_t> data
    }]>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }

    llvm::Optional<::mlir::ModuleOp> getInnerModule() {
      auto moduleOps = getBlock().getOps<::mlir::ModuleOp>();
      if (moduleOps.empty()) return llvm::None;
      return *moduleOps.begin();
    }
  }];

  let verifier = [{ return verifyExecutableBinaryOp(*this); }];
}

def HAL_ExecutableBinaryEndOp : HAL_Op<"executable.binary_end", [
    HasParent<"IREE::HAL::ExecutableBinaryOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable binary op}];
  let parser = [{ return parseRegionEndOp(parser, &result); }];
  let printer = [{ return printRegionEndOp(p, *this); }];
}

#endif  // IREE_DIALECT_HAL_OPS
