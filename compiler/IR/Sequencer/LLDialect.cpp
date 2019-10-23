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

#include "compiler/IR/Sequencer/LLDialect.h"

#include "compiler/IR/Sequencer/LLOps.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
namespace iree_compiler {

IREELLSequencerDialect::IREELLSequencerDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
#define GET_OP_LIST
  addOperations<
#include "compiler/IR/Sequencer/LLOps.cpp.inc"
      >();
}

static DialectRegistration<IREELLSequencerDialect> iree_ll_seq_dialect;

}  // namespace iree_compiler
}  // namespace mlir
