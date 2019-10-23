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

#ifndef IREE_COMPILER_TRANSLATION_INTERPRETER_INTERPRETEREXECUTABLETRANSLATION_H_
#define IREE_COMPILER_TRANSLATION_INTERPRETER_INTERPRETEREXECUTABLETRANSLATION_H_

#include <vector>

#include "iree/compiler/IR/StructureOps.h"
#include "iree/compiler/Utils/TranslationUtils.h"
#include "mlir/IR/Module.h"

namespace mlir {
namespace iree_compiler {

// Translates an MLIR module into a bytecode interpreter executable.
// These executables are stored as IREE modules as defined in the
// https://github.com/google/iree/tree/master/iree/iree/schemas/module_def.fbs
// schema.
llvm::Optional<ExecutableTranslationResult>
translateExecutableToInterpreterExecutable(
    ArrayRef<IREE::ExecutableOp> executableOps,
    ExecutableTranslationOptions options = {});

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_INTERPRETER_INTERPRETEREXECUTABLETRANSLATION_H_
