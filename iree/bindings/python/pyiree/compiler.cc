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

#include "iree/bindings/python/pyiree/compiler.h"

#include <stdexcept>

#include "iree/bindings/python/pyiree/binding.h"
#include "iree/bindings/python/pyiree/status_utils.h"
#include "iree/compiler/Translation/Sequencer/SequencerModuleTranslation.h"
#include "iree/schemas/module_def_generated.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"

namespace py = pybind11;

using namespace mlir;
using namespace mlir::iree_compiler;

using llvm::MemoryBuffer;
using llvm::MemoryBufferRef;
using llvm::StringRef;

namespace iree {
namespace python {

namespace {

OwningModuleRef parseMLIRModuleFromString(StringRef contents,
                                          MLIRContext* context) {
  std::unique_ptr<MemoryBuffer> contents_buffer;
  if (contents.back() == 0) {
    // If it has a nul terminator, just use as-is.
    contents_buffer = MemoryBuffer::getMemBuffer(contents.drop_back());
  } else {
    // Otherwise, make a copy.
    contents_buffer = MemoryBuffer::getMemBufferCopy(contents, "EMBED");
  }

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(contents_buffer), llvm::SMLoc());
  OwningModuleRef mlir_module = parseSourceFile(source_mgr, context);
  return mlir_module;
}

}  // namespace

std::shared_ptr<OpaqueBlob> CompileModuleFromAsm(const std::string& moduleAsm) {
  MLIRContext context;

  // Arrange to get a view that includes a terminating null to avoid additional
  // copy.
  const char* moduleAsmChars = moduleAsm.c_str();
  StringRef moduleAsmSr(moduleAsmChars, moduleAsm.size() + 1);

  // TODO(laurenzo): This error handling is super hoaky. Hook into the MLIR
  // error reporter and plumb through properly.
  OwningModuleRef mlirModule = parseMLIRModuleFromString(moduleAsmSr, &context);
  if (!mlirModule) {
    throw std::runtime_error("Failed to parse MLIR asm");
  }

  auto moduleBlob =
      mlir::iree_compiler::translateMlirToIreeSequencerModule(mlirModule.get());
  if (moduleBlob.empty()) {
    throw std::runtime_error("Failed to translate MLIR module");
  }
  return std::make_shared<OpaqueByteVectorBlob>(std::move(moduleBlob));
}

void SetupCompilerBindings(pybind11::module m) {
  m.def("compile_module_from_asm", CompileModuleFromAsm);
}

}  // namespace python
}  // namespace iree
