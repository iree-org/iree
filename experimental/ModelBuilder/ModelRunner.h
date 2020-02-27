// Copyright 2020 Google LLC
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

// ModelRunner.h
// -----------------------------------------------------------------------------
//
// MLIR Model Runner
//
// The ModelRunner exposes relevant core MLIR and LLVM APIs that are sufficient
// to compile an mlir::ModuleOp. This set of classes and APIs encompass:
//  1. an mlir::ExecutionEngine engine
//  2. and llvm::TargetMachine targetMachine;
//  3. a `compile` function that takes optimization levels for the llvm opt and
//  llc tools and produces LLVMIR.
//
// Usage:
// ======
//
// ```
// // Create the builder and build some mlir::FuncOp
// ModelBuilder modelBuilder(...);
//
// // Compile the function.
// ModelRunner runner(modelBuilder.getModuleRef());
// runner.compile(/*llvmOptLevel=*/3, /*llcOptLevel=*/3);
//
// // Allocate data within data structures that interoperate with the MLIR ABI
// // conventions used by codegen.
// auto inputBuffer = ...;
// auto outputBuffer = ...;
//
// // Call the funcOp name `funcName` with arguments.
// runner.engine->invoke(funcName, ...);
// ```

#ifndef IREE_EXPERIMENTAL_MODELBUILDER_MODELRUNNER_H_
#define IREE_EXPERIMENTAL_MODELBUILDER_MODELRUNNER_H_

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Module.h"

namespace llvm {
class TargetMachine;
}  // namespace llvm

namespace mlir {

class ExecutionEngine;

class ModelRunner {
 public:
  // Initialize the runner with an OwningModuleRef, typically constructed with
  // a ModelBiulder.
  ModelRunner(mlir::OwningModuleRef &m) : module(m) {}

  // Get the underlying ModuleOp.
  ModuleOp getModule() { return *module; }

  // Compile the owned `module` into LLVMIR that can be passed to the buffer.
  // For now, the MLIR passes and transformations are kept to a minimum and only
  // perform straightforward lowering to LLVMIR. An optional shared runtime
  // support library is passed to the execution engine.
  void compile(int llvmOptLevel, int llcOptLevel,
               const std::string &runtime = {});

  // Reference to the compiled module.
  mlir::OwningModuleRef &module;

  // An execution engine and an associated target machine. The latter must
  // outlive the former since it may be used by the transformation layers.
  std::unique_ptr<mlir::ExecutionEngine> engine;
  std::unique_ptr<llvm::TargetMachine> targetMachine;
};

}  // namespace mlir

#endif  // IREE_EXPERIMENTAL_MODELBUILDER_MODELRUNNER_H_
