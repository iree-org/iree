// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_CODEGENPIPELINEOPTIONS_H_
#define IREE_COMPILER_CODEGEN_UTILS_CODEGENPIPELINEOPTIONS_H_

#include "mlir/Support/TypeID.h"

namespace mlir::iree_compiler {

/// Polymorphic base for per-pipeline codegen options passed through
/// PipelineAttrInterface::buildPipeline. Supports llvm::isa/cast/dyn_cast.
struct CodegenPipelineOptions {
  virtual ~CodegenPipelineOptions();

  TypeID getTypeID() const { return typeID; }

protected:
  explicit CodegenPipelineOptions(TypeID typeID) : typeID(typeID) {}

private:
  TypeID typeID;
};

/// CRTP helper that provides the TypeID constructor and classof for concrete
/// pipeline options classes.
template <typename Derived>
struct CodegenPipelineOptionsBase : CodegenPipelineOptions {
  static bool classof(const CodegenPipelineOptions *opts) {
    return opts->getTypeID() == TypeID::get<Derived>();
  }

protected:
  CodegenPipelineOptionsBase()
      : CodegenPipelineOptions(TypeID::get<Derived>()) {}
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_CODEGENPIPELINEOPTIONS_H_
