// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

namespace mlir {
class RewritePatternSet;
class DialectRegistry;
class SymbolTableCollection;
class OpPassManager;
class Pass;
namespace iree_compiler::IREE::Flow {

std::unique_ptr<Pass> createConvertMeshToFlowPass();

void populateMeshToFlowCollectivesPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection,
    bool useNamedDefaultChannels);

void registerMeshToFlowDependencies(DialectRegistry &registry);
void registerMeshToFlowPasses();

} // namespace iree_compiler::IREE::Flow
} // namespace mlir
