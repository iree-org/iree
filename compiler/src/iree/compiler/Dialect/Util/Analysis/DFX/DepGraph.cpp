// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/DFX/DepGraph.h"

#include <atomic>

#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GraphWriter.h"

namespace llvm {

template <>
struct DOTGraphTraits<DFXDepGraph *> : public DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool isSimple = false)
      : DefaultDOTGraphTraits(isSimple) {}

  static std::string getNodeLabel(const DFXDepGraphNode *node,
                                  const DFXDepGraph *graph) {
    std::string str;
    llvm::raw_string_ostream os(str);
    node->print(os, graph->asmState);
    return str;
  }
};

} // end namespace llvm

namespace mlir::iree_compiler::DFX {

void DepGraph::print(llvm::raw_ostream &os) {
  for (auto &depElement : syntheticRoot.deps) {
    cast<AbstractElement>(depElement.getPointer())->printWithDeps(os, asmState);
  }
}

void DepGraph::dumpGraph() {
  static std::atomic<int> callTimes;
  std::string prefix = "dep_graph";
  std::string filename =
      prefix + "_" + std::to_string(callTimes.load()) + ".dot";

  llvm::outs() << "Dependency graph dump to " << filename << ".\n";

  std::error_code ec;
  llvm::raw_fd_ostream file(filename, ec, llvm::sys::fs::OF_TextWithCRLF);
  if (!ec)
    llvm::WriteGraph(file, this);

  callTimes++;
}

} // namespace mlir::iree_compiler::DFX
