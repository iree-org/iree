// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===------------------- DumpDispatchGraph.cpp ----------------------------===//
//
// Generate a graphviz graph for dispatches
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/GraphWriter.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

static const StringRef kLineStyleControlFlow = "dashed";
static const StringRef kLineStyleDataFlow = "solid";
static const StringRef kShapeNode = "box";
static const StringRef kShapeBox = "box";
static const StringRef kShapeTab = "tab";
static const StringRef kShapeNone = "plain";
static const StringRef kShapeEllipse = "ellipse";

static StringRef getShape(Operation *op) {
  if (isa<DispatchOp>(op)) return kShapeBox;

  return kShapeEllipse;
}

/// Return the size limits for eliding large attributes.
static int64_t getLargeAttributeSizeLimit() {
  // Use the default from the printer flags if possible.
  if (Optional<int64_t> limit = OpPrintingFlags().getLargeElementsAttrLimit())
    return *limit;
  return 16;
}

/// Return all values printed onto a stream as a string.
static std::string strFromOs(function_ref<void(raw_ostream &)> func) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  func(os);
  return os.str();
}

/// Escape special characters such as '\n' and quotation marks.
static std::string escapeString(std::string str) {
  return strFromOs([&](raw_ostream &os) { os.write_escaped(str); });
}

/// Put quotation marks around a given string.
static std::string quoteString(const std::string &str) {
  return "\"" + str + "\"";
}

using AttributeMap = llvm::StringMap<std::string>;

/// This struct represents a node in the DOT language. Each node has an
/// identifier and an optional identifier for the cluster (subgraph) that
/// contains the node.
/// Note: In the DOT language, edges can be drawn only from nodes to nodes, but
/// not between clusters. However, edges can be clipped to the boundary of a
/// cluster with `lhead` and `ltail` attributes. Therefore, when creating a new
/// cluster, an invisible "anchor" node is created.
struct Node {
 public:
  Node(int id = 0, Optional<int> clusterId = llvm::None)
      : id(id), clusterId(clusterId) {}

  int id;
  Optional<int> clusterId;
};

/// This pass generates a Graphviz dataflow visualization of an MLIR operation.
/// Note: See https://www.graphviz.org/doc/info/lang.html for more information
/// about the Graphviz DOT language.
class DumpDispatchGraphPass
    : public DumpDispatchGraphBase<DumpDispatchGraphPass> {
 public:
  DumpDispatchGraphPass(raw_ostream &os) : os(os) {}
  DumpDispatchGraphPass(const DumpDispatchGraphPass &o)
      : DumpDispatchGraphPass(o.os.getOStream()) {}

  void runOnOperation() override {
    auto modOp = dyn_cast<ModuleOp>(getOperation());
    if (!modOp) return;

    for (auto funcOp : modOp.getOps<func::FuncOp>()) {
      emitGraph([&]() {
        processOperation(funcOp);
        emitAllEdgeStmts();
      });
    }
  }

  /// Create a CFG graph for a region. Used in `Region::viewGraph`.
  void emitRegionCFG(Region &region) {
    printControlFlowEdges = true;
    printDataFlowEdges = false;
    emitGraph([&]() { processRegion(region); });
  }

 private:
  /// Emit all edges. This function should be called after all nodes have been
  /// emitted.
  void emitAllEdgeStmts() {
    for (const std::string &edge : edges) os << edge << ";\n";
    edges.clear();
  }

  /// Emit a cluster (subgraph). The specified builder generates the body of the
  /// cluster. Return the anchor node of the cluster.
  Node emitClusterStmt(function_ref<void()> builder, std::string label = "") {
    int clusterId = ++counter;
    os << "subgraph cluster_" << clusterId << " {\n";
    os.indent();
    // Emit invisible anchor node from/to which arrows can be drawn.
    Node anchorNode = emitNodeStmt(" ", kShapeNone);
    os << attrStmt("label", quoteString(escapeString(std::move(label))))
       << ";\n";
    builder();
    os.unindent();
    os << "}\n";
    return Node(anchorNode.id, clusterId);
  }

  /// Generate an attribute statement.
  std::string attrStmt(const Twine &key, const Twine &value) {
    return (key + " = " + value).str();
  }

  /// Emit an attribute list.
  void emitAttrList(raw_ostream &os, const AttributeMap &map) {
    os << "[";
    interleaveComma(map, os, [&](const auto &it) {
      os << this->attrStmt(it.getKey(), it.getValue());
    });
    os << "]";
  }

  // Print an MLIR attribute to `os`. Large attributes are truncated.
  void emitMlirAttr(raw_ostream &os, Attribute attr) {
    // A value used to elide large container attribute.
    int64_t largeAttrLimit = getLargeAttributeSizeLimit();

    // Always emit splat attributes.
    if (attr.isa<SplatElementsAttr>()) {
      attr.print(os);
      return;
    }

    // Elide "big" elements attributes.
    auto elements = attr.dyn_cast<ElementsAttr>();
    if (elements && elements.getNumElements() > largeAttrLimit) {
      os << std::string(elements.getType().getRank(), '[') << "..."
         << std::string(elements.getType().getRank(), ']') << " : "
         << elements.getType();
      return;
    }

    auto array = attr.dyn_cast<ArrayAttr>();
    if (array && static_cast<int64_t>(array.size()) > largeAttrLimit) {
      os << "[...]";
      return;
    }

    // Print all other attributes.
    std::string buf;
    llvm::raw_string_ostream ss(buf);
    attr.print(ss);
    os << truncateString(ss.str());
  }

  /// Append an edge to the list of edges.
  /// Note: Edges are written to the output stream via `emitAllEdgeStmts`.
  void emitEdgeStmt(Node n1, Node n2, std::string label, StringRef style) {
    AttributeMap attrs;
    attrs["style"] = style.str();
    // Do not label edges that start/end at a cluster boundary. Such edges are
    // clipped at the boundary, but labels are not. This can lead to labels
    // floating around without any edge next to them.
    if (!n1.clusterId && !n2.clusterId)
      attrs["label"] = quoteString(escapeString(std::move(label)));
    // Use `ltail` and `lhead` to draw edges between clusters.
    if (n1.clusterId)
      attrs["ltail"] = "cluster_" + std::to_string(*n1.clusterId);
    if (n2.clusterId)
      attrs["lhead"] = "cluster_" + std::to_string(*n2.clusterId);

    edges.push_back(strFromOs([&](raw_ostream &os) {
      os << llvm::format("v%i -> v%i ", n1.id, n2.id);
      emitAttrList(os, attrs);
    }));
  }

  /// Emit a graph. The specified builder generates the body of the graph.
  void emitGraph(function_ref<void()> builder) {
    os << "digraph G {\n";
    os.indent();
    // Edges between clusters are allowed only in compound mode.
    os << attrStmt("compound", "true") << ";\n";
    builder();
    os.unindent();
    os << "}\n";
  }

  /// Emit a node statement.
  Node emitNodeStmt(Operation *op) {
    int nodeId = ++counter;
    AttributeMap attrs;
    auto label = getLabel(op);
    auto shape = getShape(op);
    attrs["label"] = quoteString(escapeString(std::move(label)));
    attrs["shape"] = shape.str();
    os << llvm::format("v%i ", nodeId);
    emitAttrList(os, attrs);
    os << ";\n";
    return Node(nodeId);
  }
  Node emitNodeStmt(std::string label, StringRef shape = kShapeNode) {
    int nodeId = ++counter;
    AttributeMap attrs;
    attrs["label"] = quoteString(escapeString(std::move(label)));
    attrs["shape"] = shape.str();
    os << llvm::format("v%i ", nodeId);
    emitAttrList(os, attrs);
    os << ";\n";
    return Node(nodeId);
  }

  void printOperands(raw_ostream &os, ::mlir::Operation::operand_range operands,
                     AsmState &state) {
    for (unsigned i = 0, e = operands.size(); i != e; ++i) {
      auto operand = operands[i];
      auto op = operand.getDefiningOp();

      if (isScalarConstantOp(op)) {
        auto ty = operand.getType();
        if (ty.isa<IntegerType>()) {
          os << cast<arith::ConstantIntOp>(op).value();
        } else if (ty.isa<FloatType>()) {
          cast<arith::ConstantFloatOp>(op).value().print(os);
        } else {
          os << cast<arith::ConstantIndexOp>(op).value();
        }
      } else {
        operand.printAsOperand(os, state);
      }

      if (i != e - 1) {
        os << ", ";
      }
    }
  }

  /// Generate a label for an operation.
  std::string getLabel(Operation *op) {
    return strFromOs([&](raw_ostream &os) {
      if (op->getNumRegions() == 0) {
        auto funcOp = op->getParentOfType<func::FuncOp>();
        AsmState state(funcOp);

        for (auto result : op->getResults()) {
          result.printAsOperand(os, state);
        }
        os << " = " << op->getName();

        if (auto dispatch = dyn_cast<DispatchOp>(op)) {
          // print workload
          os << "[";
          printOperands(os, dispatch.workload(), state);
          os << "]\n";

          // print entry function name
          os << dispatch.entry_point();

          // print entry function args
          os << "(";
          printOperands(os, dispatch.operands(), state);
          os << ")\n";

        } else {
          os << "\n";
        }
      } else {
        os << op->getName() << "\n";
      }

      if (printResultTypes) {
        std::string buf;
        llvm::raw_string_ostream ss(buf);
        interleave(op->getResultTypes(), ss, "\n");
        os << ss.str();
      }
    });
  }

  /// Generate a label for a block argument.
  std::string getLabel(BlockArgument arg) {
    return "arg" + std::to_string(arg.getArgNumber());
  }

  /// Process a block. Emit a cluster and one node per block argument and
  /// operation inside the cluster.
  void processBlock(Block &block) {
    emitClusterStmt([&]() {
      for (BlockArgument &blockArg : block.getArguments())
        valueToNode[blockArg] = emitNodeStmt(getLabel(blockArg));

      // Emit a node for each operation.
      Optional<Node> prevNode;
      for (Operation &op : block) {
        Node nextNode = processOperation(&op);
        if (printControlFlowEdges && prevNode)
          emitEdgeStmt(*prevNode, nextNode, /*label=*/"",
                       kLineStyleControlFlow);
        prevNode = nextNode;
      }
    });
  }

  bool isScalarConstantOp(Operation *op) {
    if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(op))
      if (constOp.getResult().getType().isIntOrIndexOrFloat()) return true;

    return false;
  }

  /// Process an operation. If the operation has regions, emit a cluster.
  /// Otherwise, emit a node.
  Node processOperation(Operation *op) {
    Node node;

    if (isScalarConstantOp(op)) {
      // don't handle scalar constant because it creates too many edges
      // from a single constant.
      return node;
    }

    if (op->getNumRegions() == 1) {
      // do not generate a cluster when there is one region.
      processRegion(op->getRegion(0));
    } else if (op->getNumRegions() > 1) {
      // Emit cluster for op with regions.
      node = emitClusterStmt(
          [&]() {
            for (Region &region : op->getRegions()) processRegion(region);
          },
          getLabel(op));
    } else {
      node = emitNodeStmt(op);
    }

    // Insert data flow edges originating from each operand.
    if (printDataFlowEdges) {
      unsigned numOperands = op->getNumOperands();
      for (unsigned i = 0; i < numOperands; i++) {
        auto operand = op->getOperand(i);

        // a constant operand is not going to be available in the map.
        if (valueToNode.count(operand)) {
          emitEdgeStmt(valueToNode[op->getOperand(i)], node,
                       /*label=*/numOperands == 1 ? "" : std::to_string(i),
                       kLineStyleDataFlow);
        }
      }
    }

    for (Value result : op->getResults()) valueToNode[result] = node;

    return node;
  }

  /// Process a region.
  void processRegion(Region &region) {
    for (Block &block : region.getBlocks()) processBlock(block);
  }

  /// Truncate long strings.
  std::string truncateString(std::string str) {
    if (str.length() <= maxLabelLen) return str;
    return str.substr(0, maxLabelLen) + "...";
  }

  /// Output stream to write DOT file to.
  raw_indented_ostream os;
  /// A list of edges. For simplicity, should be emitted after all nodes were
  /// emitted.
  std::vector<std::string> edges;
  /// Mapping of SSA values to Graphviz nodes/clusters.
  DenseMap<Value, Node> valueToNode;
  /// Counter for generating unique node/subgraph identifiers.
  int counter = 0;
};

}  // namespace

std::unique_ptr<Pass> createDumpDispatchGraphPass(raw_ostream &os) {
  return std::make_unique<DumpDispatchGraphPass>(os);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
