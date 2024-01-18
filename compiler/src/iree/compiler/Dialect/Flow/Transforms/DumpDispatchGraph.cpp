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
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/GraphWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"

namespace mlir::iree_compiler::IREE::Flow {

namespace {

static const StringRef kLineStyleControlFlow = "dashed";
static const StringRef kLineStyleDataFlow = "solid";
static const StringRef kShapeNode = "box";
static const StringRef kShapeBox = "box";
static const StringRef kShapeTab = "tab";
static const StringRef kShapeNone = "plain";
static const StringRef kShapeEllipse = "ellipse";

static StringRef getShape(Operation *op) {
  if (isa<DispatchOp>(op))
    return kShapeBox;

  return kShapeEllipse;
}

/// Return the size limits for eliding large attributes.
static int64_t getLargeAttributeSizeLimit() {
  // Use the default from the printer flags if possible.
  if (std::optional<int64_t> limit =
          OpPrintingFlags().getLargeElementsAttrLimit())
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
  return strFromOs([&](raw_ostream &os) {
    for (unsigned char c : str) {
      switch (c) {
      case '\\':
        os << '\\' << '\\';
        break;
      case '\t':
        os << '\\' << 't';
        break;
      case '\n':
        os << '\\' << 'n';
        break;
      case '"':
        os << '\\' << '"';
        break;
      case '\r': // translate "carriage return" as "\l"
        os << '\\' << 'l';
        break;
      default:
        if (llvm::isPrint(c)) {
          os << c;
          break;
        }

        // Always use a full 3-character octal escape.
        os << '\\';
        os << char('0' + ((c >> 6) & 7));
        os << char('0' + ((c >> 3) & 7));
        os << char('0' + ((c >> 0) & 7));
      }
    }
  });
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
  Node(int id = 0, std::optional<int> clusterId = std::nullopt)
      : id(id), clusterId(clusterId) {}

  int id;
  std::optional<int> clusterId;
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
    if (!modOp)
      return;

    auto funcOps = modOp.getOps<func::FuncOp>();

    if (funcOps.empty())
      return;

    emitGraph([&]() {
      for (auto funcOp : funcOps)
        processOperation(funcOp);
      emitAllEdgeStmts();
    });
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
    for (const std::string &edge : edges)
      os << edge << ";\n";
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
    if (llvm::isa<SplatElementsAttr>(attr)) {
      attr.print(os);
      return;
    }

    // Elide "big" elements attributes.
    auto elements = llvm::dyn_cast<ElementsAttr>(attr);
    if (elements && elements.getNumElements() > largeAttrLimit) {
      auto type = cast<ShapedType>(elements.getType());
      os << std::string(type.getRank(), '[') << "..."
         << std::string(type.getRank(), ']') << " : " << type;
      return;
    }

    auto array = llvm::dyn_cast<ArrayAttr>(attr);
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

  void printResults(raw_ostream &os, Operation *op, AsmState &state) {
    for (auto result : op->getResults()) {
      result.printAsOperand(os, state);
    }
  }

  void printResultsAndName(raw_ostream &os, Operation *op, AsmState &state) {
    printResults(os, op, state);
    os << " = " << op->getName();
  }

  void printDispatchTensorLoad(raw_ostream &os, DispatchTensorLoadOp op,
                               AsmState &state) {
    printResultsAndName(os, op.getOperation(), state);
    os << " ";
    op.getSource().printAsOperand(os, state);
    os << " -> " << op.getResult().getType();
    os << "\r";
  }

  void printDispatchTensorStore(raw_ostream &os, DispatchTensorStoreOp op,
                                AsmState &state) {
    os << op->getName() << " ";
    op.getValue().printAsOperand(os, state);
    os << ", ";
    op.getTarget().printAsOperand(os, state);
    os << "\r";
  }

  void printGeneric(raw_ostream &os, linalg::GenericOp op, AsmState &state) {
    printLinalgInsOuts(os, op, state);
    for (Operation &operation : *op.getBlock()) {
      os.indent(8);
      annotateOperation(os, &operation, state);
    }
  }

  template <typename T>
  void printLinalgInsOuts(raw_ostream &os, T op, AsmState &state) {
    printResultsAndName(os, op.getOperation(), state);
    os << "[";
    llvm::interleaveComma(op.getIteratorTypesArray(), os);
    os << "] (";
    printOperands(os, op.getInputs(), state);
    os << ") -> (";
    printOperands(os, op.getOutputs(), state);
    os << ")\r";
  }

  void annotateOperation(raw_ostream &os, Operation *op, AsmState &state) {
    if (isa<arith::ConstantOp>(op))
      return;

    if (isa<func::ReturnOp>(op))
      return;

    if (auto load = dyn_cast<DispatchTensorLoadOp>(op)) {
      printDispatchTensorLoad(os, load, state);
      return;
    }

    if (auto store = dyn_cast<DispatchTensorStoreOp>(op)) {
      printDispatchTensorStore(os, store, state);
      return;
    }

    if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      printGeneric(os, generic, state);
      return;
    }

    if (auto linalgOp = dyn_cast<linalg::MatmulOp>(op)) {
      printLinalgInsOuts(os, linalgOp, state);
      return;
    }

    if (auto linalgOp = dyn_cast<linalg::BatchMatmulOp>(op)) {
      printLinalgInsOuts(os, linalgOp, state);
      return;
    }

    os << *op << "\r";
  }

  void printDispatchBody(raw_ostream &os, DispatchOp &dispatchOp) {
    // Find the entry point function from the dispatch entry point symbol
    // attribute.
    auto entryPoint = *dispatchOp.getEntryPointRefs().begin();
    auto executableOp = cast<ExecutableOp>(SymbolTable::lookupNearestSymbolFrom(
        dispatchOp, entryPoint.getRootReference()));
    if (!executableOp)
      return;

    auto calleeNameAttr = entryPoint.getLeafReference();
    auto innerModule = executableOp.getInnerModule();
    if (!innerModule)
      return;
    auto funcOps = innerModule.getOps<func::FuncOp>();
    auto funcIt = llvm::find_if(funcOps, [&](func::FuncOp op) {
      return op.getNameAttr() == calleeNameAttr;
    });
    if (funcIt == funcOps.end())
      return;

    auto callee = *funcIt;

    AsmState state(callee);

    // Iterate the operations of the function body and print important
    // operation.
    for (auto &block : callee.getBlocks()) {
      for (auto &op : block.getOperations()) {
        annotateOperation(os, &op, state);
      }
    }
  }

  void printOperands(raw_ostream &os, ::mlir::Operation::operand_range operands,
                     AsmState &state) {
    auto numOperands = operands.size();

    for (auto it : llvm::enumerate(operands)) {
      auto operand = it.value();
      auto op = operand.getDefiningOp();

      if (op && isScalarConstantOp(op)) {
        auto ty = operand.getType();
        if (llvm::isa<IntegerType>(ty)) {
          os << cast<arith::ConstantIntOp>(op).value();
        } else if (llvm::isa<FloatType>(ty)) {
          cast<arith::ConstantFloatOp>(op).value().print(os);
        } else {
          os << cast<arith::ConstantIndexOp>(op).value();
        }
      } else {
        operand.printAsOperand(os, state);
      }

      if (it.index() != numOperands - 1) {
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
        printResults(os, op, state);
        os << " = " << op->getName();

        if (auto dispatch = dyn_cast<DispatchOp>(op)) {
          // print workload
          os << "[";
          printOperands(os, dispatch.getWorkload(), state);
          os << "]\n";

          // Print entry function name, if there is only one entry function,
          // then the name space and the entry function names are the same,
          // and we can just print the function name to save space.
          auto entryPoint = *dispatch.getEntryPointRefs().begin();
          auto rootName = entryPoint.getRootReference();
          auto leafName = entryPoint.getLeafReference();
          if (rootName == leafName) {
            os << leafName;
          } else {
            os << entryPoint; // print the full name
          }

          // print entry function args
          os << "(";
          printOperands(os, dispatch.getArguments(), state);
          os << ")\n";

          printDispatchBody(os, dispatch);

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
      std::optional<Node> prevNode;
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
      if (constOp.getResult().getType().isIntOrIndexOrFloat())
        return true;

    return false;
  }

  /// Process an operation. If the operation has regions, emit a cluster.
  /// Otherwise, emit a node.
  Node processOperation(Operation *op) {
    Node node;

    // Do not handle some noisy Operations.
    if (isa<arith::ConstantOp>(op) || isa<Util::GlobalLoadOpInterface>(op)) {
      return node;
    }

    if (op->getNumRegions() == 1) {
      // do not generate a cluster when there is one region.
      processRegion(op->getRegion(0));
    } else if (op->getNumRegions() > 1) {
      // Emit cluster for op with regions.
      node = emitClusterStmt(
          [&]() {
            for (Region &region : op->getRegions())
              processRegion(region);
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

    for (Value result : op->getResults())
      valueToNode[result] = node;

    return node;
  }

  /// Process a region.
  void processRegion(Region &region) {
    for (Block &block : region.getBlocks())
      processBlock(block);
  }

  /// Truncate long strings.
  std::string truncateString(std::string str) {
    if (str.length() <= maxLabelLen)
      return str;
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

} // namespace

std::unique_ptr<Pass> createDumpDispatchGraphPass(raw_ostream &os) {
  return std::make_unique<DumpDispatchGraphPass>(os);
}

} // namespace mlir::iree_compiler::IREE::Flow
