// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTraits.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

namespace {

//===----------------------------------------------------------------------===//
// Usage analysis
//===----------------------------------------------------------------------===//

struct UsageInfo {
  // util.globals holding resources mapped by name.
  llvm::MapVector<StringRef, IREE::Util::GlobalOp> resourceGlobalOps;
  // util.buffer.constants that are (for the most part) going to end up in the
  // final binary.
  SmallVector<IREE::Util::BufferConstantOp> bufferConstantOps;

  // stream.executable ops mapped by name.
  llvm::MapVector<StringRef, IREE::Stream::ExecutableOp> executableOps;
  // stream.executable exported function -> dispatches to it.
  llvm::MapVector<mlir::func::FuncOp, SmallVector<IREE::Stream::CmdDispatchOp>>
      exportDispatchOps;

  // TODO(benvanik): resource allocations.

  // stream.cmd.execute ops containing all relevant device commands.
  SmallVector<IREE::Stream::CmdExecuteOp> executeOps;
  SmallVector<IREE::Stream::ResourceAllocaOp> allocaOps;

  // stream.timepoint.await ops indicating host/device synchronization.
  SmallVector<IREE::Stream::TimepointAwaitOp> awaitOps;

  void analyze(mlir::ModuleOp moduleOp) {
    SymbolTable symbolTable(moduleOp);
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOp>()) {
      if (llvm::isa<IREE::Stream::ResourceType>(globalOp.getType())) {
        resourceGlobalOps[globalOp.getName()] = globalOp;
      }
    }
    for (auto executableOp : moduleOp.getOps<IREE::Stream::ExecutableOp>()) {
      executableOps[executableOp.getName()] = executableOp;
    }
    for (auto funcLikeOp : moduleOp.getOps<FunctionOpInterface>()) {
      funcLikeOp.walk([&](Operation *op) {
        TypeSwitch<Operation *>(op)
            .Case<IREE::Util::BufferConstantOp>(
                [&](auto op) { bufferConstantOps.push_back(op); })
            .Case<IREE::Stream::ResourceAllocaOp>(
                [&](auto op) { allocaOps.push_back(op); })
            .Case<IREE::Stream::CmdExecuteOp>(
                [&](auto op) { executeOps.push_back(op); })
            .Case<IREE::Stream::TimepointAwaitOp>(
                [&](auto op) { awaitOps.push_back(op); });
      });
    }
    for (auto executeOp : executeOps) {
      executeOp.walk([&](IREE::Stream::CmdDispatchOp dispatchOp) {
        dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
          auto exportOp = cast<IREE::Stream::ExecutableExportOp>(
              symbolTable.lookupSymbolIn(moduleOp, entryPointAttr));
          assert(exportOp && "missing executable/export");
          auto funcOp = exportOp.lookupFunctionRef();
          assert(funcOp && "missing exported function");
          exportDispatchOps[funcOp].push_back(dispatchOp);
        });
      });
    }
  }
};

// TODO(benvanik): StaticSize helper or something for the dynamic bit.
struct Statistics {
  // Globals:
  size_t constantCount = 0;
  int64_t constantSize = 0;
  bool constantSizeDynamic = false;
  size_t variableCount = 0;
  int64_t variableSize = 0;
  bool variableSizeDynamic = false;

  // Synchronization:
  size_t awaitCount = 0;

  // Execution:
  size_t submissionCount = 0;
  int64_t transientSize = 0;
  bool transientSizeDynamic = false;
  // TODO(benvanik): add fill/copy sizes (when possible).
  size_t fillCount = 0;
  size_t copyCount = 0;
  size_t collectiveCount = 0;
  size_t dispatchCount = 0;
  size_t callCount = 0;

  // Executables:
  size_t executableCount = 0;

  void analyze(const UsageInfo &usageInfo) {
    // Globals:
    for (auto [name, globalOp] : usageInfo.resourceGlobalOps) {
      auto globalType =
          llvm::dyn_cast<IREE::Stream::ResourceType>(globalOp.getType());
      if (!globalType) continue;
      // TODO(benvanik): analyze size in UsageInfo where possible.
      switch (globalType.getLifetime()) {
        case IREE::Stream::Lifetime::Constant:
          ++constantCount;
          break;
        case IREE::Stream::Lifetime::Variable:
          ++variableCount;
          break;
        default:
          continue;
      }
    }
    for (auto constantOp : usageInfo.bufferConstantOps) {
      if (auto serializableAttr =
              constantOp.getValue()
                  .dyn_cast<IREE::Util::SerializableAttrInterface>()) {
        constantSize += serializableAttr.getStorageSize();
      }
    }

    // Synchronization:
    awaitCount = usageInfo.awaitOps.size();

    // Execution:
    submissionCount = usageInfo.executeOps.size();
    for (auto allocaOp : usageInfo.allocaOps) {
      APInt allocaSize;
      if (matchPattern(allocaOp.getStorageSize(), m_ConstantInt(&allocaSize))) {
        transientSize += allocaSize.getSExtValue();
      } else {
        transientSizeDynamic = true;
      }
    }
    for (auto executeOp : usageInfo.executeOps) {
      executeOp.walk([&](Operation *op) {
        TypeSwitch<Operation *>(op)
            .Case<IREE::Stream::CmdFillOp>([&](auto op) { ++fillCount; })
            .Case<IREE::Stream::CmdCopyOp>([&](auto op) { ++copyCount; })
            .Case<IREE::Stream::CmdCollectiveOp>(
                [&](auto op) { ++collectiveCount; })
            .Case<IREE::Stream::CmdDispatchOp>(
                [&](auto op) { ++dispatchCount; })
            .Case<IREE::Stream::CmdCallOp>([&](auto op) { ++callCount; });
      });
    }

    // Executables:
    executableCount = usageInfo.executableOps.size();
  }
};

//===----------------------------------------------------------------------===//
// Pretty printing
//===----------------------------------------------------------------------===//

static void prettyPrintOpBreadcrumb(Operation *op, llvm::raw_fd_ostream &os) {
  auto parentOp = op->getParentOp();
  if (parentOp) {
    prettyPrintOpBreadcrumb(parentOp, os);
    os << " > ";
  }
  os << op->getName();
  if (auto symbolOp = dyn_cast<SymbolOpInterface>(op)) {
    os << " @" << symbolOp.getName();
  }
}

static void prettyPrintSectionHeader(llvm::Twine header,
                                     llvm::raw_fd_ostream &os) {
  os << "//"
        "======================================================================"
        "======//\n";
  os << "// " << header << "\n";
  os << "//"
        "======================================================================"
        "======//\n";
}

static void prettyPrintItemHeader(llvm::Twine header,
                                  llvm::raw_fd_ostream &os) {
  os << "//"
        "----------------------------------------------------------------------"
        "------//\n";
  os << "// " << header << "\n";
  os << "//"
        "----------------------------------------------------------------------"
        "------//\n";
}

static void prettyPrintStatistics(const UsageInfo &usageInfo,
                                  llvm::raw_fd_ostream &os) {
  prettyPrintSectionHeader("Aggregate Statistics (static, whole-program)", os);
  os << "//\n";

  Statistics stats;
  stats.analyze(usageInfo);

  os << llvm::formatv("//   Constants: {0}, ", stats.constantCount);
  os << llvm::formatv("estimated storage of {0}{1} B ({2:F2} MiB)\n",
                      stats.constantSizeDynamic ? "minimum " : "",
                      stats.constantSize,
                      stats.constantSize / (1 * 1024 * 1024.0f));
  os << llvm::formatv("//   Variables: {0}, ", stats.variableCount);
  os << llvm::formatv("(TBD) {0}{1} B ({2:F2} MiB)\n",
                      stats.variableSizeDynamic ? "minimum " : "",
                      stats.variableSize,
                      stats.variableSize / (1 * 1024 * 1024.0f));

  os << llvm::formatv("//  D->H Syncs: {0}\n", stats.awaitCount);

  os << llvm::formatv("// Submissions: {0}, using cumulative ",
                      stats.submissionCount);
  os << llvm::formatv(
      "{0}{1} B ({2:F2} MiB)\n", stats.transientSizeDynamic ? "minimum " : "",
      stats.transientSize, stats.transientSize / (1 * 1024 * 1024.0f));

  os << llvm::formatv("//   DMA Fills: {0}\n", stats.fillCount);
  os << llvm::formatv("//  DMA Copies: {0}\n", stats.copyCount);
  os << llvm::formatv("// Collectives: {0}\n", stats.collectiveCount);
  os << llvm::formatv("//  Dispatches: {0}\n", stats.dispatchCount);
  os << llvm::formatv("// Async Calls: {0}\n", stats.callCount);

  os << llvm::formatv(
      "// Executables: {0}, {1}% reuse\n", stats.executableCount,
      (int)std::roundf(
          (1.0f - (stats.executableCount / (float)stats.dispatchCount)) *
          100.0f));

  os << "//\n";
}

static void prettyPrintGlobalInfo(const UsageInfo &usageInfo, bool verbose,
                                  llvm::raw_fd_ostream &os) {
  prettyPrintSectionHeader("Constants / Variables", os);
  os << "//\n";

  // TODO(benvanik): print global information:
  // - number of resource globals: constants/variables
  // - util.buffer.constant sizes (fed into stream.resource.try_map/map)
  // - variable allocation sizes
  os << "// TODO\n";

  os << "//\n";
}

static void prettyPrintSyncInfo(const UsageInfo &usageInfo, bool verbose,
                                llvm::raw_fd_ostream &os) {
  prettyPrintSectionHeader("Synchronization", os);
  os << "//\n";

  // TODO(benvanik): print host <-> device information:
  // - number of stream.timepoint.awaits
  // - staging buffer allocation sizes
  // - number of buffer mapping operations
  // - estimated number of submissions (execution with await in the middle)
  os << "// TODO\n";

  os << "//\n";
}

static void prettyPrintStreamInfo(const UsageInfo &usageInfo,
                                  IREE::Stream::CmdExecuteOp executeOp,
                                  llvm::raw_fd_ostream &os) {
  auto parentOp = executeOp->getParentOfType<FunctionOpInterface>();

  prettyPrintItemHeader(
      llvm::formatv("stream.cmd.execute", parentOp->getName().getStringRef()),
      os);
  os << "// ";
  prettyPrintOpBreadcrumb(executeOp, os);
  os << "\n";
  os << "//\n";

  // TODO(benvanik): print stream information (for each stream.cmd.execute):
  // - number of unique resources captured
  // - number of commands of each type
  // - % concurrently executable
  os << "// TODO\n";
}

static void prettyPrintAllStreamInfo(const UsageInfo &usageInfo, bool verbose,
                                     llvm::raw_fd_ostream &os) {
  prettyPrintSectionHeader("Streams", os);
  os << "//\n";

  // TODO(benvanik): aggregate stats:
  // - number of streams
  // - (eventually) number of streams per affinity
  // - average commands per stream
  // - streams with host dependencies/device dependencies (awaits/etc)
  os << "// TODO\n";

  os << "//\n";
  for (auto executeOp : usageInfo.executeOps) {
    prettyPrintStreamInfo(usageInfo, executeOp, os);
    os << "//\n";
  }
}

static void prettyPrintExecutableExportInfo(
    const UsageInfo &usageInfo, IREE::Stream::ExecutableOp executableOp,
    IREE::Stream::ExecutableExportOp exportOp, llvm::raw_fd_ostream &os) {
  auto funcOp = exportOp.lookupFunctionRef();
  prettyPrintItemHeader(
      llvm::formatv("stream.executable.export @{0}::@{1}",
                    executableOp.getName(), exportOp.getName()),
      os);
  os << "// ";
  prettyPrintOpBreadcrumb(funcOp, os);
  os << "\n";
  os << "//\n";

  // TODO(benvanik): interface and usage stats:
  // - operand info
  // - binding info
  //   - misaligned/unaligned/etc - big warning
  // - incoming dispatches
  //   - workload params

  // TODO(benvanik): ask codegen team if they want anything like a list of
  // linalg named ops, etc.

  os << "// TODO\n";
}

static void prettyPrintExecutableInfo(const UsageInfo &usageInfo,
                                      IREE::Stream::ExecutableOp executableOp,
                                      llvm::raw_fd_ostream &os) {
  // Today we pretty much have one export per executable here as we are
  // performing linking in the HAL. Once we link/deduplicate/etc in streams then
  // we'll want to make this segmentation nicer.
  for (auto exportOp :
       executableOp.getOps<IREE::Stream::ExecutableExportOp>()) {
    prettyPrintExecutableExportInfo(usageInfo, executableOp, exportOp, os);
  }
}

static void prettyPrintAllExecutableInfo(const UsageInfo &usageInfo,
                                         bool verbose,
                                         llvm::raw_fd_ostream &os) {
  prettyPrintSectionHeader("Executables", os);
  os << "//\n";

  // TODO(benvanik): aggregate stats:
  // - number of executables
  // - total number of exports
  // - average bindings/operands per export
  os << "// TODO\n";

  os << "//\n";
  for (auto it : usageInfo.executableOps) {
    prettyPrintExecutableInfo(usageInfo, it.second, os);
    os << "//\n";
  }
}

static void prettyPrintUsageInfo(const UsageInfo &usageInfo, bool verbose,
                                 llvm::raw_fd_ostream &os) {
  prettyPrintStatistics(usageInfo, os);
  prettyPrintGlobalInfo(usageInfo, verbose, os);
  prettyPrintSyncInfo(usageInfo, verbose, os);
  prettyPrintAllStreamInfo(usageInfo, verbose, os);
  prettyPrintAllExecutableInfo(usageInfo, verbose, os);
}

//===----------------------------------------------------------------------===//
// CSV tables
//===----------------------------------------------------------------------===//

static void dumpAggregateCSVTable(const UsageInfo &usageInfo,
                                  llvm::raw_fd_ostream &os) {
  Statistics stats;
  stats.analyze(usageInfo);

  os << R"("Constants","Constant Size","Variables","Variable Size","Awaits","Submissions","Transient Size","Fills","Copies","Dispatches","Async Calls","Executables")";
  os << "\n";

  // Globals:
  os << llvm::formatv("{0},{1},{2},{3},", stats.constantCount,
                      stats.constantSize, stats.variableCount,
                      stats.variableSize);

  // Synchronization:
  os << llvm::formatv("{0},", stats.awaitCount);

  // Execution:
  os << llvm::formatv("{0},{1},{2},{3},{4},{5},", stats.submissionCount,
                      stats.transientSize, stats.fillCount, stats.copyCount,
                      stats.dispatchCount, stats.callCount);

  // Executables:
  os << llvm::formatv("{0}", stats.executableCount);

  os << "\n";
  os << "\n";
}

static void dumpExecutionCSVTable(const UsageInfo &usageInfo,
                                  IREE::Stream::CmdExecuteOp executeOp,
                                  llvm::raw_fd_ostream &os) {
  os << "; ";
  prettyPrintOpBreadcrumb(executeOp, os);
  os << "\n";
  os << R"("Depth","Command","Symbol","Length","Invocations","Workload","Operands","Resources")";
  os << "\n";
  std::function<void(Operation *)> dumpRow;
  int depth = 0;
  dumpRow = [&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<IREE::Stream::CmdSerialOp>([&](auto op) {
          ++depth;
          for (auto &nestedOp : op.getBody().front()) dumpRow(&nestedOp);
          --depth;
        })
        .Case<IREE::Stream::CmdConcurrentOp>([&](auto op) {
          ++depth;
          for (auto &nestedOp : op.getBody().front()) dumpRow(&nestedOp);
          --depth;
        })
        .Case<IREE::Stream::CmdFillOp>([&](auto op) {
          APInt length;
          matchPattern(op.getTargetLength(), m_ConstantInt(&length));
          os << llvm::formatv(R"({0},"fill",,{1},,,,)", depth, length);
          os << "\n";
        })
        .Case<IREE::Stream::CmdCopyOp>([&](auto op) {
          APInt length;
          matchPattern(op.getLength(), m_ConstantInt(&length));
          os << llvm::formatv(R"({0},"copy",,{1},,,,)", depth, length);
          os << "\n";
        })
        .Case<IREE::Stream::CmdDispatchOp>([&](auto op) {
          auto workload = op.getWorkload();
          SmallString<32> workloadStr;
          for (unsigned i = 0; i < workload.size(); ++i) {
            if (i > 0) workloadStr.append(";");
            APInt dimValue;
            if (matchPattern(workload[i], m_ConstantInt(&dimValue))) {
              dimValue.toString(workloadStr, 10, /*signed=*/true);
            } else {
              workloadStr.append("?");
            }
          }
          APInt workloadSum = APInt(64, 1);
          for (auto dim : workload) {
            APInt dimValue;
            if (matchPattern(dim, m_ConstantInt(&dimValue))) {
              workloadSum *= dimValue;
            }
          }
          os << llvm::formatv(
              R"({0},"dispatch","{1}",,{2},"{3}",{4},{5})", depth,
              *op.getEntryPointRefs().begin(), workloadSum, workloadStr,
              op.getUniformOperands().size(), op.getResources().size());
          os << "\n";
        });
  };
  for (auto &op : executeOp.getBody().front()) {
    dumpRow(&op);
  }
  os << "\n";
}

static void dumpCSVTables(const UsageInfo &usageInfo,
                          llvm::raw_fd_ostream &os) {
  os << ";\n";
  os << "; Aggregate Statistics (static, whole-program)\n";
  os << ";\n\n";
  dumpAggregateCSVTable(usageInfo, os);

  // TODO(benvanik): globals/syncs/streams/etc.

  os << ";\n";
  os << "; Execution\n";
  os << ";\n\n";
  for (auto executeOp : usageInfo.executeOps) {
    dumpExecutionCSVTable(usageInfo, executeOp, os);
  }
}

//===----------------------------------------------------------------------===//
// JSON structures
//===----------------------------------------------------------------------===//

static void dumpAggregateJSONStructure(const UsageInfo &usageInfo,
                                       llvm::raw_fd_ostream &os) {
  Statistics stats;
  stats.analyze(usageInfo);

  const char kvPair[] = "    \"{0}\": {1},\n";
  const char kvPairNoComma[] = "    \"{0}\": {1}\n";

  os << "  \"global\": {\n";
  os << llvm::formatv(kvPair, "constant-count", stats.constantCount);
  os << llvm::formatv(kvPair, "constant-size", stats.constantSize);
  os << llvm::formatv(kvPair, "variable-count", stats.variableCount);
  os << llvm::formatv(kvPairNoComma, "variable-size", stats.variableSize);
  os << "  },\n";

  os << "  \"synchronization\": {\n";
  os << llvm::formatv(kvPairNoComma, "await-count", stats.awaitCount);
  os << "  },\n";

  os << "  \"execution\": {\n";
  os << llvm::formatv(kvPair, "submission-count", stats.submissionCount);
  os << llvm::formatv(kvPair, "transient-memory-size", stats.transientSize);
  os << llvm::formatv(kvPair, "fill-count", stats.fillCount);
  os << llvm::formatv(kvPair, "copy-count", stats.copyCount);
  os << llvm::formatv(kvPair, "dispatch-count", stats.dispatchCount);
  os << llvm::formatv(kvPairNoComma, "call-count", stats.callCount);
  os << "  },\n";

  os << "  \"executable\": {\n";
  os << llvm::formatv(kvPairNoComma, "executable-count", stats.executableCount);
  os << "  }\n";
}

static void dumpJSONStructures(const UsageInfo &usageInfo,
                               llvm::raw_fd_ostream &os) {
  os << "{\n";

  os << "\"stream-aggregate\": {\n";
  dumpAggregateJSONStructure(usageInfo, os);
  os << "}\n";

  // TODO(antiagainst): dump per-execution data if needed.

  os << "}\n";
}

//===----------------------------------------------------------------------===//
// -iree-stream-dump-statistics
//===----------------------------------------------------------------------===//

// Opens a canonical |filePath| for text output.
// An empty path can be used to target stderr and `-` will go to stdout.
// If the file cannot be opened stderr will be used.
static std::unique_ptr<llvm::raw_fd_ostream> openOutputFile(
    StringRef filePath) {
  if (filePath.empty()) {
    return std::make_unique<llvm::raw_fd_ostream>(2, false);  // stderr
  } else if (filePath == "-") {
    return std::make_unique<llvm::raw_fd_ostream>(1, false);  // stdout
  } else {
    std::error_code ec;
    auto result = std::make_unique<llvm::raw_fd_ostream>(
        filePath, ec, llvm::sys::fs::OF_TextWithCRLF);
    if (!ec) return result;
    llvm::errs() << "Error opening iree-stream-dump-statistics output file '"
                 << filePath << "'\n";
    return std::make_unique<llvm::raw_fd_ostream>(2, false);  // stderr.
  }
}

class DumpStatisticsPass : public DumpStatisticsBase<DumpStatisticsPass> {
 public:
  DumpStatisticsPass() = default;
  DumpStatisticsPass(DumpOutputFormat outputFormat, std::string outputFile) {
    this->outputFormat = outputFormat;
    this->outputFile = outputFile;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    if (outputFormat == DumpOutputFormat::None) return;

    // Open the output file we'll be streaming to.
    // Since we are processing the entire module at once we overwrite the file.
    auto os = openOutputFile(outputFile);

    // Walk the module once to accumulate everything we care about.
    auto moduleOp = getOperation();
    UsageInfo usageInfo;
    usageInfo.analyze(moduleOp);

    switch (outputFormat) {
      case DumpOutputFormat::Pretty:
      case DumpOutputFormat::Verbose:
        prettyPrintUsageInfo(usageInfo,
                             outputFormat == DumpOutputFormat::Verbose, *os);
        break;
      case DumpOutputFormat::CSV:
        dumpCSVTables(usageInfo, *os);
        break;
      case DumpOutputFormat::JSON:
        dumpJSONStructures(usageInfo, *os);
        break;
      default:
        break;
    }

    os->flush();
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createDumpStatisticsPass(
    DumpOutputFormat outputFormat, std::string outputFile) {
  return std::make_unique<DumpStatisticsPass>(outputFormat, outputFile);
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
