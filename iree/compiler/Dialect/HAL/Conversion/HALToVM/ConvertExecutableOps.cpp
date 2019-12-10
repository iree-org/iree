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

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/ConvertHALToVM.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Types.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

static FunctionType getExecutableCachingFunctionType(MLIRContext *context) {
  return FunctionType::get(
      {IREE::RefPtrType::get(IREE::HAL::DeviceType::get(context))},
      {IREE::RefPtrType::get(IREE::HAL::ExecutableType::get(context))},
      context);
}

// Converts a hal.executable to rodata and a lazy-initialization/caching getter.
// References to the executable must be converted to call the caching function.
//
// The hope is that the runtime HAL portions will cache executables globally and
// then this accessor caches internally to avoid lookups/locks and make
// debugging clearer.
//
// The resulting function is effectively:
//   func @exe(%device : !iree.ref<!hal.device>) -> !iree.ref<!hal.executable> {
//     if (%cached = @exe_cached) return %cached
//     %format = pick_supported_format(%device, exe_format_1, exe_format_2, ...)
//     switch (%format) {
//      case exe_format_1:
//        %exe = prepare_executable(%device, exe_format_1, @rodata_format_1)
//        break;
//      case exe_format_2:
//        %exe = prepare_executable(%device, exe_format_2, @rodata_format_2)
//        break;
//      default:
//       return nullptr;
//     }
//     @exe_cached = %exe
//     return %exe
//   }
//
// There are plenty of optimizations we could do here. For example:
// - group multiple executables together to prepare them as a batch
// - move the cache miss IR to its own function to let the inliner have a better
//   cache of inlining the load and compare
// - executable data compression
// - vm.switch instead of the chained cond branches (*much* simpler IR)
class ExecutableOpConversion
    : public OpConversionPattern<IREE::HAL::ExecutableOp> {
 public:
  using OpConversionPattern<IREE::HAL::ExecutableOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      IREE::HAL::ExecutableOp executableOp, llvm::ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Get the binary data for each format and create the rodata for each.
    llvm::SmallDenseMap<uint32_t, IREE::VM::RodataOp> rodataOps;
    for (auto binaryOp :
         executableOp.getBlock().getOps<IREE::HAL::ExecutableBinaryOp>()) {
      rodataOps[binaryOp.format().getZExtValue()] =
          rewriter.create<IREE::VM::RodataOp>(
              binaryOp.getLoc(),
              (executableOp.getName() + "_data_" +
               binaryOp.format().toString(8, false))
                  .str(),
              binaryOp.data());
    }

    // One global for now that represents the cached executable. We could add a
    // table for devices or something if we wanted, though that may be best done
    // as a custom type in the HAL runtime.
    auto loc = executableOp.getLoc();
    auto globalOp = rewriter.create<IREE::VM::GlobalRefOp>(
        loc, (executableOp.getName() + "_cached").str(),
        /*isMutable=*/true,
        IREE::RefPtrType::get(
            IREE::HAL::ExecutableType::get(rewriter.getContext())));

    // Caching function that creates the executable if needed.
    auto funcOp = rewriter.create<IREE::VM::FuncOp>(
        loc, executableOp.getName(),
        getExecutableCachingFunctionType(rewriter.getContext()),
        ArrayRef<NamedAttribute>{});
    auto *entryBlock = funcOp.addEntryBlock();
    auto *fastReturnBlock = funcOp.addBlock();
    auto *switchEntryBlock = funcOp.addBlock();
    auto *switchExitBlock = new Block();
    auto *failBlock = new Block();

    // if (auto cached_exe = @exe_cached) return cached_exe;
    OpBuilder funcBuilder(entryBlock);
    Value *deviceArg = entryBlock->getArgument(0);
    auto loadOp = funcBuilder.create<IREE::VM::GlobalLoadRefOp>(
        loc, globalOp.type(), funcBuilder.getSymbolRefAttr(globalOp));
    auto cmpOp = funcBuilder.create<IREE::VM::CmpNZRefOp>(
        loc, funcBuilder.getIntegerType(32), loadOp.getResult());
    funcBuilder.create<IREE::VM::CondBranchOp>(
        loc, cmpOp.result(), fastReturnBlock,
        ArrayRef<Value *>{loadOp.getResult()}, switchEntryBlock,
        ArrayRef<Value *>{});

    funcBuilder.setInsertionPointToStart(fastReturnBlock);
    funcBuilder.create<IREE::VM::ReturnOp>(
        loc, fastReturnBlock->addArgument(globalOp.type()));

    // Uncached; pick rodata and create.
    // We query which format is supported and then request that.
    funcBuilder.setInsertionPointToStart(switchEntryBlock);
    SmallVector<Value *, 4> queryCallArgs;
    queryCallArgs.push_back(deviceArg);
    for (auto formatRodataOp : rodataOps) {
      queryCallArgs.push_back(funcBuilder.createOrFold<IREE::VM::ConstI32Op>(
          loc, formatRodataOp.getFirst()));
    }
    auto queryCallOp = funcBuilder.create<IREE::VM::CallOp>(
        loc, "_hal.ex.match_supported_executable_format",
        ArrayRef<Type>{funcBuilder.getIntegerType(32)}, queryCallArgs);
    Value *selectedFormat = queryCallOp.getResult(0);

    // Switch on the result to pick the rodata.
    // We generate a big switch of if (format == xxx) goto cacheFormat(xxx);
    llvm::SmallDenseMap<uint32_t, Block *> formatMatchBlocks;
    llvm::SmallDenseMap<uint32_t, Block *> formatCacheBlocks;
    for (auto formatRodataOp : rodataOps) {
      formatMatchBlocks[formatRodataOp.getFirst()] = funcOp.addBlock();
      formatCacheBlocks[formatRodataOp.getFirst()] = funcOp.addBlock();
    }
    auto formatMatchBlockIt = formatMatchBlocks.begin();
    funcBuilder.create<IREE::VM::BranchOp>(loc, formatMatchBlockIt->getSecond(),
                                           ArrayRef<Value *>{selectedFormat});
    for (auto formatRodataOp : rodataOps) {
      uint32_t format = formatRodataOp.getFirst();
      auto &rodataOp = formatRodataOp.getSecond();

      OpBuilder caseBuilder(formatMatchBlocks[format]);
      Value *matchArg =
          caseBuilder.getBlock()->addArgument(caseBuilder.getIntegerType(32));
      auto cmpFormatOp = caseBuilder.create<IREE::VM::CmpEQI32Op>(
          loc, rewriter.getIntegerType(32), matchArg,
          caseBuilder.createOrFold<IREE::VM::ConstI32Op>(
              loc, formatRodataOp.getFirst()));
      ++formatMatchBlockIt;
      if (formatMatchBlockIt != formatMatchBlocks.end()) {
        caseBuilder.create<IREE::VM::CondBranchOp>(
            loc, cmpFormatOp.result(), formatCacheBlocks[format],
            ArrayRef<Value *>{matchArg}, formatMatchBlockIt->getSecond(),
            ArrayRef<Value *>{matchArg});
      } else {
        caseBuilder.create<IREE::VM::CondBranchOp>(
            loc, cmpFormatOp.result(), formatCacheBlocks[format],
            ArrayRef<Value *>{matchArg}, failBlock, ArrayRef<Value *>{});
      }

      OpBuilder cacheBuilder(formatCacheBlocks[format]);
      Value *formatArg =
          cacheBuilder.getBlock()->addArgument(caseBuilder.getIntegerType(32));
      auto loadRodataOp =
          cacheBuilder.create<IREE::VM::ConstRefRodataOp>(loc, rodataOp);
      auto cacheOp = cacheBuilder.create<IREE::VM::CallOp>(
          loc, "_hal.ex.cache_executable", ArrayRef<Type>{globalOp.type()},
          ArrayRef<Value *>{deviceArg, formatArg, loadRodataOp.value()});
      cacheBuilder.create<IREE::VM::BranchOp>(
          loc, switchExitBlock, ArrayRef<Value *>{cacheOp.getResult(0)});
    }

    funcOp.getBlocks().push_back(switchExitBlock);
    funcBuilder.setInsertionPointToStart(switchExitBlock);
    Value *valueArg = switchExitBlock->addArgument(globalOp.type());
    funcBuilder.create<IREE::VM::GlobalStoreRefOp>(
        loc, funcBuilder.getSymbolRefAttr(globalOp), valueArg);
    funcBuilder.create<IREE::VM::ReturnOp>(loc, valueArg);

    funcOp.getBlocks().push_back(failBlock);
    funcBuilder.setInsertionPointToStart(failBlock);
    funcBuilder.create<IREE::VM::ReturnOp>(
        loc, funcBuilder.create<IREE::VM::ConstRefZeroOp>(loc, globalOp.type())
                 .result());

    rewriter.eraseOp(executableOp);
    return matchSuccess();
  }
};

// Converts hal.ex.cache_executable to a call to the executable caching routine.
class ExCacheExecutableOpConversion
    : public OpConversionPattern<IREE::HAL::ExCacheExecutableOp> {
 public:
  using OpConversionPattern<
      IREE::HAL::ExCacheExecutableOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      IREE::HAL::ExCacheExecutableOp cacheExecutableOp,
      llvm::ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        cacheExecutableOp,
        rewriter.getSymbolRefAttr(cacheExecutableOp.executable()),
        ArrayRef<Type>{cacheExecutableOp.getResult()->getType()},
        ArrayRef<Value *>{operands[0]});
    return matchSuccess();
  }
};

}  // namespace

void populateHALExecutableToVMPatterns(MLIRContext *context,
                                       SymbolTable &importSymbols,
                                       TypeConverter &typeConverter,
                                       OwningRewritePatternList &patterns) {
  patterns.insert<ExecutableOpConversion, ExCacheExecutableOpConversion>(
      context);
}

}  // namespace iree_compiler
}  // namespace mlir
