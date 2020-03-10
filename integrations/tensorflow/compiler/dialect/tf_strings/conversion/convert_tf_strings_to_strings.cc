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

#include "integrations/tensorflow/compiler/dialect/tf_strings/conversion/convert_tf_strings_to_strings.h"

#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/ops.h"
#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/types.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Dialect.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Ops.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace TFStrings {

namespace {

Value ToStringHelper(Location loc, Value value, OpBuilder &builder) {
  auto type = value.getType();
  if (type.dyn_cast<TFStrings::StringType>()) {
    return value;
  } else if (type.isSignlessInteger(32)) {
    auto toString = builder.create<IREE::Strings::I32ToStringOp>(
        loc, builder.getType<IREE::Strings::StringType>(), value);
    return toString.getResult();
  }

  llvm_unreachable("Unsupported to_string type");
  return nullptr;
}

struct ToStringLowering : public OpRewritePattern<ToStringOp> {
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ToStringOp op,
                                     PatternRewriter &rewriter) const override {
    auto newValue = ToStringHelper(op.getLoc(), op.getOperand(), rewriter);
    rewriter.replaceOp(op, newValue);
    return matchSuccess();
  }
};

struct PrintOpLowering : public OpRewritePattern<PrintOp> {
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(PrintOp op,
                                     PatternRewriter &rewriter) const override {
    Value stringVal = nullptr;
    Type type = op.getOperand().getType();
    if (auto stringType = type.dyn_cast<TFStrings::StringType>()) {
      stringVal = op.getOperand();
    } else if (type.isSignlessInteger(32)) {
      auto toString = rewriter.create<IREE::Strings::I32ToStringOp>(
          op.getLoc(), rewriter.getType<IREE::Strings::StringType>(),
          op.getOperand());
      stringVal = toString.getResult();
    }

    rewriter.create<IREE::Strings::PrintOp>(op.getLoc(), stringVal);

    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

class IreeStringTypeConverter : public TypeConverter {
 public:
  IreeStringTypeConverter() {
    addConversion([](ShapedType type) {
      auto elementType = IREE::Strings::StringType::get(type.getContext());
      return RankedTensorType::get(type.getShape(), elementType);
    });
    addConversion([](TFStrings::StringType type) {
      return IREE::Strings::StringType::get(type.getContext());
    });
    addConversion([](Type type) -> Optional<Type> {
      if (!getElementTypeOrSelf(type).isa<TFStrings::StringType>()) {
        return type;
      }
      return llvm::None;
    });
  }

  Operation *materializeConversion(PatternRewriter &rewriter, Type resultType,
                                   ArrayRef<Value> inputs,
                                   Location loc) override {
    llvm_unreachable("unhandled materialization");
    return nullptr;
  }
};

}  // namespace

namespace {
class ConvertTFStringsToStringsPass
    : public ModulePass<ConvertTFStringsToStringsPass> {
 public:
  void runOnModule() override {
    auto module = getModule();
    OpBuilder builder(module.getContext());
    IreeStringTypeConverter typeConverter;

    OwningRewritePatternList patterns;
    populateFuncOpTypeConversionPattern(patterns, &getContext(), typeConverter);
    populateTFStringsToStringsPatterns(&getContext(), patterns);

    ConversionTarget target(getContext());
    target.addLegalDialect<IREE::Strings::StringsDialect>();
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });
    if (failed(applyPartialConversion(module, target, patterns))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<ConvertTFStringsToStringsPass> pass(
    "convert-tf_strings-to-strings",
    "Convert all XLA functions to the IREE dialect");

}  // namespace
}  // namespace TFStrings

void populateTFStringsToStringsPatterns(MLIRContext *ctx,
                                        OwningRewritePatternList &patterns) {
  patterns.insert<TFStrings::PrintOpLowering, TFStrings::ToStringLowering>(ctx);
}

}  // namespace iree_compiler
}  // namespace mlir
