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

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_dialect.h"

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/conversion/convert_flow_to_hal.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir {
namespace tf_tensorlist {

static DialectRegistration<TfTensorListDialect> registration;

LogicalResult Reserve::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    ArrayRef<NamedAttribute> attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferedReturnTypes) {
  inferedReturnTypes.push_back(TensorListType::get(context));
  return success();
}

LogicalResult SetItem::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    ArrayRef<NamedAttribute> attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferedReturnTypes) {
  inferedReturnTypes.push_back(TensorListType::get(context));
  return success();
}

LogicalResult FromVariant::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    ArrayRef<NamedAttribute> attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferedReturnTypes) {
  inferedReturnTypes.push_back(TensorListType::get(context));
  return success();
}

namespace {
class FromVariantToVariantIsIdentity : public OpRewritePattern<FromVariant> {
 public:
  using OpRewritePattern<FromVariant>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(FromVariant from_variant,
                                     PatternRewriter &rewriter) const override {
    auto to_variant =
        dyn_cast<ToVariant>(from_variant.variant().getDefiningOp());
    if (!to_variant) return matchFailure();
    from_variant.list().replaceAllUsesWith(to_variant.list());
    rewriter.eraseOp(from_variant);
    return matchSuccess();
  }
};
}  // namespace

void FromVariant::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<FromVariantToVariantIsIdentity>(context);
}

//===----------------------------------------------------------------------===//
// TfTensorListDialect Dialect
//===----------------------------------------------------------------------===//

TfTensorListDialect::TfTensorListDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addInterfaces<iree_compiler::TfTensorListToHALConversionInterface>();
  addOperations<
#define GET_OP_LIST
#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.cc.inc"
      >();
  addTypes<TensorListType>();
}

Type TfTensorListDialect::parseType(DialectAsmParser &parser) const {
  StringRef type_name;
  if (parser.parseKeyword(&type_name)) return nullptr;
  if (type_name == "list") {
    return TensorListType::get(getContext());
  }
  parser.emitError(parser.getCurrentLocation(),
                   "unknown type in `tf_tensorlist` dialect");
  return nullptr;
}

void TfTensorListDialect::printType(Type type,
                                    DialectAsmPrinter &printer) const {
  printer << "list";
}

}  // namespace tf_tensorlist
}  // namespace mlir
