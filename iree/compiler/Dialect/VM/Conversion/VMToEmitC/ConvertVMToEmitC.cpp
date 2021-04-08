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

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"

#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

SmallVector<Attribute, 4> indexSequence(int64_t n, MLIRContext *ctx) {
  return llvm::to_vector<4>(
      llvm::map_range(llvm::seq<int64_t>(0, n), [&ctx](int64_t i) -> Attribute {
        return IntegerAttr::get(IndexType::get(ctx), i);
      }));
}

template <typename AccessOpTy, typename GlobalOpTy>
GlobalOpTy lookupGlobalOp(AccessOpTy accessOp) {
  FlatSymbolRefAttr globalAttr =
      accessOp.getOperation()->template getAttrOfType<FlatSymbolRefAttr>(
          "global");
  GlobalOpTy globalOp =
      accessOp.getOperation()
          ->template getParentOfType<IREE::VM::ModuleOp>()
          .template lookupSymbol<GlobalOpTy>(globalAttr.getValue());
  return globalOp;
}

// Convert vm operations to emitc calls. The resultiong call has the ops
// operands as arguments followed by an argument for every attribute.
template <typename SrcOpTy>
class CallOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  CallOpConversion(MLIRContext *context, StringRef funcName)
      : OpConversionPattern<SrcOpTy>(context), funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto type = op.getOperation()->getResultTypes();
    StringAttr callee = rewriter.getStringAttr(funcName);

    // Default to an empty args attribute, which results in the operands being
    // printed as the arguments to the function call.
    ArrayAttr args;
    ArrayAttr templateArgs;

    // If the operation has attributes, we need to explicitely build the args
    // attribute of the emitc call op. This consists of index attributes for
    // the operands, followed by the source op attributes themselves.
    if (op->getAttrs().size() > 0) {
      SmallVector<Attribute, 4> args_ =
          indexSequence(operands.size(), op.getContext());

      for (NamedAttribute attr : op->getAttrs()) {
        args_.push_back(attr.second);
      }

      args = rewriter.getArrayAttr(args_);
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, type, callee, args,
                                               templateArgs, operands);

    return success();
  }

  StringRef funcName;
};

template <typename ConstOpTy>
class ConstOpConversion : public OpRewritePattern<ConstOpTy> {
 public:
  using OpRewritePattern<ConstOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstOpTy constOp,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::ConstOp>(constOp, constOp.getType(),
                                                constOp.value());
    return success();
  }
};

template <typename ConstZeroOpTy>
class ConstZeroOpConversion : public OpRewritePattern<ConstZeroOpTy> {
 public:
  using OpRewritePattern<ConstZeroOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstZeroOpTy constZeroOp,
                                PatternRewriter &rewriter) const final {
    auto type = constZeroOp.getType();
    IntegerAttr value = rewriter.getIntegerAttr(type, 0);

    rewriter.replaceOpWithNewOp<emitc::ConstOp>(constZeroOp, type, value);
    return success();
  }
};

class ConstRefZeroOpConversion
    : public OpRewritePattern<IREE::VM::ConstRefZeroOp> {
 public:
  using OpRewritePattern<IREE::VM::ConstRefZeroOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::VM::ConstRefZeroOp constRefZeroOp,
                                PatternRewriter &rewriter) const final {
    StringRef typeString = "iree_vm_ref_t";
    auto type = emitc::OpaqueType::get(constRefZeroOp.getContext(), typeString);

    StringRef valueString = "{0}";
    StringAttr value = rewriter.getStringAttr(valueString);

    rewriter.replaceOpWithNewOp<emitc::ConstOp>(constRefZeroOp, type, value);
    return success();
  }
};

template <typename LoadOpTy, typename GlobalOpTy>
class GlobalLoadOpConversion : public OpConversionPattern<LoadOpTy> {
  using OpConversionPattern<LoadOpTy>::OpConversionPattern;

 public:
  GlobalLoadOpConversion(MLIRContext *context, StringRef funcName)
      : OpConversionPattern<LoadOpTy>(context), funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      LoadOpTy loadOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    GlobalOpTy globalOp = lookupGlobalOp<LoadOpTy, GlobalOpTy>(loadOp);
    if (!globalOp) return loadOp.emitError() << "Unable to find GlobalOp";

    auto type = loadOp.getOperation()->getResultTypes();
    StringAttr callee = rewriter.getStringAttr(funcName);

    // TODO(simon-camp): We can't represent structs in emitc (yet maybe), so
    // the buffer where globals live after code generation as well as the
    // state struct argument name are hardcoded here.
    ArrayAttr args = rewriter.getArrayAttr(
        {rewriter.getStringAttr("state->rwdata"),
         rewriter.getUI32IntegerAttr(static_cast<uint32_t>(
             globalOp.ordinal().getValue().getZExtValue()))});
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(loadOp, type, callee, args,
                                               templateArgs, operands);

    return success();
  }

  StringRef funcName;
};

template <typename StoreOpTy, typename GlobalOpTy>
class GlobalStoreOpConversion : public OpConversionPattern<StoreOpTy> {
  using OpConversionPattern<StoreOpTy>::OpConversionPattern;

 public:
  GlobalStoreOpConversion(MLIRContext *context, StringRef funcName)
      : OpConversionPattern<StoreOpTy>(context), funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      StoreOpTy storeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    GlobalOpTy globalOp = lookupGlobalOp<StoreOpTy, GlobalOpTy>(storeOp);
    if (!globalOp) return storeOp.emitError() << "Unable to find GlobalOp";

    auto type = storeOp.getOperation()->getResultTypes();
    StringAttr callee = rewriter.getStringAttr(funcName);

    // TODO(simon-camp): We can't represent structs in emitc (yet maybe), so
    // the buffer where globals live after code generation as well as the
    // state struct argument name are hardcoded here.
    ArrayAttr args = rewriter.getArrayAttr(
        {rewriter.getStringAttr("state->rwdata"),
         rewriter.getUI32IntegerAttr(static_cast<uint32_t>(
             globalOp.ordinal().getValue().getZExtValue())),
         rewriter.getIndexAttr(0)});
    ArrayAttr templateArgs;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(storeOp, type, callee, args,
                                               templateArgs, operands);

    return success();
  }

  StringRef funcName;
};

class ListAllocOpConversion
    : public OpConversionPattern<IREE::VM::ListAllocOp> {
  using OpConversionPattern<IREE::VM::ListAllocOp>::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::ListAllocOp allocOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // clang-format off
    // The generated c code looks roughly like this.
    // iree_vm_type_def_t element_type = iree_vm_type_def_make_value_type(IREE_VM_VALUE_TYPE_I32);
    // iree_vm_type_def_t* element_type_ptr = &element_type;
    // iree_vm_list_t* list = nullptr;
    // iree_vm_list_t** list_ptr = &list;
    // iree_vm_list_create(&element_type, {initial_capacity}, state->allocator, &list);
    // clang-format on

    auto ctx = allocOp.getContext();
    auto loc = allocOp.getLoc();

    auto listType = allocOp.getType()
                        .cast<IREE::VM::RefType>()
                        .getObjectType()
                        .cast<IREE::VM::ListType>();
    auto elementType = listType.getElementType();
    std::string elementTypeStr;
    StringRef elementTypeConstructor;
    if (elementType.isa<IntegerType>()) {
      unsigned int bitWidth = elementType.getIntOrFloatBitWidth();
      elementTypeStr =
          std::string("IREE_VM_VALUE_TYPE_I") + std::to_string(bitWidth);
      elementTypeConstructor = "iree_vm_type_def_make_value_type";
    } else {
      return allocOp.emitError() << "Unhandeled element type " << elementType;
    }

    auto elementTypeOp = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_type_def_t"),
        /*callee=*/rewriter.getStringAttr(elementTypeConstructor),
        /*args=*/ArrayAttr::get(ctx, {StringAttr::get(ctx, elementTypeStr)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    auto elementTypePtrOp = rewriter.create<emitc::GetAddressOfOp>(
        /*location=*/loc,
        /*result=*/emitc::OpaqueType::get(ctx, "iree_vm_type_def_t*"),
        /*operand=*/elementTypeOp.getResult(0));

    auto listOp = rewriter.replaceOpWithNewOp<emitc::ConstOp>(
        /*op=*/allocOp,
        /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t*"),
        /*value=*/StringAttr::get(ctx, "nullptr"));

    auto listPtrOp = rewriter.create<emitc::GetAddressOfOp>(
        /*location=*/loc,
        /*result=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t**"),
        /*operand=*/listOp.getResult());

    rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/rewriter.getStringAttr("iree_vm_list_create"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
                             StringAttr::get(ctx, "state->allocator"),
                             rewriter.getIndexAttr(2)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{elementTypePtrOp.getResult(), operands[0],
                        listPtrOp.getResult()});

    return success();
  }
};

template <typename GetOpTy>
class ListGetOpConversion : public OpConversionPattern<GetOpTy> {
  using OpConversionPattern<GetOpTy>::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      GetOpTy getOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getOp.getContext();
    auto loc = getOp.getLoc();

    Optional<StringRef> valueTypeEnum;
    Optional<StringRef> valueExtractor;

    std::tie(valueTypeEnum, valueExtractor) =
        TypeSwitch<Operation *,
                   std::pair<Optional<StringRef>, Optional<StringRef>>>(
            getOp.getOperation())
            .Case<IREE::VM::ListGetI32Op>([&](auto op) {
              return std::make_pair(StringRef("IREE_VM_VALUE_TYPE_I32"),
                                    StringRef("vm_list_value_extract_i32"));
            })
            .template Case<IREE::VM::ListGetI64Op>([&](auto op) {
              return std::make_pair(StringRef("IREE_VM_VALUE_TYPE_I64"),
                                    StringRef("vm_list_value_extract_i64"));
            })
            .Default([](Operation *) { return std::make_pair(None, None); });

    if (!valueTypeEnum.hasValue() || !valueExtractor.hasValue()) {
      return getOp.emitOpError() << " not handeled";
    }

    auto valueOp = rewriter.create<emitc::ConstOp>(
        /*location=*/loc,
        /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_value_t"),
        /*value=*/StringAttr::get(ctx, ""));

    auto valuePtrOp = rewriter.create<emitc::GetAddressOfOp>(
        /*location=*/loc,
        /*result=*/emitc::OpaqueType::get(ctx, "iree_vm_value_t*"),
        /*operand=*/valueOp.getResult());

    auto getValueOp = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/rewriter.getStringAttr("iree_vm_list_get_value_as"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
                             StringAttr::get(ctx, valueTypeEnum.getValue()),
                             rewriter.getIndexAttr(2)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{getOp.list(), getOp.index(), valuePtrOp.getResult()});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/getOp,
        /*type=*/getOp.getType(),
        /*callee=*/rewriter.getStringAttr(valueExtractor.getValue()),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{valuePtrOp.getResult()});

    return success();
  }
};

template <typename SetOpTy>
class ListSetOpConversion : public OpConversionPattern<SetOpTy> {
  using OpConversionPattern<SetOpTy>::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      SetOpTy setOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = setOp.getContext();
    auto loc = setOp.getLoc();

    Optional<StringRef> valueConstructor =
        TypeSwitch<Operation *, Optional<StringRef>>(setOp.getOperation())
            .Case<IREE::VM::ListSetI32Op>(
                [&](auto op) { return StringRef("iree_vm_value_make_i32"); })
            .template Case<IREE::VM::ListSetI64Op>(
                [&](auto op) { return StringRef("iree_vm_value_make_i64"); })
            .Default([](Operation *) { return None; });

    if (!valueConstructor.hasValue()) {
      return setOp.emitOpError() << " not handeled";
    }

    auto valueOp = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_value_t"),
        /*callee=*/rewriter.getStringAttr(valueConstructor.getValue()),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{setOp.value()});

    auto valuePtrOp = rewriter.create<emitc::GetAddressOfOp>(
        /*location=*/loc,
        /*result=*/emitc::OpaqueType::get(ctx, "iree_vm_value_t*"),
        /*operand=*/valueOp.getResult(0));

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/setOp,
        /*type=*/TypeRange{},
        /*callee=*/rewriter.getStringAttr("iree_vm_list_set_value"),
        /*args=*/
        ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{setOp.list(), setOp.index(), valuePtrOp.getResult()});

    return success();
  }
};
}  // namespace

void populateVMToCPatterns(MLIRContext *context,
                           OwningRewritePatternList &patterns) {
  // Globals
  patterns.insert<
      GlobalLoadOpConversion<IREE::VM::GlobalLoadI32Op, IREE::VM::GlobalI32Op>>(
      context, "vm_global_load_i32");
  patterns.insert<GlobalStoreOpConversion<IREE::VM::GlobalStoreI32Op,
                                          IREE::VM::GlobalI32Op>>(
      context, "vm_global_store_i32");

  // Constants
  patterns.insert<ConstOpConversion<IREE::VM::ConstI32Op>>(context);
  patterns.insert<ConstZeroOpConversion<IREE::VM::ConstI32ZeroOp>>(context);
  patterns.insert<ConstRefZeroOpConversion>(context);

  // List ops
  // TODO(simon-camp): We leak memory in the generated code, as we never release
  // the lists.
  patterns.insert<ListAllocOpConversion>(context);
  patterns.insert<CallOpConversion<IREE::VM::ListReserveOp>>(
      context, "iree_vm_list_reserve");
  patterns.insert<CallOpConversion<IREE::VM::ListResizeOp>>(
      context, "iree_vm_list_resize");
  patterns.insert<CallOpConversion<IREE::VM::ListSizeOp>>(context,
                                                          "iree_vm_list_size");
  patterns.insert<ListGetOpConversion<IREE::VM::ListGetI32Op>>(context);
  patterns.insert<ListSetOpConversion<IREE::VM::ListSetI32Op>>(context);

  // Conditional assignment ops
  patterns.insert<CallOpConversion<IREE::VM::SelectI32Op>>(context,
                                                           "vm_select_i32");

  // Native integer arithmetic ops
  patterns.insert<CallOpConversion<IREE::VM::AddI32Op>>(context, "vm_add_i32");
  patterns.insert<CallOpConversion<IREE::VM::SubI32Op>>(context, "vm_sub_i32");
  patterns.insert<CallOpConversion<IREE::VM::MulI32Op>>(context, "vm_mul_i32");
  patterns.insert<CallOpConversion<IREE::VM::DivI32SOp>>(context,
                                                         "vm_div_i32s");
  patterns.insert<CallOpConversion<IREE::VM::DivI32UOp>>(context,
                                                         "vm_div_i32u");
  patterns.insert<CallOpConversion<IREE::VM::RemI32SOp>>(context,
                                                         "vm_rem_i32s");
  patterns.insert<CallOpConversion<IREE::VM::RemI32UOp>>(context,
                                                         "vm_rem_i32u");
  patterns.insert<CallOpConversion<IREE::VM::NotI32Op>>(context, "vm_not_i32");
  patterns.insert<CallOpConversion<IREE::VM::AndI32Op>>(context, "vm_and_i32");
  patterns.insert<CallOpConversion<IREE::VM::OrI32Op>>(context, "vm_or_i32");
  patterns.insert<CallOpConversion<IREE::VM::XorI32Op>>(context, "vm_xor_i32");

  // Casting and type conversion/emulation ops
  patterns.insert<CallOpConversion<IREE::VM::TruncI32I8Op>>(context,
                                                            "vm_trunc_i32i8");
  patterns.insert<CallOpConversion<IREE::VM::TruncI32I16Op>>(context,
                                                             "vm_trunc_i32i16");
  patterns.insert<CallOpConversion<IREE::VM::ExtI8I32SOp>>(context,
                                                           "vm_ext_i8i32s");
  patterns.insert<CallOpConversion<IREE::VM::ExtI8I32UOp>>(context,
                                                           "vm_ext_i8i32u");
  patterns.insert<CallOpConversion<IREE::VM::ExtI16I32SOp>>(context,
                                                            "vm_ext_i16i32s");
  patterns.insert<CallOpConversion<IREE::VM::ExtI16I32UOp>>(context,
                                                            "vm_ext_i16i32u");

  // Native bitwise shift and rotate ops
  patterns.insert<CallOpConversion<IREE::VM::ShlI32Op>>(context, "vm_shl_i32");
  patterns.insert<CallOpConversion<IREE::VM::ShrI32SOp>>(context,
                                                         "vm_shr_i32s");
  patterns.insert<CallOpConversion<IREE::VM::ShrI32UOp>>(context,
                                                         "vm_shr_i32u");

  // Comparison ops
  patterns.insert<CallOpConversion<IREE::VM::CmpEQI32Op>>(context,
                                                          "vm_cmp_eq_i32");
  patterns.insert<CallOpConversion<IREE::VM::CmpNEI32Op>>(context,
                                                          "vm_cmp_ne_i32");
  patterns.insert<CallOpConversion<IREE::VM::CmpLTI32SOp>>(context,
                                                           "vm_cmp_lt_i32s");
  patterns.insert<CallOpConversion<IREE::VM::CmpLTI32UOp>>(context,
                                                           "vm_cmp_lt_i32u");
  patterns.insert<CallOpConversion<IREE::VM::CmpNZI32Op>>(context,
                                                          "vm_cmp_nz_i32");

  // ExtI64: Constants
  patterns.insert<ConstOpConversion<IREE::VM::ConstI64Op>>(context);
  patterns.insert<ConstZeroOpConversion<IREE::VM::ConstI64ZeroOp>>(context);

  // ExtI64: List ops
  patterns.insert<ListGetOpConversion<IREE::VM::ListGetI64Op>>(context);
  patterns.insert<ListSetOpConversion<IREE::VM::ListSetI64Op>>(context);

  // ExtI64: Conditional assignment ops
  patterns.insert<CallOpConversion<IREE::VM::SelectI64Op>>(context,
                                                           "vm_select_i64");
  // ExtI64: Native integer arithmetic ops
  patterns.insert<CallOpConversion<IREE::VM::AddI64Op>>(context, "vm_add_i64");
  patterns.insert<CallOpConversion<IREE::VM::SubI64Op>>(context, "vm_sub_i64");
  patterns.insert<CallOpConversion<IREE::VM::MulI64Op>>(context, "vm_mul_i64");
  patterns.insert<CallOpConversion<IREE::VM::DivI64SOp>>(context,
                                                         "vm_div_i64s");
  patterns.insert<CallOpConversion<IREE::VM::DivI64UOp>>(context,
                                                         "vm_div_i64u");
  patterns.insert<CallOpConversion<IREE::VM::RemI64SOp>>(context,
                                                         "vm_rem_i64s");
  patterns.insert<CallOpConversion<IREE::VM::RemI64UOp>>(context,
                                                         "vm_rem_i64u");
  patterns.insert<CallOpConversion<IREE::VM::NotI64Op>>(context, "vm_not_i64");
  patterns.insert<CallOpConversion<IREE::VM::AndI64Op>>(context, "vm_and_i64");
  patterns.insert<CallOpConversion<IREE::VM::OrI64Op>>(context, "vm_or_i64");
  patterns.insert<CallOpConversion<IREE::VM::XorI64Op>>(context, "vm_xor_i64");

  // ExtI64: Casting and type conversion/emulation ops
  patterns.insert<CallOpConversion<IREE::VM::TruncI64I32Op>>(context,
                                                             "vm_trunc_i64i32");
  patterns.insert<CallOpConversion<IREE::VM::ExtI32I64SOp>>(context,
                                                            "vm_ext_i32i64s");
  patterns.insert<CallOpConversion<IREE::VM::ExtI32I64UOp>>(context,
                                                            "vm_ext_i32i64u");

  // ExtI64: Native bitwise shift and rotate ops
  patterns.insert<CallOpConversion<IREE::VM::ShlI64Op>>(context, "vm_shl_i64");
  patterns.insert<CallOpConversion<IREE::VM::ShrI64SOp>>(context,
                                                         "vm_shr_i64s");
  patterns.insert<CallOpConversion<IREE::VM::ShrI64UOp>>(context,
                                                         "vm_shr_i64u");

  // ExtI64: Comparison ops
  patterns.insert<CallOpConversion<IREE::VM::CmpEQI64Op>>(context,
                                                          "vm_cmp_eq_i64");
  patterns.insert<CallOpConversion<IREE::VM::CmpNEI64Op>>(context,
                                                          "vm_cmp_ne_i64");
  patterns.insert<CallOpConversion<IREE::VM::CmpLTI64SOp>>(context,
                                                           "vm_cmp_lt_i64s");
  patterns.insert<CallOpConversion<IREE::VM::CmpLTI64UOp>>(context,
                                                           "vm_cmp_lt_i64u");
  patterns.insert<CallOpConversion<IREE::VM::CmpNZI64Op>>(context,
                                                          "vm_cmp_nz_i64");
}

namespace IREE {
namespace VM {

namespace {

// A pass converting IREE VM operations into the EmitC dialect.
class ConvertVMToEmitCPass
    : public PassWrapper<ConvertVMToEmitCPass,
                         OperationPass<IREE::VM::ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::emitc::EmitCDialect, IREEDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    OwningRewritePatternList patterns(&getContext());
    populateVMToCPatterns(&getContext(), patterns);

    target.addLegalDialect<mlir::emitc::EmitCDialect>();
    target.addLegalDialect<iree_compiler::IREEDialect>();
    target.addIllegalDialect<IREE::VM::VMDialect>();

    // Structural ops
    target.addLegalOp<IREE::VM::ModuleOp>();
    target.addLegalOp<IREE::VM::ModuleTerminatorOp>();
    target.addLegalOp<IREE::VM::FuncOp>();
    target.addLegalOp<IREE::VM::GlobalI32Op>();
    target.addLegalOp<IREE::VM::ExportOp>();

    // Control flow ops
    target.addLegalOp<IREE::VM::BranchOp>();
    target.addLegalOp<IREE::VM::CallOp>();
    target.addLegalOp<IREE::VM::CondBranchOp>();
    // Note: We translate the fail op to two function calls in the end, but we
    // can't simply convert it here because it is a terminator.
    target.addLegalOp<IREE::VM::FailOp>();
    target.addLegalOp<IREE::VM::ReturnOp>();

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createConvertVMToEmitCPass() {
  return std::make_unique<ConvertVMToEmitCPass>();
}

}  // namespace VM
}  // namespace IREE

static PassRegistration<IREE::VM::ConvertVMToEmitCPass> pass(
    "iree-convert-vm-to-emitc", "Convert VM Ops to the EmitC dialect");

}  // namespace iree_compiler
}  // namespace mlir
