// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCBuilders.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir::iree_compiler::emitc_builders {

namespace {
std::string mapPreprocessorDirective(PreprocessorDirective directive) {
  switch (directive) {
  case PreprocessorDirective::DEFINE:
    return "#define";
  case PreprocessorDirective::UNDEF:
    return "#undef";
  case PreprocessorDirective::IFDEF:
    return "#ifdef";
  case PreprocessorDirective::IFNDEF:
    return "#ifndef";
  case PreprocessorDirective::IF:
    return "#if";
  case PreprocessorDirective::ENDIF:
    return "#endif";
  case PreprocessorDirective::ELSE:
    return "#else";
  case PreprocessorDirective::ELIF:
    return "#elif";
  case PreprocessorDirective::LINE:
    return "#line";
  case PreprocessorDirective::ERROR:
    return "#error";
  case PreprocessorDirective::INCLUDE:
    return "#include";
  case PreprocessorDirective::PRAGMA:
    return "#pragma";
  default:
    llvm_unreachable("unsupported preprocessor directive");
    return "XXX";
  }
}
} // namespace

TypedValue<emitc::LValueType> allocateVariable(OpBuilder builder,
                                               Location location, Type type,
                                               Attribute initializer) {
  Value result =
      emitc::VariableOp::create(builder,
                                /*location=*/location,
                                /*resultType=*/emitc::LValueType::get(type),
                                /*value=*/initializer)
          .getResult();
  return cast<TypedValue<emitc::LValueType>>(result);
}

TypedValue<emitc::LValueType>
allocateVariable(OpBuilder builder, Location location, Type type,
                 std::optional<StringRef> initializer) {
  auto ctx = builder.getContext();
  return allocateVariable(
      builder, location, type,
      emitc::OpaqueAttr::get(ctx, initializer.value_or("")));
}

std::pair<TypedValue<emitc::LValueType>, TypedValue<emitc::PointerType>>
allocZeroInitializedVar(OpBuilder builder, Location location, Type type) {
  auto var = allocateVariable(builder, location, type);
  auto varPtr = addressOf(builder, location, var);
  auto size = sizeOf(builder, location, TypeAttr::get(type));
  emitc_builders::memset(builder, location, varPtr, 0, size);
  return {var, varPtr};
}

TypedValue<emitc::LValueType> asLValue(OpBuilder builder, Location loc,
                                       Value value) {
  auto var = allocateVariable(builder, loc, value.getType());
  emitc::AssignOp::create(builder, loc, var, value);
  return var;
}

Value asRValue(OpBuilder builder, Location loc,
               TypedValue<emitc::LValueType> value) {
  return emitc::LoadOp::create(builder, loc, value.getType().getValueType(),
                               value);
}

void asRValues(OpBuilder builder, Location location,
               SmallVector<Value> &values) {
  for (auto &value : values) {
    if (auto lvalue = dyn_cast<TypedValue<emitc::LValueType>>(value)) {
      value = emitc_builders::asRValue(builder, location, lvalue);
    }
  }
}

TypedValue<emitc::PointerType>
addressOf(OpBuilder builder, Location location,
          TypedValue<emitc::LValueType> operand) {
  auto ctx = builder.getContext();

  auto result = emitc::ApplyOp::create(
                    builder,
                    /*location=*/location,
                    /*result=*/
                    emitc::PointerType::get(operand.getType().getValueType()),
                    /*applicableOperator=*/StringAttr::get(ctx, "&"),
                    /*operand=*/operand)
                    .getResult();

  return cast<TypedValue<emitc::PointerType>>(result);
}

Value contentsOf(OpBuilder builder, Location location, Value operand) {
  auto ctx = builder.getContext();

  Type type = operand.getType();
  assert(isa<emitc::PointerType>(type));

  return emitc::ApplyOp::create(
             builder,
             /*location=*/location,
             /*result=*/cast<emitc::PointerType>(type).getPointee(),
             /*applicableOperator=*/StringAttr::get(ctx, "*"),
             /*operand=*/operand)
      .getResult();
}

Value sizeOf(OpBuilder builder, Location loc, Value value) {
  auto ctx = builder.getContext();
  return emitc::CallOpaqueOp::create(
             builder,
             /*location=*/loc,
             /*type=*/emitc::OpaqueType::get(ctx, "iree_host_size_t"),
             /*callee=*/"sizeof",
             /*operands=*/ArrayRef<Value>{value})
      .getResult(0);
}

Value sizeOf(OpBuilder builder, Location loc, Attribute attr) {
  auto ctx = builder.getContext();
  return emitc::CallOpaqueOp::create(
             builder,
             /*location=*/loc,
             /*type=*/emitc::OpaqueType::get(ctx, "iree_host_size_t"),
             /*callee=*/"sizeof",
             /*operands=*/ArrayRef<Value>{},
             /*args=*/ArrayAttr::get(ctx, {attr}))
      .getResult(0);
}

void memcpy(OpBuilder builder, Location location, Value dest, Value src,
            Value count) {
  emitc::CallOpaqueOp::create(builder,
                              /*location=*/location,
                              /*type=*/TypeRange{},
                              /*callee=*/"memcpy",
                              /*operands=*/ArrayRef<Value>{dest, src, count});
}

void memset(OpBuilder builder, Location location, Value dest, int ch,
            Value count) {
  auto ctx = builder.getContext();
  emitc::CallOpaqueOp::create(
      builder,
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/"memset",
      /*operands=*/ArrayRef<Value>{dest, count},
      /*args=*/
      ArrayAttr::get(ctx,
                     {builder.getIndexAttr(0), builder.getUI32IntegerAttr(ch),
                      builder.getIndexAttr(1)}));
}

Value alignTo(OpBuilder builder, Location location, Value size,
              size_t alignment) {
  auto ctx = builder.getContext();
  Type hostSizeType = emitc::OpaqueType::get(ctx, "iree_host_size_t");
  Value alignValue = emitc::LiteralOp::create(builder, location, hostSizeType,
                                              std::to_string(alignment));
  return emitc::CallOpaqueOp::create(
             builder,
             /*location=*/location,
             /*type=*/hostSizeType,
             /*callee=*/"iree_host_align",
             /*operands=*/ArrayRef<Value>{size, alignValue})
      .getResult(0);
}

Value alignPtr(OpBuilder builder, Location location, Value ptr,
               size_t alignment) {
  auto ctx = builder.getContext();
  Type hostSizeType = emitc::OpaqueType::get(ctx, "iree_host_size_t");
  Type uintptrType = emitc::OpaqueType::get(ctx, "uintptr_t");
  Type bytePtrType = emitc::PointerType::get(builder.getIntegerType(8, false));

  // Cast ptr to uintptr_t.
  Value ptrAsInt =
      emitc::CastOp::create(builder, location, uintptrType, ptr).getResult();

  // Align.
  Value alignValue = emitc::LiteralOp::create(builder, location, hostSizeType,
                                              std::to_string(alignment));
  Value aligned = emitc::CallOpaqueOp::create(
                      builder, location, uintptrType, "iree_host_align",
                      ArrayRef<Value>{ptrAsInt, alignValue})
                      .getResult(0);

  // Cast back to uint8_t*.
  return emitc::CastOp::create(builder, location, bytePtrType, aligned)
      .getResult();
}

Value arrayElement(OpBuilder builder, Location location, size_t index,
                   TypedValue<emitc::PointerType> operand) {
  auto ctx = builder.getContext();
  Type type = emitc::OpaqueType::get(ctx, "iree_host_size_t");
  Value indexValue =
      emitc::LiteralOp::create(builder, location, type, std::to_string(index));
  TypedValue<emitc::LValueType> subscript =
      emitc::SubscriptOp::create(builder, location, operand, indexValue);
  return emitc::LoadOp::create(builder, location,
                               subscript.getType().getValueType(), subscript)
      .getResult();
}

Value arrayElementAddress(OpBuilder builder, Location location, size_t index,
                          TypedValue<emitc::PointerType> operand) {
  auto ctx = builder.getContext();
  Type type = emitc::OpaqueType::get(ctx, "iree_host_size_t");
  Value indexValue =
      emitc::LiteralOp::create(builder, location, type, std::to_string(index));
  return arrayElementAddress(builder, location, indexValue, operand);
}

Value arrayElementAddress(OpBuilder builder, Location location, Value index,
                          TypedValue<emitc::PointerType> operand) {
  TypedValue<emitc::LValueType> subscript =
      emitc::SubscriptOp::create(builder, location, operand, index);
  return addressOf(builder, location, subscript);
}

void arrayElementAssign(OpBuilder builder, Location location,
                        TypedValue<emitc::PointerType> array, size_t index,
                        Value value) {
  auto ctx = builder.getContext();
  Type type = emitc::OpaqueType::get(ctx, "iree_host_size_t");
  Value indexValue =
      emitc::LiteralOp::create(builder, location, type, std::to_string(index));
  TypedValue<emitc::LValueType> subscript =
      emitc::SubscriptOp::create(builder, location, array, indexValue);
  emitc::AssignOp::create(builder, location, subscript, value);
}

void structDefinition(OpBuilder builder, Location location,
                      StringRef structName, ArrayRef<StructField> fields) {
  auto ctx = builder.getContext();
  std::string decl = std::string("struct ") + structName.str() + " {";
  for (auto &field : fields) {
    decl += field.type + " " + field.name;
    if (field.isArray()) {
      decl += "[" + std::to_string(field.arraySize.value()) + "]";
    }
    decl += ";";
  }
  decl += "};";

  emitc::VerbatimOp::create(builder, location, StringAttr::get(ctx, decl));
}

Value structMember(OpBuilder builder, Location location, Type type,
                   StringRef memberName, TypedValue<mlir::Type> operand) {
  TypedValue<mlir::Type> member =
      emitc::MemberOp::create(builder, location, emitc::LValueType::get(type),
                              memberName, operand)
          .getResult();
  return emitc::LoadOp::create(builder, location, type, member).getResult();
}

TypedValue<emitc::PointerType>
structMemberAddress(OpBuilder builder, Location location,
                    emitc::PointerType type, StringRef memberName,
                    TypedValue<emitc::LValueType> operand) {
  auto member = emitc::MemberOp::create(builder, location, type.getPointee(),
                                        memberName, operand);
  return addressOf(builder, location,
                   cast<TypedValue<emitc::LValueType>>(member.getResult()));
}

void structMemberAssign(OpBuilder builder, Location location,
                        StringRef memberName, TypedValue<mlir::Type> operand,
                        Value data) {
  Value member = emitc::MemberOp::create(builder, location,
                                         emitc::LValueType::get(data.getType()),
                                         memberName, operand);
  emitc::AssignOp::create(builder, location, member, data);
}

Value structPtrMember(OpBuilder builder, Location location, Type type,
                      StringRef memberName,
                      TypedValue<emitc::LValueType> operand) {
  TypedValue<mlir::Type> member = emitc::MemberOfPtrOp::create(
      builder, location, emitc::LValueType::get(type), memberName, operand);
  return emitc::LoadOp::create(builder, location, type, member).getResult();
}

TypedValue<emitc::PointerType>
structPtrMemberAddress(OpBuilder builder, Location location,
                       emitc::PointerType type, StringRef memberName,
                       TypedValue<emitc::LValueType> operand) {
  auto member = emitc::MemberOfPtrOp::create(
      builder, location, emitc::LValueType::get(type.getPointee()), memberName,
      operand);
  return addressOf(builder, location,
                   cast<TypedValue<emitc::LValueType>>(member.getResult()));
}

void structPtrMemberAssign(OpBuilder builder, Location location,
                           StringRef memberName,
                           TypedValue<emitc::LValueType> operand, Value data) {
  Value member = emitc::MemberOfPtrOp::create(
      builder, location, emitc::LValueType::get(data.getType()), memberName,
      operand);
  emitc::AssignOp::create(builder, location, member, data);
}

Value ireeMakeCstringView(OpBuilder builder, Location location,
                          std::string str) {
  std::string escapedStr;
  llvm::raw_string_ostream os(escapedStr);
  os.write_escaped(str);
  auto quotedStr = std::string("\"") + escapedStr + std::string("\"");

  auto ctx = builder.getContext();
  return emitc::CallOpaqueOp::create(
             builder,
             /*location=*/location,
             /*type=*/emitc::OpaqueType::get(ctx, "iree_string_view_t"),
             /*callee=*/"iree_make_cstring_view",
             /*operands=*/ArrayRef<Value>{},
             /*args=*/
             ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(ctx, quotedStr)}))
      .getResult(0);
}

Value ireeOkStatus(OpBuilder builder, Location location) {
  auto ctx = builder.getContext();
  return emitc::CallOpaqueOp::create(
             builder,
             /*location=*/location,
             /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
             /*callee=*/"iree_ok_status",
             /*operands=*/ArrayRef<Value>{})
      .getResult(0);
}

Value ireeVmInstanceLookupType(OpBuilder builder, Location location,
                               Value instance, Value stringView) {
  auto ctx = builder.getContext();
  Type refType = emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t");
  return emitc::CallOpaqueOp::create(
             builder,
             /*location=*/location,
             /*type=*/refType,
             /*callee=*/"iree_vm_instance_lookup_type",
             /*operands=*/ArrayRef<Value>{instance, stringView})
      .getResult(0);
}

void ireeVmRefRelease(OpBuilder builder, Location location, Value operand) {
  emitc::CallOpaqueOp::create(builder,
                              /*location=*/location,
                              /*type=*/TypeRange{},
                              /*callee=*/"iree_vm_ref_release",
                              /*operands=*/ArrayRef<Value>{operand});
}

emitc::VerbatimOp preprocessorDirective(OpBuilder builder, Location location,
                                        PreprocessorDirective directive,
                                        StringRef value) {

  auto t = mapPreprocessorDirective(directive);
  if (!value.empty()) {
    t += " ";
    t += value;
  }

  return emitc::VerbatimOp::create(builder, location, t);
}

} // namespace mlir::iree_compiler::emitc_builders
