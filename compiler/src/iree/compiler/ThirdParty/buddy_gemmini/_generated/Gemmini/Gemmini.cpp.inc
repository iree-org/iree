/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Gemmini.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::buddy::gemmini::ComputeAccumulatedOp,
::buddy::gemmini::ComputePreloadedOp,
::buddy::gemmini::ConfigExOp,
::buddy::gemmini::ConfigLdOp,
::buddy::gemmini::ConfigNormOp,
::buddy::gemmini::ConfigStOp,
::buddy::gemmini::FlushOp,
::buddy::gemmini::ComputeAccumulated_IntrOp,
::buddy::gemmini::ComputePreloaded_IntrOp,
::buddy::gemmini::ConfigEX_IntrOp,
::buddy::gemmini::ConfigNorm_IntrOp,
::buddy::gemmini::ConfigSt_IntrOp,
::buddy::gemmini::ConifgLd_IntrOp,
::buddy::gemmini::Flush_IntrOp,
::buddy::gemmini::LoopConvWsConfig1_IntrOp,
::buddy::gemmini::LoopConvWsConfig2_IntrOp,
::buddy::gemmini::LoopConvWsConfig3_IntrOp,
::buddy::gemmini::LoopConvWsConfig4_IntrOp,
::buddy::gemmini::LoopConvWsConfig5_IntrOp,
::buddy::gemmini::LoopConvWsConfig6_IntrOp,
::buddy::gemmini::LoopConvWs_IntrOp,
::buddy::gemmini::LoopWsConfigAddrsAB_IntrOp,
::buddy::gemmini::LoopWsConfigAddrsDC_IntrOp,
::buddy::gemmini::LoopWsConfigBounds_IntrOp,
::buddy::gemmini::LoopWsConfigStridesAB_IntrOp,
::buddy::gemmini::LoopWsConfigStridesDC_IntrOp,
::buddy::gemmini::LoopWs_IntrOp,
::buddy::gemmini::Mvin2_IntrOp,
::buddy::gemmini::Mvin3_IntrOp,
::buddy::gemmini::Mvin_IntrOp,
::buddy::gemmini::Mvout_IntrOp,
::buddy::gemmini::Preload_IntrOp,
::buddy::gemmini::Mvin2Op,
::buddy::gemmini::Mvin3Op,
::buddy::gemmini::MvinOp,
::buddy::gemmini::MvoutOp,
::buddy::gemmini::PreloadOp,
::buddy::gemmini::PreloadZerosOp,
::buddy::gemmini::PrintOp,
::buddy::gemmini::TileConvOp,
::buddy::gemmini::TileMatMulOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace buddy {
namespace gemmini {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_Gemmini1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(64)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 64-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_Gemmini2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::mlir::LLVM::isCompatibleOuterType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM dialect-compatible type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_Gemmini3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::MemRefType>(type))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 2))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 2D memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_Gemmini4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::MemRefType>(type))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || (((::llvm::isa<::mlir::MemRefType>(type))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || (((::llvm::isa<::mlir::MemRefType>(type))) && ([](::mlir::Type elementType) { return (elementType.isF32()); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || (((::llvm::isa<::mlir::MemRefType>(type))) && ([](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of 8-bit signless integer values or memref of 32-bit signless integer values or memref of 32-bit float values or memref of 64-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_Gemmini5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::MemRefType>(type))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 4))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 4D memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_Gemmini6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::MemRefType>(type))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1D memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_Gemmini1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_Gemmini1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_Gemmini1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_Gemmini2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::FloatAttr>(attr))) && ((::llvm::cast<::mlir::FloatAttr>(attr).getType().isF32()))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit float attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_Gemmini2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_Gemmini2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_Gemmini3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::BoolAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: bool attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_Gemmini3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_Gemmini3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
} // namespace gemmini
} // namespace buddy
namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::ComputeAccumulatedOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ComputeAccumulatedOpAdaptor::ComputeAccumulatedOpAdaptor(ComputeAccumulatedOp op) : ComputeAccumulatedOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ComputeAccumulatedOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ComputeAccumulatedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols) {
  odsState.addOperands(aAddr);
  odsState.addOperands(bdAddr);
  odsState.addOperands(aRows);
  odsState.addOperands(aCols);
  odsState.addOperands(bdRows);
  odsState.addOperands(bdCols);
}

void ComputeAccumulatedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols) {
  odsState.addOperands(aAddr);
  odsState.addOperands(bdAddr);
  odsState.addOperands(aRows);
  odsState.addOperands(aCols);
  odsState.addOperands(bdRows);
  odsState.addOperands(bdCols);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ComputeAccumulatedOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 6u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ComputeAccumulatedOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ComputeAccumulatedOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ComputeAccumulatedOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand aAddrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aAddrOperands(&aAddrRawOperand, 1);  ::llvm::SMLoc aAddrOperandsLoc;
  (void)aAddrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bdAddrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bdAddrOperands(&bdAddrRawOperand, 1);  ::llvm::SMLoc bdAddrOperandsLoc;
  (void)bdAddrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand aRowsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aRowsOperands(&aRowsRawOperand, 1);  ::llvm::SMLoc aRowsOperandsLoc;
  (void)aRowsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand aColsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aColsOperands(&aColsRawOperand, 1);  ::llvm::SMLoc aColsOperandsLoc;
  (void)aColsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bdRowsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bdRowsOperands(&bdRowsRawOperand, 1);  ::llvm::SMLoc bdRowsOperandsLoc;
  (void)bdRowsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bdColsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bdColsOperands(&bdColsRawOperand, 1);  ::llvm::SMLoc bdColsOperandsLoc;
  (void)bdColsOperandsLoc;
  ::mlir::Type aAddrRawType{};
  ::llvm::ArrayRef<::mlir::Type> aAddrTypes(&aAddrRawType, 1);
  ::mlir::Type bdAddrRawType{};
  ::llvm::ArrayRef<::mlir::Type> bdAddrTypes(&bdAddrRawType, 1);
  ::mlir::Type aRowsRawType{};
  ::llvm::ArrayRef<::mlir::Type> aRowsTypes(&aRowsRawType, 1);
  ::mlir::Type aColsRawType{};
  ::llvm::ArrayRef<::mlir::Type> aColsTypes(&aColsRawType, 1);
  ::mlir::Type bdRowsRawType{};
  ::llvm::ArrayRef<::mlir::Type> bdRowsTypes(&bdRowsRawType, 1);
  ::mlir::Type bdColsRawType{};
  ::llvm::ArrayRef<::mlir::Type> bdColsTypes(&bdColsRawType, 1);

  aAddrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aAddrRawOperand))
    return ::mlir::failure();

  bdAddrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bdAddrRawOperand))
    return ::mlir::failure();

  aRowsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRowsRawOperand))
    return ::mlir::failure();

  aColsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aColsRawOperand))
    return ::mlir::failure();

  bdRowsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bdRowsRawOperand))
    return ::mlir::failure();

  bdColsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bdColsRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    aAddrRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bdAddrRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    aRowsRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    aColsRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bdRowsRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bdColsRawType = type;
  }
  if (parser.resolveOperands(aAddrOperands, aAddrTypes, aAddrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bdAddrOperands, bdAddrTypes, bdAddrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(aRowsOperands, aRowsTypes, aRowsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(aColsOperands, aColsTypes, aColsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bdRowsOperands, bdRowsTypes, bdRowsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bdColsOperands, bdColsTypes, bdColsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ComputeAccumulatedOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAAddr();
  _odsPrinter << ' ';
  _odsPrinter << getBdAddr();
  _odsPrinter << ' ';
  _odsPrinter << getARows();
  _odsPrinter << ' ';
  _odsPrinter << getACols();
  _odsPrinter << ' ';
  _odsPrinter << getBdRows();
  _odsPrinter << ' ';
  _odsPrinter << getBdCols();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAAddr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getBdAddr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getARows().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getACols().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getBdRows().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getBdCols().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::ComputeAccumulatedOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::ComputePreloadedOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ComputePreloadedOpAdaptor::ComputePreloadedOpAdaptor(ComputePreloadedOp op) : ComputePreloadedOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ComputePreloadedOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ComputePreloadedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols) {
  odsState.addOperands(aAddr);
  odsState.addOperands(bdAddr);
  odsState.addOperands(aRows);
  odsState.addOperands(aCols);
  odsState.addOperands(bdRows);
  odsState.addOperands(bdCols);
}

void ComputePreloadedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols) {
  odsState.addOperands(aAddr);
  odsState.addOperands(bdAddr);
  odsState.addOperands(aRows);
  odsState.addOperands(aCols);
  odsState.addOperands(bdRows);
  odsState.addOperands(bdCols);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ComputePreloadedOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 6u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ComputePreloadedOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ComputePreloadedOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ComputePreloadedOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand aAddrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aAddrOperands(&aAddrRawOperand, 1);  ::llvm::SMLoc aAddrOperandsLoc;
  (void)aAddrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bdAddrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bdAddrOperands(&bdAddrRawOperand, 1);  ::llvm::SMLoc bdAddrOperandsLoc;
  (void)bdAddrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand aRowsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aRowsOperands(&aRowsRawOperand, 1);  ::llvm::SMLoc aRowsOperandsLoc;
  (void)aRowsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand aColsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aColsOperands(&aColsRawOperand, 1);  ::llvm::SMLoc aColsOperandsLoc;
  (void)aColsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bdRowsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bdRowsOperands(&bdRowsRawOperand, 1);  ::llvm::SMLoc bdRowsOperandsLoc;
  (void)bdRowsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bdColsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bdColsOperands(&bdColsRawOperand, 1);  ::llvm::SMLoc bdColsOperandsLoc;
  (void)bdColsOperandsLoc;
  ::mlir::Type aAddrRawType{};
  ::llvm::ArrayRef<::mlir::Type> aAddrTypes(&aAddrRawType, 1);
  ::mlir::Type bdAddrRawType{};
  ::llvm::ArrayRef<::mlir::Type> bdAddrTypes(&bdAddrRawType, 1);
  ::mlir::Type aRowsRawType{};
  ::llvm::ArrayRef<::mlir::Type> aRowsTypes(&aRowsRawType, 1);
  ::mlir::Type aColsRawType{};
  ::llvm::ArrayRef<::mlir::Type> aColsTypes(&aColsRawType, 1);
  ::mlir::Type bdRowsRawType{};
  ::llvm::ArrayRef<::mlir::Type> bdRowsTypes(&bdRowsRawType, 1);
  ::mlir::Type bdColsRawType{};
  ::llvm::ArrayRef<::mlir::Type> bdColsTypes(&bdColsRawType, 1);

  aAddrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aAddrRawOperand))
    return ::mlir::failure();

  bdAddrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bdAddrRawOperand))
    return ::mlir::failure();

  aRowsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRowsRawOperand))
    return ::mlir::failure();

  aColsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aColsRawOperand))
    return ::mlir::failure();

  bdRowsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bdRowsRawOperand))
    return ::mlir::failure();

  bdColsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bdColsRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    aAddrRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bdAddrRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    aRowsRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    aColsRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bdRowsRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bdColsRawType = type;
  }
  if (parser.resolveOperands(aAddrOperands, aAddrTypes, aAddrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bdAddrOperands, bdAddrTypes, bdAddrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(aRowsOperands, aRowsTypes, aRowsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(aColsOperands, aColsTypes, aColsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bdRowsOperands, bdRowsTypes, bdRowsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bdColsOperands, bdColsTypes, bdColsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ComputePreloadedOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAAddr();
  _odsPrinter << ' ';
  _odsPrinter << getBdAddr();
  _odsPrinter << ' ';
  _odsPrinter << getARows();
  _odsPrinter << ' ';
  _odsPrinter << getACols();
  _odsPrinter << ' ';
  _odsPrinter << getBdRows();
  _odsPrinter << ' ';
  _odsPrinter << getBdCols();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAAddr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getBdAddr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getARows().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getACols().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getBdRows().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getBdCols().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::ComputePreloadedOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::ConfigExOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ConfigExOpGenericAdaptorBase::ConfigExOpGenericAdaptorBase(ConfigExOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::IntegerAttr ConfigExOpGenericAdaptorBase::getDataflowAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().dataflow);
  return attr;
}

uint64_t ConfigExOpGenericAdaptorBase::getDataflow() {
  auto attr = getDataflowAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConfigExOpGenericAdaptorBase::getSysActAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().sysAct);
  return attr;
}

uint64_t ConfigExOpGenericAdaptorBase::getSysAct() {
  auto attr = getSysActAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConfigExOpGenericAdaptorBase::getSysShiftAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().sysShift);
  return attr;
}

uint64_t ConfigExOpGenericAdaptorBase::getSysShift() {
  auto attr = getSysShiftAttr();
  return attr.getValue().getZExtValue();
}

::mlir::FloatAttr ConfigExOpGenericAdaptorBase::getSysAccScaleAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().sysAccScale);
  return attr;
}

::llvm::APFloat ConfigExOpGenericAdaptorBase::getSysAccScale() {
  auto attr = getSysAccScaleAttr();
  return attr.getValue();
}

::mlir::IntegerAttr ConfigExOpGenericAdaptorBase::getCStrideAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().cStride);
  return attr;
}

uint64_t ConfigExOpGenericAdaptorBase::getCStride() {
  auto attr = getCStrideAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConfigExOpGenericAdaptorBase::getAStrideAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().aStride);
  return attr;
}

uint64_t ConfigExOpGenericAdaptorBase::getAStride() {
  auto attr = getAStrideAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr ConfigExOpGenericAdaptorBase::getATransposeAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().aTranspose);
  return attr;
}

bool ConfigExOpGenericAdaptorBase::getATranspose() {
  auto attr = getATransposeAttr();
  return attr.getValue();
}

::mlir::BoolAttr ConfigExOpGenericAdaptorBase::getBTransposeAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().bTranspose);
  return attr;
}

bool ConfigExOpGenericAdaptorBase::getBTranspose() {
  auto attr = getBTransposeAttr();
  return attr.getValue();
}

::mlir::BoolAttr ConfigExOpGenericAdaptorBase::getSetOnlyStridesAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().setOnlyStrides);
  return attr;
}

bool ConfigExOpGenericAdaptorBase::getSetOnlyStrides() {
  auto attr = getSetOnlyStridesAttr();
  return attr.getValue();
}

} // namespace detail
ConfigExOpAdaptor::ConfigExOpAdaptor(ConfigExOp op) : ConfigExOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConfigExOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_aStride = getProperties().aStride; (void)tblgen_aStride;
  auto tblgen_aTranspose = getProperties().aTranspose; (void)tblgen_aTranspose;
  auto tblgen_bTranspose = getProperties().bTranspose; (void)tblgen_bTranspose;
  auto tblgen_cStride = getProperties().cStride; (void)tblgen_cStride;
  auto tblgen_dataflow = getProperties().dataflow; (void)tblgen_dataflow;
  auto tblgen_setOnlyStrides = getProperties().setOnlyStrides; (void)tblgen_setOnlyStrides;
  auto tblgen_sysAccScale = getProperties().sysAccScale; (void)tblgen_sysAccScale;
  auto tblgen_sysAct = getProperties().sysAct; (void)tblgen_sysAct;
  auto tblgen_sysShift = getProperties().sysShift; (void)tblgen_sysShift;

  if (tblgen_dataflow && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_dataflow))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_dataflow).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_ex' op ""attribute 'dataflow' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_sysAct && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_sysAct))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_sysAct).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_ex' op ""attribute 'sysAct' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_sysShift && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_sysShift))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_sysShift).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_ex' op ""attribute 'sysShift' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_sysAccScale && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_sysAccScale))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_sysAccScale).getType().isF32()))))
    return emitError(loc, "'gemmini.config_ex' op ""attribute 'sysAccScale' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_cStride && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_cStride))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_cStride).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_ex' op ""attribute 'cStride' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_aStride && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_aStride))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_aStride).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_ex' op ""attribute 'aStride' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_aTranspose && !((::llvm::isa<::mlir::BoolAttr>(tblgen_aTranspose))))
    return emitError(loc, "'gemmini.config_ex' op ""attribute 'aTranspose' failed to satisfy constraint: bool attribute");

  if (tblgen_bTranspose && !((::llvm::isa<::mlir::BoolAttr>(tblgen_bTranspose))))
    return emitError(loc, "'gemmini.config_ex' op ""attribute 'bTranspose' failed to satisfy constraint: bool attribute");

  if (tblgen_setOnlyStrides && !((::llvm::isa<::mlir::BoolAttr>(tblgen_setOnlyStrides))))
    return emitError(loc, "'gemmini.config_ex' op ""attribute 'setOnlyStrides' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

::llvm::LogicalResult ConfigExOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.aStride;
       auto attr = dict.get("aStride");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `aStride` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.aTranspose;
       auto attr = dict.get("aTranspose");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `aTranspose` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.bTranspose;
       auto attr = dict.get("bTranspose");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `bTranspose` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.cStride;
       auto attr = dict.get("cStride");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `cStride` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.dataflow;
       auto attr = dict.get("dataflow");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `dataflow` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.setOnlyStrides;
       auto attr = dict.get("setOnlyStrides");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `setOnlyStrides` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.sysAccScale;
       auto attr = dict.get("sysAccScale");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `sysAccScale` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.sysAct;
       auto attr = dict.get("sysAct");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `sysAct` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.sysShift;
       auto attr = dict.get("sysShift");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `sysShift` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ConfigExOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.aStride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("aStride",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.aTranspose;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("aTranspose",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.bTranspose;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("bTranspose",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.cStride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("cStride",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.dataflow;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("dataflow",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.setOnlyStrides;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("setOnlyStrides",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.sysAccScale;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("sysAccScale",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.sysAct;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("sysAct",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.sysShift;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("sysShift",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ConfigExOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.aStride.getAsOpaquePointer()), 
    llvm::hash_value(prop.aTranspose.getAsOpaquePointer()), 
    llvm::hash_value(prop.bTranspose.getAsOpaquePointer()), 
    llvm::hash_value(prop.cStride.getAsOpaquePointer()), 
    llvm::hash_value(prop.dataflow.getAsOpaquePointer()), 
    llvm::hash_value(prop.setOnlyStrides.getAsOpaquePointer()), 
    llvm::hash_value(prop.sysAccScale.getAsOpaquePointer()), 
    llvm::hash_value(prop.sysAct.getAsOpaquePointer()), 
    llvm::hash_value(prop.sysShift.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ConfigExOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "aStride")
      return prop.aStride;

    if (name == "aTranspose")
      return prop.aTranspose;

    if (name == "bTranspose")
      return prop.bTranspose;

    if (name == "cStride")
      return prop.cStride;

    if (name == "dataflow")
      return prop.dataflow;

    if (name == "setOnlyStrides")
      return prop.setOnlyStrides;

    if (name == "sysAccScale")
      return prop.sysAccScale;

    if (name == "sysAct")
      return prop.sysAct;

    if (name == "sysShift")
      return prop.sysShift;
  return std::nullopt;
}

void ConfigExOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "aStride") {
       prop.aStride = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.aStride)>>(value);
       return;
    }

    if (name == "aTranspose") {
       prop.aTranspose = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.aTranspose)>>(value);
       return;
    }

    if (name == "bTranspose") {
       prop.bTranspose = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.bTranspose)>>(value);
       return;
    }

    if (name == "cStride") {
       prop.cStride = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.cStride)>>(value);
       return;
    }

    if (name == "dataflow") {
       prop.dataflow = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.dataflow)>>(value);
       return;
    }

    if (name == "setOnlyStrides") {
       prop.setOnlyStrides = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.setOnlyStrides)>>(value);
       return;
    }

    if (name == "sysAccScale") {
       prop.sysAccScale = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.sysAccScale)>>(value);
       return;
    }

    if (name == "sysAct") {
       prop.sysAct = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.sysAct)>>(value);
       return;
    }

    if (name == "sysShift") {
       prop.sysShift = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.sysShift)>>(value);
       return;
    }
}

void ConfigExOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.aStride) attrs.append("aStride", prop.aStride);

    if (prop.aTranspose) attrs.append("aTranspose", prop.aTranspose);

    if (prop.bTranspose) attrs.append("bTranspose", prop.bTranspose);

    if (prop.cStride) attrs.append("cStride", prop.cStride);

    if (prop.dataflow) attrs.append("dataflow", prop.dataflow);

    if (prop.setOnlyStrides) attrs.append("setOnlyStrides", prop.setOnlyStrides);

    if (prop.sysAccScale) attrs.append("sysAccScale", prop.sysAccScale);

    if (prop.sysAct) attrs.append("sysAct", prop.sysAct);

    if (prop.sysShift) attrs.append("sysShift", prop.sysShift);
}

::llvm::LogicalResult ConfigExOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getAStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "aStride", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getATransposeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "aTranspose", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBTransposeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "bTranspose", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "cStride", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getDataflowAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "dataflow", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSetOnlyStridesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "setOnlyStrides", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSysAccScaleAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(attr, "sysAccScale", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSysActAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "sysAct", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSysShiftAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "sysShift", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ConfigExOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.aStride)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.aTranspose)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.bTranspose)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.cStride)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.dataflow)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.setOnlyStrides)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.sysAccScale)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.sysAct)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.sysShift)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConfigExOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.aStride);

  writer.writeOptionalAttribute(prop.aTranspose);

  writer.writeOptionalAttribute(prop.bTranspose);

  writer.writeOptionalAttribute(prop.cStride);

  writer.writeOptionalAttribute(prop.dataflow);

  writer.writeOptionalAttribute(prop.setOnlyStrides);

  writer.writeOptionalAttribute(prop.sysAccScale);

  writer.writeOptionalAttribute(prop.sysAct);

  writer.writeOptionalAttribute(prop.sysShift);
}

uint64_t ConfigExOp::getDataflow() {
  auto attr = getDataflowAttr();
  return attr.getValue().getZExtValue();
}

uint64_t ConfigExOp::getSysAct() {
  auto attr = getSysActAttr();
  return attr.getValue().getZExtValue();
}

uint64_t ConfigExOp::getSysShift() {
  auto attr = getSysShiftAttr();
  return attr.getValue().getZExtValue();
}

::llvm::APFloat ConfigExOp::getSysAccScale() {
  auto attr = getSysAccScaleAttr();
  return attr.getValue();
}

uint64_t ConfigExOp::getCStride() {
  auto attr = getCStrideAttr();
  return attr.getValue().getZExtValue();
}

uint64_t ConfigExOp::getAStride() {
  auto attr = getAStrideAttr();
  return attr.getValue().getZExtValue();
}

bool ConfigExOp::getATranspose() {
  auto attr = getATransposeAttr();
  return attr.getValue();
}

bool ConfigExOp::getBTranspose() {
  auto attr = getBTransposeAttr();
  return attr.getValue();
}

bool ConfigExOp::getSetOnlyStrides() {
  auto attr = getSetOnlyStridesAttr();
  return attr.getValue();
}

void ConfigExOp::setDataflow(uint64_t attrValue) {
  getProperties().dataflow = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigExOp::setSysAct(uint64_t attrValue) {
  getProperties().sysAct = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigExOp::setSysShift(uint64_t attrValue) {
  getProperties().sysShift = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigExOp::setSysAccScale(::llvm::APFloat attrValue) {
  getProperties().sysAccScale = ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue);
}

void ConfigExOp::setCStride(uint64_t attrValue) {
  getProperties().cStride = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigExOp::setAStride(uint64_t attrValue) {
  getProperties().aStride = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigExOp::setATranspose(bool attrValue) {
  getProperties().aTranspose = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void ConfigExOp::setBTranspose(bool attrValue) {
  getProperties().bTranspose = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void ConfigExOp::setSetOnlyStrides(bool attrValue) {
  getProperties().setOnlyStrides = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void ConfigExOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr dataflow, ::mlir::IntegerAttr sysAct, ::mlir::IntegerAttr sysShift, ::mlir::FloatAttr sysAccScale, ::mlir::IntegerAttr cStride, ::mlir::IntegerAttr aStride, ::mlir::BoolAttr aTranspose, ::mlir::BoolAttr bTranspose, ::mlir::BoolAttr setOnlyStrides) {
  if (dataflow) {
    odsState.getOrAddProperties<Properties>().dataflow = dataflow;
  }
  if (sysAct) {
    odsState.getOrAddProperties<Properties>().sysAct = sysAct;
  }
  if (sysShift) {
    odsState.getOrAddProperties<Properties>().sysShift = sysShift;
  }
  if (sysAccScale) {
    odsState.getOrAddProperties<Properties>().sysAccScale = sysAccScale;
  }
  if (cStride) {
    odsState.getOrAddProperties<Properties>().cStride = cStride;
  }
  if (aStride) {
    odsState.getOrAddProperties<Properties>().aStride = aStride;
  }
  if (aTranspose) {
    odsState.getOrAddProperties<Properties>().aTranspose = aTranspose;
  }
  if (bTranspose) {
    odsState.getOrAddProperties<Properties>().bTranspose = bTranspose;
  }
  if (setOnlyStrides) {
    odsState.getOrAddProperties<Properties>().setOnlyStrides = setOnlyStrides;
  }
}

void ConfigExOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr dataflow, ::mlir::IntegerAttr sysAct, ::mlir::IntegerAttr sysShift, ::mlir::FloatAttr sysAccScale, ::mlir::IntegerAttr cStride, ::mlir::IntegerAttr aStride, ::mlir::BoolAttr aTranspose, ::mlir::BoolAttr bTranspose, ::mlir::BoolAttr setOnlyStrides) {
  if (dataflow) {
    odsState.getOrAddProperties<Properties>().dataflow = dataflow;
  }
  if (sysAct) {
    odsState.getOrAddProperties<Properties>().sysAct = sysAct;
  }
  if (sysShift) {
    odsState.getOrAddProperties<Properties>().sysShift = sysShift;
  }
  if (sysAccScale) {
    odsState.getOrAddProperties<Properties>().sysAccScale = sysAccScale;
  }
  if (cStride) {
    odsState.getOrAddProperties<Properties>().cStride = cStride;
  }
  if (aStride) {
    odsState.getOrAddProperties<Properties>().aStride = aStride;
  }
  if (aTranspose) {
    odsState.getOrAddProperties<Properties>().aTranspose = aTranspose;
  }
  if (bTranspose) {
    odsState.getOrAddProperties<Properties>().bTranspose = bTranspose;
  }
  if (setOnlyStrides) {
    odsState.getOrAddProperties<Properties>().setOnlyStrides = setOnlyStrides;
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConfigExOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint64_t dataflow, uint64_t sysAct, uint64_t sysShift, ::llvm::APFloat sysAccScale, uint64_t cStride, uint64_t aStride, bool aTranspose, bool bTranspose, bool setOnlyStrides) {
  odsState.getOrAddProperties<Properties>().dataflow = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), dataflow);
  odsState.getOrAddProperties<Properties>().sysAct = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), sysAct);
  odsState.getOrAddProperties<Properties>().sysShift = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), sysShift);
  odsState.getOrAddProperties<Properties>().sysAccScale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), sysAccScale);
  odsState.getOrAddProperties<Properties>().cStride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), cStride);
  odsState.getOrAddProperties<Properties>().aStride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), aStride);
  odsState.getOrAddProperties<Properties>().aTranspose = odsBuilder.getBoolAttr(aTranspose);
  odsState.getOrAddProperties<Properties>().bTranspose = odsBuilder.getBoolAttr(bTranspose);
  odsState.getOrAddProperties<Properties>().setOnlyStrides = odsBuilder.getBoolAttr(setOnlyStrides);
}

void ConfigExOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint64_t dataflow, uint64_t sysAct, uint64_t sysShift, ::llvm::APFloat sysAccScale, uint64_t cStride, uint64_t aStride, bool aTranspose, bool bTranspose, bool setOnlyStrides) {
  odsState.getOrAddProperties<Properties>().dataflow = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), dataflow);
  odsState.getOrAddProperties<Properties>().sysAct = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), sysAct);
  odsState.getOrAddProperties<Properties>().sysShift = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), sysShift);
  odsState.getOrAddProperties<Properties>().sysAccScale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), sysAccScale);
  odsState.getOrAddProperties<Properties>().cStride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), cStride);
  odsState.getOrAddProperties<Properties>().aStride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), aStride);
  odsState.getOrAddProperties<Properties>().aTranspose = odsBuilder.getBoolAttr(aTranspose);
  odsState.getOrAddProperties<Properties>().bTranspose = odsBuilder.getBoolAttr(bTranspose);
  odsState.getOrAddProperties<Properties>().setOnlyStrides = odsBuilder.getBoolAttr(setOnlyStrides);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConfigExOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ConfigExOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void ConfigExOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.dataflow)
    properties.dataflow = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.sysAct)
    properties.sysAct = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.sysShift)
    properties.sysShift = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.sysAccScale)
    properties.sysAccScale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0);
  if (!properties.cStride)
    properties.cStride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1);
  if (!properties.aStride)
    properties.aStride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1);
  if (!properties.aTranspose)
    properties.aTranspose = odsBuilder.getBoolAttr(false);
  if (!properties.bTranspose)
    properties.bTranspose = odsBuilder.getBoolAttr(false);
  if (!properties.setOnlyStrides)
    properties.setOnlyStrides = odsBuilder.getBoolAttr(false);
}

::llvm::LogicalResult ConfigExOp::verifyInvariantsImpl() {
  auto tblgen_aStride = getProperties().aStride; (void)tblgen_aStride;
  auto tblgen_aTranspose = getProperties().aTranspose; (void)tblgen_aTranspose;
  auto tblgen_bTranspose = getProperties().bTranspose; (void)tblgen_bTranspose;
  auto tblgen_cStride = getProperties().cStride; (void)tblgen_cStride;
  auto tblgen_dataflow = getProperties().dataflow; (void)tblgen_dataflow;
  auto tblgen_setOnlyStrides = getProperties().setOnlyStrides; (void)tblgen_setOnlyStrides;
  auto tblgen_sysAccScale = getProperties().sysAccScale; (void)tblgen_sysAccScale;
  auto tblgen_sysAct = getProperties().sysAct; (void)tblgen_sysAct;
  auto tblgen_sysShift = getProperties().sysShift; (void)tblgen_sysShift;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_dataflow, "dataflow")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_sysAct, "sysAct")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_sysShift, "sysShift")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(*this, tblgen_sysAccScale, "sysAccScale")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_cStride, "cStride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_aStride, "aStride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_aTranspose, "aTranspose")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_bTranspose, "bTranspose")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_setOnlyStrides, "setOnlyStrides")))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::LogicalResult ConfigExOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ConfigExOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void ConfigExOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDataflowAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("dataflow");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getSysActAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("sysAct");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getSysShiftAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("sysShift");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getSysAccScaleAttr();
     if(attr && (attr == odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0)))
       elidedAttrs.push_back("sysAccScale");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getCStrideAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1)))
       elidedAttrs.push_back("cStride");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getAStrideAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1)))
       elidedAttrs.push_back("aStride");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getATransposeAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("aTranspose");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getBTransposeAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("bTranspose");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getSetOnlyStridesAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("setOnlyStrides");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::ConfigExOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::ConfigLdOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ConfigLdOpGenericAdaptorBase::ConfigLdOpGenericAdaptorBase(ConfigLdOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::FloatAttr ConfigLdOpGenericAdaptorBase::getScaleAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().scale);
  return attr;
}

::llvm::APFloat ConfigLdOpGenericAdaptorBase::getScale() {
  auto attr = getScaleAttr();
  return attr.getValue();
}

::mlir::BoolAttr ConfigLdOpGenericAdaptorBase::getShrunkAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().shrunk);
  return attr;
}

bool ConfigLdOpGenericAdaptorBase::getShrunk() {
  auto attr = getShrunkAttr();
  return attr.getValue();
}

::mlir::IntegerAttr ConfigLdOpGenericAdaptorBase::getIdAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().id);
  return attr;
}

uint64_t ConfigLdOpGenericAdaptorBase::getId() {
  auto attr = getIdAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConfigLdOpGenericAdaptorBase::getBlockMvinStrideAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().block_mvin_stride);
  return attr;
}

uint64_t ConfigLdOpGenericAdaptorBase::getBlockMvinStride() {
  auto attr = getBlockMvinStrideAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConfigLdOpGenericAdaptorBase::getPixelRepeatsAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().pixel_repeats);
  return attr;
}

uint64_t ConfigLdOpGenericAdaptorBase::getPixelRepeats() {
  auto attr = getPixelRepeatsAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ConfigLdOpAdaptor::ConfigLdOpAdaptor(ConfigLdOp op) : ConfigLdOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConfigLdOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_block_mvin_stride = getProperties().block_mvin_stride; (void)tblgen_block_mvin_stride;
  auto tblgen_id = getProperties().id; (void)tblgen_id;
  auto tblgen_pixel_repeats = getProperties().pixel_repeats; (void)tblgen_pixel_repeats;
  auto tblgen_scale = getProperties().scale; (void)tblgen_scale;
  auto tblgen_shrunk = getProperties().shrunk; (void)tblgen_shrunk;

  if (tblgen_scale && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_scale))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_scale).getType().isF32()))))
    return emitError(loc, "'gemmini.config_ld' op ""attribute 'scale' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_shrunk && !((::llvm::isa<::mlir::BoolAttr>(tblgen_shrunk))))
    return emitError(loc, "'gemmini.config_ld' op ""attribute 'shrunk' failed to satisfy constraint: bool attribute");

  if (tblgen_id && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_id))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_id).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_ld' op ""attribute 'id' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_block_mvin_stride && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_block_mvin_stride))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_block_mvin_stride).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_ld' op ""attribute 'block_mvin_stride' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_pixel_repeats && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_pixel_repeats))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_pixel_repeats).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_ld' op ""attribute 'pixel_repeats' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult ConfigLdOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.block_mvin_stride;
       auto attr = dict.get("block_mvin_stride");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `block_mvin_stride` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.id;
       auto attr = dict.get("id");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `id` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.pixel_repeats;
       auto attr = dict.get("pixel_repeats");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `pixel_repeats` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.scale;
       auto attr = dict.get("scale");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `scale` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.shrunk;
       auto attr = dict.get("shrunk");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `shrunk` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ConfigLdOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.block_mvin_stride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("block_mvin_stride",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.id;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("id",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.pixel_repeats;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("pixel_repeats",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.scale;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("scale",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.shrunk;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("shrunk",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ConfigLdOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.block_mvin_stride.getAsOpaquePointer()), 
    llvm::hash_value(prop.id.getAsOpaquePointer()), 
    llvm::hash_value(prop.pixel_repeats.getAsOpaquePointer()), 
    llvm::hash_value(prop.scale.getAsOpaquePointer()), 
    llvm::hash_value(prop.shrunk.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ConfigLdOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "block_mvin_stride")
      return prop.block_mvin_stride;

    if (name == "id")
      return prop.id;

    if (name == "pixel_repeats")
      return prop.pixel_repeats;

    if (name == "scale")
      return prop.scale;

    if (name == "shrunk")
      return prop.shrunk;
  return std::nullopt;
}

void ConfigLdOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "block_mvin_stride") {
       prop.block_mvin_stride = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.block_mvin_stride)>>(value);
       return;
    }

    if (name == "id") {
       prop.id = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.id)>>(value);
       return;
    }

    if (name == "pixel_repeats") {
       prop.pixel_repeats = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.pixel_repeats)>>(value);
       return;
    }

    if (name == "scale") {
       prop.scale = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.scale)>>(value);
       return;
    }

    if (name == "shrunk") {
       prop.shrunk = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.shrunk)>>(value);
       return;
    }
}

void ConfigLdOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.block_mvin_stride) attrs.append("block_mvin_stride", prop.block_mvin_stride);

    if (prop.id) attrs.append("id", prop.id);

    if (prop.pixel_repeats) attrs.append("pixel_repeats", prop.pixel_repeats);

    if (prop.scale) attrs.append("scale", prop.scale);

    if (prop.shrunk) attrs.append("shrunk", prop.shrunk);
}

::llvm::LogicalResult ConfigLdOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBlockMvinStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "block_mvin_stride", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getIdAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "id", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPixelRepeatsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "pixel_repeats", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getScaleAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(attr, "scale", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getShrunkAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "shrunk", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ConfigLdOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.block_mvin_stride)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.id)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.pixel_repeats)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.scale)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.shrunk)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConfigLdOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.block_mvin_stride);

  writer.writeOptionalAttribute(prop.id);

  writer.writeOptionalAttribute(prop.pixel_repeats);

  writer.writeOptionalAttribute(prop.scale);

  writer.writeOptionalAttribute(prop.shrunk);
}

::llvm::APFloat ConfigLdOp::getScale() {
  auto attr = getScaleAttr();
  return attr.getValue();
}

bool ConfigLdOp::getShrunk() {
  auto attr = getShrunkAttr();
  return attr.getValue();
}

uint64_t ConfigLdOp::getId() {
  auto attr = getIdAttr();
  return attr.getValue().getZExtValue();
}

uint64_t ConfigLdOp::getBlockMvinStride() {
  auto attr = getBlockMvinStrideAttr();
  return attr.getValue().getZExtValue();
}

uint64_t ConfigLdOp::getPixelRepeats() {
  auto attr = getPixelRepeatsAttr();
  return attr.getValue().getZExtValue();
}

void ConfigLdOp::setScale(::llvm::APFloat attrValue) {
  getProperties().scale = ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue);
}

void ConfigLdOp::setShrunk(bool attrValue) {
  getProperties().shrunk = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void ConfigLdOp::setId(uint64_t attrValue) {
  getProperties().id = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigLdOp::setBlockMvinStride(uint64_t attrValue) {
  getProperties().block_mvin_stride = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigLdOp::setPixelRepeats(uint64_t attrValue) {
  getProperties().pixel_repeats = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigLdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value stride, ::mlir::FloatAttr scale, ::mlir::BoolAttr shrunk, ::mlir::IntegerAttr id, ::mlir::IntegerAttr block_mvin_stride, ::mlir::IntegerAttr pixel_repeats) {
  odsState.addOperands(stride);
  if (scale) {
    odsState.getOrAddProperties<Properties>().scale = scale;
  }
  if (shrunk) {
    odsState.getOrAddProperties<Properties>().shrunk = shrunk;
  }
  if (id) {
    odsState.getOrAddProperties<Properties>().id = id;
  }
  if (block_mvin_stride) {
    odsState.getOrAddProperties<Properties>().block_mvin_stride = block_mvin_stride;
  }
  if (pixel_repeats) {
    odsState.getOrAddProperties<Properties>().pixel_repeats = pixel_repeats;
  }
}

void ConfigLdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value stride, ::mlir::FloatAttr scale, ::mlir::BoolAttr shrunk, ::mlir::IntegerAttr id, ::mlir::IntegerAttr block_mvin_stride, ::mlir::IntegerAttr pixel_repeats) {
  odsState.addOperands(stride);
  if (scale) {
    odsState.getOrAddProperties<Properties>().scale = scale;
  }
  if (shrunk) {
    odsState.getOrAddProperties<Properties>().shrunk = shrunk;
  }
  if (id) {
    odsState.getOrAddProperties<Properties>().id = id;
  }
  if (block_mvin_stride) {
    odsState.getOrAddProperties<Properties>().block_mvin_stride = block_mvin_stride;
  }
  if (pixel_repeats) {
    odsState.getOrAddProperties<Properties>().pixel_repeats = pixel_repeats;
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConfigLdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value stride, ::llvm::APFloat scale, bool shrunk, uint64_t id, uint64_t block_mvin_stride, uint64_t pixel_repeats) {
  odsState.addOperands(stride);
  odsState.getOrAddProperties<Properties>().scale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), scale);
  odsState.getOrAddProperties<Properties>().shrunk = odsBuilder.getBoolAttr(shrunk);
  odsState.getOrAddProperties<Properties>().id = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), id);
  odsState.getOrAddProperties<Properties>().block_mvin_stride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), block_mvin_stride);
  odsState.getOrAddProperties<Properties>().pixel_repeats = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), pixel_repeats);
}

void ConfigLdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value stride, ::llvm::APFloat scale, bool shrunk, uint64_t id, uint64_t block_mvin_stride, uint64_t pixel_repeats) {
  odsState.addOperands(stride);
  odsState.getOrAddProperties<Properties>().scale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), scale);
  odsState.getOrAddProperties<Properties>().shrunk = odsBuilder.getBoolAttr(shrunk);
  odsState.getOrAddProperties<Properties>().id = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), id);
  odsState.getOrAddProperties<Properties>().block_mvin_stride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), block_mvin_stride);
  odsState.getOrAddProperties<Properties>().pixel_repeats = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), pixel_repeats);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConfigLdOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ConfigLdOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void ConfigLdOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.scale)
    properties.scale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0);
  if (!properties.shrunk)
    properties.shrunk = odsBuilder.getBoolAttr(false);
  if (!properties.id)
    properties.id = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.block_mvin_stride)
    properties.block_mvin_stride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), -1);
  if (!properties.pixel_repeats)
    properties.pixel_repeats = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1);
}

::llvm::LogicalResult ConfigLdOp::verifyInvariantsImpl() {
  auto tblgen_block_mvin_stride = getProperties().block_mvin_stride; (void)tblgen_block_mvin_stride;
  auto tblgen_id = getProperties().id; (void)tblgen_id;
  auto tblgen_pixel_repeats = getProperties().pixel_repeats; (void)tblgen_pixel_repeats;
  auto tblgen_scale = getProperties().scale; (void)tblgen_scale;
  auto tblgen_shrunk = getProperties().shrunk; (void)tblgen_shrunk;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(*this, tblgen_scale, "scale")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_shrunk, "shrunk")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_id, "id")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_block_mvin_stride, "block_mvin_stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_pixel_repeats, "pixel_repeats")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ConfigLdOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ConfigLdOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand strideRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> strideOperands(&strideRawOperand, 1);  ::llvm::SMLoc strideOperandsLoc;
  (void)strideOperandsLoc;
  ::mlir::Type strideRawType{};
  ::llvm::ArrayRef<::mlir::Type> strideTypes(&strideRawType, 1);

  strideOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(strideRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    strideRawType = type;
  }
  if (parser.resolveOperands(strideOperands, strideTypes, strideOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConfigLdOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getStride();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getScaleAttr();
     if(attr && (attr == odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0)))
       elidedAttrs.push_back("scale");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getShrunkAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("shrunk");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getIdAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("id");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getBlockMvinStrideAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), -1)))
       elidedAttrs.push_back("block_mvin_stride");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getPixelRepeatsAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1)))
       elidedAttrs.push_back("pixel_repeats");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getStride().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::ConfigLdOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::ConfigNormOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ConfigNormOpGenericAdaptorBase::ConfigNormOpGenericAdaptorBase(ConfigNormOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::IntegerAttr ConfigNormOpGenericAdaptorBase::getQConstAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().qConst);
  return attr;
}

uint64_t ConfigNormOpGenericAdaptorBase::getQConst() {
  auto attr = getQConstAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConfigNormOpGenericAdaptorBase::getQConstTypeAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().qConstType);
  return attr;
}

uint64_t ConfigNormOpGenericAdaptorBase::getQConstType() {
  auto attr = getQConstTypeAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConfigNormOpGenericAdaptorBase::getSetStatsIdOnlyAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().setStatsIdOnly);
  return attr;
}

uint64_t ConfigNormOpGenericAdaptorBase::getSetStatsIdOnly() {
  auto attr = getSetStatsIdOnlyAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConfigNormOpGenericAdaptorBase::getActMsbAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().actMsb);
  return attr;
}

uint64_t ConfigNormOpGenericAdaptorBase::getActMsb() {
  auto attr = getActMsbAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConfigNormOpGenericAdaptorBase::getStatsIdAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().StatsId);
  return attr;
}

uint64_t ConfigNormOpGenericAdaptorBase::getStatsId() {
  auto attr = getStatsIdAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConfigNormOpGenericAdaptorBase::getIgeluQbAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().igeluQb);
  return attr;
}

uint64_t ConfigNormOpGenericAdaptorBase::getIgeluQb() {
  auto attr = getIgeluQbAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConfigNormOpGenericAdaptorBase::getIgeluQcAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().igeluQc);
  return attr;
}

uint64_t ConfigNormOpGenericAdaptorBase::getIgeluQc() {
  auto attr = getIgeluQcAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ConfigNormOpAdaptor::ConfigNormOpAdaptor(ConfigNormOp op) : ConfigNormOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConfigNormOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_StatsId = getProperties().StatsId; (void)tblgen_StatsId;
  auto tblgen_actMsb = getProperties().actMsb; (void)tblgen_actMsb;
  auto tblgen_igeluQb = getProperties().igeluQb; (void)tblgen_igeluQb;
  auto tblgen_igeluQc = getProperties().igeluQc; (void)tblgen_igeluQc;
  auto tblgen_qConst = getProperties().qConst; (void)tblgen_qConst;
  auto tblgen_qConstType = getProperties().qConstType; (void)tblgen_qConstType;
  auto tblgen_setStatsIdOnly = getProperties().setStatsIdOnly; (void)tblgen_setStatsIdOnly;

  if (tblgen_qConst && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_qConst))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_qConst).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_norm' op ""attribute 'qConst' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_qConstType && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_qConstType))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_qConstType).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_norm' op ""attribute 'qConstType' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_setStatsIdOnly && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_setStatsIdOnly))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_setStatsIdOnly).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_norm' op ""attribute 'setStatsIdOnly' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_actMsb && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_actMsb))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_actMsb).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_norm' op ""attribute 'actMsb' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_StatsId && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_StatsId))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_StatsId).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_norm' op ""attribute 'StatsId' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_igeluQb && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_igeluQb))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_igeluQb).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_norm' op ""attribute 'igeluQb' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_igeluQc && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_igeluQc))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_igeluQc).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_norm' op ""attribute 'igeluQc' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult ConfigNormOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.StatsId;
       auto attr = dict.get("StatsId");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `StatsId` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.actMsb;
       auto attr = dict.get("actMsb");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `actMsb` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.igeluQb;
       auto attr = dict.get("igeluQb");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `igeluQb` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.igeluQc;
       auto attr = dict.get("igeluQc");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `igeluQc` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.qConst;
       auto attr = dict.get("qConst");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `qConst` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.qConstType;
       auto attr = dict.get("qConstType");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `qConstType` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.setStatsIdOnly;
       auto attr = dict.get("setStatsIdOnly");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `setStatsIdOnly` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ConfigNormOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.StatsId;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("StatsId",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.actMsb;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("actMsb",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.igeluQb;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("igeluQb",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.igeluQc;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("igeluQc",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.qConst;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("qConst",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.qConstType;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("qConstType",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.setStatsIdOnly;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("setStatsIdOnly",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ConfigNormOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.StatsId.getAsOpaquePointer()), 
    llvm::hash_value(prop.actMsb.getAsOpaquePointer()), 
    llvm::hash_value(prop.igeluQb.getAsOpaquePointer()), 
    llvm::hash_value(prop.igeluQc.getAsOpaquePointer()), 
    llvm::hash_value(prop.qConst.getAsOpaquePointer()), 
    llvm::hash_value(prop.qConstType.getAsOpaquePointer()), 
    llvm::hash_value(prop.setStatsIdOnly.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ConfigNormOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "StatsId")
      return prop.StatsId;

    if (name == "actMsb")
      return prop.actMsb;

    if (name == "igeluQb")
      return prop.igeluQb;

    if (name == "igeluQc")
      return prop.igeluQc;

    if (name == "qConst")
      return prop.qConst;

    if (name == "qConstType")
      return prop.qConstType;

    if (name == "setStatsIdOnly")
      return prop.setStatsIdOnly;
  return std::nullopt;
}

void ConfigNormOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "StatsId") {
       prop.StatsId = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.StatsId)>>(value);
       return;
    }

    if (name == "actMsb") {
       prop.actMsb = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.actMsb)>>(value);
       return;
    }

    if (name == "igeluQb") {
       prop.igeluQb = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.igeluQb)>>(value);
       return;
    }

    if (name == "igeluQc") {
       prop.igeluQc = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.igeluQc)>>(value);
       return;
    }

    if (name == "qConst") {
       prop.qConst = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.qConst)>>(value);
       return;
    }

    if (name == "qConstType") {
       prop.qConstType = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.qConstType)>>(value);
       return;
    }

    if (name == "setStatsIdOnly") {
       prop.setStatsIdOnly = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.setStatsIdOnly)>>(value);
       return;
    }
}

void ConfigNormOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.StatsId) attrs.append("StatsId", prop.StatsId);

    if (prop.actMsb) attrs.append("actMsb", prop.actMsb);

    if (prop.igeluQb) attrs.append("igeluQb", prop.igeluQb);

    if (prop.igeluQc) attrs.append("igeluQc", prop.igeluQc);

    if (prop.qConst) attrs.append("qConst", prop.qConst);

    if (prop.qConstType) attrs.append("qConstType", prop.qConstType);

    if (prop.setStatsIdOnly) attrs.append("setStatsIdOnly", prop.setStatsIdOnly);
}

::llvm::LogicalResult ConfigNormOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getStatsIdAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "StatsId", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getActMsbAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "actMsb", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getIgeluQbAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "igeluQb", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getIgeluQcAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "igeluQc", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getQConstAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "qConst", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getQConstTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "qConstType", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSetStatsIdOnlyAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "setStatsIdOnly", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ConfigNormOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.StatsId)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.actMsb)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.igeluQb)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.igeluQc)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.qConst)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.qConstType)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.setStatsIdOnly)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConfigNormOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.StatsId);

  writer.writeOptionalAttribute(prop.actMsb);

  writer.writeOptionalAttribute(prop.igeluQb);

  writer.writeOptionalAttribute(prop.igeluQc);

  writer.writeOptionalAttribute(prop.qConst);

  writer.writeOptionalAttribute(prop.qConstType);

  writer.writeOptionalAttribute(prop.setStatsIdOnly);
}

uint64_t ConfigNormOp::getQConst() {
  auto attr = getQConstAttr();
  return attr.getValue().getZExtValue();
}

uint64_t ConfigNormOp::getQConstType() {
  auto attr = getQConstTypeAttr();
  return attr.getValue().getZExtValue();
}

uint64_t ConfigNormOp::getSetStatsIdOnly() {
  auto attr = getSetStatsIdOnlyAttr();
  return attr.getValue().getZExtValue();
}

uint64_t ConfigNormOp::getActMsb() {
  auto attr = getActMsbAttr();
  return attr.getValue().getZExtValue();
}

uint64_t ConfigNormOp::getStatsId() {
  auto attr = getStatsIdAttr();
  return attr.getValue().getZExtValue();
}

uint64_t ConfigNormOp::getIgeluQb() {
  auto attr = getIgeluQbAttr();
  return attr.getValue().getZExtValue();
}

uint64_t ConfigNormOp::getIgeluQc() {
  auto attr = getIgeluQcAttr();
  return attr.getValue().getZExtValue();
}

void ConfigNormOp::setQConst(uint64_t attrValue) {
  getProperties().qConst = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigNormOp::setQConstType(uint64_t attrValue) {
  getProperties().qConstType = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigNormOp::setSetStatsIdOnly(uint64_t attrValue) {
  getProperties().setStatsIdOnly = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigNormOp::setActMsb(uint64_t attrValue) {
  getProperties().actMsb = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigNormOp::setStatsId(uint64_t attrValue) {
  getProperties().StatsId = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigNormOp::setIgeluQb(uint64_t attrValue) {
  getProperties().igeluQb = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigNormOp::setIgeluQc(uint64_t attrValue) {
  getProperties().igeluQc = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigNormOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr qConst, ::mlir::IntegerAttr qConstType, ::mlir::IntegerAttr setStatsIdOnly, ::mlir::IntegerAttr actMsb, ::mlir::IntegerAttr StatsId, ::mlir::IntegerAttr igeluQb, ::mlir::IntegerAttr igeluQc) {
  if (qConst) {
    odsState.getOrAddProperties<Properties>().qConst = qConst;
  }
  if (qConstType) {
    odsState.getOrAddProperties<Properties>().qConstType = qConstType;
  }
  if (setStatsIdOnly) {
    odsState.getOrAddProperties<Properties>().setStatsIdOnly = setStatsIdOnly;
  }
  if (actMsb) {
    odsState.getOrAddProperties<Properties>().actMsb = actMsb;
  }
  if (StatsId) {
    odsState.getOrAddProperties<Properties>().StatsId = StatsId;
  }
  if (igeluQb) {
    odsState.getOrAddProperties<Properties>().igeluQb = igeluQb;
  }
  if (igeluQc) {
    odsState.getOrAddProperties<Properties>().igeluQc = igeluQc;
  }
}

void ConfigNormOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr qConst, ::mlir::IntegerAttr qConstType, ::mlir::IntegerAttr setStatsIdOnly, ::mlir::IntegerAttr actMsb, ::mlir::IntegerAttr StatsId, ::mlir::IntegerAttr igeluQb, ::mlir::IntegerAttr igeluQc) {
  if (qConst) {
    odsState.getOrAddProperties<Properties>().qConst = qConst;
  }
  if (qConstType) {
    odsState.getOrAddProperties<Properties>().qConstType = qConstType;
  }
  if (setStatsIdOnly) {
    odsState.getOrAddProperties<Properties>().setStatsIdOnly = setStatsIdOnly;
  }
  if (actMsb) {
    odsState.getOrAddProperties<Properties>().actMsb = actMsb;
  }
  if (StatsId) {
    odsState.getOrAddProperties<Properties>().StatsId = StatsId;
  }
  if (igeluQb) {
    odsState.getOrAddProperties<Properties>().igeluQb = igeluQb;
  }
  if (igeluQc) {
    odsState.getOrAddProperties<Properties>().igeluQc = igeluQc;
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConfigNormOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint64_t qConst, uint64_t qConstType, uint64_t setStatsIdOnly, uint64_t actMsb, uint64_t StatsId, uint64_t igeluQb, uint64_t igeluQc) {
  odsState.getOrAddProperties<Properties>().qConst = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), qConst);
  odsState.getOrAddProperties<Properties>().qConstType = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), qConstType);
  odsState.getOrAddProperties<Properties>().setStatsIdOnly = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), setStatsIdOnly);
  odsState.getOrAddProperties<Properties>().actMsb = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), actMsb);
  odsState.getOrAddProperties<Properties>().StatsId = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), StatsId);
  odsState.getOrAddProperties<Properties>().igeluQb = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), igeluQb);
  odsState.getOrAddProperties<Properties>().igeluQc = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), igeluQc);
}

void ConfigNormOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint64_t qConst, uint64_t qConstType, uint64_t setStatsIdOnly, uint64_t actMsb, uint64_t StatsId, uint64_t igeluQb, uint64_t igeluQc) {
  odsState.getOrAddProperties<Properties>().qConst = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), qConst);
  odsState.getOrAddProperties<Properties>().qConstType = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), qConstType);
  odsState.getOrAddProperties<Properties>().setStatsIdOnly = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), setStatsIdOnly);
  odsState.getOrAddProperties<Properties>().actMsb = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), actMsb);
  odsState.getOrAddProperties<Properties>().StatsId = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), StatsId);
  odsState.getOrAddProperties<Properties>().igeluQb = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), igeluQb);
  odsState.getOrAddProperties<Properties>().igeluQc = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), igeluQc);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConfigNormOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ConfigNormOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void ConfigNormOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.qConst)
    properties.qConst = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.qConstType)
    properties.qConstType = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.setStatsIdOnly)
    properties.setStatsIdOnly = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.actMsb)
    properties.actMsb = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.StatsId)
    properties.StatsId = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.igeluQb)
    properties.igeluQb = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.igeluQc)
    properties.igeluQc = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
}

::llvm::LogicalResult ConfigNormOp::verifyInvariantsImpl() {
  auto tblgen_StatsId = getProperties().StatsId; (void)tblgen_StatsId;
  auto tblgen_actMsb = getProperties().actMsb; (void)tblgen_actMsb;
  auto tblgen_igeluQb = getProperties().igeluQb; (void)tblgen_igeluQb;
  auto tblgen_igeluQc = getProperties().igeluQc; (void)tblgen_igeluQc;
  auto tblgen_qConst = getProperties().qConst; (void)tblgen_qConst;
  auto tblgen_qConstType = getProperties().qConstType; (void)tblgen_qConstType;
  auto tblgen_setStatsIdOnly = getProperties().setStatsIdOnly; (void)tblgen_setStatsIdOnly;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_qConst, "qConst")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_qConstType, "qConstType")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_setStatsIdOnly, "setStatsIdOnly")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_actMsb, "actMsb")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_StatsId, "StatsId")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_igeluQb, "igeluQb")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_igeluQc, "igeluQc")))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::LogicalResult ConfigNormOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ConfigNormOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void ConfigNormOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getQConstAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("qConst");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getQConstTypeAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("qConstType");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getSetStatsIdOnlyAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("setStatsIdOnly");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getActMsbAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("actMsb");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStatsIdAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("StatsId");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getIgeluQbAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("igeluQb");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getIgeluQcAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("igeluQc");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::ConfigNormOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::ConfigStOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ConfigStOpGenericAdaptorBase::ConfigStOpGenericAdaptorBase(ConfigStOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::IntegerAttr ConfigStOpGenericAdaptorBase::getActivationAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().activation);
  return attr;
}

uint64_t ConfigStOpGenericAdaptorBase::getActivation() {
  auto attr = getActivationAttr();
  return attr.getValue().getZExtValue();
}

::mlir::FloatAttr ConfigStOpGenericAdaptorBase::getScaleAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().scale);
  return attr;
}

::llvm::APFloat ConfigStOpGenericAdaptorBase::getScale() {
  auto attr = getScaleAttr();
  return attr.getValue();
}

} // namespace detail
ConfigStOpAdaptor::ConfigStOpAdaptor(ConfigStOp op) : ConfigStOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConfigStOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_activation = getProperties().activation; (void)tblgen_activation;
  auto tblgen_scale = getProperties().scale; (void)tblgen_scale;

  if (tblgen_activation && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_activation))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_activation).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.config_st' op ""attribute 'activation' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_scale && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_scale))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_scale).getType().isF32()))))
    return emitError(loc, "'gemmini.config_st' op ""attribute 'scale' failed to satisfy constraint: 32-bit float attribute");
  return ::mlir::success();
}

::llvm::LogicalResult ConfigStOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.activation;
       auto attr = dict.get("activation");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `activation` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.scale;
       auto attr = dict.get("scale");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `scale` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ConfigStOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.activation;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("activation",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.scale;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("scale",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ConfigStOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.activation.getAsOpaquePointer()), 
    llvm::hash_value(prop.scale.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ConfigStOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "activation")
      return prop.activation;

    if (name == "scale")
      return prop.scale;
  return std::nullopt;
}

void ConfigStOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "activation") {
       prop.activation = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.activation)>>(value);
       return;
    }

    if (name == "scale") {
       prop.scale = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.scale)>>(value);
       return;
    }
}

void ConfigStOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.activation) attrs.append("activation", prop.activation);

    if (prop.scale) attrs.append("scale", prop.scale);
}

::llvm::LogicalResult ConfigStOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getActivationAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "activation", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getScaleAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(attr, "scale", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ConfigStOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.activation)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.scale)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConfigStOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.activation);

  writer.writeOptionalAttribute(prop.scale);
}

uint64_t ConfigStOp::getActivation() {
  auto attr = getActivationAttr();
  return attr.getValue().getZExtValue();
}

::llvm::APFloat ConfigStOp::getScale() {
  auto attr = getScaleAttr();
  return attr.getValue();
}

void ConfigStOp::setActivation(uint64_t attrValue) {
  getProperties().activation = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConfigStOp::setScale(::llvm::APFloat attrValue) {
  getProperties().scale = ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue);
}

void ConfigStOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value stride, ::mlir::IntegerAttr activation, ::mlir::FloatAttr scale) {
  odsState.addOperands(stride);
  if (activation) {
    odsState.getOrAddProperties<Properties>().activation = activation;
  }
  if (scale) {
    odsState.getOrAddProperties<Properties>().scale = scale;
  }
}

void ConfigStOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value stride, ::mlir::IntegerAttr activation, ::mlir::FloatAttr scale) {
  odsState.addOperands(stride);
  if (activation) {
    odsState.getOrAddProperties<Properties>().activation = activation;
  }
  if (scale) {
    odsState.getOrAddProperties<Properties>().scale = scale;
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConfigStOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value stride, uint64_t activation, ::llvm::APFloat scale) {
  odsState.addOperands(stride);
  odsState.getOrAddProperties<Properties>().activation = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), activation);
  odsState.getOrAddProperties<Properties>().scale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), scale);
}

void ConfigStOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value stride, uint64_t activation, ::llvm::APFloat scale) {
  odsState.addOperands(stride);
  odsState.getOrAddProperties<Properties>().activation = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), activation);
  odsState.getOrAddProperties<Properties>().scale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), scale);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConfigStOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ConfigStOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void ConfigStOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.activation)
    properties.activation = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.scale)
    properties.scale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0);
}

::llvm::LogicalResult ConfigStOp::verifyInvariantsImpl() {
  auto tblgen_activation = getProperties().activation; (void)tblgen_activation;
  auto tblgen_scale = getProperties().scale; (void)tblgen_scale;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_activation, "activation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(*this, tblgen_scale, "scale")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ConfigStOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ConfigStOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand strideRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> strideOperands(&strideRawOperand, 1);  ::llvm::SMLoc strideOperandsLoc;
  (void)strideOperandsLoc;
  ::mlir::Type strideRawType{};
  ::llvm::ArrayRef<::mlir::Type> strideTypes(&strideRawType, 1);

  strideOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(strideRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    strideRawType = type;
  }
  if (parser.resolveOperands(strideOperands, strideTypes, strideOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConfigStOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getStride();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getActivationAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("activation");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getScaleAttr();
     if(attr && (attr == odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0)))
       elidedAttrs.push_back("scale");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getStride().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::ConfigStOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::FlushOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
FlushOpAdaptor::FlushOpAdaptor(FlushOp op) : FlushOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult FlushOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void FlushOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value skip) {
  odsState.addOperands(skip);
}

void FlushOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value skip) {
  odsState.addOperands(skip);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FlushOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult FlushOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult FlushOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FlushOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand skipRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> skipOperands(&skipRawOperand, 1);  ::llvm::SMLoc skipOperandsLoc;
  (void)skipOperandsLoc;
  ::mlir::Type skipRawType{};
  ::llvm::ArrayRef<::mlir::Type> skipTypes(&skipRawType, 1);

  skipOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(skipRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    skipRawType = type;
  }
  if (parser.resolveOperands(skipOperands, skipTypes, skipOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FlushOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSkip();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSkip().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::FlushOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::ComputeAccumulated_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ComputeAccumulated_IntrOpAdaptor::ComputeAccumulated_IntrOpAdaptor(ComputeAccumulated_IntrOp op) : ComputeAccumulated_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ComputeAccumulated_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ComputeAccumulated_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void ComputeAccumulated_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ComputeAccumulated_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ComputeAccumulated_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ComputeAccumulated_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::ComputeAccumulated_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::ComputePreloaded_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ComputePreloaded_IntrOpAdaptor::ComputePreloaded_IntrOpAdaptor(ComputePreloaded_IntrOp op) : ComputePreloaded_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ComputePreloaded_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ComputePreloaded_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void ComputePreloaded_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ComputePreloaded_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ComputePreloaded_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ComputePreloaded_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::ComputePreloaded_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::ConfigEX_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ConfigEX_IntrOpAdaptor::ConfigEX_IntrOpAdaptor(ConfigEX_IntrOp op) : ConfigEX_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConfigEX_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ConfigEX_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void ConfigEX_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConfigEX_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ConfigEX_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ConfigEX_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::ConfigEX_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::ConfigNorm_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ConfigNorm_IntrOpAdaptor::ConfigNorm_IntrOpAdaptor(ConfigNorm_IntrOp op) : ConfigNorm_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConfigNorm_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ConfigNorm_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void ConfigNorm_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConfigNorm_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ConfigNorm_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ConfigNorm_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::ConfigNorm_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::ConfigSt_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ConfigSt_IntrOpAdaptor::ConfigSt_IntrOpAdaptor(ConfigSt_IntrOp op) : ConfigSt_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConfigSt_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ConfigSt_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void ConfigSt_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConfigSt_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ConfigSt_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ConfigSt_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::ConfigSt_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::ConifgLd_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ConifgLd_IntrOpAdaptor::ConifgLd_IntrOpAdaptor(ConifgLd_IntrOp op) : ConifgLd_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConifgLd_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ConifgLd_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void ConifgLd_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConifgLd_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ConifgLd_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ConifgLd_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::ConifgLd_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::Flush_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
Flush_IntrOpAdaptor::Flush_IntrOpAdaptor(Flush_IntrOp op) : Flush_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult Flush_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void Flush_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void Flush_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Flush_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult Flush_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult Flush_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::Flush_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopConvWsConfig1_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopConvWsConfig1_IntrOpAdaptor::LoopConvWsConfig1_IntrOpAdaptor(LoopConvWsConfig1_IntrOp op) : LoopConvWsConfig1_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopConvWsConfig1_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopConvWsConfig1_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopConvWsConfig1_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopConvWsConfig1_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopConvWsConfig1_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopConvWsConfig1_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopConvWsConfig1_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopConvWsConfig2_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopConvWsConfig2_IntrOpAdaptor::LoopConvWsConfig2_IntrOpAdaptor(LoopConvWsConfig2_IntrOp op) : LoopConvWsConfig2_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopConvWsConfig2_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopConvWsConfig2_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopConvWsConfig2_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopConvWsConfig2_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopConvWsConfig2_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopConvWsConfig2_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopConvWsConfig2_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopConvWsConfig3_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopConvWsConfig3_IntrOpAdaptor::LoopConvWsConfig3_IntrOpAdaptor(LoopConvWsConfig3_IntrOp op) : LoopConvWsConfig3_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopConvWsConfig3_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopConvWsConfig3_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopConvWsConfig3_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopConvWsConfig3_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopConvWsConfig3_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopConvWsConfig3_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopConvWsConfig3_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopConvWsConfig4_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopConvWsConfig4_IntrOpAdaptor::LoopConvWsConfig4_IntrOpAdaptor(LoopConvWsConfig4_IntrOp op) : LoopConvWsConfig4_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopConvWsConfig4_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopConvWsConfig4_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopConvWsConfig4_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopConvWsConfig4_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopConvWsConfig4_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopConvWsConfig4_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopConvWsConfig4_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopConvWsConfig5_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopConvWsConfig5_IntrOpAdaptor::LoopConvWsConfig5_IntrOpAdaptor(LoopConvWsConfig5_IntrOp op) : LoopConvWsConfig5_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopConvWsConfig5_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopConvWsConfig5_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopConvWsConfig5_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopConvWsConfig5_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopConvWsConfig5_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopConvWsConfig5_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopConvWsConfig5_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopConvWsConfig6_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopConvWsConfig6_IntrOpAdaptor::LoopConvWsConfig6_IntrOpAdaptor(LoopConvWsConfig6_IntrOp op) : LoopConvWsConfig6_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopConvWsConfig6_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopConvWsConfig6_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopConvWsConfig6_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopConvWsConfig6_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopConvWsConfig6_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopConvWsConfig6_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopConvWsConfig6_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopConvWs_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopConvWs_IntrOpAdaptor::LoopConvWs_IntrOpAdaptor(LoopConvWs_IntrOp op) : LoopConvWs_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopConvWs_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopConvWs_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopConvWs_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopConvWs_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopConvWs_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopConvWs_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopConvWs_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopWsConfigAddrsAB_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopWsConfigAddrsAB_IntrOpAdaptor::LoopWsConfigAddrsAB_IntrOpAdaptor(LoopWsConfigAddrsAB_IntrOp op) : LoopWsConfigAddrsAB_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopWsConfigAddrsAB_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopWsConfigAddrsAB_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopWsConfigAddrsAB_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopWsConfigAddrsAB_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopWsConfigAddrsAB_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopWsConfigAddrsAB_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopWsConfigAddrsAB_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopWsConfigAddrsDC_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopWsConfigAddrsDC_IntrOpAdaptor::LoopWsConfigAddrsDC_IntrOpAdaptor(LoopWsConfigAddrsDC_IntrOp op) : LoopWsConfigAddrsDC_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopWsConfigAddrsDC_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopWsConfigAddrsDC_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopWsConfigAddrsDC_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopWsConfigAddrsDC_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopWsConfigAddrsDC_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopWsConfigAddrsDC_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopWsConfigAddrsDC_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopWsConfigBounds_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopWsConfigBounds_IntrOpAdaptor::LoopWsConfigBounds_IntrOpAdaptor(LoopWsConfigBounds_IntrOp op) : LoopWsConfigBounds_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopWsConfigBounds_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopWsConfigBounds_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopWsConfigBounds_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopWsConfigBounds_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopWsConfigBounds_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopWsConfigBounds_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopWsConfigBounds_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopWsConfigStridesAB_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopWsConfigStridesAB_IntrOpAdaptor::LoopWsConfigStridesAB_IntrOpAdaptor(LoopWsConfigStridesAB_IntrOp op) : LoopWsConfigStridesAB_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopWsConfigStridesAB_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopWsConfigStridesAB_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopWsConfigStridesAB_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopWsConfigStridesAB_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopWsConfigStridesAB_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopWsConfigStridesAB_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopWsConfigStridesAB_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopWsConfigStridesDC_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopWsConfigStridesDC_IntrOpAdaptor::LoopWsConfigStridesDC_IntrOpAdaptor(LoopWsConfigStridesDC_IntrOp op) : LoopWsConfigStridesDC_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopWsConfigStridesDC_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopWsConfigStridesDC_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopWsConfigStridesDC_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopWsConfigStridesDC_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopWsConfigStridesDC_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopWsConfigStridesDC_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopWsConfigStridesDC_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::LoopWs_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoopWs_IntrOpAdaptor::LoopWs_IntrOpAdaptor(LoopWs_IntrOp op) : LoopWs_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoopWs_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopWs_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void LoopWs_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopWs_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult LoopWs_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoopWs_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::LoopWs_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::Mvin2_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
Mvin2_IntrOpAdaptor::Mvin2_IntrOpAdaptor(Mvin2_IntrOp op) : Mvin2_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult Mvin2_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void Mvin2_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void Mvin2_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Mvin2_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult Mvin2_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult Mvin2_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::Mvin2_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::Mvin3_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
Mvin3_IntrOpAdaptor::Mvin3_IntrOpAdaptor(Mvin3_IntrOp op) : Mvin3_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult Mvin3_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void Mvin3_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void Mvin3_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Mvin3_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult Mvin3_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult Mvin3_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::Mvin3_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::Mvin_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
Mvin_IntrOpAdaptor::Mvin_IntrOpAdaptor(Mvin_IntrOp op) : Mvin_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult Mvin_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void Mvin_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void Mvin_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Mvin_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult Mvin_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult Mvin_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::Mvin_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::Mvout_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
Mvout_IntrOpAdaptor::Mvout_IntrOpAdaptor(Mvout_IntrOp op) : Mvout_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult Mvout_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void Mvout_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void Mvout_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Mvout_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult Mvout_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult Mvout_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::Mvout_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::Preload_IntrOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
Preload_IntrOpAdaptor::Preload_IntrOpAdaptor(Preload_IntrOp op) : Preload_IntrOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult Preload_IntrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void Preload_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
}

void Preload_IntrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Preload_IntrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult Preload_IntrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult Preload_IntrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::Preload_IntrOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::Mvin2Op definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
Mvin2OpAdaptor::Mvin2OpAdaptor(Mvin2Op op) : Mvin2OpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult Mvin2OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void Mvin2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value addr) {
  odsState.addOperands(input);
  odsState.addOperands(addr);
}

void Mvin2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value addr) {
  odsState.addOperands(input);
  odsState.addOperands(addr);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Mvin2Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult Mvin2Op::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult Mvin2Op::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult Mvin2Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand addrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> addrOperands(&addrRawOperand, 1);  ::llvm::SMLoc addrOperandsLoc;
  (void)addrOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);
  ::mlir::Type addrRawType{};
  ::llvm::ArrayRef<::mlir::Type> addrTypes(&addrRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();

  addrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addrRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    addrRawType = type;
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(addrOperands, addrTypes, addrOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void Mvin2Op::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  _odsPrinter << ' ';
  _odsPrinter << getAddr();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getAddr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::Mvin2Op)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::Mvin3Op definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
Mvin3OpAdaptor::Mvin3OpAdaptor(Mvin3Op op) : Mvin3OpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult Mvin3OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void Mvin3Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value addr) {
  odsState.addOperands(input);
  odsState.addOperands(addr);
}

void Mvin3Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value addr) {
  odsState.addOperands(input);
  odsState.addOperands(addr);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Mvin3Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult Mvin3Op::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult Mvin3Op::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult Mvin3Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand addrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> addrOperands(&addrRawOperand, 1);  ::llvm::SMLoc addrOperandsLoc;
  (void)addrOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);
  ::mlir::Type addrRawType{};
  ::llvm::ArrayRef<::mlir::Type> addrTypes(&addrRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();

  addrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addrRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    addrRawType = type;
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(addrOperands, addrTypes, addrOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void Mvin3Op::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  _odsPrinter << ' ';
  _odsPrinter << getAddr();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getAddr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::Mvin3Op)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::MvinOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
MvinOpAdaptor::MvinOpAdaptor(MvinOp op) : MvinOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult MvinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void MvinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value addr) {
  odsState.addOperands(input);
  odsState.addOperands(addr);
}

void MvinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value addr) {
  odsState.addOperands(input);
  odsState.addOperands(addr);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MvinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult MvinOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult MvinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MvinOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand addrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> addrOperands(&addrRawOperand, 1);  ::llvm::SMLoc addrOperandsLoc;
  (void)addrOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);
  ::mlir::Type addrRawType{};
  ::llvm::ArrayRef<::mlir::Type> addrTypes(&addrRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();

  addrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addrRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    addrRawType = type;
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(addrOperands, addrTypes, addrOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MvinOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  _odsPrinter << ' ';
  _odsPrinter << getAddr();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getAddr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::MvinOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::MvoutOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
MvoutOpAdaptor::MvoutOpAdaptor(MvoutOp op) : MvoutOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult MvoutOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void MvoutOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value output, ::mlir::Value addr) {
  odsState.addOperands(output);
  odsState.addOperands(addr);
}

void MvoutOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value output, ::mlir::Value addr) {
  odsState.addOperands(output);
  odsState.addOperands(addr);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MvoutOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult MvoutOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult MvoutOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MvoutOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand outputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> outputOperands(&outputRawOperand, 1);  ::llvm::SMLoc outputOperandsLoc;
  (void)outputOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand addrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> addrOperands(&addrRawOperand, 1);  ::llvm::SMLoc addrOperandsLoc;
  (void)addrOperandsLoc;
  ::mlir::Type outputRawType{};
  ::llvm::ArrayRef<::mlir::Type> outputTypes(&outputRawType, 1);
  ::mlir::Type addrRawType{};
  ::llvm::ArrayRef<::mlir::Type> addrTypes(&addrRawType, 1);

  outputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(outputRawOperand))
    return ::mlir::failure();

  addrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addrRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    outputRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    addrRawType = type;
  }
  if (parser.resolveOperands(outputOperands, outputTypes, outputOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(addrOperands, addrTypes, addrOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MvoutOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOutput();
  _odsPrinter << ' ';
  _odsPrinter << getAddr();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOutput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getAddr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::MvoutOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::PreloadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
PreloadOpAdaptor::PreloadOpAdaptor(PreloadOp op) : PreloadOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult PreloadOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void PreloadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value bdAddr, ::mlir::Value cAddr, ::mlir::Value bdRows, ::mlir::Value bdCols, ::mlir::Value cRows, ::mlir::Value cCols) {
  odsState.addOperands(bdAddr);
  odsState.addOperands(cAddr);
  odsState.addOperands(bdRows);
  odsState.addOperands(bdCols);
  odsState.addOperands(cRows);
  odsState.addOperands(cCols);
}

void PreloadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value bdAddr, ::mlir::Value cAddr, ::mlir::Value bdRows, ::mlir::Value bdCols, ::mlir::Value cRows, ::mlir::Value cCols) {
  odsState.addOperands(bdAddr);
  odsState.addOperands(cAddr);
  odsState.addOperands(bdRows);
  odsState.addOperands(bdCols);
  odsState.addOperands(cRows);
  odsState.addOperands(cCols);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PreloadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 6u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult PreloadOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult PreloadOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PreloadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand bdAddrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bdAddrOperands(&bdAddrRawOperand, 1);  ::llvm::SMLoc bdAddrOperandsLoc;
  (void)bdAddrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand cAddrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> cAddrOperands(&cAddrRawOperand, 1);  ::llvm::SMLoc cAddrOperandsLoc;
  (void)cAddrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bdRowsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bdRowsOperands(&bdRowsRawOperand, 1);  ::llvm::SMLoc bdRowsOperandsLoc;
  (void)bdRowsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bdColsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bdColsOperands(&bdColsRawOperand, 1);  ::llvm::SMLoc bdColsOperandsLoc;
  (void)bdColsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand cRowsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> cRowsOperands(&cRowsRawOperand, 1);  ::llvm::SMLoc cRowsOperandsLoc;
  (void)cRowsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand cColsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> cColsOperands(&cColsRawOperand, 1);  ::llvm::SMLoc cColsOperandsLoc;
  (void)cColsOperandsLoc;
  ::mlir::Type bdAddrRawType{};
  ::llvm::ArrayRef<::mlir::Type> bdAddrTypes(&bdAddrRawType, 1);
  ::mlir::Type cAddrRawType{};
  ::llvm::ArrayRef<::mlir::Type> cAddrTypes(&cAddrRawType, 1);
  ::mlir::Type bdRowsRawType{};
  ::llvm::ArrayRef<::mlir::Type> bdRowsTypes(&bdRowsRawType, 1);
  ::mlir::Type bdColsRawType{};
  ::llvm::ArrayRef<::mlir::Type> bdColsTypes(&bdColsRawType, 1);
  ::mlir::Type cRowsRawType{};
  ::llvm::ArrayRef<::mlir::Type> cRowsTypes(&cRowsRawType, 1);
  ::mlir::Type cColsRawType{};
  ::llvm::ArrayRef<::mlir::Type> cColsTypes(&cColsRawType, 1);

  bdAddrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bdAddrRawOperand))
    return ::mlir::failure();

  cAddrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(cAddrRawOperand))
    return ::mlir::failure();

  bdRowsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bdRowsRawOperand))
    return ::mlir::failure();

  bdColsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bdColsRawOperand))
    return ::mlir::failure();

  cRowsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(cRowsRawOperand))
    return ::mlir::failure();

  cColsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(cColsRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bdAddrRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    cAddrRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bdRowsRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bdColsRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    cRowsRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    cColsRawType = type;
  }
  if (parser.resolveOperands(bdAddrOperands, bdAddrTypes, bdAddrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(cAddrOperands, cAddrTypes, cAddrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bdRowsOperands, bdRowsTypes, bdRowsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bdColsOperands, bdColsTypes, bdColsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(cRowsOperands, cRowsTypes, cRowsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(cColsOperands, cColsTypes, cColsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PreloadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBdAddr();
  _odsPrinter << ' ';
  _odsPrinter << getCAddr();
  _odsPrinter << ' ';
  _odsPrinter << getBdRows();
  _odsPrinter << ' ';
  _odsPrinter << getBdCols();
  _odsPrinter << ' ';
  _odsPrinter << getCRows();
  _odsPrinter << ' ';
  _odsPrinter << getCCols();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getBdAddr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getCAddr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getBdRows().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getBdCols().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getCRows().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getCCols().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::PreloadOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::PreloadZerosOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
PreloadZerosOpAdaptor::PreloadZerosOpAdaptor(PreloadZerosOp op) : PreloadZerosOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult PreloadZerosOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void PreloadZerosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value addr, ::mlir::Value cRows, ::mlir::Value cCols) {
  odsState.addOperands(addr);
  odsState.addOperands(cRows);
  odsState.addOperands(cCols);
}

void PreloadZerosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, ::mlir::Value cRows, ::mlir::Value cCols) {
  odsState.addOperands(addr);
  odsState.addOperands(cRows);
  odsState.addOperands(cCols);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PreloadZerosOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult PreloadZerosOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult PreloadZerosOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PreloadZerosOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand addrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> addrOperands(&addrRawOperand, 1);  ::llvm::SMLoc addrOperandsLoc;
  (void)addrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand cRowsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> cRowsOperands(&cRowsRawOperand, 1);  ::llvm::SMLoc cRowsOperandsLoc;
  (void)cRowsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand cColsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> cColsOperands(&cColsRawOperand, 1);  ::llvm::SMLoc cColsOperandsLoc;
  (void)cColsOperandsLoc;
  ::mlir::Type addrRawType{};
  ::llvm::ArrayRef<::mlir::Type> addrTypes(&addrRawType, 1);
  ::mlir::Type cRowsRawType{};
  ::llvm::ArrayRef<::mlir::Type> cRowsTypes(&cRowsRawType, 1);
  ::mlir::Type cColsRawType{};
  ::llvm::ArrayRef<::mlir::Type> cColsTypes(&cColsRawType, 1);

  addrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addrRawOperand))
    return ::mlir::failure();

  cRowsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(cRowsRawOperand))
    return ::mlir::failure();

  cColsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(cColsRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    addrRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    cRowsRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    cColsRawType = type;
  }
  if (parser.resolveOperands(addrOperands, addrTypes, addrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(cRowsOperands, cRowsTypes, cRowsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(cColsOperands, cColsTypes, cColsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PreloadZerosOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAddr();
  _odsPrinter << ' ';
  _odsPrinter << getCRows();
  _odsPrinter << ' ';
  _odsPrinter << getCCols();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAddr().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getCRows().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getCCols().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::PreloadZerosOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::PrintOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
PrintOpAdaptor::PrintOpAdaptor(PrintOp op) : PrintOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult PrintOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input) {
  odsState.addOperands(input);
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult PrintOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult PrintOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PrintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PrintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::PrintOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::TileConvOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TileConvOpGenericAdaptorBase::TileConvOpGenericAdaptorBase(TileConvOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::FloatAttr TileConvOpGenericAdaptorBase::getScaleAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().scale);
  return attr;
}

::llvm::APFloat TileConvOpGenericAdaptorBase::getScale() {
  auto attr = getScaleAttr();
  return attr.getValue();
}

::mlir::IntegerAttr TileConvOpGenericAdaptorBase::getStrideAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().stride);
  return attr;
}

uint64_t TileConvOpGenericAdaptorBase::getStride() {
  auto attr = getStrideAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr TileConvOpGenericAdaptorBase::getInputDilationAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().inputDilation);
  return attr;
}

uint64_t TileConvOpGenericAdaptorBase::getInputDilation() {
  auto attr = getInputDilationAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr TileConvOpGenericAdaptorBase::getKernelDilationAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().kernelDilation);
  return attr;
}

uint64_t TileConvOpGenericAdaptorBase::getKernelDilation() {
  auto attr = getKernelDilationAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr TileConvOpGenericAdaptorBase::getPaddingAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().padding);
  return attr;
}

uint64_t TileConvOpGenericAdaptorBase::getPadding() {
  auto attr = getPaddingAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr TileConvOpGenericAdaptorBase::getWrot180Attr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().wrot180);
  return attr;
}

bool TileConvOpGenericAdaptorBase::getWrot180() {
  auto attr = getWrot180Attr();
  return attr.getValue();
}

::mlir::BoolAttr TileConvOpGenericAdaptorBase::getTransOutput1203Attr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().transOutput1203);
  return attr;
}

bool TileConvOpGenericAdaptorBase::getTransOutput1203() {
  auto attr = getTransOutput1203Attr();
  return attr.getValue();
}

::mlir::BoolAttr TileConvOpGenericAdaptorBase::getTransInput3120Attr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().transInput3120);
  return attr;
}

bool TileConvOpGenericAdaptorBase::getTransInput3120() {
  auto attr = getTransInput3120Attr();
  return attr.getValue();
}

::mlir::BoolAttr TileConvOpGenericAdaptorBase::getTransWeight1203Attr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().transWeight1203);
  return attr;
}

bool TileConvOpGenericAdaptorBase::getTransWeight1203() {
  auto attr = getTransWeight1203Attr();
  return attr.getValue();
}

::mlir::BoolAttr TileConvOpGenericAdaptorBase::getTransWeight0132Attr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().transWeight0132);
  return attr;
}

bool TileConvOpGenericAdaptorBase::getTransWeight0132() {
  auto attr = getTransWeight0132Attr();
  return attr.getValue();
}

::mlir::IntegerAttr TileConvOpGenericAdaptorBase::getActAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().act);
  return attr;
}

uint64_t TileConvOpGenericAdaptorBase::getAct() {
  auto attr = getActAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr TileConvOpGenericAdaptorBase::getPoolSizeAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().poolSize);
  return attr;
}

uint64_t TileConvOpGenericAdaptorBase::getPoolSize() {
  auto attr = getPoolSizeAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr TileConvOpGenericAdaptorBase::getPoolStrideAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().poolStride);
  return attr;
}

uint64_t TileConvOpGenericAdaptorBase::getPoolStride() {
  auto attr = getPoolStrideAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr TileConvOpGenericAdaptorBase::getPoolPaddingAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().poolPadding);
  return attr;
}

uint64_t TileConvOpGenericAdaptorBase::getPoolPadding() {
  auto attr = getPoolPaddingAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
TileConvOpAdaptor::TileConvOpAdaptor(TileConvOp op) : TileConvOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TileConvOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_act = getProperties().act; (void)tblgen_act;
  auto tblgen_inputDilation = getProperties().inputDilation; (void)tblgen_inputDilation;
  auto tblgen_kernelDilation = getProperties().kernelDilation; (void)tblgen_kernelDilation;
  auto tblgen_padding = getProperties().padding; (void)tblgen_padding;
  auto tblgen_poolPadding = getProperties().poolPadding; (void)tblgen_poolPadding;
  auto tblgen_poolSize = getProperties().poolSize; (void)tblgen_poolSize;
  auto tblgen_poolStride = getProperties().poolStride; (void)tblgen_poolStride;
  auto tblgen_scale = getProperties().scale; (void)tblgen_scale;
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  auto tblgen_transInput3120 = getProperties().transInput3120; (void)tblgen_transInput3120;
  auto tblgen_transOutput1203 = getProperties().transOutput1203; (void)tblgen_transOutput1203;
  auto tblgen_transWeight0132 = getProperties().transWeight0132; (void)tblgen_transWeight0132;
  auto tblgen_transWeight1203 = getProperties().transWeight1203; (void)tblgen_transWeight1203;
  auto tblgen_wrot180 = getProperties().wrot180; (void)tblgen_wrot180;

  if (tblgen_scale && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_scale))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_scale).getType().isF32()))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'scale' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_stride && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_stride))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_stride).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'stride' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_inputDilation && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_inputDilation))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_inputDilation).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'inputDilation' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_kernelDilation && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_kernelDilation))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_kernelDilation).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'kernelDilation' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_padding && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_padding))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_padding).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_wrot180 && !((::llvm::isa<::mlir::BoolAttr>(tblgen_wrot180))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'wrot180' failed to satisfy constraint: bool attribute");

  if (tblgen_transOutput1203 && !((::llvm::isa<::mlir::BoolAttr>(tblgen_transOutput1203))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'transOutput1203' failed to satisfy constraint: bool attribute");

  if (tblgen_transInput3120 && !((::llvm::isa<::mlir::BoolAttr>(tblgen_transInput3120))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'transInput3120' failed to satisfy constraint: bool attribute");

  if (tblgen_transWeight1203 && !((::llvm::isa<::mlir::BoolAttr>(tblgen_transWeight1203))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'transWeight1203' failed to satisfy constraint: bool attribute");

  if (tblgen_transWeight0132 && !((::llvm::isa<::mlir::BoolAttr>(tblgen_transWeight0132))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'transWeight0132' failed to satisfy constraint: bool attribute");

  if (tblgen_act && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_act))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_act).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'act' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_poolSize && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_poolSize))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_poolSize).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'poolSize' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_poolStride && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_poolStride))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_poolStride).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'poolStride' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_poolPadding && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_poolPadding))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_poolPadding).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.tile_conv' op ""attribute 'poolPadding' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult TileConvOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.act;
       auto attr = dict.get("act");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `act` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.inputDilation;
       auto attr = dict.get("inputDilation");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `inputDilation` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.kernelDilation;
       auto attr = dict.get("kernelDilation");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `kernelDilation` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.padding;
       auto attr = dict.get("padding");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `padding` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.poolPadding;
       auto attr = dict.get("poolPadding");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `poolPadding` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.poolSize;
       auto attr = dict.get("poolSize");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `poolSize` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.poolStride;
       auto attr = dict.get("poolStride");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `poolStride` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.scale;
       auto attr = dict.get("scale");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `scale` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.stride;
       auto attr = dict.get("stride");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `stride` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.transInput3120;
       auto attr = dict.get("transInput3120");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `transInput3120` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.transOutput1203;
       auto attr = dict.get("transOutput1203");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `transOutput1203` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.transWeight0132;
       auto attr = dict.get("transWeight0132");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `transWeight0132` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.transWeight1203;
       auto attr = dict.get("transWeight1203");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `transWeight1203` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.wrot180;
       auto attr = dict.get("wrot180");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `wrot180` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TileConvOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.act;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("act",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.inputDilation;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("inputDilation",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.kernelDilation;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("kernelDilation",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.padding;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("padding",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.poolPadding;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("poolPadding",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.poolSize;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("poolSize",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.poolStride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("poolStride",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.scale;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("scale",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.stride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("stride",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.transInput3120;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("transInput3120",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.transOutput1203;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("transOutput1203",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.transWeight0132;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("transWeight0132",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.transWeight1203;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("transWeight1203",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.wrot180;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("wrot180",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TileConvOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.act.getAsOpaquePointer()), 
    llvm::hash_value(prop.inputDilation.getAsOpaquePointer()), 
    llvm::hash_value(prop.kernelDilation.getAsOpaquePointer()), 
    llvm::hash_value(prop.padding.getAsOpaquePointer()), 
    llvm::hash_value(prop.poolPadding.getAsOpaquePointer()), 
    llvm::hash_value(prop.poolSize.getAsOpaquePointer()), 
    llvm::hash_value(prop.poolStride.getAsOpaquePointer()), 
    llvm::hash_value(prop.scale.getAsOpaquePointer()), 
    llvm::hash_value(prop.stride.getAsOpaquePointer()), 
    llvm::hash_value(prop.transInput3120.getAsOpaquePointer()), 
    llvm::hash_value(prop.transOutput1203.getAsOpaquePointer()), 
    llvm::hash_value(prop.transWeight0132.getAsOpaquePointer()), 
    llvm::hash_value(prop.transWeight1203.getAsOpaquePointer()), 
    llvm::hash_value(prop.wrot180.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TileConvOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "act")
      return prop.act;

    if (name == "inputDilation")
      return prop.inputDilation;

    if (name == "kernelDilation")
      return prop.kernelDilation;

    if (name == "padding")
      return prop.padding;

    if (name == "poolPadding")
      return prop.poolPadding;

    if (name == "poolSize")
      return prop.poolSize;

    if (name == "poolStride")
      return prop.poolStride;

    if (name == "scale")
      return prop.scale;

    if (name == "stride")
      return prop.stride;

    if (name == "transInput3120")
      return prop.transInput3120;

    if (name == "transOutput1203")
      return prop.transOutput1203;

    if (name == "transWeight0132")
      return prop.transWeight0132;

    if (name == "transWeight1203")
      return prop.transWeight1203;

    if (name == "wrot180")
      return prop.wrot180;
  return std::nullopt;
}

void TileConvOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "act") {
       prop.act = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.act)>>(value);
       return;
    }

    if (name == "inputDilation") {
       prop.inputDilation = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.inputDilation)>>(value);
       return;
    }

    if (name == "kernelDilation") {
       prop.kernelDilation = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.kernelDilation)>>(value);
       return;
    }

    if (name == "padding") {
       prop.padding = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.padding)>>(value);
       return;
    }

    if (name == "poolPadding") {
       prop.poolPadding = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.poolPadding)>>(value);
       return;
    }

    if (name == "poolSize") {
       prop.poolSize = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.poolSize)>>(value);
       return;
    }

    if (name == "poolStride") {
       prop.poolStride = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.poolStride)>>(value);
       return;
    }

    if (name == "scale") {
       prop.scale = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.scale)>>(value);
       return;
    }

    if (name == "stride") {
       prop.stride = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.stride)>>(value);
       return;
    }

    if (name == "transInput3120") {
       prop.transInput3120 = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.transInput3120)>>(value);
       return;
    }

    if (name == "transOutput1203") {
       prop.transOutput1203 = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.transOutput1203)>>(value);
       return;
    }

    if (name == "transWeight0132") {
       prop.transWeight0132 = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.transWeight0132)>>(value);
       return;
    }

    if (name == "transWeight1203") {
       prop.transWeight1203 = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.transWeight1203)>>(value);
       return;
    }

    if (name == "wrot180") {
       prop.wrot180 = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.wrot180)>>(value);
       return;
    }
}

void TileConvOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.act) attrs.append("act", prop.act);

    if (prop.inputDilation) attrs.append("inputDilation", prop.inputDilation);

    if (prop.kernelDilation) attrs.append("kernelDilation", prop.kernelDilation);

    if (prop.padding) attrs.append("padding", prop.padding);

    if (prop.poolPadding) attrs.append("poolPadding", prop.poolPadding);

    if (prop.poolSize) attrs.append("poolSize", prop.poolSize);

    if (prop.poolStride) attrs.append("poolStride", prop.poolStride);

    if (prop.scale) attrs.append("scale", prop.scale);

    if (prop.stride) attrs.append("stride", prop.stride);

    if (prop.transInput3120) attrs.append("transInput3120", prop.transInput3120);

    if (prop.transOutput1203) attrs.append("transOutput1203", prop.transOutput1203);

    if (prop.transWeight0132) attrs.append("transWeight0132", prop.transWeight0132);

    if (prop.transWeight1203) attrs.append("transWeight1203", prop.transWeight1203);

    if (prop.wrot180) attrs.append("wrot180", prop.wrot180);
}

::llvm::LogicalResult TileConvOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getActAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "act", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getInputDilationAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "inputDilation", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getKernelDilationAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "kernelDilation", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPaddingAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "padding", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPoolPaddingAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "poolPadding", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPoolSizeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "poolSize", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPoolStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "poolStride", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getScaleAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(attr, "scale", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "stride", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTransInput3120AttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "transInput3120", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTransOutput1203AttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "transOutput1203", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTransWeight0132AttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "transWeight0132", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTransWeight1203AttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "transWeight1203", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getWrot180AttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "wrot180", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TileConvOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.act)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.inputDilation)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.kernelDilation)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.padding)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.poolPadding)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.poolSize)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.poolStride)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.scale)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.stride)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.transInput3120)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.transOutput1203)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.transWeight0132)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.transWeight1203)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.wrot180)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TileConvOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.act);

  writer.writeOptionalAttribute(prop.inputDilation);

  writer.writeOptionalAttribute(prop.kernelDilation);

  writer.writeOptionalAttribute(prop.padding);

  writer.writeOptionalAttribute(prop.poolPadding);

  writer.writeOptionalAttribute(prop.poolSize);

  writer.writeOptionalAttribute(prop.poolStride);

  writer.writeOptionalAttribute(prop.scale);

  writer.writeOptionalAttribute(prop.stride);

  writer.writeOptionalAttribute(prop.transInput3120);

  writer.writeOptionalAttribute(prop.transOutput1203);

  writer.writeOptionalAttribute(prop.transWeight0132);

  writer.writeOptionalAttribute(prop.transWeight1203);

  writer.writeOptionalAttribute(prop.wrot180);
}

::llvm::APFloat TileConvOp::getScale() {
  auto attr = getScaleAttr();
  return attr.getValue();
}

uint64_t TileConvOp::getStride() {
  auto attr = getStrideAttr();
  return attr.getValue().getZExtValue();
}

uint64_t TileConvOp::getInputDilation() {
  auto attr = getInputDilationAttr();
  return attr.getValue().getZExtValue();
}

uint64_t TileConvOp::getKernelDilation() {
  auto attr = getKernelDilationAttr();
  return attr.getValue().getZExtValue();
}

uint64_t TileConvOp::getPadding() {
  auto attr = getPaddingAttr();
  return attr.getValue().getZExtValue();
}

bool TileConvOp::getWrot180() {
  auto attr = getWrot180Attr();
  return attr.getValue();
}

bool TileConvOp::getTransOutput1203() {
  auto attr = getTransOutput1203Attr();
  return attr.getValue();
}

bool TileConvOp::getTransInput3120() {
  auto attr = getTransInput3120Attr();
  return attr.getValue();
}

bool TileConvOp::getTransWeight1203() {
  auto attr = getTransWeight1203Attr();
  return attr.getValue();
}

bool TileConvOp::getTransWeight0132() {
  auto attr = getTransWeight0132Attr();
  return attr.getValue();
}

uint64_t TileConvOp::getAct() {
  auto attr = getActAttr();
  return attr.getValue().getZExtValue();
}

uint64_t TileConvOp::getPoolSize() {
  auto attr = getPoolSizeAttr();
  return attr.getValue().getZExtValue();
}

uint64_t TileConvOp::getPoolStride() {
  auto attr = getPoolStrideAttr();
  return attr.getValue().getZExtValue();
}

uint64_t TileConvOp::getPoolPadding() {
  auto attr = getPoolPaddingAttr();
  return attr.getValue().getZExtValue();
}

void TileConvOp::setScale(::llvm::APFloat attrValue) {
  getProperties().scale = ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue);
}

void TileConvOp::setStride(uint64_t attrValue) {
  getProperties().stride = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TileConvOp::setInputDilation(uint64_t attrValue) {
  getProperties().inputDilation = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TileConvOp::setKernelDilation(uint64_t attrValue) {
  getProperties().kernelDilation = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TileConvOp::setPadding(uint64_t attrValue) {
  getProperties().padding = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TileConvOp::setWrot180(bool attrValue) {
  getProperties().wrot180 = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void TileConvOp::setTransOutput1203(bool attrValue) {
  getProperties().transOutput1203 = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void TileConvOp::setTransInput3120(bool attrValue) {
  getProperties().transInput3120 = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void TileConvOp::setTransWeight1203(bool attrValue) {
  getProperties().transWeight1203 = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void TileConvOp::setTransWeight0132(bool attrValue) {
  getProperties().transWeight0132 = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void TileConvOp::setAct(uint64_t attrValue) {
  getProperties().act = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TileConvOp::setPoolSize(uint64_t attrValue) {
  getProperties().poolSize = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TileConvOp::setPoolStride(uint64_t attrValue) {
  getProperties().poolStride = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TileConvOp::setPoolPadding(uint64_t attrValue) {
  getProperties().poolPadding = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TileConvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::mlir::FloatAttr scale, ::mlir::IntegerAttr stride, ::mlir::IntegerAttr inputDilation, ::mlir::IntegerAttr kernelDilation, ::mlir::IntegerAttr padding, ::mlir::BoolAttr wrot180, ::mlir::BoolAttr transOutput1203, ::mlir::BoolAttr transInput3120, ::mlir::BoolAttr transWeight1203, ::mlir::BoolAttr transWeight0132, ::mlir::IntegerAttr act, ::mlir::IntegerAttr poolSize, ::mlir::IntegerAttr poolStride, ::mlir::IntegerAttr poolPadding) {
  odsState.addOperands(input);
  odsState.addOperands(weights);
  odsState.addOperands(bias);
  odsState.addOperands(output);
  odsState.addOperands(outRowDim);
  odsState.addOperands(outColDim);
  odsState.addOperands(kernelDim);
  if (scale) {
    odsState.getOrAddProperties<Properties>().scale = scale;
  }
  if (stride) {
    odsState.getOrAddProperties<Properties>().stride = stride;
  }
  if (inputDilation) {
    odsState.getOrAddProperties<Properties>().inputDilation = inputDilation;
  }
  if (kernelDilation) {
    odsState.getOrAddProperties<Properties>().kernelDilation = kernelDilation;
  }
  if (padding) {
    odsState.getOrAddProperties<Properties>().padding = padding;
  }
  if (wrot180) {
    odsState.getOrAddProperties<Properties>().wrot180 = wrot180;
  }
  if (transOutput1203) {
    odsState.getOrAddProperties<Properties>().transOutput1203 = transOutput1203;
  }
  if (transInput3120) {
    odsState.getOrAddProperties<Properties>().transInput3120 = transInput3120;
  }
  if (transWeight1203) {
    odsState.getOrAddProperties<Properties>().transWeight1203 = transWeight1203;
  }
  if (transWeight0132) {
    odsState.getOrAddProperties<Properties>().transWeight0132 = transWeight0132;
  }
  if (act) {
    odsState.getOrAddProperties<Properties>().act = act;
  }
  if (poolSize) {
    odsState.getOrAddProperties<Properties>().poolSize = poolSize;
  }
  if (poolStride) {
    odsState.getOrAddProperties<Properties>().poolStride = poolStride;
  }
  if (poolPadding) {
    odsState.getOrAddProperties<Properties>().poolPadding = poolPadding;
  }
}

void TileConvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::mlir::FloatAttr scale, ::mlir::IntegerAttr stride, ::mlir::IntegerAttr inputDilation, ::mlir::IntegerAttr kernelDilation, ::mlir::IntegerAttr padding, ::mlir::BoolAttr wrot180, ::mlir::BoolAttr transOutput1203, ::mlir::BoolAttr transInput3120, ::mlir::BoolAttr transWeight1203, ::mlir::BoolAttr transWeight0132, ::mlir::IntegerAttr act, ::mlir::IntegerAttr poolSize, ::mlir::IntegerAttr poolStride, ::mlir::IntegerAttr poolPadding) {
  odsState.addOperands(input);
  odsState.addOperands(weights);
  odsState.addOperands(bias);
  odsState.addOperands(output);
  odsState.addOperands(outRowDim);
  odsState.addOperands(outColDim);
  odsState.addOperands(kernelDim);
  if (scale) {
    odsState.getOrAddProperties<Properties>().scale = scale;
  }
  if (stride) {
    odsState.getOrAddProperties<Properties>().stride = stride;
  }
  if (inputDilation) {
    odsState.getOrAddProperties<Properties>().inputDilation = inputDilation;
  }
  if (kernelDilation) {
    odsState.getOrAddProperties<Properties>().kernelDilation = kernelDilation;
  }
  if (padding) {
    odsState.getOrAddProperties<Properties>().padding = padding;
  }
  if (wrot180) {
    odsState.getOrAddProperties<Properties>().wrot180 = wrot180;
  }
  if (transOutput1203) {
    odsState.getOrAddProperties<Properties>().transOutput1203 = transOutput1203;
  }
  if (transInput3120) {
    odsState.getOrAddProperties<Properties>().transInput3120 = transInput3120;
  }
  if (transWeight1203) {
    odsState.getOrAddProperties<Properties>().transWeight1203 = transWeight1203;
  }
  if (transWeight0132) {
    odsState.getOrAddProperties<Properties>().transWeight0132 = transWeight0132;
  }
  if (act) {
    odsState.getOrAddProperties<Properties>().act = act;
  }
  if (poolSize) {
    odsState.getOrAddProperties<Properties>().poolSize = poolSize;
  }
  if (poolStride) {
    odsState.getOrAddProperties<Properties>().poolStride = poolStride;
  }
  if (poolPadding) {
    odsState.getOrAddProperties<Properties>().poolPadding = poolPadding;
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TileConvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::llvm::APFloat scale, uint64_t stride, uint64_t inputDilation, uint64_t kernelDilation, uint64_t padding, bool wrot180, bool transOutput1203, bool transInput3120, bool transWeight1203, bool transWeight0132, uint64_t act, uint64_t poolSize, uint64_t poolStride, uint64_t poolPadding) {
  odsState.addOperands(input);
  odsState.addOperands(weights);
  odsState.addOperands(bias);
  odsState.addOperands(output);
  odsState.addOperands(outRowDim);
  odsState.addOperands(outColDim);
  odsState.addOperands(kernelDim);
  odsState.getOrAddProperties<Properties>().scale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), scale);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), stride);
  odsState.getOrAddProperties<Properties>().inputDilation = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), inputDilation);
  odsState.getOrAddProperties<Properties>().kernelDilation = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), kernelDilation);
  odsState.getOrAddProperties<Properties>().padding = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), padding);
  odsState.getOrAddProperties<Properties>().wrot180 = odsBuilder.getBoolAttr(wrot180);
  odsState.getOrAddProperties<Properties>().transOutput1203 = odsBuilder.getBoolAttr(transOutput1203);
  odsState.getOrAddProperties<Properties>().transInput3120 = odsBuilder.getBoolAttr(transInput3120);
  odsState.getOrAddProperties<Properties>().transWeight1203 = odsBuilder.getBoolAttr(transWeight1203);
  odsState.getOrAddProperties<Properties>().transWeight0132 = odsBuilder.getBoolAttr(transWeight0132);
  odsState.getOrAddProperties<Properties>().act = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), act);
  odsState.getOrAddProperties<Properties>().poolSize = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), poolSize);
  odsState.getOrAddProperties<Properties>().poolStride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), poolStride);
  odsState.getOrAddProperties<Properties>().poolPadding = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), poolPadding);
}

void TileConvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::llvm::APFloat scale, uint64_t stride, uint64_t inputDilation, uint64_t kernelDilation, uint64_t padding, bool wrot180, bool transOutput1203, bool transInput3120, bool transWeight1203, bool transWeight0132, uint64_t act, uint64_t poolSize, uint64_t poolStride, uint64_t poolPadding) {
  odsState.addOperands(input);
  odsState.addOperands(weights);
  odsState.addOperands(bias);
  odsState.addOperands(output);
  odsState.addOperands(outRowDim);
  odsState.addOperands(outColDim);
  odsState.addOperands(kernelDim);
  odsState.getOrAddProperties<Properties>().scale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), scale);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), stride);
  odsState.getOrAddProperties<Properties>().inputDilation = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), inputDilation);
  odsState.getOrAddProperties<Properties>().kernelDilation = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), kernelDilation);
  odsState.getOrAddProperties<Properties>().padding = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), padding);
  odsState.getOrAddProperties<Properties>().wrot180 = odsBuilder.getBoolAttr(wrot180);
  odsState.getOrAddProperties<Properties>().transOutput1203 = odsBuilder.getBoolAttr(transOutput1203);
  odsState.getOrAddProperties<Properties>().transInput3120 = odsBuilder.getBoolAttr(transInput3120);
  odsState.getOrAddProperties<Properties>().transWeight1203 = odsBuilder.getBoolAttr(transWeight1203);
  odsState.getOrAddProperties<Properties>().transWeight0132 = odsBuilder.getBoolAttr(transWeight0132);
  odsState.getOrAddProperties<Properties>().act = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), act);
  odsState.getOrAddProperties<Properties>().poolSize = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), poolSize);
  odsState.getOrAddProperties<Properties>().poolStride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), poolStride);
  odsState.getOrAddProperties<Properties>().poolPadding = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), poolPadding);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TileConvOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 7u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TileConvOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void TileConvOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.scale)
    properties.scale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0);
  if (!properties.stride)
    properties.stride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1);
  if (!properties.inputDilation)
    properties.inputDilation = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1);
  if (!properties.kernelDilation)
    properties.kernelDilation = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1);
  if (!properties.padding)
    properties.padding = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.wrot180)
    properties.wrot180 = odsBuilder.getBoolAttr(false);
  if (!properties.transOutput1203)
    properties.transOutput1203 = odsBuilder.getBoolAttr(false);
  if (!properties.transInput3120)
    properties.transInput3120 = odsBuilder.getBoolAttr(false);
  if (!properties.transWeight1203)
    properties.transWeight1203 = odsBuilder.getBoolAttr(false);
  if (!properties.transWeight0132)
    properties.transWeight0132 = odsBuilder.getBoolAttr(false);
  if (!properties.act)
    properties.act = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.poolSize)
    properties.poolSize = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.poolStride)
    properties.poolStride = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.poolPadding)
    properties.poolPadding = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
}

::llvm::LogicalResult TileConvOp::verifyInvariantsImpl() {
  auto tblgen_act = getProperties().act; (void)tblgen_act;
  auto tblgen_inputDilation = getProperties().inputDilation; (void)tblgen_inputDilation;
  auto tblgen_kernelDilation = getProperties().kernelDilation; (void)tblgen_kernelDilation;
  auto tblgen_padding = getProperties().padding; (void)tblgen_padding;
  auto tblgen_poolPadding = getProperties().poolPadding; (void)tblgen_poolPadding;
  auto tblgen_poolSize = getProperties().poolSize; (void)tblgen_poolSize;
  auto tblgen_poolStride = getProperties().poolStride; (void)tblgen_poolStride;
  auto tblgen_scale = getProperties().scale; (void)tblgen_scale;
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  auto tblgen_transInput3120 = getProperties().transInput3120; (void)tblgen_transInput3120;
  auto tblgen_transOutput1203 = getProperties().transOutput1203; (void)tblgen_transOutput1203;
  auto tblgen_transWeight0132 = getProperties().transWeight0132; (void)tblgen_transWeight0132;
  auto tblgen_transWeight1203 = getProperties().transWeight1203; (void)tblgen_transWeight1203;
  auto tblgen_wrot180 = getProperties().wrot180; (void)tblgen_wrot180;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(*this, tblgen_scale, "scale")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_inputDilation, "inputDilation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_kernelDilation, "kernelDilation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_padding, "padding")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_wrot180, "wrot180")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_transOutput1203, "transOutput1203")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_transInput3120, "transInput3120")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_transWeight1203, "transWeight1203")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_transWeight0132, "transWeight0132")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_act, "act")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_poolSize, "poolSize")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_poolStride, "poolStride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_poolPadding, "poolPadding")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TileConvOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TileConvOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand weightsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> weightsOperands(&weightsRawOperand, 1);  ::llvm::SMLoc weightsOperandsLoc;
  (void)weightsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand biasRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> biasOperands(&biasRawOperand, 1);  ::llvm::SMLoc biasOperandsLoc;
  (void)biasOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand outputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> outputOperands(&outputRawOperand, 1);  ::llvm::SMLoc outputOperandsLoc;
  (void)outputOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand outRowDimRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> outRowDimOperands(&outRowDimRawOperand, 1);  ::llvm::SMLoc outRowDimOperandsLoc;
  (void)outRowDimOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand outColDimRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> outColDimOperands(&outColDimRawOperand, 1);  ::llvm::SMLoc outColDimOperandsLoc;
  (void)outColDimOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand kernelDimRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> kernelDimOperands(&kernelDimRawOperand, 1);  ::llvm::SMLoc kernelDimOperandsLoc;
  (void)kernelDimOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);
  ::mlir::Type weightsRawType{};
  ::llvm::ArrayRef<::mlir::Type> weightsTypes(&weightsRawType, 1);
  ::mlir::Type biasRawType{};
  ::llvm::ArrayRef<::mlir::Type> biasTypes(&biasRawType, 1);
  ::mlir::Type outputRawType{};
  ::llvm::ArrayRef<::mlir::Type> outputTypes(&outputRawType, 1);
  ::mlir::Type outRowDimRawType{};
  ::llvm::ArrayRef<::mlir::Type> outRowDimTypes(&outRowDimRawType, 1);
  ::mlir::Type outColDimRawType{};
  ::llvm::ArrayRef<::mlir::Type> outColDimTypes(&outColDimRawType, 1);
  ::mlir::Type kernelDimRawType{};
  ::llvm::ArrayRef<::mlir::Type> kernelDimTypes(&kernelDimRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();

  weightsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(weightsRawOperand))
    return ::mlir::failure();

  biasOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(biasRawOperand))
    return ::mlir::failure();

  outputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(outputRawOperand))
    return ::mlir::failure();

  outRowDimOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(outRowDimRawOperand))
    return ::mlir::failure();

  outColDimOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(outColDimRawOperand))
    return ::mlir::failure();

  kernelDimOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(kernelDimRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    weightsRawType = type;
  }

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    biasRawType = type;
  }

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    outputRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    outRowDimRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    outColDimRawType = type;
  }

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    kernelDimRawType = type;
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(weightsOperands, weightsTypes, weightsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(biasOperands, biasTypes, biasOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(outputOperands, outputTypes, outputOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(outRowDimOperands, outRowDimTypes, outRowDimOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(outColDimOperands, outColDimTypes, outColDimOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(kernelDimOperands, kernelDimTypes, kernelDimOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TileConvOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  _odsPrinter << ' ';
  _odsPrinter << getWeights();
  _odsPrinter << ' ';
  _odsPrinter << getBias();
  _odsPrinter << ' ';
  _odsPrinter << getOutput();
  _odsPrinter << ' ';
  _odsPrinter << getOutRowDim();
  _odsPrinter << ' ';
  _odsPrinter << getOutColDim();
  _odsPrinter << ' ';
  _odsPrinter << getKernelDim();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getScaleAttr();
     if(attr && (attr == odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0)))
       elidedAttrs.push_back("scale");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getStrideAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1)))
       elidedAttrs.push_back("stride");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getInputDilationAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1)))
       elidedAttrs.push_back("inputDilation");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getKernelDilationAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1)))
       elidedAttrs.push_back("kernelDilation");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getPaddingAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("padding");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getWrot180Attr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("wrot180");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getTransOutput1203Attr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("transOutput1203");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getTransInput3120Attr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("transInput3120");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getTransWeight1203Attr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("transWeight1203");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getTransWeight0132Attr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("transWeight0132");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getActAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("act");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getPoolSizeAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("poolSize");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getPoolStrideAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("poolStride");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getPoolPaddingAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("poolPadding");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getWeights().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getBias().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getOutput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getOutRowDim().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getOutColDim().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getKernelDim().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::TileConvOp)

namespace buddy {
namespace gemmini {

//===----------------------------------------------------------------------===//
// ::buddy::gemmini::TileMatMulOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TileMatMulOpGenericAdaptorBase::TileMatMulOpGenericAdaptorBase(TileMatMulOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::FloatAttr TileMatMulOpGenericAdaptorBase::getAScaleFactorAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().aScaleFactor);
  return attr;
}

::llvm::APFloat TileMatMulOpGenericAdaptorBase::getAScaleFactor() {
  auto attr = getAScaleFactorAttr();
  return attr.getValue();
}

::mlir::FloatAttr TileMatMulOpGenericAdaptorBase::getBScaleFactorAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().bScaleFactor);
  return attr;
}

::llvm::APFloat TileMatMulOpGenericAdaptorBase::getBScaleFactor() {
  auto attr = getBScaleFactorAttr();
  return attr.getValue();
}

::mlir::FloatAttr TileMatMulOpGenericAdaptorBase::getDScaleFactorAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().dScaleFactor);
  return attr;
}

::llvm::APFloat TileMatMulOpGenericAdaptorBase::getDScaleFactor() {
  auto attr = getDScaleFactorAttr();
  return attr.getValue();
}

::mlir::IntegerAttr TileMatMulOpGenericAdaptorBase::getActAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().act);
  return attr;
}

uint64_t TileMatMulOpGenericAdaptorBase::getAct() {
  auto attr = getActAttr();
  return attr.getValue().getZExtValue();
}

::mlir::FloatAttr TileMatMulOpGenericAdaptorBase::getAccScaleAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().accScale);
  return attr;
}

::llvm::APFloat TileMatMulOpGenericAdaptorBase::getAccScale() {
  auto attr = getAccScaleAttr();
  return attr.getValue();
}

::mlir::FloatAttr TileMatMulOpGenericAdaptorBase::getBertScaleAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().bertScale);
  return attr;
}

::llvm::APFloat TileMatMulOpGenericAdaptorBase::getBertScale() {
  auto attr = getBertScaleAttr();
  return attr.getValue();
}

::mlir::BoolAttr TileMatMulOpGenericAdaptorBase::getRepeatingBiasAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().repeatingBias);
  return attr;
}

bool TileMatMulOpGenericAdaptorBase::getRepeatingBias() {
  auto attr = getRepeatingBiasAttr();
  return attr.getValue();
}

::mlir::BoolAttr TileMatMulOpGenericAdaptorBase::getATransposeAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().aTranspose);
  return attr;
}

bool TileMatMulOpGenericAdaptorBase::getATranspose() {
  auto attr = getATransposeAttr();
  return attr.getValue();
}

::mlir::BoolAttr TileMatMulOpGenericAdaptorBase::getBTransposeAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().bTranspose);
  return attr;
}

bool TileMatMulOpGenericAdaptorBase::getBTranspose() {
  auto attr = getBTransposeAttr();
  return attr.getValue();
}

::mlir::BoolAttr TileMatMulOpGenericAdaptorBase::getFullCAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().fullC);
  return attr;
}

bool TileMatMulOpGenericAdaptorBase::getFullC() {
  auto attr = getFullCAttr();
  return attr.getValue();
}

::mlir::BoolAttr TileMatMulOpGenericAdaptorBase::getLowDAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().lowD);
  return attr;
}

bool TileMatMulOpGenericAdaptorBase::getLowD() {
  auto attr = getLowDAttr();
  return attr.getValue();
}

::mlir::IntegerAttr TileMatMulOpGenericAdaptorBase::getWeightAAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().weightA);
  return attr;
}

uint64_t TileMatMulOpGenericAdaptorBase::getWeightA() {
  auto attr = getWeightAAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr TileMatMulOpGenericAdaptorBase::getDataflowAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().dataflow);
  return attr;
}

uint64_t TileMatMulOpGenericAdaptorBase::getDataflow() {
  auto attr = getDataflowAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
TileMatMulOpAdaptor::TileMatMulOpAdaptor(TileMatMulOp op) : TileMatMulOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TileMatMulOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_aScaleFactor = getProperties().aScaleFactor; (void)tblgen_aScaleFactor;
  auto tblgen_aTranspose = getProperties().aTranspose; (void)tblgen_aTranspose;
  auto tblgen_accScale = getProperties().accScale; (void)tblgen_accScale;
  auto tblgen_act = getProperties().act; (void)tblgen_act;
  auto tblgen_bScaleFactor = getProperties().bScaleFactor; (void)tblgen_bScaleFactor;
  auto tblgen_bTranspose = getProperties().bTranspose; (void)tblgen_bTranspose;
  auto tblgen_bertScale = getProperties().bertScale; (void)tblgen_bertScale;
  auto tblgen_dScaleFactor = getProperties().dScaleFactor; (void)tblgen_dScaleFactor;
  auto tblgen_dataflow = getProperties().dataflow; (void)tblgen_dataflow;
  auto tblgen_fullC = getProperties().fullC; (void)tblgen_fullC;
  auto tblgen_lowD = getProperties().lowD; (void)tblgen_lowD;
  auto tblgen_repeatingBias = getProperties().repeatingBias; (void)tblgen_repeatingBias;
  auto tblgen_weightA = getProperties().weightA; (void)tblgen_weightA;

  if (tblgen_aScaleFactor && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_aScaleFactor))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_aScaleFactor).getType().isF32()))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'aScaleFactor' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_bScaleFactor && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_bScaleFactor))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_bScaleFactor).getType().isF32()))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'bScaleFactor' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_dScaleFactor && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_dScaleFactor))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_dScaleFactor).getType().isF32()))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'dScaleFactor' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_act && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_act))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_act).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'act' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_accScale && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_accScale))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_accScale).getType().isF32()))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'accScale' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_bertScale && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_bertScale))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_bertScale).getType().isF32()))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'bertScale' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_repeatingBias && !((::llvm::isa<::mlir::BoolAttr>(tblgen_repeatingBias))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'repeatingBias' failed to satisfy constraint: bool attribute");

  if (tblgen_aTranspose && !((::llvm::isa<::mlir::BoolAttr>(tblgen_aTranspose))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'aTranspose' failed to satisfy constraint: bool attribute");

  if (tblgen_bTranspose && !((::llvm::isa<::mlir::BoolAttr>(tblgen_bTranspose))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'bTranspose' failed to satisfy constraint: bool attribute");

  if (tblgen_fullC && !((::llvm::isa<::mlir::BoolAttr>(tblgen_fullC))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'fullC' failed to satisfy constraint: bool attribute");

  if (tblgen_lowD && !((::llvm::isa<::mlir::BoolAttr>(tblgen_lowD))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'lowD' failed to satisfy constraint: bool attribute");

  if (tblgen_weightA && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_weightA))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_weightA).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'weightA' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_dataflow && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_dataflow))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_dataflow).getType().isSignlessInteger(64)))))
    return emitError(loc, "'gemmini.tile_matmul' op ""attribute 'dataflow' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult TileMatMulOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.aScaleFactor;
       auto attr = dict.get("aScaleFactor");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `aScaleFactor` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.aTranspose;
       auto attr = dict.get("aTranspose");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `aTranspose` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.accScale;
       auto attr = dict.get("accScale");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `accScale` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.act;
       auto attr = dict.get("act");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `act` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.bScaleFactor;
       auto attr = dict.get("bScaleFactor");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `bScaleFactor` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.bTranspose;
       auto attr = dict.get("bTranspose");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `bTranspose` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.bertScale;
       auto attr = dict.get("bertScale");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `bertScale` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.dScaleFactor;
       auto attr = dict.get("dScaleFactor");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `dScaleFactor` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.dataflow;
       auto attr = dict.get("dataflow");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `dataflow` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.fullC;
       auto attr = dict.get("fullC");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `fullC` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.lowD;
       auto attr = dict.get("lowD");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `lowD` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.repeatingBias;
       auto attr = dict.get("repeatingBias");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `repeatingBias` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.weightA;
       auto attr = dict.get("weightA");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `weightA` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TileMatMulOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.aScaleFactor;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("aScaleFactor",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.aTranspose;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("aTranspose",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.accScale;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("accScale",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.act;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("act",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.bScaleFactor;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("bScaleFactor",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.bTranspose;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("bTranspose",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.bertScale;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("bertScale",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.dScaleFactor;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("dScaleFactor",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.dataflow;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("dataflow",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.fullC;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("fullC",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.lowD;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("lowD",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.repeatingBias;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("repeatingBias",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.weightA;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("weightA",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TileMatMulOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.aScaleFactor.getAsOpaquePointer()), 
    llvm::hash_value(prop.aTranspose.getAsOpaquePointer()), 
    llvm::hash_value(prop.accScale.getAsOpaquePointer()), 
    llvm::hash_value(prop.act.getAsOpaquePointer()), 
    llvm::hash_value(prop.bScaleFactor.getAsOpaquePointer()), 
    llvm::hash_value(prop.bTranspose.getAsOpaquePointer()), 
    llvm::hash_value(prop.bertScale.getAsOpaquePointer()), 
    llvm::hash_value(prop.dScaleFactor.getAsOpaquePointer()), 
    llvm::hash_value(prop.dataflow.getAsOpaquePointer()), 
    llvm::hash_value(prop.fullC.getAsOpaquePointer()), 
    llvm::hash_value(prop.lowD.getAsOpaquePointer()), 
    llvm::hash_value(prop.repeatingBias.getAsOpaquePointer()), 
    llvm::hash_value(prop.weightA.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TileMatMulOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "aScaleFactor")
      return prop.aScaleFactor;

    if (name == "aTranspose")
      return prop.aTranspose;

    if (name == "accScale")
      return prop.accScale;

    if (name == "act")
      return prop.act;

    if (name == "bScaleFactor")
      return prop.bScaleFactor;

    if (name == "bTranspose")
      return prop.bTranspose;

    if (name == "bertScale")
      return prop.bertScale;

    if (name == "dScaleFactor")
      return prop.dScaleFactor;

    if (name == "dataflow")
      return prop.dataflow;

    if (name == "fullC")
      return prop.fullC;

    if (name == "lowD")
      return prop.lowD;

    if (name == "repeatingBias")
      return prop.repeatingBias;

    if (name == "weightA")
      return prop.weightA;
  return std::nullopt;
}

void TileMatMulOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "aScaleFactor") {
       prop.aScaleFactor = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.aScaleFactor)>>(value);
       return;
    }

    if (name == "aTranspose") {
       prop.aTranspose = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.aTranspose)>>(value);
       return;
    }

    if (name == "accScale") {
       prop.accScale = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.accScale)>>(value);
       return;
    }

    if (name == "act") {
       prop.act = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.act)>>(value);
       return;
    }

    if (name == "bScaleFactor") {
       prop.bScaleFactor = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.bScaleFactor)>>(value);
       return;
    }

    if (name == "bTranspose") {
       prop.bTranspose = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.bTranspose)>>(value);
       return;
    }

    if (name == "bertScale") {
       prop.bertScale = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.bertScale)>>(value);
       return;
    }

    if (name == "dScaleFactor") {
       prop.dScaleFactor = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.dScaleFactor)>>(value);
       return;
    }

    if (name == "dataflow") {
       prop.dataflow = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.dataflow)>>(value);
       return;
    }

    if (name == "fullC") {
       prop.fullC = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.fullC)>>(value);
       return;
    }

    if (name == "lowD") {
       prop.lowD = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.lowD)>>(value);
       return;
    }

    if (name == "repeatingBias") {
       prop.repeatingBias = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.repeatingBias)>>(value);
       return;
    }

    if (name == "weightA") {
       prop.weightA = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.weightA)>>(value);
       return;
    }
}

void TileMatMulOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.aScaleFactor) attrs.append("aScaleFactor", prop.aScaleFactor);

    if (prop.aTranspose) attrs.append("aTranspose", prop.aTranspose);

    if (prop.accScale) attrs.append("accScale", prop.accScale);

    if (prop.act) attrs.append("act", prop.act);

    if (prop.bScaleFactor) attrs.append("bScaleFactor", prop.bScaleFactor);

    if (prop.bTranspose) attrs.append("bTranspose", prop.bTranspose);

    if (prop.bertScale) attrs.append("bertScale", prop.bertScale);

    if (prop.dScaleFactor) attrs.append("dScaleFactor", prop.dScaleFactor);

    if (prop.dataflow) attrs.append("dataflow", prop.dataflow);

    if (prop.fullC) attrs.append("fullC", prop.fullC);

    if (prop.lowD) attrs.append("lowD", prop.lowD);

    if (prop.repeatingBias) attrs.append("repeatingBias", prop.repeatingBias);

    if (prop.weightA) attrs.append("weightA", prop.weightA);
}

::llvm::LogicalResult TileMatMulOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getAScaleFactorAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(attr, "aScaleFactor", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getATransposeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "aTranspose", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getAccScaleAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(attr, "accScale", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getActAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "act", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBScaleFactorAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(attr, "bScaleFactor", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBTransposeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "bTranspose", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBertScaleAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(attr, "bertScale", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getDScaleFactorAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(attr, "dScaleFactor", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getDataflowAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "dataflow", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getFullCAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "fullC", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getLowDAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "lowD", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getRepeatingBiasAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(attr, "repeatingBias", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getWeightAAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(attr, "weightA", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TileMatMulOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.aScaleFactor)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.aTranspose)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.accScale)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.act)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.bScaleFactor)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.bTranspose)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.bertScale)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.dScaleFactor)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.dataflow)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.fullC)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.lowD)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.repeatingBias)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.weightA)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TileMatMulOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.aScaleFactor);

  writer.writeOptionalAttribute(prop.aTranspose);

  writer.writeOptionalAttribute(prop.accScale);

  writer.writeOptionalAttribute(prop.act);

  writer.writeOptionalAttribute(prop.bScaleFactor);

  writer.writeOptionalAttribute(prop.bTranspose);

  writer.writeOptionalAttribute(prop.bertScale);

  writer.writeOptionalAttribute(prop.dScaleFactor);

  writer.writeOptionalAttribute(prop.dataflow);

  writer.writeOptionalAttribute(prop.fullC);

  writer.writeOptionalAttribute(prop.lowD);

  writer.writeOptionalAttribute(prop.repeatingBias);

  writer.writeOptionalAttribute(prop.weightA);
}

::llvm::APFloat TileMatMulOp::getAScaleFactor() {
  auto attr = getAScaleFactorAttr();
  return attr.getValue();
}

::llvm::APFloat TileMatMulOp::getBScaleFactor() {
  auto attr = getBScaleFactorAttr();
  return attr.getValue();
}

::llvm::APFloat TileMatMulOp::getDScaleFactor() {
  auto attr = getDScaleFactorAttr();
  return attr.getValue();
}

uint64_t TileMatMulOp::getAct() {
  auto attr = getActAttr();
  return attr.getValue().getZExtValue();
}

::llvm::APFloat TileMatMulOp::getAccScale() {
  auto attr = getAccScaleAttr();
  return attr.getValue();
}

::llvm::APFloat TileMatMulOp::getBertScale() {
  auto attr = getBertScaleAttr();
  return attr.getValue();
}

bool TileMatMulOp::getRepeatingBias() {
  auto attr = getRepeatingBiasAttr();
  return attr.getValue();
}

bool TileMatMulOp::getATranspose() {
  auto attr = getATransposeAttr();
  return attr.getValue();
}

bool TileMatMulOp::getBTranspose() {
  auto attr = getBTransposeAttr();
  return attr.getValue();
}

bool TileMatMulOp::getFullC() {
  auto attr = getFullCAttr();
  return attr.getValue();
}

bool TileMatMulOp::getLowD() {
  auto attr = getLowDAttr();
  return attr.getValue();
}

uint64_t TileMatMulOp::getWeightA() {
  auto attr = getWeightAAttr();
  return attr.getValue().getZExtValue();
}

uint64_t TileMatMulOp::getDataflow() {
  auto attr = getDataflowAttr();
  return attr.getValue().getZExtValue();
}

void TileMatMulOp::setAScaleFactor(::llvm::APFloat attrValue) {
  getProperties().aScaleFactor = ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue);
}

void TileMatMulOp::setBScaleFactor(::llvm::APFloat attrValue) {
  getProperties().bScaleFactor = ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue);
}

void TileMatMulOp::setDScaleFactor(::llvm::APFloat attrValue) {
  getProperties().dScaleFactor = ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue);
}

void TileMatMulOp::setAct(uint64_t attrValue) {
  getProperties().act = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TileMatMulOp::setAccScale(::llvm::APFloat attrValue) {
  getProperties().accScale = ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue);
}

void TileMatMulOp::setBertScale(::llvm::APFloat attrValue) {
  getProperties().bertScale = ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue);
}

void TileMatMulOp::setRepeatingBias(bool attrValue) {
  getProperties().repeatingBias = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void TileMatMulOp::setATranspose(bool attrValue) {
  getProperties().aTranspose = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void TileMatMulOp::setBTranspose(bool attrValue) {
  getProperties().bTranspose = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void TileMatMulOp::setFullC(bool attrValue) {
  getProperties().fullC = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void TileMatMulOp::setLowD(bool attrValue) {
  getProperties().lowD = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void TileMatMulOp::setWeightA(uint64_t attrValue) {
  getProperties().weightA = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TileMatMulOp::setDataflow(uint64_t attrValue) {
  getProperties().dataflow = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void TileMatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::mlir::FloatAttr aScaleFactor, ::mlir::FloatAttr bScaleFactor, ::mlir::FloatAttr dScaleFactor, ::mlir::IntegerAttr act, ::mlir::FloatAttr accScale, ::mlir::FloatAttr bertScale, ::mlir::BoolAttr repeatingBias, ::mlir::BoolAttr aTranspose, ::mlir::BoolAttr bTranspose, ::mlir::BoolAttr fullC, ::mlir::BoolAttr lowD, ::mlir::IntegerAttr weightA, ::mlir::IntegerAttr dataflow) {
  odsState.addOperands(aArray);
  odsState.addOperands(bArray);
  odsState.addOperands(cArray);
  odsState.addOperands(dArray);
  if (aScaleFactor) {
    odsState.getOrAddProperties<Properties>().aScaleFactor = aScaleFactor;
  }
  if (bScaleFactor) {
    odsState.getOrAddProperties<Properties>().bScaleFactor = bScaleFactor;
  }
  if (dScaleFactor) {
    odsState.getOrAddProperties<Properties>().dScaleFactor = dScaleFactor;
  }
  if (act) {
    odsState.getOrAddProperties<Properties>().act = act;
  }
  if (accScale) {
    odsState.getOrAddProperties<Properties>().accScale = accScale;
  }
  if (bertScale) {
    odsState.getOrAddProperties<Properties>().bertScale = bertScale;
  }
  if (repeatingBias) {
    odsState.getOrAddProperties<Properties>().repeatingBias = repeatingBias;
  }
  if (aTranspose) {
    odsState.getOrAddProperties<Properties>().aTranspose = aTranspose;
  }
  if (bTranspose) {
    odsState.getOrAddProperties<Properties>().bTranspose = bTranspose;
  }
  if (fullC) {
    odsState.getOrAddProperties<Properties>().fullC = fullC;
  }
  if (lowD) {
    odsState.getOrAddProperties<Properties>().lowD = lowD;
  }
  if (weightA) {
    odsState.getOrAddProperties<Properties>().weightA = weightA;
  }
  if (dataflow) {
    odsState.getOrAddProperties<Properties>().dataflow = dataflow;
  }
}

void TileMatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::mlir::FloatAttr aScaleFactor, ::mlir::FloatAttr bScaleFactor, ::mlir::FloatAttr dScaleFactor, ::mlir::IntegerAttr act, ::mlir::FloatAttr accScale, ::mlir::FloatAttr bertScale, ::mlir::BoolAttr repeatingBias, ::mlir::BoolAttr aTranspose, ::mlir::BoolAttr bTranspose, ::mlir::BoolAttr fullC, ::mlir::BoolAttr lowD, ::mlir::IntegerAttr weightA, ::mlir::IntegerAttr dataflow) {
  odsState.addOperands(aArray);
  odsState.addOperands(bArray);
  odsState.addOperands(cArray);
  odsState.addOperands(dArray);
  if (aScaleFactor) {
    odsState.getOrAddProperties<Properties>().aScaleFactor = aScaleFactor;
  }
  if (bScaleFactor) {
    odsState.getOrAddProperties<Properties>().bScaleFactor = bScaleFactor;
  }
  if (dScaleFactor) {
    odsState.getOrAddProperties<Properties>().dScaleFactor = dScaleFactor;
  }
  if (act) {
    odsState.getOrAddProperties<Properties>().act = act;
  }
  if (accScale) {
    odsState.getOrAddProperties<Properties>().accScale = accScale;
  }
  if (bertScale) {
    odsState.getOrAddProperties<Properties>().bertScale = bertScale;
  }
  if (repeatingBias) {
    odsState.getOrAddProperties<Properties>().repeatingBias = repeatingBias;
  }
  if (aTranspose) {
    odsState.getOrAddProperties<Properties>().aTranspose = aTranspose;
  }
  if (bTranspose) {
    odsState.getOrAddProperties<Properties>().bTranspose = bTranspose;
  }
  if (fullC) {
    odsState.getOrAddProperties<Properties>().fullC = fullC;
  }
  if (lowD) {
    odsState.getOrAddProperties<Properties>().lowD = lowD;
  }
  if (weightA) {
    odsState.getOrAddProperties<Properties>().weightA = weightA;
  }
  if (dataflow) {
    odsState.getOrAddProperties<Properties>().dataflow = dataflow;
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TileMatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::llvm::APFloat aScaleFactor, ::llvm::APFloat bScaleFactor, ::llvm::APFloat dScaleFactor, uint64_t act, ::llvm::APFloat accScale, ::llvm::APFloat bertScale, bool repeatingBias, bool aTranspose, bool bTranspose, bool fullC, bool lowD, uint64_t weightA, uint64_t dataflow) {
  odsState.addOperands(aArray);
  odsState.addOperands(bArray);
  odsState.addOperands(cArray);
  odsState.addOperands(dArray);
  odsState.getOrAddProperties<Properties>().aScaleFactor = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), aScaleFactor);
  odsState.getOrAddProperties<Properties>().bScaleFactor = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), bScaleFactor);
  odsState.getOrAddProperties<Properties>().dScaleFactor = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), dScaleFactor);
  odsState.getOrAddProperties<Properties>().act = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), act);
  odsState.getOrAddProperties<Properties>().accScale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), accScale);
  odsState.getOrAddProperties<Properties>().bertScale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), bertScale);
  odsState.getOrAddProperties<Properties>().repeatingBias = odsBuilder.getBoolAttr(repeatingBias);
  odsState.getOrAddProperties<Properties>().aTranspose = odsBuilder.getBoolAttr(aTranspose);
  odsState.getOrAddProperties<Properties>().bTranspose = odsBuilder.getBoolAttr(bTranspose);
  odsState.getOrAddProperties<Properties>().fullC = odsBuilder.getBoolAttr(fullC);
  odsState.getOrAddProperties<Properties>().lowD = odsBuilder.getBoolAttr(lowD);
  odsState.getOrAddProperties<Properties>().weightA = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), weightA);
  odsState.getOrAddProperties<Properties>().dataflow = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), dataflow);
}

void TileMatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::llvm::APFloat aScaleFactor, ::llvm::APFloat bScaleFactor, ::llvm::APFloat dScaleFactor, uint64_t act, ::llvm::APFloat accScale, ::llvm::APFloat bertScale, bool repeatingBias, bool aTranspose, bool bTranspose, bool fullC, bool lowD, uint64_t weightA, uint64_t dataflow) {
  odsState.addOperands(aArray);
  odsState.addOperands(bArray);
  odsState.addOperands(cArray);
  odsState.addOperands(dArray);
  odsState.getOrAddProperties<Properties>().aScaleFactor = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), aScaleFactor);
  odsState.getOrAddProperties<Properties>().bScaleFactor = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), bScaleFactor);
  odsState.getOrAddProperties<Properties>().dScaleFactor = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), dScaleFactor);
  odsState.getOrAddProperties<Properties>().act = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), act);
  odsState.getOrAddProperties<Properties>().accScale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), accScale);
  odsState.getOrAddProperties<Properties>().bertScale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), bertScale);
  odsState.getOrAddProperties<Properties>().repeatingBias = odsBuilder.getBoolAttr(repeatingBias);
  odsState.getOrAddProperties<Properties>().aTranspose = odsBuilder.getBoolAttr(aTranspose);
  odsState.getOrAddProperties<Properties>().bTranspose = odsBuilder.getBoolAttr(bTranspose);
  odsState.getOrAddProperties<Properties>().fullC = odsBuilder.getBoolAttr(fullC);
  odsState.getOrAddProperties<Properties>().lowD = odsBuilder.getBoolAttr(lowD);
  odsState.getOrAddProperties<Properties>().weightA = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), weightA);
  odsState.getOrAddProperties<Properties>().dataflow = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), dataflow);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TileMatMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TileMatMulOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void TileMatMulOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.aScaleFactor)
    properties.aScaleFactor = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0);
  if (!properties.bScaleFactor)
    properties.bScaleFactor = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0);
  if (!properties.dScaleFactor)
    properties.dScaleFactor = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0);
  if (!properties.act)
    properties.act = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.accScale)
    properties.accScale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0);
  if (!properties.bertScale)
    properties.bertScale = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 0.0);
  if (!properties.repeatingBias)
    properties.repeatingBias = odsBuilder.getBoolAttr(false);
  if (!properties.aTranspose)
    properties.aTranspose = odsBuilder.getBoolAttr(false);
  if (!properties.bTranspose)
    properties.bTranspose = odsBuilder.getBoolAttr(false);
  if (!properties.fullC)
    properties.fullC = odsBuilder.getBoolAttr(false);
  if (!properties.lowD)
    properties.lowD = odsBuilder.getBoolAttr(false);
  if (!properties.weightA)
    properties.weightA = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0);
  if (!properties.dataflow)
    properties.dataflow = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1);
}

::llvm::LogicalResult TileMatMulOp::verifyInvariantsImpl() {
  auto tblgen_aScaleFactor = getProperties().aScaleFactor; (void)tblgen_aScaleFactor;
  auto tblgen_aTranspose = getProperties().aTranspose; (void)tblgen_aTranspose;
  auto tblgen_accScale = getProperties().accScale; (void)tblgen_accScale;
  auto tblgen_act = getProperties().act; (void)tblgen_act;
  auto tblgen_bScaleFactor = getProperties().bScaleFactor; (void)tblgen_bScaleFactor;
  auto tblgen_bTranspose = getProperties().bTranspose; (void)tblgen_bTranspose;
  auto tblgen_bertScale = getProperties().bertScale; (void)tblgen_bertScale;
  auto tblgen_dScaleFactor = getProperties().dScaleFactor; (void)tblgen_dScaleFactor;
  auto tblgen_dataflow = getProperties().dataflow; (void)tblgen_dataflow;
  auto tblgen_fullC = getProperties().fullC; (void)tblgen_fullC;
  auto tblgen_lowD = getProperties().lowD; (void)tblgen_lowD;
  auto tblgen_repeatingBias = getProperties().repeatingBias; (void)tblgen_repeatingBias;
  auto tblgen_weightA = getProperties().weightA; (void)tblgen_weightA;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(*this, tblgen_aScaleFactor, "aScaleFactor")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(*this, tblgen_bScaleFactor, "bScaleFactor")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(*this, tblgen_dScaleFactor, "dScaleFactor")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_act, "act")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(*this, tblgen_accScale, "accScale")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini2(*this, tblgen_bertScale, "bertScale")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_repeatingBias, "repeatingBias")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_aTranspose, "aTranspose")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_bTranspose, "bTranspose")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_fullC, "fullC")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini3(*this, tblgen_lowD, "lowD")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_weightA, "weightA")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Gemmini1(*this, tblgen_dataflow, "dataflow")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Gemmini3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TileMatMulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TileMatMulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand aArrayRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aArrayOperands(&aArrayRawOperand, 1);  ::llvm::SMLoc aArrayOperandsLoc;
  (void)aArrayOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bArrayRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bArrayOperands(&bArrayRawOperand, 1);  ::llvm::SMLoc bArrayOperandsLoc;
  (void)bArrayOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand cArrayRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> cArrayOperands(&cArrayRawOperand, 1);  ::llvm::SMLoc cArrayOperandsLoc;
  (void)cArrayOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand dArrayRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dArrayOperands(&dArrayRawOperand, 1);  ::llvm::SMLoc dArrayOperandsLoc;
  (void)dArrayOperandsLoc;
  ::mlir::Type aArrayRawType{};
  ::llvm::ArrayRef<::mlir::Type> aArrayTypes(&aArrayRawType, 1);
  ::mlir::Type bArrayRawType{};
  ::llvm::ArrayRef<::mlir::Type> bArrayTypes(&bArrayRawType, 1);
  ::mlir::Type cArrayRawType{};
  ::llvm::ArrayRef<::mlir::Type> cArrayTypes(&cArrayRawType, 1);
  ::mlir::Type dArrayRawType{};
  ::llvm::ArrayRef<::mlir::Type> dArrayTypes(&dArrayRawType, 1);

  aArrayOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aArrayRawOperand))
    return ::mlir::failure();

  bArrayOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bArrayRawOperand))
    return ::mlir::failure();

  cArrayOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(cArrayRawOperand))
    return ::mlir::failure();

  dArrayOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dArrayRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    aArrayRawType = type;
  }

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bArrayRawType = type;
  }

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    cArrayRawType = type;
  }

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    dArrayRawType = type;
  }
  if (parser.resolveOperands(aArrayOperands, aArrayTypes, aArrayOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bArrayOperands, bArrayTypes, bArrayOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(cArrayOperands, cArrayTypes, cArrayOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dArrayOperands, dArrayTypes, dArrayOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TileMatMulOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAArray();
  _odsPrinter << ' ';
  _odsPrinter << getBArray();
  _odsPrinter << ' ';
  _odsPrinter << getCArray();
  _odsPrinter << ' ';
  _odsPrinter << getDArray();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getAScaleFactorAttr();
     if(attr && (attr == odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0)))
       elidedAttrs.push_back("aScaleFactor");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getBScaleFactorAttr();
     if(attr && (attr == odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0)))
       elidedAttrs.push_back("bScaleFactor");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDScaleFactorAttr();
     if(attr && (attr == odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0)))
       elidedAttrs.push_back("dScaleFactor");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getActAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("act");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getAccScaleAttr();
     if(attr && (attr == odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 1.0)))
       elidedAttrs.push_back("accScale");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getBertScaleAttr();
     if(attr && (attr == odsBuilder.getFloatAttr(odsBuilder.getF32Type(), 0.0)))
       elidedAttrs.push_back("bertScale");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getRepeatingBiasAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("repeatingBias");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getATransposeAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("aTranspose");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getBTransposeAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("bTranspose");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getFullCAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("fullC");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getLowDAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("lowD");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getWeightAAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("weightA");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getDataflowAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1)))
       elidedAttrs.push_back("dataflow");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAArray().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getBArray().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getCArray().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  {
    auto type = getDArray().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace gemmini
} // namespace buddy
MLIR_DEFINE_EXPLICIT_TYPE_ID(::buddy::gemmini::TileMatMulOp)


#endif  // GET_OP_CLASSES

