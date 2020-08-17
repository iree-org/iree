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

#include "experimental/ModelBuilder/ModelBuilder.h"

#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::ops;
using namespace mlir::edsc::intrinsics;

thread_local MLIRContext mlir::ModelBuilder::ctx;

void ModelBuilder::registerAllDialects() {
  registerDialect<AffineDialect>();
  registerDialect<gpu::GPUDialect>();
  registerDialect<LLVM::LLVMDialect>();
  registerDialect<linalg::LinalgDialect>();
  registerDialect<scf::SCFDialect>();
  registerDialect<omp::OpenMPDialect>();
  registerDialect<spirv::SPIRVDialect>();
  registerDialect<StandardOpsDialect>();
  registerDialect<vector::VectorDialect>();
}

mlir::ModelBuilder::ModelBuilder()
    : OpBuilder(&ctx),
      module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx))),
      symbolTable(*module),
      loc(module->getLoc()),
      i8(IntegerType::get(8, &ctx)),
      f32(FloatType::getF32(&ctx)),
      f64(FloatType::getF64(&ctx)) {}

Value mlir::ModelBuilder::constant_f32(float v) {
  return std_constant_float(llvm::APFloat(v),
                            FloatType::getF32(ScopedContext::getContext()));
}

Value mlir::ModelBuilder::constant_f64(double v) {
  return std_constant_float(llvm::APFloat(v),
                            FloatType::getF64(ScopedContext::getContext()));
}

Value mlir::ModelBuilder::constant_index(int64_t v) {
  return std_constant_index(v);
}

FuncOp mlir::ModelBuilder::makeFunction(StringRef name, ArrayRef<Type> results,
                                        ArrayRef<Type> args,
                                        MLIRFuncOpConfig config) {
  FunctionType ft = FunctionType::get(args, results, &ctx);
  auto function = FuncOp::create(loc, name, ft);
  config.apply(function);
  module->push_back(function);
  return function;
}
FuncOp mlir::ModelBuilder::makeFunction(
    std::function<std::string(FunctionType)> nameBuilder,
    ArrayRef<Type> results, ArrayRef<Type> args, MLIRFuncOpConfig config) {
  FunctionType ft = FunctionType::get(args, results, &ctx);
  return makeFunction(nameBuilder(ft), results, args, config);
}

static spirv::TargetEnvAttr getTargetEnv(MLIRContext *context) {
  auto triple = spirv::VerCapExtAttr::get(
      spirv::Version::V_1_0,
      {spirv::Capability::Shader, spirv::Capability::CooperativeMatrixNV,
       spirv::Capability::Int8, spirv::Capability::Float16,
       spirv::Capability::StorageBuffer8BitAccess,
       spirv::Capability::Float16Buffer},
      {spirv::Extension::SPV_KHR_storage_buffer_storage_class,
       spirv::Extension::SPV_NV_cooperative_matrix,
       spirv::Extension::SPV_KHR_8bit_storage,
       spirv::Extension::SPV_KHR_16bit_storage},
      context);
  return spirv::TargetEnvAttr::get(triple,
                                   spirv::getDefaultResourceLimits(context));
}

gpu::GPUModuleOp mlir::ModelBuilder::makeGPUModule(StringRef name) {
  // Add module attributes required first.
  addGPUAttr();
  OpBuilder b(&module->getBodyRegion());
  auto kernelModule = b.create<gpu::GPUModuleOp>(loc, name);
  return kernelModule;
}

void mlir::ModelBuilder::addGPUAttr() {
  // Add module attributes required first.
  module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                  UnitAttr::get(module->getContext()));
  spirv::TargetEnvAttr targetEnv = getTargetEnv(module->getContext());
  module->setAttr(spirv::getTargetEnvAttrName(), targetEnv);
}

gpu::GPUFuncOp mlir::ModelBuilder::makeGPUKernel(
    StringRef name, gpu::GPUModuleOp GPUModule, ArrayRef<int32_t> workgroupSize,
    ArrayRef<Type> args, ArrayRef<Type> results) {
  auto fnType = FunctionType::get(args, results, module->getContext());
  OpBuilder b(&GPUModule.body());
  auto kernelFunc = b.create<gpu::GPUFuncOp>(loc, name, fnType);
  kernelFunc.setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
  kernelFunc.setAttr(
      spirv::getEntryPointABIAttrName(),
      spirv::getEntryPointABIAttr(workgroupSize, module->getContext()));
  return kernelFunc;
}

VectorType mlir::ModelBuilder::getVectorType(ArrayRef<int64_t> shape,
                                             Type elementalType) {
  return VectorType::get(shape, elementalType);
}

MemRefType mlir::ModelBuilder::getMemRefType(ArrayRef<int64_t> shape,
                                             Type elementType,
                                             unsigned addressSpace) {
  return MemRefType::get(shape, elementType, {}, addressSpace);
}

RankedTensorType mlir::ModelBuilder::getRankedTensorType(
    ArrayRef<int64_t> shape, Type elementType) {
  return RankedTensorType::get(shape, elementType);
}

Value mlir::ModelBuilder::fusedBiasTanh(Value x, Value bias) {
  using edsc::op::operator+;
  using edsc::op::operator*;
  using edsc::intrinsics::std_call;
  assert(x.getType().isF32() && bias.getType().isF32() && "f32 expected");
  Value half = constant_f32(0.5f);
  return x + half * call_tanhf((x + bias) * half) + half;
}

Value mlir::ModelBuilder::FCBiasTanh(std::array<Value, 3> fcArgs,
                                     Value biasValueArg) {
  //==========================================================================//
  // Layer 1: FC
  //==========================================================================//
  Value I = fcArgs[0], W = fcArgs[1], O = fcArgs[2];
  // Emit a linalg.generic op that implements matmul:
  linalg_generic_matmul(I, W, O);

  //==========================================================================//
  // Layer 2: BiasAddTanh Block
  //==========================================================================//
  // Build and capture AffineExpr i and j for building index expressions.
  AffineExpr i, j;
  bindDims(&ctx, i, j);

  // Emit a linalg.generic op that implements pointwise with `opBuilder` for:
  //   `0.5f * tanh(0.5f * (x + bias)) + 0.5f`
  //
  // This performs the (inplace) computation:
  //   `o[i, j] <- pointwise(bias[j], o[i, j])`
  //
  // in which bias is broadcast along `i`.
  StructuredIndexed o(O), bias(biasValueArg);
  linalg_generic_pointwise(fusedBiasTanh, o({i, j}), bias({j}), o({i, j}));

  return O;
}

Value ModelBuilder::FCBiasTanhTensors(RankedTensorType outputTensorType,
                                      std::array<Value, 2> fcArgs,
                                      Value biasValueArg) {
  //==========================================================================//
  // Layer 1: FC
  //==========================================================================//
  Value I = fcArgs[0], W = fcArgs[1];
  Value O2 = linalg_generic_matmul(I, W, outputTensorType)->getResult(0);

  //==========================================================================//
  // Layer 2: BiasAddTanh Block
  //==========================================================================//
  AffineExpr i, j;
  bindDims(&ctx, i, j);
  // in-place with explicit bias broacast
  StructuredIndexed o2(O2), bias(biasValueArg), o3Type(outputTensorType);
  return linalg_generic_pointwise(fusedBiasTanh, o2({i, j}), bias({j}),
                                  o3Type({i, j}))
      ->getResult(0);
}

Value ModelBuilder::call_tanhf(Value v) {
  assert(v.getType().isF32() && "f32 expected");
  return emitCallToRegisteredSymbol("tanhf", v.getType(), v)->getResult(0);
}

void ModelBuilder::call_print_memref_f32(Value v) {
  auto &builder = ScopedContext::getBuilderRef();
  auto loc = builder.getInsertionBlock()
                 ->getParent()
                 ->getParentOfType<FuncOp>()
                 .getLoc();
  auto elementType = v.getType().cast<MemRefType>().getElementType();
  auto unrankedType = UnrankedMemRefType::get(elementType, 0);
  auto castMemRef = builder.create<MemRefCastOp>(loc, v, unrankedType);
  if (elementType.isF32())
    emitCallToRegisteredSymbol("print_memref_f32", {}, {castMemRef});
  else
    llvm_unreachable("Incorrect argument type for print_memref_f32");
}

Operation *ModelBuilder::emitCallToRegisteredSymbol(StringRef functionName,
                                                    ArrayRef<Type> returnTypes,
                                                    ValueRange values) {
  auto &builder = ScopedContext::getBuilderRef();
  auto funcOp =
      builder.getInsertionBlock()->getParent()->getParentOfType<FuncOp>();
  Operation *func = SymbolTable::lookupNearestSymbolFrom(funcOp, functionName);
  if (!func) {
    OpBuilder::InsertionGuard insertGuard(builder);
    auto module = funcOp.getParentOfType<ModuleOp>();
    builder.setInsertionPointToStart(module.getBody());
    func = builder.create<FuncOp>(
        module.getLoc(), functionName,
        FunctionType::get(SmallVector<Type, 4>(values.getTypes()), returnTypes,
                          builder.getContext()));
  }
  return std_call(builder.getSymbolRefAttr(func), returnTypes, values);
}

MLIRFuncOpConfig &MLIRFuncOpConfig::setNoInline(bool v) {
  noInline = v;
  return *this;
}
MLIRFuncOpConfig &MLIRFuncOpConfig::setPreferAvx512(bool v) {
  preferAvx512 = v;
  return *this;
}
MLIRFuncOpConfig &MLIRFuncOpConfig::setTargetCpu(StringRef s) {
  targetCpu = std::string(s);
  return *this;
}
MLIRFuncOpConfig &MLIRFuncOpConfig::setDeclOnly(bool v) {
  declOnly = v;
  return *this;
}
MLIRFuncOpConfig &MLIRFuncOpConfig::setEmitCInterface(bool v) {
  emitCInterface = v;
  return *this;
}

void MLIRFuncOpConfig::apply(FuncOp &f) {
  MLIRContext *ctx = f.getContext();
  SmallVector<Attribute, 8> attrs;
  if (noInline) attrs.push_back(StringAttr::get("noinline", ctx));
  if (preferAvx512)
    attrs.push_back(ArrayAttr::get({StringAttr::get("prefer-vector-width", ctx),
                                    StringAttr::get("512", ctx)},
                                   ctx));
  if (!targetCpu.empty())
    attrs.push_back(ArrayAttr::get(
        {StringAttr::get("target-cpu", ctx), StringAttr::get(targetCpu, ctx)},
        ctx));
  if (!attrs.empty()) f.setAttr("passthrough", ArrayAttr::get(attrs, ctx));

  if (emitCInterface)
    f.setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(ctx));

  if (!declOnly) f.addEntryBlock();
}

// -----------------------------------------------------------------------------
// EDSC extensions.
// -----------------------------------------------------------------------------
template <typename Lambda>
static SmallVector<Value, 4> valueRangeOperatorImpl(Lambda fun, ValueRange a,
                                                    ValueRange b) {
  SmallVector<Value, 4> res;
  res.reserve(std::min(a.size(), b.size()));
  for (auto it : llvm::zip(a, b))
    res.push_back(fun(std::get<0>(it), std::get<1>(it)));
  return res;
}
SmallVector<Value, 4> mlir::edsc::extensions::operator-(ValueRange a,
                                                        ValueRange b) {
  return valueRangeOperatorImpl(edsc::op::operator-, a, b);
}
SmallVector<Value, 4> mlir::edsc::extensions::operator+(ValueRange a,
                                                        ValueRange b) {
  return valueRangeOperatorImpl(edsc::op::operator+, a, b);
}
SmallVector<Value, 4> mlir::edsc::extensions::std_max(ValueRange a,
                                                      ValueRange b) {
  using edsc::op::slt;
  auto fun = [](Value va, Value vb) { return slt(va, vb) ? vb : va; };
  return valueRangeOperatorImpl(fun, a, b);
}
SmallVector<Value, 4> mlir::edsc::extensions::std_min(ValueRange a,
                                                      ValueRange b) {
  using edsc::op::slt;
  auto fun = [](Value va, Value vb) { return slt(va, vb) ? va : vb; };
  return valueRangeOperatorImpl(fun, a, b);
}
SmallVector<Value, 4> mlir::edsc::extensions::affine_max(ValueRange a,
                                                         ValueRange b) {
  // TODO(ntv): cleanup when affine_max accepts has more idiomatic builders.
  MLIRContext *ctx = ScopedContext::getContext();
  auto map = AffineMap::get(
      2, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)}, ctx);
  auto fun = [&](Value va, Value vb) {
    return intrinsics::affine_max(map, ValueRange{va, vb});
  };
  return valueRangeOperatorImpl(fun, a, b);
}
SmallVector<Value, 4> mlir::edsc::extensions::affine_min(ValueRange a,
                                                         ValueRange b) {
  // TODO(ntv): cleanup when affine_min accepts has more idiomatic builders.
  MLIRContext *ctx = ScopedContext::getContext();
  auto map = AffineMap::get(
      2, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)}, ctx);
  auto fun = [&](Value va, Value vb) {
    return intrinsics::affine_min(map, ValueRange{va, vb});
  };
  return valueRangeOperatorImpl(fun, a, b);
}
