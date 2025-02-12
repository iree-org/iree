// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::TFLite {

// Wraps each model entry point in a "_tflite_xx" function that matches the
// expectations of the IREE TFLite C bindings and materializes shape query and
// calculation functions for dynamically shaped I/O.
//
// For each exported function we produce:
// - `_tflite_xx_argN`/`retN` globals carrying shape dimensions
// - `_tflite_xx` entry function wrapping the existing export
// - `_tflite_xx_calculate_shapes` shape calculation function
// - `_tflite_xx_query_input_shape` shape query function
// - `_tflite_xx_query_output_shape` shape query function
//
// Each I/O of the function gets one global per dynamic dimension storing the
// provided or calculated dimension value at runtime. For example:
//   (%arg0: tensor<1x?x?xf32>)
// ->
//   // no dim0 as it is static 1.
//   util.global private mutable @_tflite_xx_arg0_dim1 : index
//   util.global private mutable @_tflite_xx_arg0_dim2 : index
class WrapEntryPointsPass
    : public PassWrapper<WrapEntryPointsPass, OperationPass<ModuleOp>> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                    IREE::HAL::HALDialect, IREE::Util::UtilDialect>();
  }

  StringRef getArgument() const override {
    return "iree-tflite-wrap-entry-points";
  }

  StringRef getDescription() const override {
    return "Wraps model entry points in functions compatible with the tflite "
           "bindings";
  }

  static StringAttr getArgId(IREE::Util::FuncOp funcOp, int i) {
    StringAttr id =
        funcOp.getArgAttrOfType<StringAttr>(i, "ml_program.identifier");
    return id ? id : funcOp.getArgAttrOfType<StringAttr>(i, "iree.abi.name");
  }

  static StringAttr getResultId(IREE::Util::FuncOp funcOp, int i) {
    StringAttr id =
        funcOp.getResultAttrOfType<StringAttr>(i, "ml_program.identifier");
    return id ? id : funcOp.getResultAttrOfType<StringAttr>(i, "iree.abi.name");
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    SmallVector<IREE::Util::FuncOp> entryFuncOps;
    for (auto funcOp : moduleOp.getOps<IREE::Util::FuncOp>()) {
      if (funcOp.isPublic() && !funcOp->hasAttr("iree.abi.stub")) {
        entryFuncOps.push_back(funcOp);
      }
    }
    if (entryFuncOps.size() == 0) {
      moduleOp.emitError()
          << "no entry points found; the tflite bindings "
             "require exactly 1 entry point (function with public visibility)";
      signalPassFailure();
      return;
    } else if (entryFuncOps.size() > 1) {
      moduleOp.emitError()
          << "multiple entry points found; the tflite bindings require exactly "
             "1 entry point (function with public visibility)";
      signalPassFailure();
      return;
    }
    wrapEntryPoint(entryFuncOps.front());
  }

private:
  // Globals representing each dynamic dimension of an IO tensor.
  struct DynamicDims {
    TensorType tensorType;
    mutable SmallVector<IREE::Util::GlobalOp> globalOps;

    SmallVector<Value> loadDynamicDims(OpBuilder &builder) const {
      SmallVector<Value> dims;
      unsigned dynamicDimIdx = 0;
      for (unsigned i = 0; i < tensorType.getRank(); ++i) {
        if (tensorType.isDynamicDim(i)) {
          auto globalOp = globalOps[dynamicDimIdx++];
          dims.push_back(globalOp.createLoadOp(globalOp.getLoc(), builder)
                             .getLoadedGlobalValue());
        }
      }
      return dims;
    }
  };

  // Creates one util.global index op for each |tensorType| dynamic dimension.
  static DynamicDims createDynamicDimGlobals(Location loc, StringRef namePrefix,
                                             TensorType tensorType,
                                             OpBuilder &moduleBuilder) {
    DynamicDims dynamicDims;
    dynamicDims.tensorType = tensorType;
    for (unsigned i = 0; i < tensorType.getRank(); ++i) {
      if (tensorType.isDynamicDim(i)) {
        auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
            loc, (namePrefix + "_dim" + std::to_string(i)).str(),
            /*isMutable=*/true, moduleBuilder.getIndexType());
        globalOp.setPrivate();
        dynamicDims.globalOps.push_back(globalOp);
      }
    }
    return dynamicDims;
  }

  // Creates dynamic dim globals for each input and output of |funcOp|.
  static std::pair<SmallVector<DynamicDims>, SmallVector<DynamicDims>>
  createDynamicDimGlobals(Location loc, StringRef namePrefix,
                          IREE::Util::FuncOp funcOp, OpBuilder &moduleBuilder) {
    auto funcType = funcOp.getFunctionType();

    // TFLite requires the tensor names at runtime. If they've previously been
    // extracted into iree.abi.names we use those and otherwise fallback to
    // a generic naming scheme that matches the IR (somewhat).
    SmallVector<std::string> inputNames;
    SmallVector<std::string> outputNames;
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      auto identifier = getArgId(funcOp, i);
      if (identifier) {
        inputNames.push_back(identifier.getValue().str());
      } else {
        inputNames.push_back(std::string("arg") + std::to_string(i));
      }
    }
    for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
      auto identifier = getResultId(funcOp, i);
      if (identifier) {
        outputNames.push_back(identifier.getValue().str());
      } else {
        outputNames.push_back(std::string("ret") + std::to_string(i));
      }
    }

    SmallVector<DynamicDims> inputDynamicDims;
    for (auto [arg, inputName, inputType] : llvm::zip_equal(
             funcOp.getArguments(), inputNames, funcType.getInputs())) {
      auto fullName = (namePrefix + "_" + inputName + "_shape").str();
      auto tensorType = llvm::dyn_cast<TensorType>(inputType);
      assert(tensorType && "expecting only tensors in tflite function I/O");
      inputDynamicDims.push_back(createDynamicDimGlobals(
          arg.getLoc(), fullName, tensorType, moduleBuilder));
    }
    SmallVector<DynamicDims> outputDynamicDims;
    for (auto [outputName, outputType] :
         llvm::zip_equal(outputNames, funcType.getResults())) {
      auto fullName = (namePrefix + "_" + outputName + "_shape").str();
      auto tensorType = llvm::dyn_cast<TensorType>(outputType);
      assert(tensorType && "expecting only tensors in tflite function I/O");
      outputDynamicDims.push_back(
          createDynamicDimGlobals(loc, fullName, tensorType, moduleBuilder));
    }

    return std::make_pair(inputDynamicDims, outputDynamicDims);
  }

  // Derives a shape calculation function from the given entry point |funcOp|.
  static IREE::Util::FuncOp createShapeCalculationFunc(
      Location loc, StringRef namePrefix, IREE::Util::FuncOp funcOp,
      ArrayRef<DynamicDims> inputDynamicDims,
      ArrayRef<DynamicDims> outputDynamicDims,
      IREE::Util::GlobalOp dirtyGlobalOp, OpBuilder &moduleBuilder) {
    // Clone the entire entry function with all its IR.
    auto calcFuncOp =
        cast<IREE::Util::FuncOp>(moduleBuilder.clone(*funcOp.getOperation()));
    calcFuncOp.setName(
        moduleBuilder.getStringAttr(namePrefix.str() + "_calculate_shapes"));
    calcFuncOp.setPrivate();
    // TODO(benvanik): find a better way to strip these attributes.
    calcFuncOp->removeAttr("iree.abi.stub");
    calcFuncOp->removeAttr("iree.reflection");
    auto &entryBlock = calcFuncOp.front();
    auto entryBuilder = OpBuilder::atBlockBegin(&entryBlock);

    // Go back and insert a check for the dirty flag.
    auto dirtyValue =
        dirtyGlobalOp.createLoadOp(loc, entryBuilder).getLoadedGlobalValue();
    auto *recalculateBlock = calcFuncOp.addBlock();
    auto *returnBlock = calcFuncOp.addBlock();
    entryBuilder.create<mlir::cf::CondBranchOp>(loc, dirtyValue,
                                                recalculateBlock, returnBlock);
    auto *followBlock = entryBlock.splitBlock(entryBuilder.getInsertionPoint());

    auto bufferType = entryBuilder.getType<IREE::HAL::BufferType>();

    // Turn inputs into placeholder values and kill all return values.
    // DCE then has an easy time ripping the tensor values all out.
    // We need to tie the input variable shapes to the placeholders so shape
    // propagation can use them.
    auto recalculateBuilder = OpBuilder::atBlockBegin(recalculateBlock);
    calcFuncOp.setType(
        recalculateBuilder.getFunctionType(/*inputs=*/TypeRange{},
                                           /*outputs=*/TypeRange{}));
    for (auto [inputValue, inputDynamicDims] :
         llvm::zip_equal(entryBlock.getArguments(), inputDynamicDims)) {
      auto inputPlaceholder =
          recalculateBuilder.createOrFold<IREE::Util::NullOp>(loc, bufferType);
      auto dynamicDims = inputDynamicDims.loadDynamicDims(recalculateBuilder);
      auto castOp = recalculateBuilder.create<IREE::HAL::TensorImportOp>(
          loc, inputValue.getType(), inputPlaceholder, inputValue.getType(),
          dynamicDims, /*wait_fence=*/Value{}, /*name=*/nullptr,
          /*affinity=*/nullptr);
      inputValue.replaceAllUsesWith(castOp.getTarget());
    }
    while (entryBlock.getNumArguments() > 0) {
      entryBlock.eraseArgument(entryBlock.getNumArguments() - 1);
    }
    recalculateBuilder.create<mlir::cf::BranchOp>(loc, followBlock);
    recalculateBlock->moveBefore(followBlock);

    // Replace each exit from the function with a storage back to the shape
    // variables.
    for (auto returnOp :
         llvm::to_vector(calcFuncOp.getOps<IREE::Util::ReturnOp>())) {
      auto exitLoc = returnOp.getLoc();
      OpBuilder exitBuilder(returnOp);

      // Store the derived shape values into the output shape variables.
      // We do this per exit-site so that if the function has multiple code
      // paths that may return different shape sizes we capture them all.
      for (auto [outputValue, outputDynamicDims] :
           llvm::zip_equal(returnOp.getOperands(), outputDynamicDims)) {
        SmallVector<Value> dynamicDims;
        for (int64_t i = 0; i < outputDynamicDims.globalOps.size(); ++i) {
          auto dimValue =
              exitBuilder.createOrFold<tensor::DimOp>(exitLoc, outputValue, i);
          outputDynamicDims.globalOps[i].createStoreOp(exitLoc, dimValue,
                                                       exitBuilder);
        }
      }

      // Clear the dirty flag now that the shapes have been updated.
      auto falseValue =
          exitBuilder.createOrFold<arith::ConstantIntOp>(exitLoc, 0, 1);
      dirtyGlobalOp.createStoreOp(exitLoc, falseValue, exitBuilder);
      exitBuilder.create<IREE::Util::ReturnOp>(exitLoc);
      returnOp.erase();
    }

    OpBuilder::atBlockBegin(returnBlock).create<IREE::Util::ReturnOp>(loc);

    return calcFuncOp;
  }

  // Builds a switch-statement-like chain of blocks starting at |builder|.
  // Returns a block that execution resumes at after the switch.
  static Block *buildSwitch(
      Location loc, Value indexValue, size_t caseCount,
      std::function<void(size_t i, OpBuilder &caseBuilder)> caseGenerator,
      OpBuilder &builder) {
    auto *entryBlock = builder.getBlock();
    auto ip = builder.saveInsertionPoint();
    auto *exitBlock = builder.createBlock(entryBlock->getParent(),
                                          ++Region::iterator(entryBlock));
    if (caseCount == 0) {
      builder.create<mlir::cf::BranchOp>(loc, exitBlock);
      return exitBlock;
    }
    SmallVector<Block *> compareBlocks;
    SmallVector<Block *> caseBlocks;
    for (size_t i = 0; i < caseCount; ++i) {
      compareBlocks.push_back(builder.createBlock(exitBlock));
      caseBlocks.push_back(builder.createBlock(exitBlock));
    }
    builder.restoreInsertionPoint(ip);
    builder.create<mlir::cf::BranchOp>(loc, compareBlocks[0]);
    for (size_t i = 0; i < caseCount; ++i) {
      auto compareBuilder = OpBuilder::atBlockBegin(compareBlocks[i]);
      auto caseValue =
          compareBuilder.createOrFold<arith::ConstantIndexOp>(loc, i);
      auto eqValue = compareBuilder.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, indexValue, caseValue);
      compareBuilder.create<mlir::cf::CondBranchOp>(
          loc, eqValue, caseBlocks[i],
          i < caseCount - 1 ? compareBlocks[i + 1] : exitBlock);

      auto caseBuilder = OpBuilder::atBlockBegin(caseBlocks[i]);
      caseGenerator(i, caseBuilder);
      caseBuilder.create<mlir::cf::BranchOp>(loc, exitBlock);
    }
    builder = OpBuilder::atBlockBegin(exitBlock);
    return exitBlock;
  }

  // Packs a shape into a list.
  void packShape(Location loc, const DynamicDims &dynamicDims, Value listValue,
                 OpBuilder &builder) {
    auto shapeType = dynamicDims.tensorType;
    builder.create<IREE::Util::ListResizeOp>(
        loc, listValue,
        builder.createOrFold<arith::ConstantIndexOp>(loc, shapeType.getRank()));
    unsigned dynamicDimIdx = 0;
    for (unsigned i = 0; i < shapeType.getRank(); ++i) {
      Value dimValue;
      if (shapeType.isDynamicDim(i)) {
        dimValue = dynamicDims.globalOps[dynamicDimIdx++]
                       .createLoadOp(loc, builder)
                       .getLoadedGlobalValue();
      } else {
        dimValue = builder.createOrFold<arith::ConstantIndexOp>(
            loc, shapeType.getDimSize(i));
      }
      builder.create<IREE::Util::ListSetOp>(
          loc, listValue, builder.createOrFold<arith::ConstantIndexOp>(loc, i),
          dimValue);
    }
  }

  // Unpacks a shape from a list.
  void unpackShape(Location loc, Value listValue,
                   const DynamicDims &dynamicDims, OpBuilder &builder) {
    auto shapeType = dynamicDims.tensorType;
    unsigned dynamicDimIdx = 0;
    for (unsigned i = 0; i < shapeType.getRank(); ++i) {
      if (!shapeType.isDynamicDim(i))
        continue;
      auto dimValue =
          builder
              .create<IREE::Util::ListGetOp>(
                  loc, builder.getIndexType(), listValue,
                  builder.createOrFold<arith::ConstantIndexOp>(loc, i))
              .getResult();
      dynamicDims.globalOps[dynamicDimIdx++].createStoreOp(loc, dimValue,
                                                           builder);
    }
  }

  // Creates a function to query the |inputGlobalOps| at runtime by the
  // bindings.
  //
  // func.func @_query_input_shape(%index : index, %shape : !util.list<index>)
  void createQueryInputShapeFunc(Location loc, StringRef namePrefix,
                                 ArrayRef<DynamicDims> inputDynamicDims,
                                 OpBuilder &moduleBuilder) {
    auto queryFuncOp = moduleBuilder.create<IREE::Util::FuncOp>(
        loc, namePrefix.str() + "_query_input_shape",
        moduleBuilder.getFunctionType(/*inputs=*/
                                      TypeRange{
                                          moduleBuilder.getIndexType(),
                                          IREE::Util::ListType::get(
                                              moduleBuilder.getIndexType()),
                                      },
                                      /*outputs=*/TypeRange{}));
    queryFuncOp->setAttr("iree.abi.stub", moduleBuilder.getUnitAttr());
    auto *entryBlock = queryFuncOp.addEntryBlock();
    auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);
    auto listValue = entryBlock->getArgument(1);

    auto *exitBlock = buildSwitch(
        loc, entryBlock->getArgument(0), inputDynamicDims.size(),
        [&](size_t i, OpBuilder &caseBuilder) {
          packShape(loc, inputDynamicDims[i], listValue, caseBuilder);
        },
        entryBuilder);

    auto exitBuilder = OpBuilder::atBlockBegin(exitBlock);
    exitBuilder.create<IREE::Util::ReturnOp>(loc);
  }

  // Creates a function to resize |inputGlobalOps| and sets the |dirtyGlobalOp|
  // flag.
  //
  // func.func @_resize_input_shape(%index : index, %shape : !util.list<index>)
  void createResizeInputShapeFunc(Location loc, StringRef namePrefix,
                                  ArrayRef<DynamicDims> inputDynamicDims,
                                  IREE::Util::GlobalOp dirtyGlobalOp,
                                  OpBuilder &moduleBuilder) {
    auto resizeFuncOp = moduleBuilder.create<IREE::Util::FuncOp>(
        loc, namePrefix.str() + "_resize_input_shape",
        moduleBuilder.getFunctionType(/*inputs=*/
                                      TypeRange{
                                          moduleBuilder.getIndexType(),
                                          IREE::Util::ListType::get(
                                              moduleBuilder.getIndexType()),
                                      },
                                      /*outputs=*/TypeRange{}));
    resizeFuncOp->setAttr("iree.abi.stub", moduleBuilder.getUnitAttr());
    auto *entryBlock = resizeFuncOp.addEntryBlock();
    auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);
    auto listValue = entryBlock->getArgument(1);

    auto *exitBlock = buildSwitch(
        loc, entryBlock->getArgument(0), inputDynamicDims.size(),
        [&](size_t i, OpBuilder &caseBuilder) {
          unpackShape(loc, listValue, inputDynamicDims[i], caseBuilder);
        },
        entryBuilder);

    // Set the dirty flag so that shapes get recalculated as needed.
    auto exitBuilder = OpBuilder::atBlockBegin(exitBlock);
    auto trueValue = exitBuilder.createOrFold<arith::ConstantIntOp>(loc, 1, 1);
    dirtyGlobalOp.createStoreOp(loc, trueValue, exitBuilder);
    exitBuilder.create<IREE::Util::ReturnOp>(loc);
  }

  // Creates a function to query the |outputGlobalOps| at runtime by the
  // bindings.
  //
  // func.func @_query_output_shape(%index : index, %shape : !util.list<index>)
  void createQueryOutputShapeFunc(Location loc, StringRef namePrefix,
                                  ArrayRef<DynamicDims> outputDynamicDims,
                                  IREE::Util::FuncOp calculateShapeFuncOp,
                                  OpBuilder &moduleBuilder) {
    auto queryFuncOp = moduleBuilder.create<IREE::Util::FuncOp>(
        loc, namePrefix.str() + "_query_output_shape",
        moduleBuilder.getFunctionType(/*inputs=*/
                                      TypeRange{
                                          moduleBuilder.getIndexType(),
                                          IREE::Util::ListType::get(
                                              moduleBuilder.getIndexType()),
                                      },
                                      /*outputs=*/TypeRange{}));
    queryFuncOp->setAttr("iree.abi.stub", moduleBuilder.getUnitAttr());
    auto *entryBlock = queryFuncOp.addEntryBlock();
    auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);
    auto listValue = entryBlock->getArgument(1);

    // Always call the recalculation function - it checks for whether it needs
    // to run based on the dirty flag value.
    entryBuilder.create<IREE::Util::CallOp>(loc, calculateShapeFuncOp,
                                            ValueRange{});

    auto *exitBlock = buildSwitch(
        loc, entryBlock->getArgument(0), outputDynamicDims.size(),
        [&](size_t i, OpBuilder &caseBuilder) {
          packShape(loc, outputDynamicDims[i], listValue, caseBuilder);
        },
        entryBuilder);

    auto exitBuilder = OpBuilder::atBlockBegin(exitBlock);
    exitBuilder.create<IREE::Util::ReturnOp>(loc);
  }

  // Creates the corresponding wrapper function for the given entry point.
  // The wrapper function will contain the reflection metadata required at
  // runtime to get input/output tensor names, quantization parameters, etc.
  //
  // We do this by creating a new function just for the bindings and calling the
  // existing entry point. This allows us to support multiple binding schemes as
  // transforms from other bindings can also perform their own equivalent
  // wrapping.
  //
  // NOTE: today we only support a single entry point; with minor tweaks we
  // could fix this up to support multiple if we wanted.
  void createWrapperFunc(StringRef namePrefix, IREE::Util::FuncOp entryFuncOp,
                         ArrayRef<DynamicDims> inputDynamicDims,
                         ArrayRef<DynamicDims> outputDynamicDims,
                         IREE::Util::GlobalOp dirtyGlobalOp,
                         OpBuilder &moduleBuilder) {
    // NOTE: this is where we could change our signature to provide additional
    // values from the runtime bindings as may be required - like semaphores for
    // async behavior or cancellation.
    auto entryFuncType = entryFuncOp.getFunctionType();
    auto bufferType = moduleBuilder.getType<IREE::HAL::BufferType>();
    SmallVector<Type> inputTypes(entryFuncType.getNumInputs(), bufferType);
    SmallVector<Type> outputTypes(entryFuncType.getNumResults(), bufferType);
    auto wrapperFuncType =
        moduleBuilder.getFunctionType(inputTypes, outputTypes);

    auto wrapperFuncOp = moduleBuilder.create<IREE::Util::FuncOp>(
        entryFuncOp.getLoc(), "_tflite_main", wrapperFuncType);
    wrapperFuncOp.setPublic();
    wrapperFuncOp.getOperation()->setAttr("iree.abi.stub",
                                          moduleBuilder.getUnitAttr());

    SmallVector<DictionaryAttr> argAttrDict;
    entryFuncOp.getAllArgAttrs(argAttrDict);
    wrapperFuncOp.setAllArgAttrs(argAttrDict);
    SmallVector<DictionaryAttr> resultAttrDict;
    entryFuncOp.getAllResultAttrs(resultAttrDict);
    wrapperFuncOp.setAllResultAttrs(resultAttrDict);

    populateReflectionAttrs(entryFuncOp, wrapperFuncOp);
    if (auto affinityAttr = entryFuncOp->getAttr("stream.affinity"))
      wrapperFuncOp->setAttr("stream.affinity", affinityAttr);

    // Call the entryFuncOp and return the results.
    // If we wanted to perform additional work here to invalidate cached shapes
    // from the shape support functions or validate the inputs we'd do that
    // here. Format conversion/decomposition (interleaved complex ->
    // deinterleaved, float <-> quantized conversions, etc) can also be inserted
    // such that other bindings that don't need such things aren't impacted.
    //
    // To make the interface concrete we insert casts to HAL buffers so that
    // in the final program we know they end up as the iree_hal_buffer_t we
    // expect in the runtime.
    auto *entryBlock = wrapperFuncOp.addEntryBlock();
    auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);
    SmallVector<Value> callOperands;
    for (auto [arg, inputDynamicDims] :
         llvm::zip_equal(entryBlock->getArguments(), inputDynamicDims)) {
      SmallVector<Value> dynamicDims;
      for (auto globalOp : inputDynamicDims.globalOps) {
        dynamicDims.push_back(globalOp.createLoadOp(arg.getLoc(), entryBuilder)
                                  .getLoadedGlobalValue());
      }
      callOperands.push_back(entryBuilder.create<IREE::HAL::TensorImportOp>(
          arg.getLoc(), inputDynamicDims.tensorType, arg,
          TypeAttr::get(inputDynamicDims.tensorType), dynamicDims,
          /*wait_fence=*/Value{}, /*name=*/nullptr,
          /*affinity=*/nullptr));
    }
    auto callOp = entryBuilder.create<IREE::Util::CallOp>(
        entryFuncOp.getLoc(), entryFuncOp, callOperands);
    SmallVector<Value> callResults;
    for (auto [result, outputDynamicDims] :
         llvm::zip_equal(callOp.getResults(), outputDynamicDims)) {
      SmallVector<Value> dynamicDims;
      for (unsigned i = 0; i < outputDynamicDims.tensorType.getRank(); ++i) {
        if (outputDynamicDims.tensorType.isDynamicDim(i)) {
          dynamicDims.push_back(
              entryBuilder.create<tensor::DimOp>(result.getLoc(), result, i));
        }
      }
      callResults.push_back(entryBuilder.create<IREE::HAL::TensorExportOp>(
          result.getLoc(), bufferType, result, outputDynamicDims.tensorType,
          dynamicDims, /*name=*/nullptr, /*affinity=*/nullptr));
      for (auto [dynamicDim, globalOp] :
           llvm::zip_equal(dynamicDims, outputDynamicDims.globalOps)) {
        globalOp.createStoreOp(result.getLoc(), dynamicDim, entryBuilder);
      }
    }

    // We recomputed the shapes of the outputs and can clear the dirty flag.
    dirtyGlobalOp.createStoreOp(
        entryFuncOp.getLoc(),
        entryBuilder.create<arith::ConstantIntOp>(entryFuncOp.getLoc(), 0, 1),
        entryBuilder);

    entryBuilder.create<IREE::Util::ReturnOp>(entryFuncOp.getLoc(),
                                              callResults);
  }

  void wrapEntryPoint(IREE::Util::FuncOp funcOp) {
    auto loc = funcOp.getLoc();
    auto namePrefix = ("_tflite_" + funcOp.getName()).str();
    OpBuilder moduleBuilder(funcOp);

    // Create a variable for each input and output dynamic dim. These variables
    // may represent fully static shapes - in which case they'll get constant
    // propagated - or dynamic shapes that will eventually get turned into
    // dynamic runtime values.
    auto dynamicDimGlobals =
        createDynamicDimGlobals(loc, namePrefix, funcOp, moduleBuilder);

    // Create internal shape calculation function that updates output shapes if
    // needed. This is only required if there are dynamic shapes.
    auto dirtyGlobalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, namePrefix + "_shapes_dirty",
        /*isMutable=*/true, moduleBuilder.getI1Type(),
        moduleBuilder.getIntegerAttr(moduleBuilder.getI1Type(), 1));
    dirtyGlobalOp.setPrivate();
    auto calculateShapeFuncOp = createShapeCalculationFunc(
        loc, namePrefix, funcOp, dynamicDimGlobals.first,
        dynamicDimGlobals.second, dirtyGlobalOp, moduleBuilder);

    // Create input query function (just reads variables).
    createQueryInputShapeFunc(loc, namePrefix, dynamicDimGlobals.first,
                              moduleBuilder);

    // Create input resize function (updates variables, set dirty flag).
    createResizeInputShapeFunc(loc, namePrefix, dynamicDimGlobals.first,
                               dirtyGlobalOp, moduleBuilder);

    // Create output query function (if dirty recalculates shapes).
    createQueryOutputShapeFunc(loc, namePrefix, dynamicDimGlobals.second,
                               calculateShapeFuncOp, moduleBuilder);

    // Create a wrapper function for the entry point.
    funcOp.setPrivate();
    createWrapperFunc(namePrefix, funcOp, dynamicDimGlobals.first,
                      dynamicDimGlobals.second, dirtyGlobalOp, moduleBuilder);
  }

  // Populates attributes on |wrapperFuncOp| to support runtime reflection like
  // IO tensor names and quantization information.
  void populateReflectionAttrs(IREE::Util::FuncOp entryFuncOp,
                               IREE::Util::FuncOp wrapperFuncOp) {
    SmallVector<NamedAttribute, 1> attrs;
    attrs.push_back(buildIONamesAttr(entryFuncOp));
    // TODO(#3972): tfl.io.quant: quantization information.
    // TODO(#3978): tfl.io.types: tensor types (complex/strings/etc).
    auto reflectionAttr = DictionaryAttr::get(&getContext(), attrs);
    wrapperFuncOp->setAttr("iree.reflection", reflectionAttr);
  }

  // Constructs an attribute containing all of the input and output identifiers:
  //   tfl.io.names=arg0;arg1;ret0;ret1
  //
  // Default names will be used if no identifiers are set on the function.
  NamedAttribute buildIONamesAttr(IREE::Util::FuncOp entryFuncOp) {
    SmallVector<std::string> pieces;
    for (int i = 0; i < entryFuncOp.getNumArguments(); ++i) {
      auto identifierAttr = getArgId(entryFuncOp, i);
      if (!identifierAttr || identifierAttr.getValue().empty()) {
        pieces.push_back("arg" + std::to_string(i));
      } else {
        pieces.push_back(identifierAttr.getValue().str());
      }
    }
    for (int i = 0; i < entryFuncOp.getNumResults(); ++i) {
      auto identifierAttr = getResultId(entryFuncOp, i);
      if (!identifierAttr || identifierAttr.getValue().empty()) {
        pieces.push_back("ret" + std::to_string(i));
      } else {
        pieces.push_back(identifierAttr.getValue().str());
      }
    }
    return NamedAttribute{
        "tfl.io.names",
        StringAttr::get(&getContext(), llvm::join(pieces, ";"))};
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createWrapEntryPointsPass() {
  return std::make_unique<WrapEntryPointsPass>();
}

static PassRegistration<WrapEntryPointsPass> pass;

} // namespace mlir::iree_compiler::IREE::TFLite
