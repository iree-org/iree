// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace TFLite {

// Materializes the shape query and manipulation functions used by the
// bindings. In tflite the specification of input shapes is performed
// independently of execution and the input shapes are stateful.
//
// We do this by adding one variable for each I/O shape that stores the
// current shape and functions to allow the bindings to query/manipulate those
// variables. We then generate a function to perform the same kind of shape
// propagation that the tflite runtime would have performed, only we need no
// runtime support and can do so much more efficiently :)
class MaterializeShapeSupportPass
    : public PassWrapper<MaterializeShapeSupportPass, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<iree_compiler::IREE::Flow::FlowDialect>();
    registry.insert<iree_compiler::IREE::Util::UtilDialect>();
    registry.insert<iree_compiler::ShapeDialect>();
    registry.insert<StandardOpsDialect, mlir::arith::ArithmeticDialect>();
  }

  StringRef getArgument() const override {
    return "iree-tflite-materialize-shape-support";
  }

  StringRef getDescription() const override {
    return "Materializes support functions for the tflite runtime bindings";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    for (auto funcOp : llvm::to_vector<4>(moduleOp.getOps<FuncOp>())) {
      if (!funcOp.isPublic()) {
        continue;
      }
      if (failed(materializeShapeSupport(funcOp, moduleBuilder))) {
        signalPassFailure();
        return;
      }
    }
  }

 private:
  // Materializes all of the state and supporting functions to track the shapes
  // of the inputs and outputs of |funcOp|.
  LogicalResult materializeShapeSupport(FuncOp funcOp,
                                        OpBuilder &moduleBuilder) {
    auto loc = funcOp.getLoc();
    auto namePrefix = funcOp.getName();

    // Create a variable for each input and output to store the ranked shape.
    // These variables may represent fully static shapes - in which case they'll
    // get constant propagated - or dynamic shapes that will eventually get
    // turned into dynamic runtime values.
    SmallVector<IREE::Util::GlobalOp, 4> inputGlobalOps;
    SmallVector<IREE::Util::GlobalOp, 4> outputGlobalOps;
    createShapeVariables(loc, namePrefix, funcOp, inputGlobalOps,
                         outputGlobalOps, moduleBuilder);

    // Create internal shape calculation function that updates output shapes if
    // needed. This is only required if there are dynamic shapes.
    auto dirtyGlobalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, funcOp.getName().str() + "_shapes_dirty",
        /*isMutable=*/true, moduleBuilder.getI1Type(),
        moduleBuilder.getIntegerAttr(moduleBuilder.getI1Type(), 1));
    dirtyGlobalOp.setPrivate();
    auto calculateShapeFuncOp = createShapeCalculationFunc(
        loc, namePrefix, funcOp, inputGlobalOps, outputGlobalOps, dirtyGlobalOp,
        moduleBuilder);

    // Create input query function (just reads variables).
    createQueryInputShapeFunc(loc, namePrefix, inputGlobalOps, moduleBuilder);

    // Create input resize function (updates variables, set dirty flag).
    createResizeInputShapeFunc(loc, namePrefix, inputGlobalOps, dirtyGlobalOp,
                               moduleBuilder);

    // Create output query function (if dirty recalculates shapes).
    createQueryOutputShapeFunc(loc, namePrefix, outputGlobalOps,
                               calculateShapeFuncOp, moduleBuilder);

    return success();
  }

  // Creates and initializes to default values one util.global for each I/O
  // shape of the given |funcOp|. |inputGlobalOps| and |outputGlobalOps| will be
  // populated with the created variables.
  void createShapeVariables(
      Location loc, StringRef namePrefix, FuncOp funcOp,
      SmallVectorImpl<IREE::Util::GlobalOp> &inputGlobalOps,
      SmallVectorImpl<IREE::Util::GlobalOp> &outputGlobalOps,
      OpBuilder &moduleBuilder) {
    auto funcType = funcOp.getType();

    // TFLite requires the tensor names at runtime. If they've previously been
    // extracted into iree.identifiers we use those and otherwise fallback to
    // a generic naming scheme that matches the IR (somewhat).
    SmallVector<std::string, 4> inputNames;
    SmallVector<std::string, 4> outputNames;
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      auto identifier =
          funcOp.getArgAttrOfType<StringAttr>(i, "iree.identifier");
      if (identifier) {
        inputNames.push_back(identifier.getValue().str());
      } else {
        inputNames.push_back(std::string("arg") + std::to_string(i));
      }
    }
    for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
      auto identifier =
          funcOp.getResultAttrOfType<StringAttr>(i, "iree.identifier");
      if (identifier) {
        outputNames.push_back(identifier.getValue().str());
      } else {
        outputNames.push_back(std::string("ret") + std::to_string(i));
      }
    }

    for (auto input : llvm::zip(inputNames, funcType.getInputs())) {
      auto name = funcOp.getName().str() + "_" + std::get<0>(input) + "_shape";
      auto type = std::get<1>(input);
      auto tensorType = type.dyn_cast<TensorType>();
      assert(tensorType && "expecting only tensors in tflite function I/O");
      inputGlobalOps.push_back(
          createShapeVariable(loc, name, tensorType, moduleBuilder));
    }
    for (auto output : llvm::zip(outputNames, funcType.getResults())) {
      auto name = funcOp.getName().str() + "_" + std::get<0>(output) + "_shape";
      auto type = std::get<1>(output);
      auto tensorType = type.dyn_cast<TensorType>();
      assert(tensorType && "expecting only tensors in tflite function I/O");
      outputGlobalOps.push_back(
          createShapeVariable(loc, name, tensorType, moduleBuilder));
    }
  }

  // Declares a global variable that holds a shape for the given |tensorType|.
  IREE::Util::GlobalOp createShapeVariable(Location loc, StringRef name,
                                           TensorType tensorType,
                                           OpBuilder &moduleBuilder) {
    auto shapeType = Shape::RankedShapeType::get(tensorType.getShape(),
                                                 moduleBuilder.getContext());
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, name, /*isMutable=*/true, shapeType);
    globalOp.setPrivate();
    return globalOp;
  }

  // Derives a shape calculation function from the given entry point |funcOp|.
  FuncOp createShapeCalculationFunc(
      Location loc, StringRef namePrefix, FuncOp funcOp,
      ArrayRef<IREE::Util::GlobalOp> inputGlobalOps,
      ArrayRef<IREE::Util::GlobalOp> outputGlobalOps,
      IREE::Util::GlobalOp dirtyGlobalOp, OpBuilder &moduleBuilder) {
    // Clone the entire entry function with all its IR.
    auto calcFuncOp = cast<FuncOp>(moduleBuilder.clone(*funcOp.getOperation()));
    mlir::StringAttr nameAttr = mlir::StringAttr::get(
        loc.getContext(), namePrefix.str() + "_calculate_shapes");
    calcFuncOp.setName(nameAttr);
    calcFuncOp.setPrivate();
    // TODO(benvanik): find a better way to strip these attributes.
    calcFuncOp->removeAttr("iree.abi.stub");
    calcFuncOp->removeAttr("iree.reflection");
    auto &entryBlock = calcFuncOp.front();
    auto entryBuilder = OpBuilder::atBlockBegin(&entryBlock);

    // Go back and insert a check for the dirty flag.
    auto dirtyValue = entryBuilder.createOrFold<IREE::Util::GlobalLoadOp>(
        loc, dirtyGlobalOp.type(), dirtyGlobalOp.getName());
    auto *recalculateBlock = calcFuncOp.addBlock();
    auto *returnBlock = calcFuncOp.addBlock();
    entryBuilder.create<CondBranchOp>(loc, dirtyValue, recalculateBlock,
                                      returnBlock);
    auto *followBlock = entryBlock.splitBlock(entryBuilder.getInsertionPoint());

    // Turn inputs into placeholder values and kill all return values.
    // DCE then has an easy time ripping the tensor values all out.
    // We need to tie the input variable shapes to the placeholders so shape
    // propagation can use them.
    auto recalculateBuilder = OpBuilder::atBlockBegin(recalculateBlock);
    calcFuncOp.setType(
        recalculateBuilder.getFunctionType(/*inputs=*/TypeRange{},
                                           /*outputs=*/TypeRange{}));
    for (auto inputValueVar :
         llvm::zip(entryBlock.getArguments(), inputGlobalOps)) {
      auto inputValue = std::get<0>(inputValueVar);
      auto inputGlobalOp = std::get<1>(inputValueVar);
      auto inputPlaceholder =
          recalculateBuilder.createOrFold<IREE::Util::NullOp>(
              loc, inputValue.getType());
      auto inputShapeValue =
          recalculateBuilder.createOrFold<IREE::Util::GlobalLoadOp>(
              loc, inputGlobalOp.type(), inputGlobalOp.getName());
      auto tiedValue = recalculateBuilder.create<Shape::TieShapeOp>(
          loc, inputPlaceholder, inputShapeValue);
      inputValue.replaceAllUsesWith(tiedValue);
    }
    while (entryBlock.getNumArguments() > 0) {
      entryBlock.eraseArgument(entryBlock.getNumArguments() - 1);
    }
    recalculateBuilder.create<BranchOp>(loc, followBlock);
    recalculateBlock->moveBefore(followBlock);

    // Replace each exit from the function with a storage back to the shape
    // variables.
    for (auto returnOp : llvm::to_vector<4>(calcFuncOp.getOps<ReturnOp>())) {
      auto exitLoc = returnOp.getLoc();
      OpBuilder exitBuilder(returnOp);

      // Store the derived shape values into the output shape variables.
      // We do this per exit-site so that if the function has multiple code
      // paths that may return different shape sizes we capture them all.
      for (auto outputValueVar :
           llvm::zip(returnOp.getOperands(), outputGlobalOps)) {
        auto outputValue = std::get<0>(outputValueVar);
        auto outputGlobalOp = std::get<1>(outputValueVar);
        auto outputShapeValue =
            exitBuilder.createOrFold<Shape::GetRankedShapeOp>(exitLoc,
                                                              outputValue);
        exitBuilder.create<IREE::Util::GlobalStoreOp>(exitLoc, outputShapeValue,
                                                      outputGlobalOp.getName());
      }

      // Clear the dirty flag now that the shapes have been updated.
      auto falseValue =
          exitBuilder.createOrFold<arith::ConstantIntOp>(exitLoc, 0, 1);
      exitBuilder.create<IREE::Util::GlobalStoreOp>(exitLoc, falseValue,
                                                    dirtyGlobalOp.getName());
      exitBuilder.create<ReturnOp>(exitLoc);
      returnOp.erase();
    }

    OpBuilder::atBlockBegin(returnBlock).create<ReturnOp>(loc);

    return calcFuncOp;
  }

  // Builds a switch-statement-like chain of blocks starting at |builder|.
  // Returns a block that execution resumes at after the switch.
  Block *buildSwitch(
      Location loc, Value indexValue, size_t caseCount,
      std::function<void(size_t i, OpBuilder &caseBuilder)> caseGenerator,
      OpBuilder &builder) {
    auto *entryBlock = builder.getBlock();
    auto ip = builder.saveInsertionPoint();
    auto *exitBlock = builder.createBlock(entryBlock->getParent(),
                                          ++Region::iterator(entryBlock));
    if (caseCount == 0) {
      builder.create<BranchOp>(loc, exitBlock);
      return exitBlock;
    }
    SmallVector<Block *, 4> compareBlocks;
    SmallVector<Block *, 4> caseBlocks;
    for (size_t i = 0; i < caseCount; ++i) {
      compareBlocks.push_back(builder.createBlock(exitBlock));
      caseBlocks.push_back(builder.createBlock(exitBlock));
    }
    builder.restoreInsertionPoint(ip);
    builder.create<BranchOp>(loc, compareBlocks[0]);
    for (size_t i = 0; i < caseCount; ++i) {
      auto compareBuilder = OpBuilder::atBlockBegin(compareBlocks[i]);
      auto caseValue =
          compareBuilder.createOrFold<arith::ConstantIndexOp>(loc, i);
      auto eqValue = compareBuilder.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, indexValue, caseValue);
      compareBuilder.create<CondBranchOp>(
          loc, eqValue, caseBlocks[i],
          i < caseCount - 1 ? compareBlocks[i + 1] : exitBlock);

      auto caseBuilder = OpBuilder::atBlockBegin(caseBlocks[i]);
      caseGenerator(i, caseBuilder);
      caseBuilder.create<BranchOp>(loc, exitBlock);
    }
    builder = OpBuilder::atBlockBegin(exitBlock);
    return exitBlock;
  }

  // Packs a shape into a list.
  void packShape(Location loc, Shape::RankedShapeType shapeType,
                 Value shapeValue, Value listValue, OpBuilder &builder) {
    builder.create<IREE::Util::ListResizeOp>(
        loc, listValue,
        builder.createOrFold<arith::ConstantIndexOp>(loc, shapeType.getRank()));
    for (int i = 0; i < shapeType.getRank(); ++i) {
      auto dimValue =
          builder.createOrFold<Shape::RankedDimOp>(loc, shapeValue, i);
      builder.create<IREE::Util::ListSetOp>(
          loc, listValue, builder.createOrFold<arith::ConstantIndexOp>(loc, i),
          dimValue);
    }
  }

  // Unpacks a shape from a list.
  Value unpackShape(Location loc, Shape::RankedShapeType shapeType,
                    Value listValue, OpBuilder &builder) {
    SmallVector<Value, 4> dynamicDims;
    for (int i = 0; i < shapeType.getRank(); ++i) {
      if (!shapeType.isDimDynamic(i)) continue;
      dynamicDims.push_back(builder.createOrFold<IREE::Util::ListGetOp>(
          loc, builder.getIndexType(), listValue,
          builder.createOrFold<arith::ConstantIndexOp>(loc, i)));
    }
    return builder.createOrFold<Shape::MakeRankedShapeOp>(loc, shapeType,
                                                          dynamicDims);
  }

  // Creates a function to query the |inputGlobalOps| at runtime by the
  // bindings.
  //
  // func @_query_input_shape(%index : index, %shape : !util.list<index>)
  void createQueryInputShapeFunc(Location loc, StringRef namePrefix,
                                 ArrayRef<IREE::Util::GlobalOp> inputGlobalOps,
                                 OpBuilder &moduleBuilder) {
    auto queryFuncOp = moduleBuilder.create<FuncOp>(
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
        loc, entryBlock->getArgument(0), inputGlobalOps.size(),
        [&](size_t i, OpBuilder &caseBuilder) {
          auto inputGlobalOp = inputGlobalOps[i];
          auto shapeType = inputGlobalOp.type().cast<Shape::RankedShapeType>();
          auto shapeValue = caseBuilder.createOrFold<IREE::Util::GlobalLoadOp>(
              loc, inputGlobalOp.type(), inputGlobalOp.getName());
          packShape(loc, shapeType, shapeValue, listValue, caseBuilder);
        },
        entryBuilder);

    auto exitBuilder = OpBuilder::atBlockBegin(exitBlock);
    exitBuilder.create<ReturnOp>(loc);
  }

  // Creates a function to resize |inputGlobalOps| and sets the |dirtyGlobalOp|
  // flag.
  //
  // func @_resize_input_shape(%index : index, %shape : !util.list<index>)
  void createResizeInputShapeFunc(Location loc, StringRef namePrefix,
                                  ArrayRef<IREE::Util::GlobalOp> inputGlobalOps,
                                  IREE::Util::GlobalOp dirtyGlobalOp,
                                  OpBuilder &moduleBuilder) {
    auto resizeFuncOp = moduleBuilder.create<FuncOp>(
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
        loc, entryBlock->getArgument(0), inputGlobalOps.size(),
        [&](size_t i, OpBuilder &caseBuilder) {
          auto inputGlobalOp = inputGlobalOps[i];
          auto shapeType = inputGlobalOp.type().cast<Shape::RankedShapeType>();
          auto shapeValue = unpackShape(loc, shapeType, listValue, caseBuilder);
          caseBuilder.create<IREE::Util::GlobalStoreOp>(
              loc, shapeValue, inputGlobalOp.getName());
        },
        entryBuilder);

    // Set the dirty flag so that shapes get recalculated as needed.
    auto exitBuilder = OpBuilder::atBlockBegin(exitBlock);
    auto trueValue = exitBuilder.createOrFold<arith::ConstantIntOp>(loc, 1, 1);
    exitBuilder.create<IREE::Util::GlobalStoreOp>(loc, trueValue,
                                                  dirtyGlobalOp.getName());
    exitBuilder.create<ReturnOp>(loc);
  }

  // Creates a function to query the |outputGlobalOps| at runtime by the
  // bindings.
  //
  // func @_query_output_shape(%index : index, %shape : !util.list<index>)
  void createQueryOutputShapeFunc(
      Location loc, StringRef namePrefix,
      ArrayRef<IREE::Util::GlobalOp> outputGlobalOps,
      FuncOp calculateShapeFuncOp, OpBuilder &moduleBuilder) {
    auto queryFuncOp = moduleBuilder.create<FuncOp>(
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
    entryBuilder.create<CallOp>(loc, calculateShapeFuncOp);

    auto *exitBlock = buildSwitch(
        loc, entryBlock->getArgument(0), outputGlobalOps.size(),
        [&](size_t i, OpBuilder &caseBuilder) {
          auto outputGlobalOp = outputGlobalOps[i];
          auto shapeType = outputGlobalOp.type().cast<Shape::RankedShapeType>();
          auto shapeValue = caseBuilder.createOrFold<IREE::Util::GlobalLoadOp>(
              loc, outputGlobalOp.type(), outputGlobalOp.getName());
          packShape(loc, shapeType, shapeValue, listValue, caseBuilder);
        },
        entryBuilder);

    auto exitBuilder = OpBuilder::atBlockBegin(exitBlock);
    exitBuilder.create<ReturnOp>(loc);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createMaterializeShapeSupportPass() {
  return std::make_unique<MaterializeShapeSupportPass>();
}

static PassRegistration<MaterializeShapeSupportPass> pass;

}  // namespace TFLite
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
