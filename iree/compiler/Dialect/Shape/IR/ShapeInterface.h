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

#ifndef IREE_COMPILER_DIALECT_SHAPE_IR_SHAPEINTERFACE_H_
#define IREE_COMPILER_DIALECT_SHAPE_IR_SHAPEINTERFACE_H_

#include <vector>

#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// Plugin for materializing calculations for custom op shape calculations.
// The intended purpose of this is as an escape hatch for hard to represent
// ops.
class CustomOpShapeBuilder {
 public:
  virtual ~CustomOpShapeBuilder() = default;

  // Builds the computation for computing the given result RankedShapeType,
  // given that 'inputOperation' is responsible for defining this result.
  virtual Value buildRankedShape(RankedShapeType resultShape,
                                 Operation *inputOperation,
                                 OpBuilder &builder) {
    return nullptr;
  }
};

class CustomOpShapeBuilderList {
  using BuilderListTy = std::vector<std::unique_ptr<CustomOpShapeBuilder>>;

 public:
  BuilderListTy::iterator begin() { return builders.begin(); }
  BuilderListTy::iterator end() { return builders.end(); }
  BuilderListTy::const_iterator begin() const { return builders.begin(); }
  BuilderListTy::const_iterator end() const { return builders.end(); }

  void insert(std::unique_ptr<CustomOpShapeBuilder> b) {
    builders.push_back(std::move(b));
  }

  template <typename BuilderTy, typename... ConstructorArgs>
  // TODO(suderman): Re-enable clang-format when new version migrates.
  // clang-format off
  BuilderTy &make(ConstructorArgs &&...args) {
    // clang-format on
    auto instance =
        std::make_unique<BuilderTy>(std::forward<ConstructorArgs>(args)...);
    BuilderTy *unowned = instance.get();
    builders.push_back(std::move(instance));
    return *unowned;
  }

 private:
  BuilderListTy builders;
};

// An implementation of CustomOpShapeBuilder which accepts pattern-match
// callback based on the 'inputOperation' name.
class CallbackCustomOpShapeBuilder : public CustomOpShapeBuilder {
 public:
  using RankedShapeBuilder =
      std::function<Value(RankedShapeType resultShape,
                          Operation *inputOperation, OpBuilder &builder)>;

  // Inserts a callback to handle the given operation name. Only the last
  // callback for an operation is retained.
  void insertRankedShapeBuilder(llvm::StringRef operationName,
                                RankedShapeBuilder callback);

  // Convenience for specifying a typed builder given an Op class.
  template <typename OpTy>
  void insertOpRankedShapeBuilder(
      std::function<Value(RankedShapeType, OpTy op, OpBuilder &builder)>
          callback) {
    insertRankedShapeBuilder(
        OpTy::getOperationName(),
        [callback](RankedShapeType resultShape, Operation *inputOperation,
                   OpBuilder &builder) {
          return callback(resultShape, llvm::cast<OpTy>(inputOperation),
                          builder);
        });
  }

  Value buildRankedShape(RankedShapeType resultShape, Operation *inputOperation,
                         OpBuilder &builder) final;

 private:
  llvm::StringMap<RankedShapeBuilder> rankedShapeBuilders;
};

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_IR_SHAPEINTERFACE_H_
