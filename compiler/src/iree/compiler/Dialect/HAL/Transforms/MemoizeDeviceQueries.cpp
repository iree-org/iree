// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_MEMOIZEDEVICEQUERIESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-memoize-device-queries
//===----------------------------------------------------------------------===//

// All queries for a particular !hal.device global.
struct DeviceQueries {
  // Global !hal.device.
  IREE::Util::GlobalOpInterface deviceOp;
  // [category, key, default] used for lookup/indexing.
  SmallVector<Attribute> queryKeys;
  // Ops performing queries against the device by [category, key, default].
  DenseMap<Attribute, SmallVector<IREE::HAL::DeviceQueryOp>> queryOps;
};

// A query being replaced by global lookups.
struct Query {
  Query(Location loc) : loc(loc) {}
  Location loc;
  IREE::Util::GlobalOp okGlobalOp;
  IREE::Util::GlobalOp valueGlobalOp;
  StringAttr categoryAttr;
  StringAttr keyAttr;
  TypedAttr defaultValueAttr;
};

static std::string getDeviceNamePrefix(IREE::Util::GlobalOpInterface deviceOp) {
  StringRef deviceName = deviceOp.getGlobalName().getValue();
  if (deviceName.starts_with("__")) {
    return deviceName.str();
  }
  auto prefixedName = "__" + deviceName;
  return prefixedName.str();
}

struct MemoizeDeviceQueriesPass
    : public IREE::HAL::impl::MemoizeDeviceQueriesPassBase<
          MemoizeDeviceQueriesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Analyze the module to determine which devices are used where.
    DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    // Prepare device table indexed by symbol name.
    DenseMap<Attribute, DeviceQueries> allDeviceQueries;
    for (auto deviceOp : deviceAnalysis.getDeviceGlobals()) {
      allDeviceQueries[deviceOp.getGlobalName()].deviceOp = deviceOp;
    }

    // Find all query ops we want to memoize and group them together.
    // This lets us easily replace all usages of a match with a single variable.
    for (auto callableOp : moduleOp.getOps<mlir::CallableOpInterface>()) {
      callableOp.walk([&](IREE::HAL::DeviceQueryOp queryOp) {
        // Try to find the device this query is made on. If analysis failed then
        // we can't memoize the query today.
        auto deviceGlobals =
            deviceAnalysis.lookupDeviceGlobals(queryOp.getDevice());
        if (!deviceGlobals || deviceGlobals->size() != 1)
          return WalkResult::advance();
        IREE::Util::GlobalOpInterface deviceGlobalOp = deviceGlobals->front();

        // Construct key used to dedupe/lookup the query.
        auto fullKey = ArrayAttr::get(
            moduleOp.getContext(),
            {
                StringAttr::get(moduleOp.getContext(),
                                queryOp.getCategory() + queryOp.getKey()),
                queryOp.getDefaultValue().has_value()
                    ? queryOp.getDefaultValueAttr()
                    : Attribute{},
            });

        // Track the query on the device.
        auto &deviceQueries = allDeviceQueries[deviceGlobalOp.getGlobalName()];
        auto lookup = deviceQueries.queryOps.try_emplace(
            fullKey, SmallVector<IREE::HAL::DeviceQueryOp>{});
        if (lookup.second) {
          deviceQueries.queryKeys.push_back(std::move(fullKey));
        }
        lookup.first->second.push_back(queryOp);

        return WalkResult::advance();
      });
    }

    // Create each query variable and replace the uses with loads.
    SymbolTable symbolTable(moduleOp);
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    for (auto deviceOp : deviceAnalysis.getDeviceGlobals()) {
      auto &deviceQueries = allDeviceQueries[deviceOp.getGlobalName()];
      if (deviceQueries.queryKeys.empty()) {
        // No queries against this device.
        continue;
      }

      // Create one global per unique query key against the device.
      SmallVector<Query> queries;
      moduleBuilder.setInsertionPointAfter(deviceOp);
      for (auto [i, queryKey] : llvm::enumerate(deviceQueries.queryKeys)) {
        auto &queryOps = deviceQueries.queryOps[queryKey];
        auto queryLoc = moduleBuilder.getFusedLoc(llvm::map_to_vector(
            queryOps, [&](auto queryOp) { return queryOp.getLoc(); }));

        // Create a global for the ok flag and the queried value.
        // TODO(benvanik): create a better name based on the key.
        auto anyQueryOp = queryOps.front();
        auto queryType = anyQueryOp.getValue().getType();
        std::string variableName =
            getDeviceNamePrefix(deviceOp) + "_query_" + std::to_string(i) +
            "_" + sanitizeSymbolName(anyQueryOp.getCategory()) + "_" +
            sanitizeSymbolName(anyQueryOp.getKey());
        auto okGlobalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
            queryLoc, variableName + "_ok",
            /*isMutable=*/false, moduleBuilder.getI1Type());
        symbolTable.insert(okGlobalOp);
        okGlobalOp.setPrivate();
        auto valueGlobalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
            queryLoc, variableName,
            /*isMutable=*/false, queryType);
        symbolTable.insert(valueGlobalOp);
        valueGlobalOp.setPrivate();

        // Stash the globals for initialization.
        Query query(queryLoc);
        query.okGlobalOp = okGlobalOp;
        query.valueGlobalOp = valueGlobalOp;
        query.categoryAttr = anyQueryOp.getCategoryAttr();
        query.keyAttr = anyQueryOp.getKeyAttr();
        query.defaultValueAttr = anyQueryOp.getDefaultValueAttr();
        queries.push_back(query);

        // Replace all queries with loads of the global values.
        for (auto queryOp : queryOps) {
          OpBuilder replaceBuilder(queryOp);
          auto okLoadOp =
              okGlobalOp.createLoadOp(queryOp.getLoc(), replaceBuilder);
          auto resultLoadOp =
              valueGlobalOp.createLoadOp(queryOp.getLoc(), replaceBuilder);
          queryOp.replaceAllUsesWith(ValueRange{
              okLoadOp.getLoadedGlobalValue(),
              resultLoadOp.getLoadedGlobalValue(),
          });
          queryOp.erase();
        }
      }

      // Create an initializer for the device where we will perform all queries.
      auto fusedLoc = moduleBuilder.getFusedLoc(
          llvm::map_to_vector(queries, [&](auto &query) { return query.loc; }));
      auto initializerOp =
          moduleBuilder.create<IREE::Util::InitializerOp>(fusedLoc);
      auto funcBuilder = OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
      Value device =
          deviceOp.createLoadOp(fusedLoc, funcBuilder).getLoadedGlobalValue();
      for (auto [i, queryKey] : llvm::enumerate(deviceQueries.queryKeys)) {
        auto &query = queries[i];
        auto queryOp = funcBuilder.create<IREE::HAL::DeviceQueryOp>(
            fusedLoc, funcBuilder.getI1Type(),
            query.valueGlobalOp.getGlobalType(), device, query.categoryAttr,
            query.keyAttr, query.defaultValueAttr);
        query.okGlobalOp.createStoreOp(fusedLoc, queryOp.getOk(), funcBuilder);
        query.valueGlobalOp.createStoreOp(fusedLoc, queryOp.getValue(),
                                          funcBuilder);
      }
      funcBuilder.create<IREE::Util::ReturnOp>(fusedLoc);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
