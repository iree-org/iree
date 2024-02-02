// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/instruments/dispatch.h"
#include "iree/schemas/instruments/dispatch_def_builder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_MATERIALIZEDISPATCHINSTRUMENTATIONPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

static std::string getAttrStr(Attribute attr) {
  if (!attr)
    return "";
  std::string result;
  llvm::raw_string_ostream os(result);
  attr.print(os, /*elideType=*/true);
  return result;
}

static std::string getOpStr(Operation *op) {
  std::string result;
  llvm::raw_string_ostream os(result);
  OpPrintingFlags flags;
  flags.useLocalScope();
  flags.assumeVerified();
  op->print(os, flags);
  return result;
}

// Returns a data vector containing a iree_idbts_chunk_header_t with |type|.
// The declared |contentLength| excludes padding.
static Value createChunkHeader(Location loc, iree_idbts_chunk_type_t type,
                               uint64_t contentLength, OpBuilder &builder) {
  iree_idbts_chunk_header_t header;
  header.magic = IREE_IDBTS_CHUNK_MAGIC;
  header.type = type;
  header.version = 0;
  header.content_length = contentLength;

  auto dataAttr = DenseElementsAttr::getFromRawBuffer(
      VectorType::get({sizeof(header)}, builder.getI8Type()),
      ArrayRef<char>(reinterpret_cast<const char *>(&header), sizeof(header)));

  return builder.create<IREE::Util::BufferConstantOp>(
      loc, /*name=*/nullptr, dataAttr, builder.getIndexAttr(16),
      /*mimeType=*/nullptr);
}

// Returns a zero padding vector if |unalignedLength| needs alignment or null.
static Value createPadding(Location loc, uint64_t unalignedLength,
                           OpBuilder &builder) {
  uint64_t padding = llvm::alignTo(unalignedLength, 16) - unalignedLength;
  if (!padding)
    return nullptr;
  auto i8Type = builder.getI8Type();
  auto zeroAttr = IntegerAttr::get(i8Type, 0);
  auto dataAttr = DenseElementsAttr::get(
      VectorType::get({(int64_t)padding}, i8Type), zeroAttr);
  return builder.create<IREE::Util::BufferConstantOp>(
      loc, /*name=*/nullptr, dataAttr, builder.getIndexAttr(16),
      /*mimeType=*/nullptr);
}

static void appendListItems(Location loc, Value list, ArrayRef<Value> items,
                            OpBuilder &builder) {
  Value oldLength = builder.create<IREE::Util::ListSizeOp>(loc, list);
  Value newLength = builder.create<arith::AddIOp>(
      loc, oldLength,
      builder.create<arith::ConstantIndexOp>(loc, items.size()));
  builder.create<IREE::Util::ListResizeOp>(loc, list, newLength);
  for (size_t i = 0; i < items.size(); ++i) {
    Value idx = builder.create<arith::AddIOp>(
        loc, oldLength, builder.create<arith::ConstantIndexOp>(loc, i));
    builder.create<IREE::Util::ListSetOp>(loc, list, idx, items[i]);
  }
}

//===----------------------------------------------------------------------===//
// --iree-hal-materialize-dispatch-instrumentation
//===----------------------------------------------------------------------===//

struct MaterializeDispatchInstrumentationPass
    : public IREE::HAL::impl::MaterializeDispatchInstrumentationPassBase<
          MaterializeDispatchInstrumentationPass> {
  using IREE::HAL::impl::MaterializeDispatchInstrumentationPassBase<
      MaterializeDispatchInstrumentationPass>::
      MaterializeDispatchInstrumentationPassBase;
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;

    auto moduleBuilder = OpBuilder(&moduleOp.getBody()->front());
    auto i8Type = moduleBuilder.getI8Type();
    auto i32Type = moduleBuilder.getI32Type();
    auto indexType = moduleBuilder.getIndexType();

    // Used for all instrumentation.
    auto loc = moduleBuilder.getUnknownLoc();

    // Currently statically sized to avoid disturbing too much by adding
    // additional arguments to dispatches that need to be marshaled.
    // We need to use the base power of two size for storage then add some
    // padding for overflows and the write head location.
    //
    // [power of two storage buffer]
    // [56 bytes of padding, may get overflow data]
    // [8 bytes of write head]
    int64_t totalBufferSize =
        bufferSize.value + IREE_INSTRUMENT_DISPATCH_PADDING;
    auto bufferSizeAttr = moduleBuilder.getIndexAttr(totalBufferSize);
    auto bufferType = MemRefType::get({totalBufferSize}, i8Type);

    // Create global device-side instrumentation resource.
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, "__dispatch_instrumentation",
        /*isMutable=*/false,
        moduleBuilder.getType<IREE::Stream::ResourceType>(
            IREE::Stream::Lifetime::External));
    {
      auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
      auto initializerBuilder =
          OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
      Value bufferSize =
          initializerBuilder.create<arith::ConstantOp>(loc, bufferSizeAttr);
      Value buffer = initializerBuilder.create<IREE::Stream::ResourceAllocOp>(
          loc, globalOp.getType(), bufferSize,
          /*uninitialized=*/true, /*affinity=*/nullptr);
      globalOp.createStoreOp(loc, buffer, initializerBuilder);
      initializerBuilder.create<IREE::Util::ReturnOp>(loc);
    }

    FlatbufferBuilder metadataBuilder;

    // Update all executable export signatures to include the instrumentation
    // binding. We don't actually use it yet but ensure that it's available
    // during translation. We keep track of which exports we instrument as some
    // may be external declarations we can't modify.
    SmallVector<iree_instruments_DispatchFunctionDef_ref_t>
        dispatchFunctionRefs;
    DenseMap<SymbolRefAttr, uint32_t> instrumentedExports;
    auto bindingType = moduleBuilder.getType<IREE::Stream::BindingType>();
    auto alignmentKey = moduleBuilder.getStringAttr("stream.alignment");
    auto alignment64 = moduleBuilder.getIndexAttr(64);
    for (auto executableOp : moduleOp.getOps<IREE::Stream::ExecutableOp>()) {
      for (auto exportOp :
           executableOp.getOps<IREE::Stream::ExecutableExportOp>()) {
        auto funcOp = exportOp.lookupFunctionRef();
        if (!funcOp)
          continue;

        // Capture the source before we mess with it.
        auto originalSource = getOpStr(funcOp);

        // Mark as instrumented.
        instrumentedExports[SymbolRefAttr::get(
            executableOp.getNameAttr(), {SymbolRefAttr::get(exportOp)})] =
            instrumentedExports.size();

        // Update function signature to add the ringbuffer and dispatch ID.
        SmallVector<Type> argTypes(funcOp.getArgumentTypes());
        argTypes.push_back(bindingType);
        argTypes.push_back(i32Type);
        funcOp.setType(FunctionType::get(&getContext(), argTypes,
                                         funcOp.getResultTypes()));
        auto bindingArg = funcOp.front().addArgument(bindingType, loc);
        auto dispatchIdArg = funcOp.front().addArgument(i32Type, loc);
        funcOp.setArgAttrs(
            bindingArg.getArgNumber(),
            DictionaryAttr::get(&getContext(),
                                {NamedAttribute(alignmentKey, alignment64)}));

        // Insert the workgroup instrumentation. Note that this happens before
        // codegen would do tile-and-distribute, but that's ok as it should just
        // ignore these ops when doing that. This only works because we aren't
        // capturing workgroup ID yet with this instrumentation op and instead
        // leave that until late in the codegen pipeline.
        auto funcBuilder = OpBuilder::atBlockBegin(&funcOp.front());
        Value zero = funcBuilder.create<arith::ConstantIndexOp>(loc, 0);
        auto subspanOp = funcBuilder.create<IREE::Stream::BindingSubspanOp>(
            loc, bufferType, bindingArg, /*byteOffset=*/zero, ValueRange{});
        funcBuilder.create<IREE::HAL::InstrumentWorkgroupOp>(
            loc, indexType, subspanOp.getResult(), dispatchIdArg);

        // Build function metadata.
        auto nameRef = metadataBuilder.createString(exportOp.getName());
        auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(exportOp);
        auto targetRef = metadataBuilder.createString(getAttrStr(targetAttr));
        auto sourceRef = metadataBuilder.createString(originalSource);
        iree_instruments_DispatchFunctionDef_start(metadataBuilder);
        iree_instruments_DispatchFunctionDef_name_add(metadataBuilder, nameRef);
        iree_instruments_DispatchFunctionDef_target_add(metadataBuilder,
                                                        targetRef);
        iree_instruments_DispatchFunctionDef_source_add(metadataBuilder,
                                                        sourceRef);
        dispatchFunctionRefs.push_back(
            iree_instruments_DispatchFunctionDef_end(metadataBuilder));
      }
    }

    // Find all dispatches to exports that we've instrumented and pass along the
    // instrumentation buffer.
    SmallVector<iree_instruments_DispatchSiteDef_ref_t> dispatchSiteRefs;
    uint32_t dispatchSiteCount = 0;
    for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
      funcOp.walk([&](IREE::Stream::CmdExecuteOp executeOp) {
        auto parentBuilder = OpBuilder(executeOp);

        // Load the ringbuffer and capture it for use within the execute region.
        auto loadedValue =
            globalOp.createLoadOp(loc, parentBuilder).getLoadedGlobalValue();
        Value zero = parentBuilder.create<arith::ConstantIndexOp>(loc, 0);
        Value bufferSize =
            parentBuilder.create<arith::ConstantOp>(loc, bufferSizeAttr);
        executeOp.getResourceOperandsMutable().append(loadedValue);
        executeOp.getResourceOperandSizesMutable().append(bufferSize);
        auto bufferArg =
            executeOp.getBody().addArgument(loadedValue.getType(), loc);

        // Walk dispatches and pass them the ringbuffer and their unique ID.
        executeOp.walk([&](IREE::Stream::CmdDispatchOp dispatchOp) {
          // NOTE: we just choose the first instrumented export for attribution
          // as that's good enough for all current use cases. If we start
          // specializing really early we may want to fix that.
          std::optional<uint32_t> functionId;
          for (auto entryPointAttr : dispatchOp.getEntryPointRefs()) {
            auto it = instrumentedExports.find(entryPointAttr);
            if (it != instrumentedExports.end()) {
              // Found the first instrumented export.
              functionId = it->second;
              break;
            }
          }
          if (!functionId)
            return; // not instrumented

          // Append dispatch site ID to correlate this op with where it lives in
          // the program and what is being dispatched. Note that multiple
          // dispatch ops may reference the same dispatch function after
          // deduplication.
          uint32_t dispatchSiteId = dispatchSiteCount++;
          dispatchOp.getUniformOperandsMutable().append(
              parentBuilder
                  .create<arith::ConstantIntOp>(loc, dispatchSiteId, 32)
                  .getResult());

          // Record dispatch site to the host-side metadata.
          iree_instruments_DispatchSiteDef_start(metadataBuilder);
          // TODO(benvanik): source loc to identify the site.
          iree_instruments_DispatchSiteDef_function_add(metadataBuilder,
                                                        *functionId);
          dispatchSiteRefs.push_back(
              iree_instruments_DispatchSiteDef_end(metadataBuilder));

          // Append ringbuffer for storing the instrumentation data.
          dispatchOp.getResourcesMutable().append(bufferArg);
          dispatchOp.getResourceOffsetsMutable().append(zero);
          dispatchOp.getResourceLengthsMutable().append(bufferSize);
          dispatchOp.getResourceSizesMutable().append(bufferSize);
          SmallVector<Attribute> accesses(
              dispatchOp.getResourceAccesses().getValue());
          accesses.push_back(IREE::Stream::ResourceAccessBitfieldAttr::get(
              &getContext(), IREE::Stream::ResourceAccessBitfield::Read |
                                 IREE::Stream::ResourceAccessBitfield::Write));
          dispatchOp.setResourceAccessesAttr(
              parentBuilder.getArrayAttr(accesses));
        });
      });
    }

    auto dispatchFunctionsRef = iree_instruments_DispatchFunctionDef_vec_create(
        metadataBuilder, dispatchFunctionRefs.data(),
        dispatchFunctionRefs.size());
    auto dispatchSitesRef = iree_instruments_DispatchSiteDef_vec_create(
        metadataBuilder, dispatchSiteRefs.data(), dispatchSiteRefs.size());
    iree_instruments_DispatchInstrumentDef_start_as_root(metadataBuilder);
    iree_instruments_DispatchInstrumentDef_version_add(metadataBuilder, 0);
    iree_instruments_DispatchInstrumentDef_flags_add(metadataBuilder, 0);
    iree_instruments_DispatchInstrumentDef_functions_add(metadataBuilder,
                                                         dispatchFunctionsRef);
    iree_instruments_DispatchInstrumentDef_sites_add(metadataBuilder,
                                                     dispatchSitesRef);
    iree_instruments_DispatchInstrumentDef_end_as_root(metadataBuilder);
    auto metadataAttr = metadataBuilder.getBufferAttr(&getContext());

    // Create query function for getting the instrumentation data.
    auto listType = moduleBuilder.getType<IREE::Util::ListType>(
        moduleBuilder.getType<IREE::Util::VariantType>());
    auto queryOp = moduleBuilder.create<func::FuncOp>(
        loc, "__query_instruments",
        moduleBuilder.getFunctionType({listType}, {}));
    {
      queryOp.setPublic();
      auto *entryBlock = queryOp.addEntryBlock();
      auto queryBuilder = OpBuilder::atBlockBegin(entryBlock);
      auto listArg = entryBlock->getArgument(0);

      SmallVector<Value> iovecs;

      iovecs.push_back(
          createChunkHeader(loc, IREE_IDBTS_CHUNK_TYPE_DISPATCH_METADATA,
                            metadataAttr.size(), queryBuilder));

      // Grab the read-only dispatch metadata.
      iovecs.push_back(queryBuilder.create<IREE::Util::BufferConstantOp>(
          loc, queryBuilder.getStringAttr("dispatch_instrument.fb"),
          metadataAttr, queryBuilder.getIndexAttr(16),
          queryBuilder.getStringAttr("application/x-flatbuffers")));

      if (Value metadataPadding =
              createPadding(loc, metadataAttr.size(), queryBuilder)) {
        iovecs.push_back(metadataPadding);
      }

      iovecs.push_back(
          createChunkHeader(loc, IREE_IDBTS_CHUNK_TYPE_DISPATCH_RINGBUFFER,
                            totalBufferSize, queryBuilder));

      // Export the device buffer containing the instrument data.
      Value buffer =
          globalOp.createLoadOp(loc, queryBuilder).getLoadedGlobalValue();
      Value bufferSize =
          queryBuilder.create<arith::ConstantOp>(loc, bufferSizeAttr);
      auto bufferViewType = moduleBuilder.getType<IREE::HAL::BufferViewType>();
      auto exportOp = queryBuilder.create<IREE::Stream::TensorExportOp>(
          loc, bufferViewType, buffer,
          RankedTensorType::get({totalBufferSize}, queryBuilder.getI8Type()),
          ValueRange{}, bufferSize,
          /*affinity=*/nullptr);
      iovecs.push_back(exportOp.getResult());

      if (Value ringbufferPadding =
              createPadding(loc, totalBufferSize, queryBuilder)) {
        iovecs.push_back(ringbufferPadding);
      }

      appendListItems(loc, listArg, iovecs, queryBuilder);
      queryBuilder.create<func::ReturnOp>(loc);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
