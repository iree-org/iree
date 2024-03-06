// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/DialectImplementation.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Stream/IR/StreamAttrs.cpp.inc" // IWYU pragma: keep
#include "iree/compiler/Dialect/Stream/IR/StreamEnums.cpp.inc" // IWYU pragma: keep
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::Stream {

static llvm::cl::opt<Favor> clPartitioningFavor(
    "iree-stream-partitioning-favor",
    llvm::cl::desc("Default stream partitioning favor configuration."),
    llvm::cl::init(Favor::MinPeakMemory),
    llvm::cl::values(
        clEnumValN(Favor::Debug, "debug",
                   "Force debug partitioning (no concurrency or pipelining)."),
        clEnumValN(Favor::MinPeakMemory, "min-peak-memory",
                   "Favor minimizing memory consumption at the cost of "
                   "additional concurrency."),
        clEnumValN(Favor::MaxConcurrency, "max-concurrency",
                   "Favor maximizing concurrency at the cost of additional "
                   "memory consumption.")));

// TODO(#8042): properly choose this value based on target devices. We don't
// yet have the device information up in stream and thus for targets that have
// high alignment requirements (128/256/etc) we are not picking the right
// value here. Having the max sizes set to 64-bit also creates a grey area
// where a program may not have any particular resource that needs 64-bit
// but may trip the limit when packing. For now if a program is on the edge
// it'll result in runtime failures if the max sizes are smaller than INT_MAX.
static llvm::cl::opt<uint64_t> clResourceMaxAllocationSize(
    "iree-stream-resource-max-allocation-size",
    llvm::cl::desc("Maximum size of an individual memory allocation."),
    llvm::cl::init(INT64_MAX));
static llvm::cl::opt<uint64_t> clResourceMinOffsetAlignment(
    "iree-stream-resource-min-offset-alignment",
    llvm::cl::desc("Minimum required alignment in bytes for resource offsets."),
    llvm::cl::init(64ull));
static llvm::cl::opt<uint64_t> clResourceMaxRange(
    "iree-stream-resource-max-range",
    llvm::cl::desc("Maximum range of a resource binding; may be less than the "
                   "max allocation size."),
    llvm::cl::init(INT64_MAX));
static llvm::cl::opt<unsigned> clResourceIndexBits(
    "iree-stream-resource-index-bits",
    llvm::cl::desc("Bit width of indices used to reference resource offsets."),
    llvm::cl::init(64));
static llvm::cl::opt<bool> clResourceAliasMutableBindings(
    "iree-stream-resource-alias-mutable-bindings",
    llvm::cl::desc(
        "Fuses bindings that are mutable instead of leaving them split."),
    llvm::cl::init(false));
// TODO(#15522): Change this to discrete once task system scalability limits
// are corrected.
static llvm::cl::opt<IREE::Stream::MemoryModel> clResourceMemoryModel(
    "iree-stream-resource-memory-model",
    llvm::cl::desc("Memory model used for host-device resource memory access."),
    llvm::cl::values(
        clEnumValN(IREE::Stream::MemoryModel::Unified, "unified",
                   "Host and device memory are unified and there's "
                   "(practically) no performance cost for cross-access."),
        clEnumValN(IREE::Stream::MemoryModel::Discrete, "discrete",
                   "Host and device memory are discrete and cross-access is "
                   "expensive.")),
    llvm::cl::init(IREE::Stream::MemoryModel::Unified));

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

void AsyncAccessRange::print(llvm::raw_ostream &os, AsmState &asmState) {
  os << stringifyResourceAccessBitfield(access) << " ";
  resource.printAsOperand(os, asmState);
  os << "[";
  start.printAsOperand(os, asmState);
  os << " to ";
  end.printAsOperand(os, asmState);
  os << " for ";
  length.printAsOperand(os, asmState);
  os << "]";
}

// static
bool AsyncAccessRange::mayOverlap(const AsyncAccessRange &lhs,
                                  const AsyncAccessRange &rhs) {
  // Different resources do not overlap for this purpose. They may still alias
  // at various points but that's beyond the analysis we can do here.
  if (lhs.resource != rhs.resource)
    return false;

  // Check for adjacent but not overlapping.
  if (lhs.end == rhs.start || lhs.start == rhs.end) {
    return false;
  }

  // _May_ overlap. More analysis required.
  return true;
}

//===----------------------------------------------------------------------===//
// custom<ParameterReference>($scope, $key)
//===----------------------------------------------------------------------===//

ParseResult parseParameterReference(AsmParser &parser, StringAttr &scopeAttr,
                                    StringAttr &keyAttr) {
  auto builder = parser.getBuilder();
  StringAttr firstAttr;
  if (failed(parser.parseCustomAttributeWithFallback(firstAttr,
                                                     builder.getNoneType()))) {
    return failure();
  }
  if (failed(parser.parseOptionalColon())) {
    keyAttr = firstAttr;
    return success();
  }
  scopeAttr = firstAttr;
  if (failed(parser.parseColon()) ||
      failed(parser.parseCustomAttributeWithFallback(keyAttr,
                                                     builder.getNoneType()))) {
    return failure();
  }
  return success();
}

void printParameterReference(AsmPrinter &p, StringAttr scopeAttr,
                             StringAttr keyAttr) {
  if (scopeAttr) {
    p << "\"" << scopeAttr.getValue() << "\"";
    p << "::";
  }
  p << "\"" << keyAttr.getValue() << "\"";
}

//===----------------------------------------------------------------------===//
// #stream.resource_config<...>
//===----------------------------------------------------------------------===//

// static
Attribute ResourceConfigAttr::parse(AsmParser &p, Type type) {
  if (failed(p.parseLess()) || failed(p.parseLBrace()))
    return {};

  int64_t maxAllocationSize = 0;
  int64_t minBufferOffsetAlignment = 0;
  int64_t maxBufferRange = 0;
  int64_t minBufferRangeAlignment = 0;
  int64_t indexBits = 32;
  bool aliasMutableBindings = false;
  auto memoryModel = IREE::Stream::MemoryModel::Discrete;
  while (failed(p.parseOptionalRBrace())) {
    StringRef key;
    if (failed(p.parseKeyword(&key)) || failed(p.parseEqual())) {
      return {};
    }
    if (key == "max_allocation_size") {
      if (failed(p.parseInteger(maxAllocationSize)))
        return {};
    } else if (key == "min_buffer_offset_alignment") {
      if (failed(p.parseInteger(minBufferOffsetAlignment)))
        return {};
    } else if (key == "max_buffer_range") {
      if (failed(p.parseInteger(maxBufferRange)))
        return {};
    } else if (key == "min_buffer_range_alignment") {
      if (failed(p.parseInteger(minBufferRangeAlignment)))
        return {};
    } else if (key == "index_bits") {
      if (failed(p.parseInteger(indexBits)))
        return {};
    } else if (key == "alias_mutable_bindings") {
      StringRef value;
      if (failed(p.parseKeyword(&value)))
        return {};
      if (value == "true")
        aliasMutableBindings = true;
      else if (value == "false")
        aliasMutableBindings = false;
      else
        return {};
    } else if (key == "memory_model") {
      StringRef value;
      if (failed(p.parseKeyword(&value)))
        return {};
      auto enumValue = symbolizeMemoryModel(value);
      if (!enumValue.has_value())
        return {};
      memoryModel = enumValue.value();
    }
    (void)p.parseOptionalComma();
  }
  if (failed(p.parseGreater()))
    return {};

  return ResourceConfigAttr::get(p.getContext(), maxAllocationSize,
                                 minBufferOffsetAlignment, maxBufferRange,
                                 minBufferRangeAlignment, indexBits,
                                 aliasMutableBindings, memoryModel);
}

void ResourceConfigAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<{";
  os << "max_allocation_size = " << getMaxAllocationSize() << ", ";
  os << "min_buffer_offset_alignment = " << getMinBufferOffsetAlignment()
     << ", ";
  os << "max_buffer_range = " << getMaxBufferRange() << ", ";
  os << "min_buffer_range_alignment = " << getMinBufferRangeAlignment() << ", ";
  os << "index_bits = " << getIndexBits() << ", ";
  os << "alias_mutable_bindings = " << getAliasMutableBindings() << ", ";
  os << "memory_model = " << stringifyMemoryModel(getMemoryModel());
  os << "}>";
}

// static
ResourceConfigAttr
ResourceConfigAttr::intersectBufferConstraints(ResourceConfigAttr lhs,
                                               ResourceConfigAttr rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  Builder b(lhs.getContext());
  return ResourceConfigAttr::get(
      b.getContext(),
      std::min(lhs.getMaxAllocationSize(), rhs.getMaxAllocationSize()),
      std::max(lhs.getMinBufferOffsetAlignment(),
               rhs.getMinBufferOffsetAlignment()),
      std::min(lhs.getMaxBufferRange(), rhs.getMaxBufferRange()),
      std::max(lhs.getMinBufferRangeAlignment(),
               rhs.getMinBufferRangeAlignment()),
      std::max(lhs.getIndexBits(), rhs.getIndexBits()),
      rhs.getAliasMutableBindings() && lhs.getAliasMutableBindings(),
      (lhs.getMemoryModel() == IREE::Stream::MemoryModel::Unified &&
       rhs.getMemoryModel() == IREE::Stream::MemoryModel::Unified)
          ? IREE::Stream::MemoryModel::Unified
          : IREE::Stream::MemoryModel::Discrete);
}

// static
ResourceConfigAttr
ResourceConfigAttr::getDefaultHostConstraints(MLIRContext *context) {
  // Picked to represent what we kind of want on CPU today.
  // We should be able to get rid of queries for this from real programs and
  // only use this during testing by ensuring affinities are always assigned.
  return ResourceConfigAttr::get(
      context, clResourceMaxAllocationSize, clResourceMinOffsetAlignment,
      clResourceMaxRange, clResourceMinOffsetAlignment, clResourceIndexBits,
      clResourceAliasMutableBindings, clResourceMemoryModel);
}

// static
ResourceConfigAttr ResourceConfigAttr::lookup(Operation *op) {
  auto *context = op->getContext();
  auto attrId = StringAttr::get(context, "stream.resources");
  while (op) {
    // Use an override if specified.
    auto attr = op->getAttrOfType<ResourceConfigAttr>(attrId);
    if (attr)
      return attr;
    // See if the affinity specified provides a resource configuration.
    if (auto affinityOp = llvm::dyn_cast<AffinityOpInterface>(op)) {
      auto affinityAttr = affinityOp.getAffinity();
      if (affinityAttr) {
        auto attr = affinityAttr.getResourceConfigAttr();
        if (attr)
          return attr;
      }
    }
    op = op->getParentOp();
  }
  // No config found; use conservative host config.
  return getDefaultHostConstraints(context);
}

//===----------------------------------------------------------------------===//
// #stream.parameter.named<...>
//===----------------------------------------------------------------------===//

int64_t NamedParameterAttr::getStorageSize() const {
  if (auto configAttr = getConfig()) {
    if (auto lengthAttr = configAttr.getAs<IntegerAttr>("length")) {
      return lengthAttr.getInt();
    }
  }
  if (auto shapedType = getType().dyn_cast<ShapedType>()) {
    return IREE::Util::getRoundedPhysicalStorageSize(shapedType);
  } else {
    return IREE::Util::getTypePhysicalStorageBitWidth(getType());
  }
}

//===----------------------------------------------------------------------===//
// #stream.timepoint<...>
//===----------------------------------------------------------------------===//

Attribute TimepointAttr::parse(AsmParser &p, Type type) {
  StringRef timeStr;
  if (failed(p.parseLess()))
    return {};
  if (failed(p.parseKeyword(&timeStr))) {
    return {};
  }
  if (failed(p.parseGreater()))
    return {};
  if (timeStr != "immediate") {
    p.emitError(p.getCurrentLocation(),
                "only immediate timepoint attrs are supported");
    return {};
  }
  return TimepointAttr::get(p.getContext(), TimepointType::get(p.getContext()));
}

void TimepointAttr::print(AsmPrinter &p) const {
  p << "<";
  p << "immediate";
  p << ">";
}

//===----------------------------------------------------------------------===//
// #stream.affinity
//===----------------------------------------------------------------------===//

AffinityAttr AffinityAttr::lookup(Operation *op) {
  auto attrId = StringAttr::get(op->getContext(), "stream.affinity");
  while (op) {
    if (auto affinityOp = llvm::dyn_cast<AffinityOpInterface>(op)) {
      auto affinity = affinityOp.getAffinity();
      if (affinity)
        return affinity;
    }
    auto attr = op->getAttrOfType<AffinityAttr>(attrId);
    if (attr)
      return attr;
    op = op->getParentOp();
  }
  return {}; // No affinity found; let caller decide what to do.
}

// static
bool AffinityAttr::areCompatible(AffinityAttr desiredAffinity,
                                 AffinityAttr requiredAffinity) {
  if (desiredAffinity == requiredAffinity)
    return true;
  if ((desiredAffinity && !requiredAffinity) ||
      (requiredAffinity && !desiredAffinity)) {
    return true;
  }
  // We could do a fuzzier match here (interface isCompatible() etc).
  return false;
}

// static
bool AffinityAttr::canExecuteTogether(AffinityAttr lhs, AffinityAttr rhs) {
  if (lhs == rhs)
    return true;
  if ((lhs && !rhs) || (rhs && !lhs))
    return true;
  return lhs.isExecutableWith(rhs);
}

//===----------------------------------------------------------------------===//
// #stream.partitioning_config
//===----------------------------------------------------------------------===//

Attribute PartitioningConfigAttr::parse(AsmParser &p, Type type) {
  std::string favorStr;
  if (failed(p.parseLess()))
    return {};
  if (succeeded(p.parseOptionalStar())) {
    favorStr = "size";
  } else if (failed(p.parseString(&favorStr))) {
    return {};
  }
  if (failed(p.parseGreater()))
    return {};
  auto favor = symbolizeFavor(favorStr);
  if (!favor.has_value()) {
    p.emitError(p.getNameLoc(), "unknown favor value: ") << favorStr;
    return {};
  }
  return PartitioningConfigAttr::get(
      FavorAttr::get(p.getContext(), favor.value()));
}

void PartitioningConfigAttr::print(AsmPrinter &p) const {
  p << "<";
  p << "favor-";
  p << stringifyFavor(getFavor().getValue());
  p << ">";
}

PartitioningConfigAttr PartitioningConfigAttr::lookup(Operation *op) {
  auto attrId = StringAttr::get(op->getContext(), "stream.partitioning");
  while (op) {
    auto attr = op->getAttrOfType<PartitioningConfigAttr>(attrId);
    if (attr)
      return attr;
    op = op->getParentOp();
  }
  // No config found; use defaults.
  auto favorAttr = FavorAttr::get(attrId.getContext(), clPartitioningFavor);
  return PartitioningConfigAttr::get(favorAttr);
}

//===----------------------------------------------------------------------===//
// !stream.resource<lifetime>
//===----------------------------------------------------------------------===//

static std::optional<Lifetime> parseLifetime(StringRef str) {
  if (str == "*") {
    return Lifetime::Unknown;
  } else if (str == "external") {
    return Lifetime::External;
  } else if (str == "staging") {
    return Lifetime::Staging;
  } else if (str == "transient") {
    return Lifetime::Transient;
  } else if (str == "variable") {
    return Lifetime::Variable;
  } else if (str == "constant") {
    return Lifetime::Constant;
  } else {
    return std::nullopt;
  }
}

static void printLifetime(Lifetime lifetime, llvm::raw_ostream &os) {
  if (lifetime == Lifetime::Unknown) {
    os << "*";
  } else {
    os << stringifyLifetime(lifetime).lower();
  }
}

Type ResourceType::parse(AsmParser &p) {
  StringRef lifetimeStr;
  if (failed(p.parseLess()))
    return {};
  if (succeeded(p.parseOptionalStar())) {
    lifetimeStr = "*";
  } else if (failed(p.parseKeyword(&lifetimeStr))) {
    return {};
  }
  if (failed(p.parseGreater()))
    return {};
  auto lifetime = parseLifetime(lifetimeStr);
  if (!lifetime.has_value()) {
    p.emitError(p.getNameLoc(), "unknown lifetime value: ") << lifetimeStr;
    return {};
  }
  return ResourceType::get(p.getContext(), lifetime.value());
}

void ResourceType::print(AsmPrinter &p) const {
  p << "<";
  printLifetime(getLifetime(), p.getStream());
  p << ">";
}

bool ResourceType::isAccessStorageCompatible(Type accessType) const {
  if (auto resourceType = llvm::dyn_cast<ResourceType>(accessType)) {
    // We could allow widening loads or stores here but today we require
    // transfers to accomplish that.
    return accessType == resourceType;
  }
  return llvm::isa<ShapedType>(accessType);
}

Value ResourceType::inferSizeFromValue(Location loc, Value value,
                                       OpBuilder &builder) const {
  return builder.createOrFold<IREE::Stream::ResourceSizeOp>(
      loc, builder.getIndexType(), value);
}

Value ResourceType::createSubrangeOp(Location loc, Value resource,
                                     Value resourceSize, Value subrangeOffset,
                                     Value subrangeLength,
                                     OpBuilder &builder) const {
  return builder.create<IREE::Stream::ResourceSubviewOp>(
      loc, resource, resourceSize, subrangeOffset, subrangeLength);
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Stream/IR/StreamAttrInterfaces.cpp.inc" // IWYU pragma: export
#include "iree/compiler/Dialect/Stream/IR/StreamOpInterfaces.cpp.inc" // IWYU pragma: keep
#include "iree/compiler/Dialect/Stream/IR/StreamTypeInterfaces.cpp.inc" // IWYU pragma: keep

void StreamDialect::registerAttributes() {
  // Register command line flags:
  (void)clPartitioningFavor;
  (void)clResourceMaxAllocationSize;
  (void)clResourceMinOffsetAlignment;
  (void)clResourceMaxRange;
  (void)clResourceIndexBits;

  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/Stream/IR/StreamAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

void StreamDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::Stream
