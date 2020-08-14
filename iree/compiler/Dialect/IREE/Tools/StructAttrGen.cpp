// Copyright 2019 Google LLC
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

namespace mlir {
namespace tblgen {
namespace iree_compiler {
namespace {

using llvm::formatv;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::StringRef;

class StructFieldAttr {
 public:
  explicit StructFieldAttr(const llvm::Record *record) : def(record) {
    assert(def->isSubClassOf("IREE_StructFieldAttr") &&
           "must be subclass of TableGen 'IREE_StructFieldAttr' class");
  }
  explicit StructFieldAttr(const llvm::Record &record)
      : StructFieldAttr(&record) {}
  explicit StructFieldAttr(const llvm::DefInit *init)
      : StructFieldAttr(init->getDef()) {}

  StringRef getName() const { return def->getValueAsString("name"); }
  Attribute getType() const {
    auto init = def->getValueInit("type");
    return tblgen::Attribute(cast<llvm::DefInit>(init));
  }

 private:
  const llvm::Record *def;
};

class StructAttr : public Attribute {
 public:
  explicit StructAttr(const llvm::Record *record) : Attribute(record) {
    assert(isSubClassOf("IREE_StructAttr") &&
           "must be subclass of TableGen 'IREE_StructAttr' class");
  }
  explicit StructAttr(const llvm::Record &record) : StructAttr(&record) {}
  explicit StructAttr(const llvm::DefInit *init) : StructAttr(init->getDef()) {}

  StringRef getStructKind() const { return def->getValueAsString("kind"); }
  StringRef getStructClassName() const {
    return def->getValueAsString("className");
  }
  StringRef getCppNamespace() const {
    if (def->isValueUnset("cppNamespace")) {
      Dialect dialect(def->getValueAsDef("structDialect"));
      return dialect.getCppNamespace();
    } else {
      return def->getValueAsString("cppNamespace");
    }
  }

  std::vector<StructFieldAttr> getAllFields() const {
    std::vector<StructFieldAttr> attributes;
    const auto *inits = def->getValueAsListInit("fields");
    attributes.reserve(inits->size());
    for (const llvm::Init *init : *inits) {
      attributes.emplace_back(cast<llvm::DefInit>(init));
    }
    return attributes;
  }
};

static void emitStructClass(const StructAttr &structAttr, raw_ostream &os) {
  if (!structAttr.getAllFields().empty()) {
    os << formatv(R"(
namespace detail {
struct {0}Storage;
}  // namespace detail
)",
                  structAttr.getStructClassName());
  }
  os << formatv(R"(
// {0}
class {1} : public mlir::Attribute::AttrBase<{1}, mlir::Attribute, {3}Storage> {
 public:
  using Base::Base;

  static StringRef getKindName() { return "{2}"; }

)",
                structAttr.getDescription(), structAttr.getStructClassName(),
                structAttr.getStructKind(),
                structAttr.getAllFields().empty()
                    ? "Attribute"
                    : "detail::" + structAttr.getStructClassName());

  if (!structAttr.getAllFields().empty()) {
    os << "  static LogicalResult verifyConstructionInvariants(\n";
    os << "      Location loc,\n";
    interleave(
        structAttr.getAllFields(), os,
        [&](StructFieldAttr field) {
          auto type = field.getType();
          os << formatv("      {0} {1}", type.getStorageType(),
                        field.getName());
        },
        ",\n");
    os << ");\n\n";
  }

  // Attribute storage type constructor (IntegerAttr, etc).
  os << formatv("  static {0} get(", structAttr.getStructClassName());
  if (structAttr.getAllFields().empty()) {
    os << "mlir::MLIRContext* context";
  } else {
    interleaveComma(structAttr.getAllFields(), os, [&](StructFieldAttr field) {
      auto type = field.getType();
      os << formatv("\n      {0} {1}", type.getStorageType(), field.getName());
    });
  }
  os << ");\n\n";

  // Attribute return type constructor (APInt, etc).
  if (!structAttr.getAllFields().empty()) {
    os << formatv("  static {0} get(\n", structAttr.getStructClassName());
    for (auto field : structAttr.getAllFields()) {
      auto type = field.getType();
      os << formatv("      {0} {1},\n", type.getReturnType(), field.getName());
    }
    os << "      mlir::MLIRContext* context);\n";
  }

  os << R"(
  static Attribute parse(DialectAsmParser &p);
  void print(DialectAsmPrinter &p) const;

)";

  for (auto field : structAttr.getAllFields()) {
    auto type = field.getType();
    // Attribute storage type accessors (IntegerAttr, etc).
    os << formatv("  {0} {1}Attr() const;\n", type.getStorageType(),
                  field.getName());
    // Attribute return type accessors (APInt, etc).
    os << formatv("  {0} {1}() const;\n", type.getReturnType(),
                  field.getName());
  }

  os << "  void walkStorage(const llvm::function_ref<void(mlir::Attribute "
        "elementAttr)> &fn) const;\n";

  os << "};\n\n";
}

static void emitStructDecl(const Record &structDef, raw_ostream &os) {
  StructAttr structAttr(&structDef);

  // Forward declarations (to make including easier).
  os << R"(namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
}  // namespace mlir

)";

  // Wrap in the appropriate namespace.
  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(structAttr.getCppNamespace(), namespaces, "::");

  for (auto ns : namespaces) {
    os << "namespace " << ns << " {\n";
  }

  // Emit the struct class definition
  emitStructClass(structAttr, os);

  // Close the declared namespace.
  for (auto ns : namespaces) {
    os << "} // namespace " << ns << "\n";
  }
}

static bool emitStructDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("Struct Attr Declarations", os);
  auto defs = recordKeeper.getAllDerivedDefinitions("IREE_StructAttr");
  for (const auto *def : defs) {
    emitStructDecl(*def, os);
  }
  return false;
}

static void emitStorageDef(const StructAttr &structAttr, raw_ostream &os) {
  os << "namespace detail {\n";
  os << formatv("struct {0}Storage : public mlir::AttributeStorage {{\n",
                structAttr.getStructClassName());

  os << "  using KeyTy = std::tuple<";
  interleaveComma(structAttr.getAllFields(), os, [&](StructFieldAttr field) {
    auto type = field.getType();
    os << type.getStorageType();
  });
  os << ">;\n\n";

  os << formatv("  {0}Storage(", structAttr.getStructClassName());
  interleaveComma(structAttr.getAllFields(), os, [&](StructFieldAttr field) {
    auto type = field.getType();
    os << formatv("{0} {1}", type.getStorageType(), field.getName());
  });
  os << ") : ";
  interleaveComma(structAttr.getAllFields(), os, [&](StructFieldAttr field) {
    os << formatv("{0}({0})", field.getName());
  });
  os << " {}\n\n";

  os << "  bool operator==(const KeyTy &key) const {\n";
  os << "    return ";
  int i = 0;
  interleave(
      structAttr.getAllFields(), os,
      [&](StructFieldAttr field) {
        os << formatv("std::get<{0}>(key) == {1}", i++, field.getName());
      },
      " && ");
  os << ";\n  }\n\n";

  os << "  static llvm::hash_code hashKey(const KeyTy &key) {\n";
  os << "    return llvm::hash_combine(";
  i = 0;
  interleaveComma(structAttr.getAllFields(), os, [&](StructFieldAttr field) {
    os << formatv("std::get<{0}>(key)", i++, field.getName());
  });
  os << ");\n";
  os << "}\n\n";

  os << formatv(
      "  static {0}Storage *construct(AttributeStorageAllocator &allocator, "
      "const KeyTy &key) {{\n",
      structAttr.getStructClassName());
  os << formatv(
      "    return new (allocator.allocate<{0}Storage>()) {0}Storage(\n",
      structAttr.getStructClassName());
  i = 0;
  os << "        ";
  interleaveComma(structAttr.getAllFields(), os, [&](StructFieldAttr field) {
    os << formatv("std::get<{0}>(key)", i++, field.getName());
  });
  os << ");\n";
  os << "  }\n\n";

  for (auto field : structAttr.getAllFields()) {
    auto type = field.getType();
    os << formatv("  {0} {1};\n", type.getStorageType(), field.getName());
  }

  os << "};\n";
  os << "}  // namespace detail\n\n";
}

static void emitVerifierDef(const StructAttr &structAttr, raw_ostream &os) {
  os << "// static\n";
  os << formatv("LogicalResult {0}::verifyConstructionInvariants(\n",
                structAttr.getStructClassName());
  os << "    Location loc,\n";
  interleave(
      structAttr.getAllFields(), os,
      [&](StructFieldAttr field) {
        auto type = field.getType();
        os << formatv("    {0} {1}", type.getStorageType(), field.getName());
      },
      ",\n");
  os << ") {\n";

  for (auto field : structAttr.getAllFields()) {
    FmtContext fmt;
    auto type = field.getType();
    os << formatv(R"(
  if (!{0}) {{
    return emitError(loc) << "'{1}' must be {2} but got " << {1}.getType();
  }
)",
                  tgfmt(type.getConditionTemplate(),
                        &fmt.withSelf(field.getName()), field.getName()),
                  field.getName(), type.getDescription());
  }

  os << "  return success();\n";
  os << "}\n\n";
}

static void emitAttrFactoryDef(const StructAttr &structAttr, raw_ostream &os) {
  os << "// static\n";
  os << formatv("{0} {0}::get(", structAttr.getStructClassName());
  if (structAttr.getAllFields().empty()) {
    os << "mlir::MLIRContext* context";
  } else {
    interleaveComma(structAttr.getAllFields(), os, [&](StructFieldAttr field) {
      auto type = field.getType();
      os << formatv("\n    {0} {1}", type.getStorageType(), field.getName());
    });
  }
  os << ") {\n";

  for (auto field : structAttr.getAllFields()) {
    if (!field.getType().isOptional()) {
      os << formatv("  assert({0} && \"{0} is required\");\n", field.getName());
    }
  }

  if (!structAttr.getAllFields().empty()) {
    os << formatv("  auto *context = {0}.getContext();\n",
                  structAttr.getAllFields().front().getName());
  }

  os << formatv("  return Base::get(context, AttrKind::{0}",
                structAttr.getStructClassName());
  if (!structAttr.getAllFields().empty()) {
    os << ",\n                   ";
    interleaveComma(structAttr.getAllFields(), os,
                    [&](StructFieldAttr field) { os << field.getName(); });
  }
  os << ");\n";

  os << "}\n\n";
}

// Replaces all occurrences of `match` in `str` with `substitute`.
static std::string replaceAllSubstrs(std::string str, const std::string &match,
                                     const std::string &substitute) {
  std::string::size_type scanLoc = 0, matchLoc = std::string::npos;
  while ((matchLoc = str.find(match, scanLoc)) != std::string::npos) {
    str = str.replace(matchLoc, match.size(), substitute);
    scanLoc = matchLoc + substitute.size();
  }
  return str;
}

static void emitTypedFactoryDef(const StructAttr &structAttr, raw_ostream &os) {
  os << "// static\n";
  os << formatv("{0} {0}::get(", structAttr.getStructClassName());
  for (auto field : structAttr.getAllFields()) {
    auto type = field.getType();
    os << formatv("\n    {0} {1},", type.getReturnType(), field.getName());
  }
  os << "\n    mlir::MLIRContext* context) {\n";
  os << "  mlir::Builder b(context);\n";

  FmtContext ctx;
  ctx.withBuilder("b");
  for (auto field : structAttr.getAllFields()) {
    auto type = field.getType();

    // For StringAttr, its constant builder call will wrap the input in
    // quotes, which is correct for normal string literals, but incorrect
    // here given we use function arguments. So we need to strip the
    // wrapping quotes.
    std::string builderTemplate = type.getConstBuilderTemplate().str();
    if (StringRef(builderTemplate).contains("\"$0\"")) {
      builderTemplate = replaceAllSubstrs(builderTemplate, "\"$0\"", "$0");
    }

    os << formatv("  auto {0}Attr = {1};\n", field.getName(),
                  tgfmt(builderTemplate, &ctx, field.getName()));
  }

  os << "  return get(";
  if (structAttr.getAllFields().empty()) {
    os << "context";
  } else {
    interleaveComma(structAttr.getAllFields(), os, [&](StructFieldAttr attr) {
      os << attr.getName() << "Attr";
    });
  }
  os << ");\n";

  os << "}\n";
}

static void emitAccessorDefs(const StructAttr &structAttr,
                             const StructFieldAttr &field, raw_ostream &os) {
  auto type = field.getType();

  // Attribute storage type accessors (IntegerAttr, etc).
  os << formatv(R"(
{1} {0}::{2}Attr() const {{
  return getImpl()->{2};
}
)",
                structAttr.getStructClassName(), type.getStorageType(),
                field.getName());

  // Attribute return type accessors (APInt, etc).
  FmtContext ctx;
  os << formatv(
      R"(
{1} {0}::{2}() const {{
  return {3};
}
)",
      structAttr.getStructClassName(), type.getReturnType(), field.getName(),
      tgfmt(type.getConvertFromStorageCall(),
            &ctx.withSelf(field.getName() + "Attr()")));
}

static void emitWalkStorageDef(const StructAttr &structAttr, raw_ostream &os) {
  os << formatv(
      "void {0}::walkStorage(const llvm::function_ref<void(mlir::Attribute "
      "elementAttr)> &fn) const {{\n",
      structAttr.getStructClassName());
  for (auto field : structAttr.getAllFields()) {
    os << formatv("  fn({0}Attr());\n", field.getName());
  }
  os << "}\n";
}

static void emitStructDef(const Record &structDef, raw_ostream &os) {
  StructAttr structAttr(&structDef);
  StringRef cppNamespace = structAttr.getCppNamespace();

  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(cppNamespace, namespaces, "::");

  for (auto ns : namespaces) {
    os << "namespace " << ns << " {\n";
  }
  os << "\n";

  if (!structAttr.getAllFields().empty()) {
    emitStorageDef(structAttr, os);
    emitVerifierDef(structAttr, os);
  }
  emitAttrFactoryDef(structAttr, os);
  if (!structAttr.getAllFields().empty()) {
    emitTypedFactoryDef(structAttr, os);
    for (auto field : structAttr.getAllFields()) {
      emitAccessorDefs(structAttr, field, os);
    }
  }
  emitWalkStorageDef(structAttr, os);

  os << "\n";
  for (auto ns : llvm::reverse(namespaces)) {
    os << "} // namespace " << ns << "\n";
  }
}

static bool emitStructDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("Struct Attr Definitions", os);
  auto defs = recordKeeper.getAllDerivedDefinitions("IREE_StructAttr");
  for (const auto *def : defs) {
    emitStructDef(*def, os);
  }
  return false;
}

// Registers the struct utility generator to mlir-tblgen.
static GenRegistration genStructDecls("gen-iree-struct-attr-decls",
                                      "Generate struct attr declarations",
                                      [](const RecordKeeper &records,
                                         raw_ostream &os) {
                                        return emitStructDecls(records, os);
                                      });

// Registers the struct utility generator to mlir-tblgen.
static GenRegistration genStructDefs("gen-iree-struct-attr-defs",
                                     "Generate struct attr definitions",
                                     [](const RecordKeeper &records,
                                        raw_ostream &os) {
                                       return emitStructDefs(records, os);
                                     });

}  // namespace
}  // namespace iree_compiler
}  // namespace tblgen
}  // namespace mlir
