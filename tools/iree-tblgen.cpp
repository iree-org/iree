// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adapted from mlir-tblgen.cpp. Simply delegates through to MlirTblgenMain.

#include <optional>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Dialect.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// JSON Documentation Generation
//===----------------------------------------------------------------------===//

// Strips consistent leading whitespace from description text.
// Detects the pattern `= [{\n  ...` and removes that leading whitespace
// from all lines. Preserves relative indentation for ASCII art, etc.
// Returns original text if parsing fails at any point.
static std::string cleanDescription(StringRef desc) {
  if (desc.empty()) {
    return "";
  }

  SmallVector<StringRef> lines;
  desc.split(lines, '\n');

  // Find minimum indentation (excluding empty lines).
  // Use std::optional to avoid npos sentinel value bugs.
  std::optional<size_t> minIndent;
  for (StringRef line : lines) {
    size_t firstNonWS = line.find_first_not_of(" \t");
    if (firstNonWS != StringRef::npos) {
      // Non-empty line found.
      if (!minIndent || firstNonWS < *minIndent) {
        minIndent = firstNonWS;
      }
    }
  }

  // If no non-empty lines or indentation is zero, return original.
  if (!minIndent || *minIndent == 0) {
    return desc.str();
  }

  // Strip the common indentation from all lines.
  std::string result;
  for (size_t i = 0; i < lines.size(); ++i) {
    StringRef line = lines[i];
    if (line.find_first_not_of(" \t") == StringRef::npos) {
      // Empty/whitespace-only line - preserve it.
      if (i > 0) {
        result += '\n';
      }
    } else {
      // Non-empty line - strip common indentation.
      if (line.size() < *minIndent) {
        // Line is shorter than expected indentation - bail cleanly.
        // This shouldn't happen if logic is correct, but handle defensively.
        return desc.str();
      }
      if (i > 0) {
        result += '\n';
      }
      result += line.substr(*minIndent).str();
    }
  }

  // Trim leading/trailing blank lines.
  while (!result.empty() && result.front() == '\n') {
    result.erase(0, 1);
  }
  while (!result.empty() && result.back() == '\n') {
    result.pop_back();
  }

  return result;
}

// Cleans assembly format by replacing newlines with spaces.
// Assembly format strings in TableGen files may have newlines for readability,
// but they're ignored by the parser so we normalize to single line.
static std::string cleanAssemblyFormat(StringRef format) {
  if (format.empty()) {
    return "";
  }

  std::string result = format.str();
  // Replace all newlines with spaces.
  for (char &c : result) {
    if (c == '\n') {
      c = ' ';
    }
  }

  // Collapse multiple spaces into single space.
  size_t writePos = 0;
  bool lastWasSpace = false;
  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i] == ' ') {
      if (!lastWasSpace) {
        result[writePos++] = ' ';
        lastWasSpace = true;
      }
    } else {
      result[writePos++] = result[i];
      lastWasSpace = false;
    }
  }
  result.resize(writePos);

  // Trim leading/trailing spaces.
  while (!result.empty() && result.front() == ' ') {
    result.erase(0, 1);
  }
  while (!result.empty() && result.back() == ' ') {
    result.pop_back();
  }

  return result;
}

// Resolves relative path to absolute path using the defining record's location.
// This is best-effort resolution - returns original path if resolution fails.
// Paths in metadata are relative to the repository root (e.g.,
// "test/foo.mlir"), so we walk up from the .td file location to find the
// repository root. Note: Assumes IREE-style repo structure with "compiler/src/"
// directory.
static std::string resolveRelativePath(StringRef relativePath,
                                       const Record *definingRecord) {
  if (relativePath.empty()) {
    return "";
  }

  // Get source location of the record (the .td file).
  ArrayRef<SMLoc> locs = definingRecord->getLoc();
  if (locs.empty() || !locs[0].isValid()) {
    // No valid location - can't resolve, return original path.
    return relativePath.str();
  }

  // Find buffer containing this location.
  unsigned bufferID = llvm::SrcMgr.FindBufferContainingLoc(locs[0]);
  if (bufferID == 0) {
    // Buffer not found - return original path.
    return relativePath.str();
  }

  // Get .td file path from buffer.
  const MemoryBuffer *buffer = llvm::SrcMgr.getMemoryBuffer(bufferID);
  if (!buffer) {
    // Defensive: buffer should exist if bufferID != 0, but check anyway.
    return relativePath.str();
  }
  StringRef tdFilePath = buffer->getBufferIdentifier();

  // Find repository root by walking up from .td file directory.
  // The repository root is assumed to be the directory containing
  // "compiler/src/".
  SmallString<256> currentDir(tdFilePath);
  llvm::sys::path::remove_filename(
      currentDir);  // Start from .td file's directory.

  SmallString<256> repoRoot;
  bool foundRoot = false;

  // Walk up the directory tree looking for "compiler/src/".
  while (!currentDir.empty() && currentDir != "/") {
    SmallString<256> compilerSrcPath(currentDir);
    llvm::sys::path::append(compilerSrcPath, "compiler", "src");

    // Check if this directory contains "compiler/src/".
    bool isDir = false;
    std::error_code ec = llvm::sys::fs::is_directory(compilerSrcPath, isDir);
    if (!ec && isDir) {
      // Found the repository root.
      repoRoot = currentDir;
      foundRoot = true;
      break;
    }
    // If error checking directory, continue walking up (graceful degradation).

    // Move up one directory.
    llvm::sys::path::remove_filename(currentDir);
  }

  // Fallback: if root-finding failed, try extracting from .td file path.
  if (!foundRoot) {
    // Try to find "compiler/src/" in the .td file path itself.
    StringRef tdFileStr(tdFilePath);
    size_t compilerSrcPos = tdFileStr.find("compiler/src/");
    if (compilerSrcPos != StringRef::npos) {
      // Extract everything before "compiler/src/" as repo root.
      repoRoot = tdFileStr.substr(0, compilerSrcPos);
      foundRoot = true;
    } else {
      // Could not determine repository root - return original path unchanged.
      // This is acceptable for non-IREE projects or different layouts.
      return relativePath.str();
    }
  }

  // Resolve relative path against repository root.
  SmallString<256> absolutePath(repoRoot);
  llvm::sys::path::append(absolutePath, relativePath);

  // Normalize to remove . and ..
  llvm::sys::path::remove_dots(absolutePath, /*remove_dot_dot=*/true);

  return std::string(absolutePath.str());
}

// Emits source location information for a record.
// Gracefully degrades: if location can't be determined, emits nothing.
// This is acceptable as source location is supplementary information.
static void emitSourceLocation(const Record *def, const RecordKeeper &records,
                               json::OStream &J) {
  (void)records;  // Not needed - using global llvm::SrcMgr.

  // Get the primary location (first element of location array).
  ArrayRef<SMLoc> locs = def->getLoc();
  if (locs.empty() || !locs[0].isValid()) {
    // No location available - skip emitting sourceLocation attribute.
    return;
  }

  SMLoc loc = locs[0];

  // Find which buffer contains this location.
  unsigned bufferID = llvm::SrcMgr.FindBufferContainingLoc(loc);
  if (bufferID == 0) {
    // Location not associated with a buffer - skip sourceLocation.
    return;
  }

  // Get the file path from buffer.
  const MemoryBuffer *buffer = llvm::SrcMgr.getMemoryBuffer(bufferID);
  if (!buffer) {
    // Defensive: buffer should exist if bufferID != 0, but check anyway.
    return;
  }
  StringRef filename = buffer->getBufferIdentifier();

  // Get line and column numbers.
  std::pair<unsigned, unsigned> lineCol =
      llvm::SrcMgr.getLineAndColumn(loc, bufferID);

  // Emit as JSON.
  J.attributeObject("sourceLocation", [&] {
    J.attribute("file", filename);
    J.attribute("line", lineCol.first);
    J.attribute("column", lineCol.second);
  });
}

// Emits parameters for a type or attribute.
static void emitParameters(ArrayRef<AttrOrTypeParameter> params,
                           json::OStream &J) {
  if (params.empty()) {
    return;
  }

  J.attributeArray("parameters", [&] {
    for (const auto &param : params) {
      J.object([&] {
        if (!param.isAnonymous()) {
          J.attribute("name", param.getName());
        }
        J.attribute("cppType", param.getCppType());

        if (auto summary = param.getSummary()) {
          J.attribute("summary", *summary);
        }

        if (param.isOptional()) {
          J.attribute("isOptional", true);
          if (auto defaultValue = param.getDefaultValue()) {
            J.attribute("defaultValue", *defaultValue);
          }
        } else {
          J.attribute("isOptional", false);
        }

        if (auto constraint = param.getConstraint()) {
          J.attributeObject("constraint", [&] {
            if (!constraint->getSummary().empty()) {
              J.attribute("summary", constraint->getSummary());
            }
            if (!constraint->getDescription().empty()) {
              J.attribute("description",
                          cleanDescription(constraint->getDescription()));
            }
          });
        }
      });
    }
  });
}

// Emits traits for a type or attribute.
static void emitTraits(ArrayRef<Trait> traits, json::OStream &J) {
  if (traits.empty()) {
    return;
  }

  J.attributeArray("traits", [&] {
    for (const Trait &trait : traits) {
      J.object([&] {
        if (const auto *nt = dyn_cast<NativeTrait>(&trait)) {
          J.attribute("kind", "NativeTrait");
          J.attribute("trait", nt->getFullyQualifiedTraitName());
        } else if (const auto *it = dyn_cast<InterfaceTrait>(&trait)) {
          J.attribute("kind", "Interface");
          J.attribute("trait", it->getFullyQualifiedTraitName());
        } else {
          J.attribute("kind", "Other");
        }
      });
    }
  });
}

// Emits builders for an operation.
static void emitBuilders(ArrayRef<Builder> builders, json::OStream &J) {
  if (builders.empty()) {
    return;
  }

  J.attributeArray("builders", [&] {
    for (const Builder &builder : builders) {
      J.object([&] {
        // Emit parameters.
        J.attributeArray("parameters", [&] {
          for (const Builder::Parameter &param : builder.getParameters()) {
            J.object([&] {
              J.attribute("type", param.getCppType());

              if (auto name = param.getName()) {
                J.attribute("name", *name);
              }

              if (auto defaultValue = param.getDefaultValue()) {
                J.attribute("defaultValue", *defaultValue);
              }
            });
          }
        });

        // Check if deprecated.
        if (auto deprecatedMsg = builder.getDeprecatedMessage()) {
          J.attribute("deprecated", true);
          J.attribute("deprecationMessage", *deprecatedMsg);
        }
      });
    }
  });
}

// Emits interface methods.
static void emitInterfaceMethods(ArrayRef<InterfaceMethod> methods,
                                 json::OStream &J) {
  if (methods.empty()) {
    return;
  }

  J.attributeArray("methods", [&] {
    for (const InterfaceMethod &method : methods) {
      J.object([&] {
        J.attribute("name", method.getName());
        J.attribute("returnType", method.getReturnType());

        J.attributeArray("arguments", [&] {
          for (const auto &arg : method.getArguments()) {
            J.object([&] {
              J.attribute("type", arg.type);
              J.attribute("name", arg.name);
            });
          }
        });

        J.attribute("isStatic", method.isStatic());

        if (auto desc = method.getDescription()) {
          J.attribute("description", cleanDescription(*desc));
        }

        // Check if method has a default implementation.
        bool hasDefault = method.getDefaultImplementation().has_value();
        J.attribute("hasDefaultImpl", hasDefault);
      });
    }
  });
}

// Returns all operation definitions.
static std::vector<const Record *> getOpDefinitions(
    const RecordKeeper &records) {
  const Record *classDef = records.getClass("Op");
  if (!classDef) {
    return {};
  }

  std::vector<const Record *> defs;
  for (const auto &def : records.getDefs()) {
    if (def.second->isSubClassOf(classDef)) {
      defs.push_back(def.second.get());
    }
  }
  return defs;
}

// Returns all type definitions.
static std::vector<const Record *> getTypeDefinitions(
    const RecordKeeper &records) {
  const Record *classDef = records.getClass("TypeDef");
  if (!classDef) {
    return {};
  }

  std::vector<const Record *> defs;
  for (const auto &def : records.getDefs()) {
    if (def.second->isSubClassOf(classDef)) {
      defs.push_back(def.second.get());
    }
  }
  return defs;
}

// Returns all attribute definitions.
static std::vector<const Record *> getAttrDefinitions(
    const RecordKeeper &records) {
  const Record *classDef = records.getClass("AttrDef");
  if (!classDef) {
    return {};
  }

  std::vector<const Record *> defs;
  for (const auto &def : records.getDefs()) {
    if (def.second->isSubClassOf(classDef)) {
      defs.push_back(def.second.get());
    }
  }
  return defs;
}

// Returns all interface definitions (Op, Type, and Attr interfaces).
static std::vector<const Record *> getInterfaceDefinitions(
    const RecordKeeper &records) {
  std::vector<const Record *> defs;

  // OpInterface, AttrInterface, TypeInterface.
  for (const char *className :
       {"OpInterface", "AttrInterface", "TypeInterface"}) {
    const Record *classDef = records.getClass(className);
    if (!classDef) {
      continue;
    }

    for (const auto &def : records.getDefs()) {
      if (def.second->isSubClassOf(classDef)) {
        // Skip anonymous records generated by DeclareInterfaceMethods helpers.
        // These are implementation details that duplicate named interfaces.
        StringRef name = def.second->getName();
        if (name.starts_with("anonymous_")) {
          continue;
        }
        defs.push_back(def.second.get());
      }
    }
  }
  return defs;
}

// Returns all dialect definitions.
static std::vector<Dialect> getDialects(const RecordKeeper &records) {
  auto dialectDefs = records.getAllDerivedDefinitions("Dialect");
  std::vector<Dialect> dialects;
  dialects.reserve(dialectDefs.size());
  for (const Record *def : dialectDefs) {
    dialects.emplace_back(def);
  }
  return dialects;
}

// Extracts dialect name from a fully qualified interface name.
// Tries multiple common namespace patterns:
// 1. ::mlir::iree_compiler::IREE::DialectName::ClassName (IREE pattern)
// 2. ::mlir::DialectName::ClassName (simple MLIR pattern)
// 3. Generic: second-to-last namespace component before ClassName
// Returns empty optional if extraction fails.
static std::optional<std::string> extractDialectFromNamespace(StringRef fqn) {
  // Try IREE-specific pattern: ::mlir::iree_compiler::IREE::DialectName::
  StringRef ireePrefix = "::mlir::iree_compiler::IREE::";
  if (fqn.starts_with(ireePrefix)) {
    StringRef afterPrefix = fqn.substr(ireePrefix.size());
    size_t nextColon = afterPrefix.find("::");
    if (nextColon != StringRef::npos) {
      return afterPrefix.substr(0, nextColon).lower();
    }
  }

  // Try simple MLIR pattern: ::mlir::DialectName::ClassName
  StringRef mlirPrefix = "::mlir::";
  if (fqn.starts_with(mlirPrefix)) {
    StringRef afterMlir = fqn.substr(mlirPrefix.size());
    size_t firstColon = afterMlir.find("::");
    if (firstColon != StringRef::npos) {
      StringRef candidate = afterMlir.substr(0, firstColon);
      // Make sure there's another :: after this (i.e., ClassName exists).
      StringRef afterCandidate = afterMlir.substr(firstColon + 2);
      if (!afterCandidate.empty() &&
          afterCandidate.find("::") == StringRef::npos) {
        // Pattern matches: ::mlir::DialectName::ClassName
        return candidate.lower();
      }
    }
  }

  // Generic fallback: extract second-to-last namespace component.
  // Find last :: to get ClassName, then find previous :: for DialectName.
  size_t lastColons = fqn.rfind("::");
  if (lastColons != StringRef::npos && lastColons > 0) {
    StringRef beforeLast = fqn.substr(0, lastColons);
    size_t secondLastColons = beforeLast.rfind("::");
    if (secondLastColons != StringRef::npos) {
      StringRef dialectName = beforeLast.substr(secondLastColons + 2);
      if (!dialectName.empty()) {
        return dialectName.lower();
      }
    }
  }

  // Could not extract dialect name - return empty.
  return std::nullopt;
}

// Emits structured constraint information for type unions.
// Extracts the allowedTypes list from AnyTypeOf/AllOfType constraints.
static void emitConstraintMetadata(const tblgen::Constraint &constraint,
                                   json::OStream &J) {
  const Record &def = constraint.getDef();

  // Check if this is a union-type constraint (AnyTypeOf, AllOfType, etc.).
  // These constraints have an "allowedTypes" field with the constituent types.
  if (!def.getValue("allowedTypes")) {
    return;  // Not a union constraint, nothing to emit.
  }

  std::vector<const Record *> allowedTypes =
      def.getValueAsListOfDefs("allowedTypes");
  if (allowedTypes.empty()) {
    return;  // No types listed, skip emission.
  }

  // Emit structured constraint object.
  J.attributeObject("constraint", [&] {
    // Determine constraint kind from the class hierarchy.
    std::string kind = "UnionType";
    if (def.isSubClassOf("AnyTypeOf")) {
      kind = "AnyTypeOf";
    } else if (def.isSubClassOf("AllOfType")) {
      kind = "AllOfType";
    } else if (def.isSubClassOf("TypeConstraint")) {
      kind = "TypeConstraint";
    }
    J.attribute("kind", kind);

    // Emit the list of allowed type names.
    J.attributeArray("allowedTypes", [&] {
      for (const Record *typeRecord : allowedTypes) {
        J.value(typeRecord->getName());
      }
    });
  });
}

// Emits common metadata fields (used by types, attrs, interfaces).
static void emitCommonMetadata(const Record *metadata,
                               const Record *definingRecord, json::OStream &J,
                               const char *relatedListName) {
  if (!metadata) {
    return;
  }

  J.attributeObject("metadata", [&] {
    if (StringRef category = metadata->getValueAsString("category");
        !category.empty()) {
      J.attribute("category", category);
    }

    auto relatedList = metadata->getValueAsListOfStrings(relatedListName);
    if (!relatedList.empty()) {
      J.attributeArray(relatedListName, [&] {
        for (StringRef related : relatedList) {
          J.value(related);
        }
      });
    }

    if (StringRef testFile = metadata->getValueAsString("testFile");
        !testFile.empty()) {
      std::string absolutePath = resolveRelativePath(testFile, definingRecord);
      J.attribute("testFile", absolutePath);
    }

    auto exampleRefs = metadata->getValueAsListOfStrings("exampleRefs");
    if (!exampleRefs.empty()) {
      J.attributeArray("exampleRefs", [&] {
        for (StringRef ref : exampleRefs) {
          J.value(ref);
        }
      });
    }

    auto textualExamples = metadata->getValueAsListOfStrings("textualExamples");
    if (!textualExamples.empty()) {
      J.attributeArray("textualExamples", [&] {
        for (StringRef example : textualExamples) {
          J.value(example);
        }
      });
    }

    auto tags = metadata->getValueAsListOfStrings("tags");
    if (!tags.empty()) {
      J.attributeArray("tags", [&] {
        for (StringRef tag : tags) {
          J.value(tag);
        }
      });
    }
  });
}

// Emits a single operation as JSON.
static void emitOperatorJSON(const Operator &op, const RecordKeeper &records,
                             json::OStream &J) {
  const Record &def = op.getDef();

  J.object([&] {
    J.attribute("name", op.getOperationName());

    // Dialect reference.
    J.attribute("dialect", op.getDialect().getName());

    // C++ class names.
    J.attribute("cppClassName", op.getCppClassName());
    std::string fullyQualifiedName = op.getDialect().getCppNamespace().str() +
                                     "::" + op.getCppClassName().str();
    J.attribute("fullyQualifiedName", fullyQualifiedName);

    // Source location.
    emitSourceLocation(&def, records, J);

    if (op.hasSummary()) {
      J.attribute("summary", cleanDescription(op.getSummary()));
    }
    if (op.hasDescription()) {
      J.attribute("description", cleanDescription(op.getDescription()));
    }
    if (op.hasAssemblyFormat()) {
      J.attribute("assemblyFormat",
                  cleanAssemblyFormat(op.getAssemblyFormat()));
    }

    // Custom metadata extraction (Util_OpDocMetadata).
    if (const Record *metadata = def.getValueAsOptionalDef("docMetadata")) {
      J.attributeObject("metadata", [&] {
        // Extract category from OpDocGroup if present.
        if (const Record *docGroup =
                metadata->getValueAsOptionalDef("docGroup")) {
          if (StringRef summary = docGroup->getValueAsString("summary");
              !summary.empty()) {
            J.attribute("category", summary);
          }
        }

        auto relatedOps = metadata->getValueAsListOfStrings("relatedOps");
        if (!relatedOps.empty()) {
          J.attributeArray("relatedOps", [&] {
            for (StringRef relOp : relatedOps) {
              J.value(relOp);
            }
          });
        }

        if (StringRef testFile = metadata->getValueAsString("testFile");
            !testFile.empty()) {
          std::string absolutePath = resolveRelativePath(testFile, &def);
          J.attribute("testFile", absolutePath);
        }

        auto exampleRefs = metadata->getValueAsListOfStrings("exampleRefs");
        if (!exampleRefs.empty()) {
          J.attributeArray("exampleRefs", [&] {
            for (StringRef exampleRef : exampleRefs) {
              J.value(exampleRef);
            }
          });
        }

        auto textualExamples =
            metadata->getValueAsListOfStrings("textualExamples");
        if (!textualExamples.empty()) {
          J.attributeArray("textualExamples", [&] {
            for (StringRef example : textualExamples) {
              J.value(example);
            }
          });
        }

        auto tags = metadata->getValueAsListOfStrings("tags");
        if (!tags.empty()) {
          J.attributeArray("tags", [&] {
            for (StringRef tag : tags) {
              J.value(tag);
            }
          });
        }
      });
    }

    // Operands.
    J.attributeArray("operands", [&] {
      for (const auto &operand : op.getOperands()) {
        J.object([&] {
          J.attribute("name", operand.name);

          // Basic type information.
          if (!operand.constraint.getSummary().empty()) {
            J.attribute("summary", operand.constraint.getSummary());
          }
          if (!operand.constraint.getDescription().empty()) {
            J.attribute("description",
                        cleanDescription(operand.constraint.getDescription()));
          }

          // Type constraint details.
          // Skip anonymous TableGen-generated type constraint names.
          if (StringRef defName = operand.constraint.getDefName();
              !defName.empty() && !defName.starts_with("anonymous_")) {
            J.attribute("defName", defName);
          }

          // C++ type information.
          if (StringRef cppType = operand.constraint.getCppType();
              !cppType.empty()) {
            J.attribute("cppType", cppType);
          }

          // Emit structured constraint metadata if this is a union type.
          emitConstraintMetadata(operand.constraint, J);

          // Variadic/optional flags.
          J.attributeObject("properties", [&] {
            J.attribute("isOptional", operand.isOptional());
            J.attribute("isVariadic", operand.isVariadic());
            J.attribute("isVariadicOfVariadic", operand.isVariadicOfVariadic());
          });
        });
      }
    });

    // Results.
    J.attributeArray("results", [&] {
      for (const auto &result : op.getResults()) {
        J.object([&] {
          J.attribute("name", result.name);

          // Basic type information.
          if (!result.constraint.getSummary().empty()) {
            J.attribute("summary", result.constraint.getSummary());
          }
          if (!result.constraint.getDescription().empty()) {
            J.attribute("description",
                        cleanDescription(result.constraint.getDescription()));
          }

          // Type constraint details.
          // Skip anonymous TableGen-generated type constraint names.
          if (StringRef defName = result.constraint.getDefName();
              !defName.empty() && !defName.starts_with("anonymous_")) {
            J.attribute("defName", defName);
          }

          // C++ type information.
          if (StringRef cppType = result.constraint.getCppType();
              !cppType.empty()) {
            J.attribute("cppType", cppType);
          }

          // Emit structured constraint metadata if this is a union type.
          emitConstraintMetadata(result.constraint, J);

          // Variadic/optional flags.
          J.attributeObject("properties", [&] {
            J.attribute("isOptional", result.isOptional());
            J.attribute("isVariadic", result.isVariadic());
            J.attribute("isVariadicOfVariadic", result.isVariadicOfVariadic());
          });
        });
      }
    });

    // Traits.
    J.attributeArray("traits", [&] {
      for (const Trait &trait : op.getTraits()) {
        if (const auto *nt = dyn_cast<NativeTrait>(&trait)) {
          J.value(nt->getFullyQualifiedTraitName());
        } else if (const auto *it = dyn_cast<InterfaceTrait>(&trait)) {
          J.value(it->getFullyQualifiedTraitName());
        }
      }
    });

    // Builders.
    emitBuilders(op.getBuilders(), J);
  });
}

// Emits a single type as JSON.
static void emitTypeJSON(const Record &def, const RecordKeeper &records,
                         json::OStream &J) {
  // Wrap in TypeDef to access TableGen APIs.
  TypeDef type(&def);

  J.object([&] {
    J.attribute("name", def.getName());

    // Dialect reference.
    Dialect dialect = type.getDialect();
    J.attribute("dialect", dialect.getName());

    // C++ class names.
    J.attribute("cppClassName", type.getCppClassName());
    std::string fullyQualifiedName =
        (dialect.getCppNamespace() + "::" + type.getCppClassName()).str();
    J.attribute("fullyQualifiedName", fullyQualifiedName);

    // Source location.
    emitSourceLocation(&def, records, J);

    // Mnemonic and assembly format.
    if (def.getValue("mnemonic")) {
      if (StringRef mnemonic = def.getValueAsString("mnemonic");
          !mnemonic.empty()) {
        J.attribute("mnemonic", mnemonic);
      }
    }

    if (auto assemblyFormat = type.getAssemblyFormat()) {
      J.attribute("assemblyFormat", cleanAssemblyFormat(*assemblyFormat));
    }

    // Documentation.
    if (def.getValue("summary")) {
      if (StringRef summary = def.getValueAsString("summary");
          !summary.empty()) {
        J.attribute("summary", cleanDescription(summary));
      }
    }

    if (def.getValue("description")) {
      if (StringRef desc = def.getValueAsString("description"); !desc.empty()) {
        J.attribute("description", cleanDescription(desc));
      }
    }

    // Parameters.
    emitParameters(type.getParameters(), J);

    // Traits.
    emitTraits(type.getTraits(), J);

    // Emit custom metadata (only if the field exists).
    if (def.getValue("docMetadata")) {
      if (const Record *metadata = def.getValueAsOptionalDef("docMetadata")) {
        emitCommonMetadata(metadata, &def, J, "relatedTypes");
      }
    }
  });
}

// Emits a single attribute as JSON.
static void emitAttrJSON(const Record &def, const RecordKeeper &records,
                         json::OStream &J) {
  // Wrap in AttrDef to access TableGen APIs.
  AttrDef attr(&def);

  J.object([&] {
    J.attribute("name", def.getName());

    // Dialect reference.
    Dialect dialect = attr.getDialect();
    J.attribute("dialect", dialect.getName());

    // C++ class names.
    J.attribute("cppClassName", attr.getCppClassName());
    std::string fullyQualifiedName =
        (dialect.getCppNamespace() + "::" + attr.getCppClassName()).str();
    J.attribute("fullyQualifiedName", fullyQualifiedName);

    // Source location.
    emitSourceLocation(&def, records, J);

    // Mnemonic and assembly format.
    if (def.getValue("mnemonic")) {
      if (StringRef mnemonic = def.getValueAsString("mnemonic");
          !mnemonic.empty()) {
        J.attribute("mnemonic", mnemonic);
      }
    }

    if (auto assemblyFormat = attr.getAssemblyFormat()) {
      J.attribute("assemblyFormat", cleanAssemblyFormat(*assemblyFormat));
    }

    // Documentation.
    if (def.getValue("summary")) {
      if (StringRef summary = def.getValueAsString("summary");
          !summary.empty()) {
        J.attribute("summary", cleanDescription(summary));
      }
    }

    if (def.getValue("description")) {
      if (StringRef desc = def.getValueAsString("description"); !desc.empty()) {
        J.attribute("description", cleanDescription(desc));
      }
    }

    // Parameters.
    emitParameters(attr.getParameters(), J);

    // Traits.
    emitTraits(attr.getTraits(), J);

    // Emit custom metadata (only if the field exists).
    if (def.getValue("docMetadata")) {
      if (const Record *metadata = def.getValueAsOptionalDef("docMetadata")) {
        emitCommonMetadata(metadata, &def, J, "relatedAttrs");
      }
    }
  });
}

// Emits a single interface as JSON.
static void emitInterfaceJSON(const Record &def, const RecordKeeper &records,
                              json::OStream &J) {
  // Wrap in Interface to access TableGen APIs.
  Interface interface(&def);

  J.object([&] {
    J.attribute("name", def.getName());

    // C++ class names.
    std::string fullyQualifiedName = interface.getFullyQualifiedName();
    StringRef fqnRef(fullyQualifiedName);

    // Try to extract dialect name from namespace using generic patterns.
    if (auto dialectName = extractDialectFromNamespace(fqnRef)) {
      J.attribute("dialect", *dialectName);
    }

    // Extract class name from fully qualified name (part after last "::").
    size_t lastColons = fqnRef.rfind("::");
    if (lastColons != StringRef::npos) {
      J.attribute("cppClassName", fqnRef.substr(lastColons + 2));
    } else {
      J.attribute("cppClassName", fqnRef);
    }

    J.attribute("fullyQualifiedName", fullyQualifiedName);

    // Source location.
    emitSourceLocation(&def, records, J);

    // Determine interface type.
    std::string interfaceType = "unknown";
    if (def.isSubClassOf("OpInterface")) {
      interfaceType = "op";
    } else if (def.isSubClassOf("TypeInterface")) {
      interfaceType = "type";
    } else if (def.isSubClassOf("AttrInterface")) {
      interfaceType = "attr";
    }
    J.attribute("interfaceType", interfaceType);

    // Documentation.
    if (def.getValue("summary")) {
      if (StringRef summary = def.getValueAsString("summary");
          !summary.empty()) {
        J.attribute("summary", cleanDescription(summary));
      }
    }

    if (auto desc = interface.getDescription()) {
      J.attribute("description", cleanDescription(*desc));
    }

    // Base interfaces.
    auto baseInterfaces = interface.getBaseInterfaces();
    if (!baseInterfaces.empty()) {
      J.attributeArray("baseInterfaces", [&] {
        for (const Interface &base : baseInterfaces) {
          J.value(base.getFullyQualifiedName());
        }
      });
    }

    // Methods.
    emitInterfaceMethods(interface.getMethods(), J);

    // Emit custom metadata (only if the field exists).
    if (def.getValue("docMetadata")) {
      if (const Record *metadata = def.getValueAsOptionalDef("docMetadata")) {
        emitCommonMetadata(metadata, &def, J, "relatedInterfaces");
      }
    }
  });
}

// Main JSON generation function.
static bool emitDialectJSON(const RecordKeeper &records, raw_ostream &os) {
  // Get all definitions.
  auto dialects = getDialects(records);
  auto opDefs = getOpDefinitions(records);
  auto typeDefs = getTypeDefinitions(records);
  auto attrDefs = getAttrDefinitions(records);
  auto interfaceDefs = getInterfaceDefinitions(records);

  // Collect categories from operations.
  std::map<std::string, std::string> categories;
  for (const Record *def : opDefs) {
    if (const Record *metadata = def->getValueAsOptionalDef("docMetadata")) {
      if (const Record *docGroup =
              metadata->getValueAsOptionalDef("docGroup")) {
        StringRef categoryName = docGroup->getValueAsString("summary");
        StringRef categoryDesc = docGroup->getValueAsString("description");
        if (!categoryName.empty()) {
          // Store category if not already present.
          categories.try_emplace(categoryName.str(), categoryDesc.str());
        }
      }
    }
  }

  json::OStream J(os, 2);  // Indent with 2 spaces.
  J.object([&] {
    // Dialects.
    if (!dialects.empty()) {
      J.attributeObject("dialects", [&] {
        for (const Dialect &dialect : dialects) {
          J.attributeObject(dialect.getName(), [&] {
            // C++ class names.
            J.attribute("cppClassName", dialect.getCppClassName());
            std::string fullyQualifiedName =
                (dialect.getCppNamespace() + "::" + dialect.getCppClassName())
                    .str();
            J.attribute("fullyQualifiedName", fullyQualifiedName);

            // Source location.
            emitSourceLocation(dialect.getDef(), records, J);

            if (!dialect.getSummary().empty()) {
              J.attribute("summary", cleanDescription(dialect.getSummary()));
            }

            if (!dialect.getDescription().empty()) {
              J.attribute("description",
                          cleanDescription(dialect.getDescription()));
            }

            // Dependent dialects.
            ArrayRef<StringRef> deps = dialect.getDependentDialects();
            if (!deps.empty()) {
              J.attributeArray("dependentDialects", [&] {
                for (StringRef dep : deps) {
                  J.value(dep);
                }
              });
            }

            // Feature flags (only emit if any are non-default).
            if (dialect.hasCanonicalizer() ||
                dialect.hasConstantMaterializer() ||
                !dialect.useDefaultAttributePrinterParser() ||
                !dialect.useDefaultTypePrinterParser() ||
                dialect.isExtensible()) {
              J.attributeObject("features", [&] {
                J.attribute("hasCanonicalizer", dialect.hasCanonicalizer());
                J.attribute("hasConstantMaterializer",
                            dialect.hasConstantMaterializer());
                J.attribute("useDefaultAttributePrinterParser",
                            dialect.useDefaultAttributePrinterParser());
                J.attribute("useDefaultTypePrinterParser",
                            dialect.useDefaultTypePrinterParser());
                J.attribute("isExtensible", dialect.isExtensible());
              });
            }
          });
        }
      });
    }

    // Categories.
    if (!categories.empty()) {
      J.attributeObject("categories", [&] {
        for (const auto &category : categories) {
          J.attributeObject(category.first, [&] {
            if (!category.second.empty()) {
              J.attribute("description", category.second);
            }
          });
        }
      });
    }

    // Operations.
    if (!opDefs.empty()) {
      J.attributeArray("operations", [&] {
        for (const Record *def : opDefs) {
          emitOperatorJSON(Operator(def), records, J);
        }
      });
    }

    // Types.
    if (!typeDefs.empty()) {
      J.attributeArray("types", [&] {
        for (const Record *def : typeDefs) {
          emitTypeJSON(*def, records, J);
        }
      });
    }

    // Attributes.
    if (!attrDefs.empty()) {
      J.attributeArray("attributes", [&] {
        for (const Record *def : attrDefs) {
          emitAttrJSON(*def, records, J);
        }
      });
    }

    // Interfaces.
    if (!interfaceDefs.empty()) {
      J.attributeArray("interfaces", [&] {
        for (const Record *def : interfaceDefs) {
          emitInterfaceJSON(*def, records, J);
        }
      });
    }
  });

  return false;
}

// Register the JSON dialect doc generation backend.
GenRegistration genDialectJSON("gen-dialect-json",
                               "Generate JSON documentation for a dialect",
                               emitDialectJSON);

//===----------------------------------------------------------------------===//
// Other Generators
//===----------------------------------------------------------------------===//

// Generator that prints records.
GenRegistration genPrintRecords("print-records", "Print all records to stdout",
                                [](const RecordKeeper &records,
                                   raw_ostream &os) {
                                  os << records;
                                  return false;
                                });

int main(int argc, char **argv) { return MlirTblgenMain(argc, argv); }
