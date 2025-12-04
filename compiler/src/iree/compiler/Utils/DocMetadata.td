// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_DOC_METADATA
#define IREE_COMPILER_UTILS_DOC_METADATA

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Documentation metadata classes
//===----------------------------------------------------------------------===//
//
// These classes provide rich metadata for generating comprehensive
// documentation from TableGen definitions. The metadata is extracted by
// iree-tblgen's --gen-dialect-json backend and used to generate cross-linked
// Markdown documentation that is useful for both humans and LLMs navigating the
// codebase as well as sharded JSON files for use with tools like `jq`.
//
// Usage Guidelines:
// ----------------
//
// 1. Operation Metadata:
//    - Reference an OpDocGroup for category organization (reuses upstream MLIR)
//    - Use exampleRefs to point to test functions (e.g., ["@resourceAlloc"])
//    - The doc generator extracts entire // ----- regions from test files
//      containing those symbols, allowing for well-documented example cases
//
//    Example:
//      def MyDialect_MyOpDocMetadata : Util_OpDocMetadata {
//        let docGroup = MyOpsGroup;
//        let testFile = "test/my_ops.mlir";
//        let exampleRefs = ["@basicExample", "@advancedUsage"];
//        let relatedOps = ["other.op", "another.op"];
//        let tags = ["experimental"];
//      }
//
// 2. Type/Attribute/Interface Metadata:
//    - Use category string for grouping (e.g., "Resource Types")
//    - Follow same exampleRefs pattern as operations
//
// 3. Test File Example Extraction:
//    - Examples are split by // ----- delimiters
//    - FileCheck comments (// CHECK*) are automatically stripped
//    - Regular comments are preserved in documentation
//
//===----------------------------------------------------------------------===//

// Metadata for operation documentation.
// Captures rich information for generating comprehensive operation documentation.
class Util_OpDocMetadata {
  // Category group (reuses upstream MLIR OpDocGroup).
  OpDocGroup docGroup = ?;
  // Cross-references to related operations.
  list<string> relatedOps = [];
  // Path to primary test file with examples.
  string testFile = "";
  // Function symbols in testFile to extract as examples (e.g.,
  // ["@resourceAlloc", "@resourceAllocWithAffinity"]).
  // Doc generator extracts the entire // ----- split region, strips CHECK
  // comments.
  list<string> exampleRefs = [];
  // Non-code illustrative examples (prose, diagrams, ASCII art, etc.).
  list<string> textualExamples = [];
  // Semantic tags (experimental, deprecated, performance-critical, etc.).
  list<string> tags = [];
}

// Metadata for pass documentation.
// Used to document compiler passes and their transformations.
class Util_PassDocMetadata {
  // Pass category (e.g., "Optimization Passes", "Lowering Passes").
  string category = "";
  // Path to primary test file.
  string testFile = "";
  // Function symbols in testFile to extract (e.g., ["@beforePass",
  // "@afterPass"]). Doc generator extracts // ----- regions,
  // strips CHECK comments.
  list<string> exampleRefs = [];
  // Non-code illustrative examples (prose, transformation diagrams, etc.).
  list<string> textualExamples = [];
  // Which pipeline uses this pass (e.g., "Flow", "Stream", "HAL").
  string pipelineStage = "";
  // Related passes (often run before/after in sequence).
  list<string> relatedPasses = [];
  // Semantic tags.
  list<string> tags = [];
}

// Metadata for type documentation.
// Used to document custom dialect types and their usage.
class Util_TypeDocMetadata {
  // Type category (e.g., "Resource Types", "Handle Types").
  string category = "";
  // Cross-references to related types.
  list<string> relatedTypes = [];
  // Path to primary test file.
  string testFile = "";
  // Function symbols in testFile showing usage examples.
  list<string> exampleRefs = [];
  // Non-code illustrative examples.
  list<string> textualExamples = [];
  // Semantic tags.
  list<string> tags = [];
}

// Metadata for attribute documentation.
// Used to document custom dialect attributes and their semantics.
class Util_AttrDocMetadata {
  // Attribute category (e.g., "Execution Attributes", "Layout Attributes").
  string category = "";
  // Cross-references to related attributes.
  list<string> relatedAttrs = [];
  // Path to primary test file.
  string testFile = "";
  // Function symbols in testFile showing usage examples.
  list<string> exampleRefs = [];
  // Non-code illustrative examples.
  list<string> textualExamples = [];
  // Semantic tags.
  list<string> tags = [];
}

// Metadata for interface documentation.
// Used to document op/type/attr interfaces and their contracts.
class Util_InterfaceDocMetadata {
  // Interface category (e.g., "Op Interfaces", "Type Interfaces").
  string category = "";
  // Cross-references to related interfaces.
  list<string> relatedInterfaces = [];
  // Path to primary test file.
  string testFile = "";
  // Function symbols in testFile showing implementations.
  list<string> exampleRefs = [];
  // Non-code illustrative examples.
  list<string> textualExamples = [];
  // Semantic tags.
  list<string> tags = [];
}

#endif  // IREE_COMPILER_UTILS_DOC_METADATA
