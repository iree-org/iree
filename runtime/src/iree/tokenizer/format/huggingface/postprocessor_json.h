// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HuggingFace tokenizer.json post_processor section parser.
//
// This header provides parsing for the "post_processor" object in HuggingFace
// tokenizer.json files. The post-processor configures special token insertion
// around encoded sequences (BOS, EOS, CLS, SEP, etc.) and offset trimming.
//
// All HuggingFace post-processor types are compiled at parse time into a single
// flat precomputed representation (iree_tokenizer_postprocessor_t). No
// type-specific dispatch occurs at runtime.
//
// Supported post-processor types:
//   - TemplateProcessing: Flexible piece-array templates with special_tokens
//   map
//   - BertProcessing: [CLS] $A [SEP] / [CLS] $A [SEP] $B [SEP]
//   - RobertaProcessing: Like BERT but double-sep infix, all type_ids=0
//   - ByteLevel: No special tokens, offset trimming only
//   - Sequence: Chains processors (extracts ByteLevel + first token-producing)

#ifndef IREE_TOKENIZER_FORMAT_HUGGINGFACE_POSTPROCESSOR_JSON_H_
#define IREE_TOKENIZER_FORMAT_HUGGINGFACE_POSTPROCESSOR_JSON_H_

#include "iree/base/api.h"
#include "iree/tokenizer/postprocessor.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Post-Processor Parser
//===----------------------------------------------------------------------===//

// Parses a post-processor from its JSON value (the object containing "type").
//
// Dispatches to type-specific parsers based on the "type" field. Each type is
// compiled down to the flat prefix/infix/suffix template representation at
// parse time.
//
// For TemplateProcessing: walks single/pair piece arrays, resolves SpecialToken
// names via the special_tokens map to get numeric IDs, and extracts type_ids.
// Multi-token special tokens are supported (ids array with multiple entries).
//
// For Sequence: recursively parses child processors, extracts trim_offsets from
// any ByteLevel child, and compiles the first token-producing child.
//
// The |postprocessor_value| is the JSON text of the post_processor object
// itself (e.g.,
// '{"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}'). Pass
// "null" to leave the postprocessor at its zero-initialized state (no-op).
//
// The |out_postprocessor| is caller-owned storage that will be initialized on
// success. Call iree_tokenizer_postprocessor_deinitialize() when done.
//
// Returns:
//   - IREE_STATUS_OK on success
//   - IREE_STATUS_INVALID_ARGUMENT for malformed JSON or constraint violations
//   - IREE_STATUS_UNIMPLEMENTED for unsupported post-processor types
iree_status_t iree_tokenizer_huggingface_parse_postprocessor(
    iree_string_view_t postprocessor_value,
    iree_tokenizer_postprocessor_t* out_postprocessor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_FORMAT_HUGGINGFACE_POSTPROCESSOR_JSON_H_
