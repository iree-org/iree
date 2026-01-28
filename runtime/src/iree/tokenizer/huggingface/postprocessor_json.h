// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON parsing for token post-processors from HuggingFace tokenizer.json.
//
// Parses the `post_processor` field from tokenizer.json files into
// iree_tokenizer_postprocessor_t instances.
//
// Supported post-processor types:
//   - null/missing: No special tokens added (GPT-2 style)
//   - TemplateProcessing: Configurable template-based insertion (BERT, LLaMA)
//   - RobertaProcessing: RoBERTa-specific handling
//   - Sequence: Chain of processors (Llama 3 style)
//   - ByteLevel: For Sequence chains
//
// Usage:
//   iree_tokenizer_postprocessor_t pp;
//   iree_status_t status = iree_tokenizer_postprocessor_parse_json(
//       json_root, allocator, &pp);
//   // ... use pp ...
//   iree_tokenizer_postprocessor_deinitialize(&pp);

#ifndef IREE_TOKENIZER_HUGGINGFACE_POSTPROCESSOR_JSON_H_
#define IREE_TOKENIZER_HUGGINGFACE_POSTPROCESSOR_JSON_H_

#include "iree/tokenizer/postprocessor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses a post_processor object from tokenizer.json into a postprocessor.
//
// |json_root| is the root JSON object (containing "post_processor" field).
// |allocator| is used for any heap allocations (templates, sequences).
// |out_pp| receives the parsed post-processor on success.
//
// If the post_processor field is missing or null, returns a NONE postprocessor.
// If the type is unsupported, returns IREE_STATUS_UNIMPLEMENTED.
// If the JSON structure is invalid, returns IREE_STATUS_INVALID_ARGUMENT.
//
// The caller must call iree_tokenizer_postprocessor_deinitialize() when done.
iree_status_t iree_tokenizer_postprocessor_parse_json(
    iree_string_view_t json_root, iree_allocator_t allocator,
    iree_tokenizer_postprocessor_t* out_pp);

// Parses a single post_processor object (not the root).
// Used for parsing nested processors in Sequence.
iree_status_t iree_tokenizer_postprocessor_parse_field(
    iree_string_view_t pp_json, iree_allocator_t allocator,
    iree_tokenizer_postprocessor_t* out_pp);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_HUGGINGFACE_POSTPROCESSOR_JSON_H_
