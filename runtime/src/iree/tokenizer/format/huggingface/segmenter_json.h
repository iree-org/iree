// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HuggingFace tokenizer.json pre_tokenizer section parser.
//
// This header provides parsing for the "pre_tokenizer" object in HuggingFace
// tokenizer.json files. The pre_tokenizer controls how text is split into
// initial segments before the model's tokenization algorithm runs.
//
// HuggingFace calls this "pre_tokenizer"; IREE calls the result a "segmenter"
// since it segments text into chunks for independent model processing.
//
// Supported pre_tokenizer types:
//   - ByteLevel: GPT-2 regex splitting (creates Split segmenter with DFA)
//   - Metaspace: SentencePiece-style whitespace replacement and splitting
//   - Punctuation: Splits on punctuation chars with configurable behavior

#ifndef IREE_TOKENIZER_FORMAT_HUGGINGFACE_SEGMENTER_JSON_H_
#define IREE_TOKENIZER_FORMAT_HUGGINGFACE_SEGMENTER_JSON_H_

#include "iree/base/api.h"
#include "iree/tokenizer/format/huggingface/types.h"
#include "iree/tokenizer/segmenter.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Segmenter Parser (HuggingFace "pre_tokenizer")
//===----------------------------------------------------------------------===//

// Parses a pre_tokenizer from its JSON value (the object containing "type").
//
// Dispatches to type-specific parsers based on the "type" field:
//   - "ByteLevel": Compiles GPT-2 regex to DFA, creates Split segmenter
//     (when use_regex=true; otherwise returns NULL for passthrough)
//   - "Metaspace": Creates Metaspace segmenter with replacement codepoint
//   - "Split": Compiles regex pattern to DFA, creates Split segmenter
//   - "Sequence": Chains multiple child pre-tokenizers
//
// The |pre_tokenizer_value| is the JSON text of the pre_tokenizer object
// (e.g., '{"type":"Metaspace","replacement":"▁","split":true}').
// Pass "null" to get a NULL (passthrough) segmenter.
//
// On success, |out_flags| receives format-level properties discovered during
// parsing (e.g., ByteLevel presence). These flags are for the orchestrator to
// route to other section parsers; they do not affect the segmenter itself.
//
// Returns:
//   - IREE_STATUS_OK on success (out_segmenter may be NULL for passthrough)
//   - IREE_STATUS_INVALID_ARGUMENT for malformed JSON or invalid regex
//   - IREE_STATUS_UNIMPLEMENTED for unsupported pre_tokenizer types
iree_status_t iree_tokenizer_huggingface_parse_segmenter(
    iree_string_view_t pre_tokenizer_value, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter,
    iree_tokenizer_huggingface_pre_tokenizer_flags_t* out_flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_FORMAT_HUGGINGFACE_SEGMENTER_JSON_H_
