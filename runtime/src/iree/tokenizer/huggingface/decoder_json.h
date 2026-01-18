// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON parsing for token decoders from HuggingFace tokenizer.json.
//
// Parses the `decoder` field from tokenizer.json files into
// iree_tokenizer_decoder_t instances.
//
// Supported decoder types:
//   - WordPiece: Strip "##" prefix from continuation tokens
//   - Metaspace: Replace ▁ (U+2581) with space
//   - ByteLevel: Reverse GPT-2 byte encoding
//   - BPE: Basic concatenation (passthrough)
//
// Usage:
//   iree_tokenizer_decoder_t decoder;
//   iree_status_t status = iree_tokenizer_decoder_parse_json(
//       json_root, &decoder);
//   // ... use decoder ...
//   iree_tokenizer_decoder_deinitialize(&decoder);

#ifndef IREE_TOKENIZER_HUGGINGFACE_DECODER_JSON_H_
#define IREE_TOKENIZER_HUGGINGFACE_DECODER_JSON_H_

#include "iree/tokenizer/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses a decoder object from tokenizer.json into a decoder.
//
// |json_root| is the root JSON object (containing "decoder" field).
// |out_decoder| receives the parsed decoder on success.
//
// If the decoder field is missing or null, returns a NONE decoder.
// If the type is unsupported, returns IREE_STATUS_UNIMPLEMENTED.
// If the JSON structure is invalid, returns IREE_STATUS_INVALID_ARGUMENT.
//
// The caller must call iree_tokenizer_decoder_deinitialize() when done.
iree_status_t iree_tokenizer_decoder_parse_json(
    iree_string_view_t json_root, iree_tokenizer_decoder_t* out_decoder);

// Parses a single decoder object (not the root).
iree_status_t iree_tokenizer_decoder_parse_decoder_field(
    iree_string_view_t decoder_json, iree_tokenizer_decoder_t* out_decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_HUGGINGFACE_DECODER_JSON_H_
