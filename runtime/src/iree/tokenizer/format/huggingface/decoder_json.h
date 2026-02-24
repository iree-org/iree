// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HuggingFace tokenizer.json decoder section parser.
//
// This header provides parsing for the "decoder" object in HuggingFace
// tokenizer.json files. The decoder transforms token IDs back into text
// (reversing pre-tokenizer transformations like Metaspace, byte-level, etc.).
//
// Supported decoder types:
//   - Sequence: Chains multiple decoders in order
//   - ByteFallback: Reconstructs bytes from <0xNN> tokens
//   - ByteLevel: Reverses GPT-2 byte-to-unicode encoding
//   - CTC: Speech-to-text decoder (Wav2Vec2, HuBERT)
//   - Metaspace: Replaces ▁ with spaces (SentencePiece-style)
//   - Replace: String pattern replacement (String patterns only, not Regex)
//   - Strip: Removes characters from token boundaries
//   - WordPiece: Handles ## continuation prefixes
//   - Fuse: No-op in streaming architecture (returns NULL)

#ifndef IREE_TOKENIZER_FORMAT_HUGGINGFACE_DECODER_JSON_H_
#define IREE_TOKENIZER_FORMAT_HUGGINGFACE_DECODER_JSON_H_

#include "iree/base/api.h"
#include "iree/tokenizer/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Decoder Parser
//===----------------------------------------------------------------------===//

// Parses a decoder from its JSON value (the object containing "type").
//
// Dispatches to type-specific parsers based on the "type" field. For Sequence
// decoders, recursively parses children and compacts out NULL (no-op) children.
// A sequence with 0 children after compaction returns NULL, and a sequence
// with 1 child returns that child directly (no wrapper).
//
// The "Fuse" decoder type returns NULL (no-op) because our streaming
// architecture handles first-token detection via state, not array indexing.
//
// The |decoder_value| is the JSON text of the decoder object itself
// (e.g., '{"type":"ByteFallback"}'), not the full tokenizer.json file.
// Pass "null" to get a NULL (passthrough) decoder.
//
// Returns:
//   - IREE_STATUS_OK on success (out_decoder may be NULL for passthrough)
//   - IREE_STATUS_INVALID_ARGUMENT for malformed JSON
//   - IREE_STATUS_NOT_FOUND for missing required fields (e.g., CTC tokens)
//   - IREE_STATUS_UNIMPLEMENTED for unsupported decoder types or Regex Replace
iree_status_t iree_tokenizer_huggingface_parse_decoder(
    iree_string_view_t decoder_value, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_FORMAT_HUGGINGFACE_DECODER_JSON_H_
