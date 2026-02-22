// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Split segmenter for regex-based pre-tokenization.
//
// Splits text using a compiled regex DFA pattern. Supports all HuggingFace
// Split behaviors: REMOVED, ISOLATED, MERGED_WITH_PREVIOUS, MERGED_WITH_NEXT,
// and CONTIGUOUS. This enables:
//
//   - GPT-2 tokenizers with `use_regex=true` (ByteLevel pre-tokenizer)
//   - Explicit `Split` pre-tokenizers in HuggingFace tokenizer.json
//   - Custom regex-based splitting patterns
//
// Behavior Overview:
//
//   ISOLATED (GPT-2 default): Regex matches become independent segments.
//     Pattern matches TOKENS (words, punctuation, whitespace groups).
//     Each match is emitted as its own segment; gaps between matches are
//     also emitted (for non-matching text between tokens).
//     Example: "Hello world" with word pattern -> ["Hello", " ", "world"]
//
//   REMOVED: Discard matched delimiters entirely.
//     "a,b" with pattern "," -> ["a", "b"]
//
//   MERGED_WITH_PREVIOUS: Append delimiter to previous segment.
//     "a,b" with pattern "," -> ["a,", "b"]
//
//   MERGED_WITH_NEXT: Prepend delimiter to next segment.
//     "a,b" with pattern "," -> ["a", ",b"]
//
//   CONTIGUOUS: Merge consecutive delimiters into one segment.
//     "a,,b" with pattern "," -> ["a", ",,", "b"]
//
// Invert Mode:
//   When |invert| is true, the pattern semantics flip:
//   - Normal: Pattern matches DELIMITERS (split ON matches)
//   - Invert: Pattern matches TOKENS (matches ARE segments)
//
//   GPT-2 uses invert=false with a token-matching pattern in ISOLATED mode,
//   so matches become the segments directly (no delimiter concept).
//
// Streaming Behavior:
//   - Uses regex executor's streaming API for cross-chunk matching
//   - Buffers partial matches that span chunk boundaries
//   - process() returns relative offsets; finalize() returns absolute offsets

#ifndef IREE_TOKENIZER_SEGMENTER_SPLIT_H_
#define IREE_TOKENIZER_SEGMENTER_SPLIT_H_

#include "iree/base/api.h"
#include "iree/tokenizer/regex/exec.h"
#include "iree/tokenizer/segmenter.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// iree_tokenizer_segmenter_split_t
//===----------------------------------------------------------------------===//

// Allocates a segmenter that splits text using a compiled regex DFA pattern.
// The |dfa| handle must point into |dfa_data| as returned by the regex
// compiler. Ownership of |dfa_data| transfers to the segmenter on success and
// the segmenter will free it when destroyed; on failure the caller remains
// responsible for freeing |dfa_data|.
//
// The |behavior| controls how regex matches are converted to segments. With
// ISOLATED behavior (GPT-2 default) each match becomes its own segment with
// gaps between matches also emitted. With REMOVED behavior matches are
// discarded. MERGED_WITH_PREVIOUS and MERGED_WITH_NEXT attach the matched
// delimiter to the adjacent segment. CONTIGUOUS merges consecutive matches.
//
// When |invert| is true the pattern semantics are flipped: matches become
// segment boundaries rather than segment content.
iree_status_t iree_tokenizer_segmenter_split_allocate(
    iree_tokenizer_regex_dfa_t dfa, uint8_t* dfa_data,
    iree_tokenizer_regex_split_behavior_t behavior, bool invert,
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter);

// Allocates a segmenter that splits text on literal string matches.
// Unlike the regex version, this uses efficient byte-by-byte substring search.
// The |pattern| is copied and owned by the segmenter.
//
// The |behavior| and |invert| flags work identically to the regex version:
// - ISOLATED: Each match becomes its own segment (with gaps between)
// - REMOVED: Matches are discarded, only gaps are emitted
// - MERGED_WITH_PREVIOUS/NEXT: Match merges with adjacent segment
// - CONTIGUOUS: Consecutive matches merge into one segment
iree_status_t iree_tokenizer_segmenter_split_literal_allocate(
    iree_string_view_t pattern, iree_tokenizer_regex_split_behavior_t behavior,
    bool invert, iree_allocator_t allocator,
    iree_tokenizer_segmenter_t** out_segmenter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_SEGMENTER_SPLIT_H_
