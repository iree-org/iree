// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON parsing for literal tokens.
//
// Parses the "added_tokens" array from HuggingFace tokenizer.json files
// into a literals collection. Handles:
// - Token ID and content extraction
// - Flag parsing (lstrip, rstrip, single_word, normalized)
// - Special token detection

#ifndef IREE_TOKENIZER_HUGGINGFACE_LITERALS_JSON_H_
#define IREE_TOKENIZER_HUGGINGFACE_LITERALS_JSON_H_

#include "iree/base/api.h"
#include "iree/tokenizer/literals.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses literals from the "added_tokens" array in tokenizer.json.
//
// |json_root| is the root JSON object (containing "added_tokens").
// |literals| must be initialized before calling this function.
// Returns OK if added_tokens is missing (it's optional).
//
// JSON format expected:
// {
//   "added_tokens": [
//     {
//       "id": 50264,
//       "content": "<mask>",
//       "special": true,
//       "lstrip": true,
//       "rstrip": false,
//       "single_word": false,
//       "normalized": false
//     },
//     ...
//   ]
// }
iree_status_t iree_tokenizer_literals_parse_json(
    iree_string_view_t json_root, iree_tokenizer_literals_t* literals);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_HUGGINGFACE_LITERALS_JSON_H_
