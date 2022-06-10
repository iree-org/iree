// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_YAML_UTIL_H_
#define IREE_TOOLING_YAML_UTIL_H_

#include "iree/base/api.h"

#define YAML_DECLARE_STATIC
#include <yaml.h>  // IWYU pragma: export

// Wraps a YAML parser error in an iree_status_t.
iree_status_t iree_status_from_yaml_parser_error(yaml_parser_t* parser);

// Returns the given scalar |node| as an iree_string_view_t or empty string if
// the node is not scalar.
iree_string_view_t iree_yaml_node_as_string(yaml_node_t* node);

// Returns true if the scalar |node| is equal to |value|.
bool iree_yaml_string_equal(yaml_node_t* node, iree_string_view_t value);

// Finds the mapping pair with |key|, returning OK and *|out_value| == NULL if
// it does not exist within |node|.
iree_status_t iree_yaml_mapping_try_find(yaml_document_t* document,
                                         yaml_node_t* node,
                                         iree_string_view_t key,
                                         yaml_node_t** out_value);

// Finds the mapping pair with |key|, returning NOT_FOUND if it does not exist
// within |node|.
iree_status_t iree_yaml_mapping_find(yaml_document_t* document,
                                     yaml_node_t* node, iree_string_view_t key,
                                     yaml_node_t** out_value);

// Calculates the size, in bytes, of |source| once decoded as a base64 value.
size_t iree_yaml_base64_calculate_size(iree_string_view_t source);

// Decodes |source| as a base64 value into |target| which must be at least the
// size returned by iree_yaml_base64_calculate_size.
iree_status_t iree_yaml_base64_decode(iree_string_view_t source,
                                      iree_byte_span_t target);

#endif  // IREE_TOOLING_YAML_UTIL_H_
