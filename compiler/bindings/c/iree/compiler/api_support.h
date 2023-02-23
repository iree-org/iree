// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_API_SUPPORT_H
#define IREE_COMPILER_API_SUPPORT_H

#if (defined(_WIN32) || defined(__CYGWIN__))
// Visibility annotations disabled.
#define IREE_EMBED_EXPORTED
#elif defined(_WIN32) || defined(__CYGWIN__)
// Windows visibility declarations.
#if IREE_EMBED_BUILDING_LIBRARY
#define IREE_EMBED_EXPORTED __declspec(dllexport)
#else
#define IREE_EMBED_EXPORTED __declspec(dllimport)
#endif
#else
// Non-windows: use visibility attributes.
#define IREE_EMBED_EXPORTED __attribute__((visibility("default")))
#endif

#endif  // IREE_COMPILER_API_SUPPORT_H
