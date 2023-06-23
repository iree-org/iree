// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_DATATILINGUNIVERSALPADDING_H_
#define IREE_COMPILER_UTILS_DATATILINGUNIVERSALPADDING_H_

namespace mlir {
namespace iree_compiler {

// When using data-tiling, during Flow, the SetEncoding pass must ensure that
// allocated buffers will be large enough for the eventual padded-and-tiled
// buffers. Those will only be created in the MaterializeEncoding pass, in HAL.
// Until then, the exact tile sizes aren't know. Our short-term approach to
// unblock this is to let SetEncoding pad everything to the next multiple of
// a "universal" padding size. In order for this to work, this universal padding
// value must be greater than or equal to any actual tile size that can occur.
//
// This widening of tensors is particularly problematic for narrow tensors. For
// example, it is inefficient to rewrite a tensor<1x1024xf32> into
// tensor<16x1024xf32>, using only row 0, leaving the other 15 rows unused. To
// remedy that in the short term until a better solution is found, we have the
// following contract: for any dimension that is statically sized and whose size
// is less than DataTilingUniversalPadding, the largest tile size that
// MaterializeEncoding is allowed to choose is the original dimension size
// rounded up to the next power of two.
//
// Example. If DataTilingUniversalPadding=16, then:
//
// For the source tensor type | MaterializeEncoding can choose tile sizes up to
// -------------------------- | -----------------------------------------------
// tensor<20x40xf32>          | 16x16
// tensor<20x1xf32>           | 16x1
// tensor<1x40xf32>           | 1x16
// tensor<1x1xf32>            | 1x1
// tensor<20x2xf32>           | 16x2
// tensor<20x3xf32>           | 16x4
// tensor<20x4xf32>           | 16x4
// tensor<20x5xf32>           | 16x8
//
// TODO(#11632) - find a way to do without universal padding.
const int DataTilingUniversalPadding = 16;

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_DATATILINGUNIVERSALPADDING_H_