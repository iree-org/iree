// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_LATCH_H
#define _LIBCUDACXX___CUDA_LATCH_H

#ifndef __cuda_std__
#error "<__cuda/latch> should only be included in from <cuda/std/latch>"
#endif // __cuda_std__

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template<thread_scope _Sco>
class latch : public _CUDA_VSTD::__latch_base<_Sco> {
public:
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
    latch(_CUDA_VSTD::ptrdiff_t __count)
        : _CUDA_VSTD::__latch_base<_Sco>(__count) {
    }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___CUDA_LATCH_H
