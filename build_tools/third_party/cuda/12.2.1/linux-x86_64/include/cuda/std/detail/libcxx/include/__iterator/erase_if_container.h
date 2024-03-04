// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ERASE_IF_CONTAINER_H
#define _LIBCUDACXX___ITERATOR_ERASE_IF_CONTAINER_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Container, class _Predicate>
_LIBCUDACXX_INLINE_VISIBILITY
typename _Container::size_type
__libcpp_erase_if_container(_Container& __c, _Predicate& __pred) {
  typename _Container::size_type __old_size = __c.size();

  const typename _Container::iterator __last = __c.end();
  for (typename _Container::iterator __iter = __c.begin(); __iter != __last;) {
    if (__pred(*__iter))
      __iter = __c.erase(__iter);
    else
      ++__iter;
  }

  return __old_size - __c.size();
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_ERASE_IF_CONTAINER_H
