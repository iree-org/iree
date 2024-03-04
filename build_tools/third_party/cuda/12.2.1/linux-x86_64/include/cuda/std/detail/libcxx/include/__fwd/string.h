//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FWD_STRING_H
#define _LIBCUDACXX___FWD_STRING_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__fwd/memory_resource.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _CharT>
struct _LIBCUDACXX_TEMPLATE_VIS char_traits;
template <>
struct char_traits<char>;

#ifndef _LIBCUDACXX_NO_HAS_CHAR8_T
template <>
struct char_traits<char8_t>;
#endif

template <>
struct char_traits<char16_t>;
template <>
struct char_traits<char32_t>;

#ifndef _LIBCUDACXX_HAS_NO_WIDE_CHARACTERS
template <>
struct char_traits<wchar_t>;
#endif

template <class _Tp>
class _LIBCUDACXX_TEMPLATE_VIS allocator;

template <class _CharT, class _Traits = char_traits<_CharT>, class _Allocator = allocator<_CharT> >
class _LIBCUDACXX_TEMPLATE_VIS basic_string;

using string = basic_string<char>;

#ifndef _LIBCUDACXX_HAS_NO_WIDE_CHARACTERS
using wstring = basic_string<wchar_t>;
#endif

#ifndef _LIBCUDACXX_NO_HAS_CHAR8_T
using u8string = basic_string<char8_t>;
#endif

using u16string = basic_string<char16_t>;
using u32string = basic_string<char32_t>;

#if _LIBCUDACXX_STD_VER >= 17

namespace pmr {
template <class _CharT, class _Traits = char_traits<_CharT>>
using basic_string = std::basic_string<_CharT, _Traits, polymorphic_allocator<_CharT>>;

using string = basic_string<char>;

#  ifndef _LIBCUDACXX_HAS_NO_WIDE_CHARACTERS
using wstring = basic_string<wchar_t>;
#  endif

#  ifndef _LIBCUDACXX_NO_HAS_CHAR8_T
using u8string = basic_string<char8_t>;
#  endif

using u16string = basic_string<char16_t>;
using u32string = basic_string<char32_t>;

} // namespace pmr

#endif // _LIBCUDACXX_STD_VER >= 17

// clang-format off
template <class _CharT, class _Traits, class _Allocator>
class _LIBCUDACXX_PREFERRED_NAME(string)
#ifndef _LIBCUDACXX_HAS_NO_WIDE_CHARACTERS
      _LIBCUDACXX_PREFERRED_NAME(wstring)
#endif
#ifndef _LIBCUDACXX_NO_HAS_CHAR8_T
      _LIBCUDACXX_PREFERRED_NAME(u8string)
#endif
      _LIBCUDACXX_PREFERRED_NAME(u16string)
      _LIBCUDACXX_PREFERRED_NAME(u32string)
#if _LIBCUDACXX_STD_VER >= 17
      _LIBCUDACXX_PREFERRED_NAME(pmr::string)
#  ifndef _LIBCUDACXX_HAS_NO_WIDE_CHARACTERS
      _LIBCUDACXX_PREFERRED_NAME(pmr::wstring)
#  endif
#  ifndef _LIBCUDACXX_NO_HAS_CHAR8_T
      _LIBCUDACXX_PREFERRED_NAME(pmr::u8string)
#  endif
      _LIBCUDACXX_PREFERRED_NAME(pmr::u16string)
      _LIBCUDACXX_PREFERRED_NAME(pmr::u32string)
#endif
      basic_string;
// clang-format on

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FWD_STRING_H
