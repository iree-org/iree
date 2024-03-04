/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef _LIBCUDACXX___MDSPAN_DEFAULT_ACCESSOR_HPP
#define _LIBCUDACXX___MDSPAN_DEFAULT_ACCESSOR_HPP

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__mdspan/macros.h"
#include "../__type_traits/is_convertible.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11

template <class _ElementType>
struct default_accessor {

  using offset_policy = default_accessor;
  using element_type = _ElementType;
  using reference = _ElementType&;
  using data_handle_type = _ElementType*;

  __MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr default_accessor() noexcept = default;

  __MDSPAN_TEMPLATE_REQUIRES(
    class _OtherElementType,
    /* requires */ (
      _LIBCUDACXX_TRAIT(is_convertible, _OtherElementType(*)[], element_type(*)[])
    )
  )
  __MDSPAN_INLINE_FUNCTION
  constexpr default_accessor(default_accessor<_OtherElementType>) noexcept {}

  __MDSPAN_INLINE_FUNCTION
  constexpr data_handle_type
  offset(data_handle_type __p, size_t __i) const noexcept {
    return __p + __i;
  }

  __MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference access(data_handle_type __p, size_t __i) const noexcept {
    return __p[__i];
  }

};

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MDSPAN_DEFAULT_ACCESSOR_HPP
