// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_POINTER_TRAITS_H
#define _LIBCUDACXX___MEMORY_POINTER_TRAITS_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__memory/addressof.h"
#include "../__type_traits/conjunction.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_class.h"
#include "../__type_traits/is_function.h"
#include "../__type_traits/is_void.h"
#include "../__type_traits/void_t.h"
#include "../__utility/declval.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp, class = void>
struct __has_element_type : false_type {};

template <class _Tp>
struct __has_element_type<_Tp,
              __void_t<typename _Tp::element_type>> : true_type {};

template <class _Ptr, bool = __has_element_type<_Ptr>::value>
struct __pointer_traits_element_type;

template <class _Ptr>
struct __pointer_traits_element_type<_Ptr, true>
{
    typedef _LIBCUDACXX_NODEBUG_TYPE typename _Ptr::element_type type;
};

#ifndef _LIBCUDACXX_HAS_NO_VARIADICS

template <template <class, class...> class _Sp, class _Tp, class ..._Args>
struct __pointer_traits_element_type<_Sp<_Tp, _Args...>, true>
{
    typedef _LIBCUDACXX_NODEBUG_TYPE typename _Sp<_Tp, _Args...>::element_type type;
};

template <template <class, class...> class _Sp, class _Tp, class ..._Args>
struct __pointer_traits_element_type<_Sp<_Tp, _Args...>, false>
{
    typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;
};

#else  // _LIBCUDACXX_HAS_NO_VARIADICS

template <template <class> class _Sp, class _Tp>
struct __pointer_traits_element_type<_Sp<_Tp>, true>
{
    typedef typename _Sp<_Tp>::element_type type;
};

template <template <class> class _Sp, class _Tp>
struct __pointer_traits_element_type<_Sp<_Tp>, false>
{
    typedef _Tp type;
};

template <template <class, class> class _Sp, class _Tp, class _A0>
struct __pointer_traits_element_type<_Sp<_Tp, _A0>, true>
{
    typedef typename _Sp<_Tp, _A0>::element_type type;
};

template <template <class, class> class _Sp, class _Tp, class _A0>
struct __pointer_traits_element_type<_Sp<_Tp, _A0>, false>
{
    typedef _Tp type;
};

template <template <class, class, class> class _Sp, class _Tp, class _A0, class _A1>
struct __pointer_traits_element_type<_Sp<_Tp, _A0, _A1>, true>
{
    typedef typename _Sp<_Tp, _A0, _A1>::element_type type;
};

template <template <class, class, class> class _Sp, class _Tp, class _A0, class _A1>
struct __pointer_traits_element_type<_Sp<_Tp, _A0, _A1>, false>
{
    typedef _Tp type;
};

template <template <class, class, class, class> class _Sp, class _Tp, class _A0,
                                                           class _A1, class _A2>
struct __pointer_traits_element_type<_Sp<_Tp, _A0, _A1, _A2>, true>
{
    typedef typename _Sp<_Tp, _A0, _A1, _A2>::element_type type;
};

template <template <class, class, class, class> class _Sp, class _Tp, class _A0,
                                                           class _A1, class _A2>
struct __pointer_traits_element_type<_Sp<_Tp, _A0, _A1, _A2>, false>
{
    typedef _Tp type;
};

#endif  // _LIBCUDACXX_HAS_NO_VARIADICS

template <class _Tp, class = void>
struct __has_difference_type : false_type {};

template <class _Tp>
struct __has_difference_type<_Tp,
            __void_t<typename _Tp::difference_type>> : true_type {};

template <class _Ptr, bool = __has_difference_type<_Ptr>::value>
struct __pointer_traits_difference_type
{
    typedef _LIBCUDACXX_NODEBUG_TYPE ptrdiff_t type;
};

template <class _Ptr>
struct __pointer_traits_difference_type<_Ptr, true>
{
    typedef _LIBCUDACXX_NODEBUG_TYPE typename _Ptr::difference_type type;
};

template <class _Tp, class _Up>
struct __has_rebind
{
private:
    template <class _Xp> _LIBCUDACXX_INLINE_VISIBILITY static false_type __test(...);
    _LIBCUDACXX_SUPPRESS_DEPRECATED_PUSH
    template <class _Xp> _LIBCUDACXX_INLINE_VISIBILITY static true_type __test(typename _Xp::template rebind<_Up>* = 0);
    _LIBCUDACXX_SUPPRESS_DEPRECATED_POP
public:
    static const bool value = decltype(__test<_Tp>(0))::value;
};

template <class _Tp, class _Up, bool = __has_rebind<_Tp, _Up>::value>
struct __pointer_traits_rebind
{
#ifndef _LIBCUDACXX_CXX03_LANG
    typedef _LIBCUDACXX_NODEBUG_TYPE typename _Tp::template rebind<_Up> type;
#else
    typedef _LIBCUDACXX_NODEBUG_TYPE typename _Tp::template rebind<_Up>::other type;
#endif
};

#ifndef _LIBCUDACXX_HAS_NO_VARIADICS

template <template <class, class...> class _Sp, class _Tp, class ..._Args, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _Args...>, _Up, true>
{
#ifndef _LIBCUDACXX_CXX03_LANG
    typedef _LIBCUDACXX_NODEBUG_TYPE typename _Sp<_Tp, _Args...>::template rebind<_Up> type;
#else
    typedef _LIBCUDACXX_NODEBUG_TYPE typename _Sp<_Tp, _Args...>::template rebind<_Up>::other type;
#endif
};

template <template <class, class...> class _Sp, class _Tp, class ..._Args, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _Args...>, _Up, false>
{
    typedef _Sp<_Up, _Args...> type;
};

#else  // _LIBCUDACXX_HAS_NO_VARIADICS

template <template <class> class _Sp, class _Tp, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp>, _Up, true>
{
#ifndef _LIBCUDACXX_CXX03_LANG
    typedef typename _Sp<_Tp>::template rebind<_Up> type;
#else
    typedef typename _Sp<_Tp>::template rebind<_Up>::other type;
#endif
};

template <template <class> class _Sp, class _Tp, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp>, _Up, false>
{
    typedef _Sp<_Up> type;
};

template <template <class, class> class _Sp, class _Tp, class _A0, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _A0>, _Up, true>
{
#ifndef _LIBCUDACXX_CXX03_LANG
    typedef typename _Sp<_Tp, _A0>::template rebind<_Up> type;
#else
    typedef typename _Sp<_Tp, _A0>::template rebind<_Up>::other type;
#endif
};

template <template <class, class> class _Sp, class _Tp, class _A0, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _A0>, _Up, false>
{
    typedef _Sp<_Up, _A0> type;
};

template <template <class, class, class> class _Sp, class _Tp, class _A0,
                                         class _A1, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _A0, _A1>, _Up, true>
{
#ifndef _LIBCUDACXX_CXX03_LANG
    typedef typename _Sp<_Tp, _A0, _A1>::template rebind<_Up> type;
#else
    typedef typename _Sp<_Tp, _A0, _A1>::template rebind<_Up>::other type;
#endif
};

template <template <class, class, class> class _Sp, class _Tp, class _A0,
                                         class _A1, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _A0, _A1>, _Up, false>
{
    typedef _Sp<_Up, _A0, _A1> type;
};

template <template <class, class, class, class> class _Sp, class _Tp, class _A0,
                                                class _A1, class _A2, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _A0, _A1, _A2>, _Up, true>
{
#ifndef _LIBCUDACXX_CXX03_LANG
    typedef typename _Sp<_Tp, _A0, _A1, _A2>::template rebind<_Up> type;
#else
    typedef typename _Sp<_Tp, _A0, _A1, _A2>::template rebind<_Up>::other type;
#endif
};

template <template <class, class, class, class> class _Sp, class _Tp, class _A0,
                                                class _A1, class _A2, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _A0, _A1, _A2>, _Up, false>
{
    typedef _Sp<_Up, _A0, _A1, _A2> type;
};

#endif  // _LIBCUDACXX_HAS_NO_VARIADICS

template <class _Ptr>
struct _LIBCUDACXX_TEMPLATE_VIS pointer_traits
{
    typedef _Ptr                                                     pointer;
    typedef typename __pointer_traits_element_type<pointer>::type    element_type;
    typedef typename __pointer_traits_difference_type<pointer>::type difference_type;

#ifndef _LIBCUDACXX_CXX03_LANG
    template <class _Up> using rebind = typename __pointer_traits_rebind<pointer, _Up>::type;
#else
    template <class _Up> struct rebind
        {typedef typename __pointer_traits_rebind<pointer, _Up>::type other;};
#endif // _LIBCUDACXX_CXX03_LANG

private:
    struct __nat {};
public:
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    static pointer pointer_to(__conditional_t<is_void<element_type>::value, __nat, element_type>& __r)
        {return pointer::pointer_to(__r);}
};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS pointer_traits<_Tp*>
{
    typedef _Tp*      pointer;
    typedef _Tp       element_type;
    typedef ptrdiff_t difference_type;

#ifndef _LIBCUDACXX_CXX03_LANG
    template <class _Up> using rebind = _Up*;
#else
    template <class _Up> struct rebind {typedef _Up* other;};
#endif

private:
    struct __nat {};
public:
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    static pointer pointer_to(__conditional_t<is_void<element_type>::value, __nat, element_type>& __r) _NOEXCEPT
        {return _CUDA_VSTD::addressof(__r);}
};

template <class _From, class _To>
struct __rebind_pointer {
#ifndef _LIBCUDACXX_CXX03_LANG
    typedef typename pointer_traits<_From>::template rebind<_To>        type;
#else
    typedef typename pointer_traits<_From>::template rebind<_To>::other type;
#endif
};

// to_address

template <class _Pointer, class = void>
struct __to_address_helper;

template <class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
_Tp* __to_address(_Tp* __p) _NOEXCEPT {
    static_assert(!is_function<_Tp>::value, "_Tp is a function type");
    return __p;
}

template <class _Pointer, class = void>
struct _HasToAddress : false_type {};

template <class _Pointer>
struct _HasToAddress<_Pointer,
    decltype((void)pointer_traits<_Pointer>::to_address(declval<const _Pointer&>()))
> : true_type {};

template <class _Pointer, class = void>
struct _HasArrow : false_type {};

template <class _Pointer>
struct _HasArrow<_Pointer,
    decltype((void)declval<const _Pointer&>().operator->())
> : true_type {};

template <class _Pointer>
struct _IsFancyPointer {
  static const bool value = _HasArrow<_Pointer>::value || _HasToAddress<_Pointer>::value;
};

// enable_if is needed here to avoid instantiating checks for fancy pointers on raw pointers
template <class _Pointer, class = __enable_if_t<
    _And<is_class<_Pointer>, _IsFancyPointer<_Pointer> >::value
> >
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
__decay_t<decltype(__to_address_helper<_Pointer>::__call(declval<const _Pointer&>()))>
__to_address(const _Pointer& __p) _NOEXCEPT {
    return __to_address_helper<_Pointer>::__call(__p);
}

template <class _Pointer, class>
struct __to_address_helper {
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
    static decltype(_CUDA_VSTD::__to_address(declval<const _Pointer&>().operator->()))
    __call(const _Pointer& __p) _NOEXCEPT {
        return _CUDA_VSTD::__to_address(__p.operator->());
    }
};

template <class _Pointer>
struct __to_address_helper<_Pointer, decltype((void)pointer_traits<_Pointer>::to_address(declval<const _Pointer&>()))> {
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
    static decltype(pointer_traits<_Pointer>::to_address(declval<const _Pointer&>()))
    __call(const _Pointer& __p) _NOEXCEPT {
        return pointer_traits<_Pointer>::to_address(__p);
    }
};

#if _LIBCUDACXX_STD_VER > 11
template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr
auto to_address(_Tp *__p) noexcept {
    return _CUDA_VSTD::__to_address(__p);
}

template <class _Pointer>
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr
auto to_address(const _Pointer& __p) noexcept -> decltype(_CUDA_VSTD::__to_address(__p)) {
    return _CUDA_VSTD::__to_address(__p);
}
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MEMORY_POINTER_TRAITS_H
