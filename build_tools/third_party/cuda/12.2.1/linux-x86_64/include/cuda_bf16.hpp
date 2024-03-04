/*
* Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/

#if !defined(__CUDA_BF16_HPP__)
#define __CUDA_BF16_HPP__

#if !defined(__CUDA_BF16_H__)
#error "Do not include this file directly. Instead, include cuda_bf16.h."
#endif

#if !defined(_MSC_VER) && __cplusplus >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_BF16
#elif _MSC_FULL_VER >= 190024210 && _MSVC_LANG >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_BF16
#endif

/* C++11 header for std::move. 
 * In RTC mode, std::move is provided implicitly; don't include the header
 */
#if defined(__CPP_VERSION_AT_LEAST_11_BF16) && !defined(__CUDACC_RTC__)
#include <utility>
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) && !defined(__CUDACC_RTC__) */

/* C++ header for std::memcpy (used for type punning in host-side implementations).
 * When compiling as a CUDA source file memcpy is provided implicitly.
 * !defined(__CUDACC__) implies !defined(__CUDACC_RTC__).
 */
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <cstring>
#endif /* defined(__cplusplus) && !defined(__CUDACC__) */

// implicitly provided by NVRTC
#if !defined(__CUDACC_RTC__)
#include <nv/target>
#endif  /* !defined(__CUDACC_RTC__) */

#if !defined(IF_DEVICE_OR_CUDACC)
#if defined(__CUDACC__)
    #define IF_DEVICE_OR_CUDACC(d, c, f) NV_IF_ELSE_TARGET(NV_IS_DEVICE, d, c)
#else
    #define IF_DEVICE_OR_CUDACC(d, c, f) NV_IF_ELSE_TARGET(NV_IS_DEVICE, d, f)
#endif
#endif

/* Set up function decorations */
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define __CUDA_BF16_DECL__ static __device__ __inline__
#define __CUDA_HOSTDEVICE_BF16_DECL__ static __host__ __device__ __inline__
#define __CUDA_HOSTDEVICE__ __host__ __device__
#else /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
#if defined(__GNUC__)
#define __CUDA_HOSTDEVICE_BF16_DECL__ static __attribute__ ((unused))
#else
#define __CUDA_HOSTDEVICE_BF16_DECL__ static
#endif /* defined(__GNUC__) */
#define __CUDA_HOSTDEVICE__
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */

/* Set up structure-alignment attribute */
#if defined(__CUDACC__)
#define __CUDA_ALIGN__(align) __align__(align)
#else
/* Define alignment macro based on compiler type (cannot assume C11 "_Alignas" is available) */
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
#define __CUDA_ALIGN__(n) alignas(n)    /* C++11 kindly gives us a keyword for this */
#else /* defined(__CPP_VERSION_AT_LEAST_11_BF16)*/
#if defined(__GNUC__)
#define __CUDA_ALIGN__(n) __attribute__ ((aligned(n)))
#elif defined(_MSC_VER)
#define __CUDA_ALIGN__(n) __declspec(align(n))
#else
#define __CUDA_ALIGN__(n)
#endif /* defined(__GNUC__) */
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */
#endif /* defined(__CUDACC__) */

/* Macros to allow nv_bfloat16 & nv_bfloat162 to be used by inline assembly */
#define __BFLOAT16_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __BFLOAT16_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __BFLOAT162_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __BFLOAT162_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))

// define __CUDA_BF16_CONSTEXPR__ in order to
// use constexpr where possible, with supporting C++ dialects
// undef after use
#if (defined __CPP_VERSION_AT_LEAST_11_BF16)
#define __CUDA_BF16_CONSTEXPR__   constexpr
#else
#define __CUDA_BF16_CONSTEXPR__
#endif

/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief __nv_bfloat16_raw data type
 * \details Type allows static initialization of \p nv_bfloat16 until it becomes
 * a builtin type.
 * 
 * - Note: this initialization is as a bit-field representation of \p nv_bfloat16,
 * and not a conversion from \p short to \p nv_bfloat16.
 * Such representation will be deprecated in a future version of CUDA.
 * 
 * - Note: this is visible to non-nvcc compilers, including C-only compilations
 */
typedef struct __CUDA_ALIGN__(2) {
    unsigned short x;
} __nv_bfloat16_raw;

/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief __nv_bfloat162_raw data type
 * \details Type allows static initialization of \p nv_bfloat162 until it becomes
 * a builtin type.
 * 
 * - Note: this initialization is as a bit-field representation of \p nv_bfloat162,
 * and not a conversion from \p short2 to \p nv_bfloat162.
 * Such representation will be deprecated in a future version of CUDA.
 * 
 * - Note: this is visible to non-nvcc compilers, including C-only compilations
 */
typedef struct __CUDA_ALIGN__(4) {
    unsigned short x;
    unsigned short y;
} __nv_bfloat162_raw;

/* All other definitions in this file are only visible to C++ compilers */
#if defined(__cplusplus)

/* Hide GCC member initialization list warnings because of host/device in-function init requirement */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Weffc++"
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

/* class' : multiple assignment operators specified
   The class has multiple assignment operators of a single type. This warning is informational */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning( push )
#pragma warning( disable:4522 )
#endif /* defined(__GNUC__) */

/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16_CONSTANTS
 * \brief Defines floating-point positive infinity value for the \p nv_bfloat16 data type
 */
#define CUDART_INF_BF16            __ushort_as_bfloat16((unsigned short)0x7F80U)
/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16_CONSTANTS
 * \brief Defines canonical NaN value for the \p nv_bfloat16 data type
 */
#define CUDART_NAN_BF16            __ushort_as_bfloat16((unsigned short)0x7FFFU)
/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16_CONSTANTS
 * \brief Defines a minimum representable (denormalized) value for the \p nv_bfloat16 data type
 */
#define CUDART_MIN_DENORM_BF16     __ushort_as_bfloat16((unsigned short)0x0001U)
/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16_CONSTANTS
 * \brief Defines a maximum representable value for the \p nv_bfloat16 data type
 */
#define CUDART_MAX_NORMAL_BF16     __ushort_as_bfloat16((unsigned short)0x7F7FU)
/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16_CONSTANTS
 * \brief Defines a negative zero value for the \p nv_bfloat16 data type
 */
#define CUDART_NEG_ZERO_BF16       __ushort_as_bfloat16((unsigned short)0x8000U)
/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16_CONSTANTS
 * \brief Defines a positive zero value for the \p nv_bfloat16 data type
 */
#define CUDART_ZERO_BF16           __ushort_as_bfloat16((unsigned short)0x0000U)
/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16_CONSTANTS
 * \brief Defines a value of 1.0 for the \p nv_bfloat16 data type
 */
#define CUDART_ONE_BF16            __ushort_as_bfloat16((unsigned short)0x3F80U)

/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief nv_bfloat16 datatype 
 * 
 * \details This structure implements the datatype for storing 
 * nv_bfloat16 floating-point numbers. The structure implements 
 * assignment operators and type conversions. 16 bits are being 
 * used in total: 1 sign bit, 8 bits for the exponent, and 
 * the significand is being stored in 7 bits. The total 
 * precision is 8 bits.
 * 
 */
struct __CUDA_ALIGN__(2) __nv_bfloat16 {
protected:
    /**
     * Protected storage variable contains the bits of floating-point data.
     */
    unsigned short __x;

public:
    /**
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
    __nv_bfloat16() = default;
#else
    __CUDA_HOSTDEVICE__ __nv_bfloat16() { }
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */

    /* Convert to/from __nv_bfloat16_raw */
    /**
     * Constructor from \p __nv_bfloat16_raw.
     */
    __CUDA_HOSTDEVICE__ __CUDA_BF16_CONSTEXPR__ __nv_bfloat16(const __nv_bfloat16_raw &hr) : __x(hr.x) { }
    /**
     * Assignment operator from \p __nv_bfloat16_raw.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(const __nv_bfloat16_raw &hr) { __x = hr.x; return *this; }
    /**
     * Assignment operator from \p __nv_bfloat16_raw to \p volatile \p __nv_bfloat16.
     */
    __CUDA_HOSTDEVICE__ volatile __nv_bfloat16 &operator=(const __nv_bfloat16_raw &hr) volatile { __x = hr.x; return *this; }
    /**
     * Assignment operator from \p volatile \p __nv_bfloat16_raw to \p volatile \p __nv_bfloat16.
     */
    __CUDA_HOSTDEVICE__ volatile __nv_bfloat16 &operator=(const volatile __nv_bfloat16_raw &hr) volatile { __x = hr.x; return *this; }
    /**
     * Type cast to \p __nv_bfloat16_raw operator.
     */
    __CUDA_HOSTDEVICE__ operator __nv_bfloat16_raw() const { __nv_bfloat16_raw ret; ret.x = __x; return ret; }
    /**
     * Type cast to \p __nv_bfloat16_raw operator with \p volatile input.
     */
    __CUDA_HOSTDEVICE__ operator __nv_bfloat16_raw() const volatile { __nv_bfloat16_raw ret; ret.x = __x; return ret; }

#if !defined(__CUDA_NO_BFLOAT16_CONVERSIONS__)
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
    /**
     * Construct \p __nv_bfloat16 from \p __half input using default round-to-nearest-even rounding mode.
     */
    explicit __CUDA_HOSTDEVICE__ __nv_bfloat16(const __half f);
#endif /* #if defined(__CPP_VERSION_AT_LEAST_11_BF16) */

    /* Construct from float/double */
    /**
     * Construct \p __nv_bfloat16 from \p float input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(const float f) { __x = __float2bfloat16(f).__x;  }
    /**
     * Construct \p __nv_bfloat16 from \p double input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(const double f) { __x = __double2bfloat16(f).__x;  }
    /**
     * Type cast to \p float operator.
     */
    __CUDA_HOSTDEVICE__ operator float() const { return __bfloat162float(*this); }
    /**
     * Type cast to \p __nv_bfloat16 assignment operator from \p float input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(const float f) { __x = __float2bfloat16(f).__x; return *this; }

    /* We omit "cast to double" operator, so as to not be ambiguous about up-cast */
    /**
     * Type cast to \p __nv_bfloat16 assignment operator from \p double input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(const double f) { __x = __double2bfloat16(f).__x; return *this; }

/*
 * Implicit type conversions to/from integer types were only available to nvcc compilation.
 * Introducing them for all compilers is a potentially breaking change that may affect
 * overloads resolution and will require users to update their code.
 * Define __CUDA_BF16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__ to opt-out.
 */
#if !(defined __CUDA_BF16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__) || (defined __CUDACC__)
    /* Allow automatic construction from types supported natively in hardware */
    /* Note we do avoid constructor init-list because of special host/device compilation rules */

    /**
     * Construct \p __nv_bfloat16 from \p short integer input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(short val) { __x = __short2bfloat16_rn(val).__x;  }
    /**
     * Construct \p __nv_bfloat16 from \p unsigned \p short integer input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(unsigned short val) { __x = __ushort2bfloat16_rn(val).__x;  }
    /**
     * Construct \p __nv_bfloat16 from \p int input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(int val) { __x = __int2bfloat16_rn(val).__x;  }
    /**
     * Construct \p __nv_bfloat16 from \p unsigned \p int input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(unsigned int val) { __x = __uint2bfloat16_rn(val).__x;  }


    /**
     * Construct \p __nv_bfloat16 from \p long input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(const long val) {

        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (sizeof(long) == sizeof(long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (default: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */

        {
            __x = __ll2bfloat16_rn(static_cast<long long>(val)).__x;
        } else {
            __x = __int2bfloat16_rn(static_cast<int>(val)).__x;
        }
    }

    /**
     * Construct \p __nv_bfloat16 from \p unsigned \p long input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(const unsigned long val) {

        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (sizeof(unsigned long) == sizeof(unsigned long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (default: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */

        {
            __x = __ull2bfloat16_rn(static_cast<unsigned long long>(val)).__x;
        } else {
            __x = __uint2bfloat16_rn(static_cast<unsigned int>(val)).__x;
        }
    }


    /**
     * Construct \p __nv_bfloat16 from \p long \p long input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(long long val) { __x = __ll2bfloat16_rn(val).__x;  }
    /**
     * Construct \p __nv_bfloat16 from \p unsigned \p long \p long input using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16(unsigned long long val) { __x = __ull2bfloat16_rn(val).__x; }

    /* Allow automatic casts to supported builtin types, matching all that are permitted with float */

    /**
     * Conversion operator to \p signed \p char data type.
     * Using round-toward-zero rounding mode.
     * 
     * See __bfloat162char_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator signed char() const { return __bfloat162char_rz(*this); }
    /**
     * Conversion operator to \p unsigned \p char data type.
     * Using round-toward-zero rounding mode.
     * 
     * See __bfloat162uchar_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator unsigned char() const { return __bfloat162uchar_rz(*this); }
    /**
     * Conversion operator to an implementation defined \p char data type.
     * Using round-toward-zero rounding mode.
     * 
     * Detects signedness of the \p char type and proceeds accordingly, see
     * further details in signed and unsigned char operators.
     */
    __CUDA_HOSTDEVICE__ operator char() const {
        char value;
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (((char)-1) < (char)0)
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (default: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            value = static_cast<char>(__bfloat162char_rz(*this));
        }
        else
        {
            value = static_cast<char>(__bfloat162uchar_rz(*this));
        }
        return value;
    }
    /**
     * Conversion operator to \p short data type.
     * Using round-toward-zero rounding mode.
     * 
     * See __bfloat162short_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator short() const { return __bfloat162short_rz(*this); }
    /**
     * Conversion operator to \p unsigned \p short data type.
     * Using round-toward-zero rounding mode.
     * 
     * See __bfloat162ushort_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator unsigned short() const { return __bfloat162ushort_rz(*this); }
    /**
     * Conversion operator to \p int data type.
     * Using round-toward-zero rounding mode.
     * 
     * See __bfloat162int_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator int() const { return __bfloat162int_rz(*this); }
    /**
     * Conversion operator to \p unsigned \p int data type.
     * Using round-toward-zero rounding mode.
     * 
     * See __bfloat162uint_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator unsigned int() const { return __bfloat162uint_rz(*this); }

    /**
     * Conversion operator to \p long data type.
     * Using round-toward-zero rounding mode.
     */
    __CUDA_HOSTDEVICE__ operator long() const {
        long retval;
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (sizeof(long) == sizeof(long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (default: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            retval = static_cast<long>(__bfloat162ll_rz(*this));
        }
        else
        {
            retval = static_cast<long>(__bfloat162int_rz(*this));
        }
        return retval;
    }

    /**
     * Conversion operator to \p unsigned \p long data type.
     * Using round-toward-zero rounding mode.
     */
    __CUDA_HOSTDEVICE__ operator unsigned long() const {
        unsigned long retval;
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (sizeof(unsigned long) == sizeof(unsigned long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (default: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            retval = static_cast<unsigned long>(__bfloat162ull_rz(*this));
        }
        else
        {
            retval = static_cast<unsigned long>(__bfloat162uint_rz(*this));
        }
        return retval;
    }

    /**
     * Conversion operator to \p long \p long data type.
     * Using round-toward-zero rounding mode.
     * 
     * See __bfloat162ll_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator long long() const { return __bfloat162ll_rz(*this); }
    /**
     * Conversion operator to \p unsigned \p long \p long data type.
     * Using round-toward-zero rounding mode.
     * 
     * See __bfloat162ull_rz(__nv_bfloat16) for further details
     */
    __CUDA_HOSTDEVICE__ operator unsigned long long() const { return __bfloat162ull_rz(*this); }

    /**
     * Type cast from \p short assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(short val) { __x = __short2bfloat16_rn(val).__x; return *this; }
    /**
     * Type cast from \p unsigned \p short assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(unsigned short val) { __x = __ushort2bfloat16_rn(val).__x; return *this; }
    /**
     * Type cast from \p int assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(int val) { __x = __int2bfloat16_rn(val).__x; return *this; }
   /**
     * Type cast from \p unsigned \p int assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(unsigned int val) { __x = __uint2bfloat16_rn(val).__x; return *this; }
    /**
     * Type cast from \p long \p long assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(long long val) { __x = __ll2bfloat16_rn(val).__x; return *this; }
    /**
     * Type cast from \p unsigned \p long \p long assignment operator, using default round-to-nearest-even rounding mode.
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat16 &operator=(unsigned long long val) { __x = __ull2bfloat16_rn(val).__x; return *this; }

    /**
     * Conversion operator to \p bool data type.
     * +0 and -0 inputs convert to \p false.
     * Non-zero inputs convert to \p true.
     */
    __CUDA_HOSTDEVICE__ __CUDA_BF16_CONSTEXPR__ operator bool() const { return (__x & 0x7FFFU) != 0U; }
#endif /* !(defined __CUDA_BF16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__) || (defined __CUDACC__) */
#endif /* !defined(__CUDA_NO_BFLOAT16_CONVERSIONS__) */
};

#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
#if !defined(__CUDA_NO_BFLOAT16_CONVERSIONS__)
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16::__nv_bfloat16(const __half f)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("{  cvt.rn.bf16.f16 %0, %1;}\n" : "=h"(__x) : "h"(__BFLOAT16_TO_CUS(f)));
,
    __x = __float2bfloat16(__half2float(f)).__x;
)
}
#endif

#if !defined(__CUDA_NO_HALF_CONVERSIONS__)
__CUDA_HOSTDEVICE__ __forceinline__ __half::__half(const __nv_bfloat16 f)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("{  cvt.rn.f16.bf16 %0, %1;}\n" : "=h"(__x) : "h"(__BFLOAT16_TO_CUS(f)));
,
    __x = __float2half_rn(__bfloat162float(f)).__x;
)
}
#endif
#endif /* #if defined(__CPP_VERSION_AT_LEAST_11_BF16) */

#if !defined(__CUDA_NO_BFLOAT16_OPERATORS__)
/* Some basic arithmetic operations expected of a builtin */
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 addition operation.
 * See also __hadd(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16 operator+(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) { return __hadd(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 subtraction operation.
 * See also __hsub(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16 operator-(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) { return __hsub(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 multiplication operation.
 * See also __hmul(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16 operator*(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) { return __hmul(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 division operation.
 * See also __hdiv(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16 operator/(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) { return __hdiv(lh, rh); }

/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 compound assignment with addition operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16 &operator+=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh) { lh = __hadd(lh, rh); return lh; }
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 compound assignment with subtraction operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16 &operator-=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh) { lh = __hsub(lh, rh); return lh; }
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 compound assignment with multiplication operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16 &operator*=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh) { lh = __hmul(lh, rh); return lh; }
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 compound assignment with division operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16 &operator/=(__nv_bfloat16 &lh, const __nv_bfloat16 &rh) { lh = __hdiv(lh, rh); return lh; }

/* Note for increment and decrement we use the raw value 0x3F80U equating to nv_bfloat16(1.0F), to avoid the extra conversion */
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 prefix increment operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16 &operator++(__nv_bfloat16 &h)      { __nv_bfloat16_raw one; one.x = 0x3F80U; h += one; return h; }
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 prefix decrement operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16 &operator--(__nv_bfloat16 &h)      { __nv_bfloat16_raw one; one.x = 0x3F80U; h -= one; return h; }
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 postfix increment operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16  operator++(__nv_bfloat16 &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __nv_bfloat16 ret = h;
    __nv_bfloat16_raw one;
    one.x = 0x3F80U;
    h += one;
    return ret;
}
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Performs \p nv_bfloat16 postfix decrement operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16  operator--(__nv_bfloat16 &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __nv_bfloat16 ret = h;
    __nv_bfloat16_raw one;
    one.x = 0x3F80U;
    h -= one;
    return ret;
}
/* Unary plus and inverse operators */
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Implements \p nv_bfloat16 unary plus operator, returns input value.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16 operator+(const __nv_bfloat16 &h) { return h; }
/**
 * \ingroup CUDA_MATH__BFLOAT16_ARITHMETIC
 * Implements \p nv_bfloat16 unary minus operator.
 * See also __hneg(__nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat16 operator-(const __nv_bfloat16 &h) { return __hneg(h); }

/* Some basic comparison operations to make it look like a builtin */
/**
 * \ingroup CUDA_MATH__BFLOAT16_COMPARISON
 * Performs \p nv_bfloat16 ordered compare equal operation.
 * See also __heq(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __forceinline__ bool operator==(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) { return __heq(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT16_COMPARISON
 * Performs \p nv_bfloat16 unordered compare not-equal operation.
 * See also __hneu(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __forceinline__ bool operator!=(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) { return __hneu(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT16_COMPARISON
 * Performs \p nv_bfloat16 ordered greater-than compare operation.
 * See also __hgt(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __forceinline__ bool operator> (const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) { return __hgt(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT16_COMPARISON
 * Performs \p nv_bfloat16 ordered less-than compare operation.
 * See also __hlt(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __forceinline__ bool operator< (const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) { return __hlt(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT16_COMPARISON
 * Performs \p nv_bfloat16 ordered greater-or-equal compare operation.
 * See also __hge(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __forceinline__ bool operator>=(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) { return __hge(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT16_COMPARISON
 * Performs \p nv_bfloat16 ordered less-or-equal compare operation.
 * See also __hle(__nv_bfloat16, __nv_bfloat16)
 */
__CUDA_HOSTDEVICE__ __forceinline__ bool operator<=(const __nv_bfloat16 &lh, const __nv_bfloat16 &rh) { return __hle(lh, rh); }
#endif /* !defined(__CUDA_NO_BFLOAT16_OPERATORS__) */

/**
* \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief nv_bfloat162 datatype
 * \details This structure implements the datatype for storing two 
 * nv_bfloat16 floating-point numbers. 
 * The structure implements assignment, arithmetic and comparison
 * operators, and type conversions. 
 * 
 * - NOTE: __nv_bfloat162 is visible to non-nvcc host compilers
 */
struct __CUDA_ALIGN__(4) __nv_bfloat162 {
    /**
     * Storage field holding lower \p __nv_bfloat16 part.
     */
    __nv_bfloat16 x;
    /**
     * Storage field holding upper \p __nv_bfloat16 part.
     */
    __nv_bfloat16 y;

    // All construct/copy/assign/move
public:
    /**
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
    __nv_bfloat162() = default;
    /**
     * Move constructor, available for \p C++11 and later dialects
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat162(__nv_bfloat162 &&src) { __BFLOAT162_TO_UI(*this) = std::move(__BFLOAT162_TO_CUI(src)); }
    /**
     * Move assignment operator, available for \p C++11 and later dialects
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat162 &operator=(__nv_bfloat162 &&src) { __BFLOAT162_TO_UI(*this) = std::move(__BFLOAT162_TO_CUI(src)); return *this; }
#else
    __CUDA_HOSTDEVICE__ __nv_bfloat162() { }
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */

    /**
     * Constructor from two \p __nv_bfloat16 variables
     */
    __CUDA_HOSTDEVICE__ __CUDA_BF16_CONSTEXPR__ __nv_bfloat162(const __nv_bfloat16 &a, const __nv_bfloat16 &b) : x(a), y(b) { }
    /**
     * Copy constructor
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat162(const __nv_bfloat162 &src) { __BFLOAT162_TO_UI(*this) = __BFLOAT162_TO_CUI(src); }
    /**
     * Copy assignment operator
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat162 &operator=(const __nv_bfloat162 &src) { __BFLOAT162_TO_UI(*this) = __BFLOAT162_TO_CUI(src); return *this; }

    /* Convert to/from __nv_bfloat162_raw */
    /**
     * Constructor from \p __nv_bfloat162_raw
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat162(const __nv_bfloat162_raw &h2r ) { __BFLOAT162_TO_UI(*this) = __BFLOAT162_TO_CUI(h2r); }
    /**
     * Assignment operator from \p __nv_bfloat162_raw
     */
    __CUDA_HOSTDEVICE__ __nv_bfloat162 &operator=(const __nv_bfloat162_raw &h2r) { __BFLOAT162_TO_UI(*this) = __BFLOAT162_TO_CUI(h2r); return *this; }
    /**
     * Conversion operator to \p __nv_bfloat162_raw
     */
    __CUDA_HOSTDEVICE__ operator __nv_bfloat162_raw() const { __nv_bfloat162_raw ret; ret.x = 0U; ret.y = 0U; __BFLOAT162_TO_UI(ret) = __BFLOAT162_TO_CUI(*this); return ret; }
};

#if !defined(__CUDA_NO_BFLOAT162_OPERATORS__)
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 addition operation.
 * See also __hadd2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162 operator+(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) { return __hadd2(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 subtraction operation.
 * See also __hsub2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162 operator-(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) { return __hsub2(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 multiplication operation.
 * See also __hmul2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162 operator*(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) { return __hmul2(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 division operation.
 * See also __h2div(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162 operator/(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) { return __h2div(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 compound assignment with addition operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162& operator+=(__nv_bfloat162 &lh, const __nv_bfloat162 &rh) { lh = __hadd2(lh, rh); return lh; }
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 compound assignment with subtraction operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162& operator-=(__nv_bfloat162 &lh, const __nv_bfloat162 &rh) { lh = __hsub2(lh, rh); return lh; }
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 compound assignment with multiplication operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162& operator*=(__nv_bfloat162 &lh, const __nv_bfloat162 &rh) { lh = __hmul2(lh, rh); return lh; }
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 compound assignment with division operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162& operator/=(__nv_bfloat162 &lh, const __nv_bfloat162 &rh) { lh = __h2div(lh, rh); return lh; }
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 prefix increment operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162 &operator++(__nv_bfloat162 &h)      { __nv_bfloat162_raw one; one.x = 0x3F80U; one.y = 0x3F80U; h = __hadd2(h, one); return h; }
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 prefix decrement operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162 &operator--(__nv_bfloat162 &h)      { __nv_bfloat162_raw one; one.x = 0x3F80U; one.y = 0x3F80U; h = __hsub2(h, one); return h; }
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 postfix increment operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162  operator++(__nv_bfloat162 &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __nv_bfloat162 ret = h;
    __nv_bfloat162_raw one;
    one.x = 0x3F80U;
    one.y = 0x3F80U;
    h = __hadd2(h, one);
    return ret;
}
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Performs packed \p nv_bfloat16 postfix decrement operation.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162  operator--(__nv_bfloat162 &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __nv_bfloat162 ret = h;
    __nv_bfloat162_raw one;
    one.x = 0x3F80U;
    one.y = 0x3F80U;
    h = __hsub2(h, one);
    return ret;
}
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Implements packed \p nv_bfloat16 unary plus operator, returns input value.
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162 operator+(const __nv_bfloat162 &h) { return h; }
/**
 * \ingroup CUDA_MATH__BFLOAT162_ARITHMETIC
 * Implements packed \p nv_bfloat16 unary minus operator.
 * See also __hneg2(__nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __forceinline__ __nv_bfloat162 operator-(const __nv_bfloat162 &h) { return __hneg2(h); }
/**
 * \ingroup CUDA_MATH__BFLOAT162_COMPARISON
 * Performs packed \p nv_bfloat16 ordered compare equal operation.
 * See also __hbeq2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __forceinline__ bool operator==(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) { return __hbeq2(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT162_COMPARISON
 * Performs packed \p nv_bfloat16 unordered compare not-equal operation.
 * See also __hbneu2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __forceinline__ bool operator!=(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) { return __hbneu2(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT162_COMPARISON
 * Performs packed \p nv_bfloat16 ordered greater-than compare operation.
 * See also __hbgt2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __forceinline__ bool operator>(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) { return __hbgt2(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT162_COMPARISON
 * Performs packed \p nv_bfloat16 ordered less-than compare operation.
 * See also __hblt2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __forceinline__ bool operator<(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) { return __hblt2(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT162_COMPARISON
 * Performs packed \p nv_bfloat16 ordered greater-or-equal compare operation.
 * See also __hbge2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __forceinline__ bool operator>=(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) { return __hbge2(lh, rh); }
/**
 * \ingroup CUDA_MATH__BFLOAT162_COMPARISON
 * Performs packed \p nv_bfloat16 ordered less-or-equal compare operation.
 * See also __hble2(__nv_bfloat162, __nv_bfloat162)
 */
__CUDA_HOSTDEVICE__ __forceinline__ bool operator<=(const __nv_bfloat162 &lh, const __nv_bfloat162 &rh) { return __hble2(lh, rh); }

#endif /* !defined(__CUDA_NO_BFLOAT162_OPERATORS__) */

/* Restore warning for multiple assignment operators */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning( pop )
#endif /* defined(_MSC_VER) && _MSC_VER >= 1500 */

/* Restore -Weffc++ warnings from here on */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

#undef __CUDA_HOSTDEVICE__
#undef __CUDA_ALIGN__

__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __internal_float_as_uint(const float f)
{
    unsigned int u;
IF_DEVICE_OR_CUDACC(
    u = __float_as_uint(f);
,
    memcpy(&u, &f, sizeof(f));
,
    std::memcpy(&u, &f, sizeof(f));
)
    return u;
}

__CUDA_HOSTDEVICE_BF16_DECL__ float __internal_uint_as_float(const unsigned int u)
{
    float f;
IF_DEVICE_OR_CUDACC(
    f = __uint_as_float(u);
,
    memcpy(&f, &u, sizeof(u));
,
    std::memcpy(&f, &u, sizeof(u));
)
    return f;
}

__CUDA_HOSTDEVICE_BF16_DECL__ unsigned short __internal_float2bfloat16(const float f, unsigned int &sign, unsigned int &remainder)
{
    unsigned int x;

    x = __internal_float_as_uint(f);

    if ((x & 0x7fffffffU) > 0x7f800000U) {
        sign = 0U;
        remainder = 0U;
        return static_cast<unsigned short>(0x7fffU);
    }
    sign = x >> 31U;
    remainder = x << 16U;
    return static_cast<unsigned short>(x >> 16U);
}

__CUDA_HOSTDEVICE_BF16_DECL__ float __internal_double2float_rn(const double x)
{
    float r;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.rn.f32.f64 %0, %1;" : "=f"(r) : "d"(x));
,
    r = static_cast<float>(x);
)
    return r;
}
__CUDA_HOSTDEVICE_BF16_DECL__ double __internal_float2double(const float x)
{
    double r;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("cvt.f64.f32 %0, %1;" : "=d"(r) : "f"(x));
,
    r = static_cast<double>(x);
)
    return r;
}

__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __double2bfloat16(const double x)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 val;
    asm("{  cvt.rn.bf16.f64 %0, %1;}\n" : "=h"(__BFLOAT16_TO_US(val)) : "d"(x));
    return val;
,
    float f = __internal_double2float_rn(x);
    const double d = __internal_float2double(f);
    unsigned int u = __internal_float_as_uint(f);

    bool x_is_not_nan = ((u << (unsigned)1U) <= (unsigned)0xFF000000U);


    if ((x > 0.0) && (d > x)) {
        u--;
    }
    if ((x < 0.0) && (d < x)) {
        u--;
    }
    if ((d != x) && x_is_not_nan) {
        u |= 1U;
    }

    f = __internal_uint_as_float(u);

    return __float2bfloat16(f);
)
}

__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __float2bfloat16(const float a)
{
    __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm("{  cvt.rn.bf16.f32 %0, %1;}\n" : "=h"(__BFLOAT16_TO_US(val)) : "f"(a));
,
    __nv_bfloat16_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
        r.x++;
    }
    val = r;
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __float2bfloat16_rn(const float a)
{
    __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm("{  cvt.rn.bf16.f32 %0, %1;}\n" : "=h"(__BFLOAT16_TO_US(val)) : "f"(a));
,
    __nv_bfloat16_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
        r.x++;
    }
    val = r;
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __float2bfloat16_rz(const float a)
{
    __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm("{  cvt.rz.bf16.f32 %0, %1;}\n" : "=h"(__BFLOAT16_TO_US(val)) : "f"(a));
,
    __nv_bfloat16_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    val = r;
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __float2bfloat16_rd(const float a)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 val;
    asm("{  cvt.rm.bf16.f32 %0, %1;}\n" : "=h"(__BFLOAT16_TO_US(val)) : "f"(a));
    return val;
,
    __nv_bfloat16 val;
    __nv_bfloat16_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder != 0U) && (sign != 0U)) {
        r.x++;
    }
    val = r;
    return val;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __float2bfloat16_ru(const float a)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 val;
    asm("{  cvt.rp.bf16.f32 %0, %1;}\n" : "=h"(__BFLOAT16_TO_US(val)) : "f"(a));
    return val;
,
    __nv_bfloat16 val;
    __nv_bfloat16_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder != 0U) && (sign == 0U)) {
        r.x++;
    }
    val = r;
    return val;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __float2bfloat162_rn(const float a)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm("{.reg .b16 low;\n"
        "  cvt.rn.bf16.f32 low, %1;\n"
        "  mov.b32 %0, {low,low};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "f"(a));
,
    val = __nv_bfloat162(__float2bfloat16_rn(a), __float2bfloat16_rn(a));
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __floats2bfloat162_rn(const float a, const float b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm("{ cvt.rn.bf16x2.f32 %0, %2, %1;}\n"
        : "=r"(__BFLOAT162_TO_UI(val)) : "f"(a), "f"(b));
,
    val = __nv_bfloat162(__float2bfloat16_rn(a), __float2bfloat16_rn(b));
)
    return val;
}

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ float __internal_device_bfloat162float(const unsigned short h)
{
    float f;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("{ cvt.f32.bf16 %0, %1;}\n" : "=f"(f) : "h"(h));
,
    asm("{ mov.b32 %0, {0,%1};}\n" : "=f"(f) : "h"(h));
)
    return f;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */

__CUDA_HOSTDEVICE_BF16_DECL__ float __internal_bfloat162float(const unsigned short h)
{
    float f;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    f = __internal_device_bfloat162float(h);
,
    unsigned int u = static_cast<unsigned int>(h) << 16;
    f = __internal_uint_as_float(u);
)
    return f;
}

__CUDA_HOSTDEVICE_BF16_DECL__ float __bfloat162float(const __nv_bfloat16 a)
{
    return __internal_bfloat162float(static_cast<__nv_bfloat16_raw>(a).x);
}
__CUDA_HOSTDEVICE_BF16_DECL__ float __low2float(const __nv_bfloat162 a)
{
    return __internal_bfloat162float(static_cast<__nv_bfloat162_raw>(a).x);
}

__CUDA_HOSTDEVICE_BF16_DECL__ float __high2float(const __nv_bfloat162 a)
{
    return __internal_bfloat162float(static_cast<__nv_bfloat162_raw>(a).y);
}

/* CUDA vector-types compatible vector creation function (note returns __nv_bfloat162, not nv_bfloat162) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 make_bfloat162(const __nv_bfloat16 x, const __nv_bfloat16 y)
{
    __nv_bfloat162 t; t.x = x; t.y = y; return t;
}

/* Definitions of intrinsics */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __float22bfloat162_rn(const float2 a)
{
    __nv_bfloat162 val = __floats2bfloat162_rn(a.x, a.y);
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ float2 __bfloat1622float2(const __nv_bfloat162 a)
{
    float hi_float;
    float lo_float;
    lo_float = __internal_bfloat162float(((__nv_bfloat162_raw)a).x);
    hi_float = __internal_bfloat162float(((__nv_bfloat162_raw)a).y);
    return make_float2(lo_float, hi_float);
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ int __bfloat162int_rn(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    int val;
    asm("{  cvt.rni.s32.bf16 %0, %1;}\n" : "=r"(val) : "h"(__BFLOAT16_TO_CUS(h)));
    return val;
,
    return __float2int_rn(__bfloat162float(h));
)
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */

__CUDA_HOSTDEVICE_BF16_DECL__ int __internal_bfloat162int_rz(const __nv_bfloat16 h)
{
    const float f = __bfloat162float(h);
    int   i;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    i = __float2int_rz(f);
,
    const int max_val = (int)0x7fffffffU;
    const int min_val = (int)0x80000000U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__nv_bfloat16_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xFF00U) {
        // NaN
        i = 0;
    } else if (f >= static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        i = static_cast<int>(f);
    }
)
    return i;
}

__CUDA_HOSTDEVICE_BF16_DECL__ int __bfloat162int_rz(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    int val;
    asm("{  cvt.rzi.s32.bf16 %0, %1;}\n" : "=r"(val) : "h"(__BFLOAT16_TO_CUS(h)));
    return val;
,
    return __internal_bfloat162int_rz(h);
)
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ int __bfloat162int_rd(const __nv_bfloat16 h)
{
    int val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("{  cvt.rmi.s32.bf16 %0, %1;}\n" : "=r"(val) : "h"(__BFLOAT16_TO_CUS(h)));
,
    const float f = __bfloat162float(h);
    asm("cvt.rmi.s32.f32 %0, %1;" : "=r"(val) : "f"(f));
)
    return val;
}
__CUDA_BF16_DECL__ int __bfloat162int_ru(const __nv_bfloat16 h)
{
    int val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("{  cvt.rpi.s32.bf16 %0, %1;}\n" : "=r"(val) : "h"(__BFLOAT16_TO_CUS(h)));
,
    const float f = __bfloat162float(h);
    asm("cvt.rpi.s32.f32 %0, %1;" : "=r"(val) : "f"(f));
)
    return val;
}

__CUDA_BF16_DECL__ __nv_bfloat16 __internal_device_int2bfloat16_rn(const int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
        __nv_bfloat16 val;
       asm("cvt.rn.bf16.s32 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "r"(i));
       return val;
,
        const float ru = __int2float_ru(i);
        const float rd = __int2float_rd(i);
        float rz = __int2float_rz(i);
        if (ru != rd) {
            rz = __uint_as_float(__float_as_uint(rz) | 1U);
        }
        return __float2bfloat16_rn(rz);
)
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __int2bfloat16_rn(const int i)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_int2bfloat16_rn(i);
,
    const double d = static_cast<double>(i);
    return __double2bfloat16(d);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ signed char __bfloat162char_rz(const __nv_bfloat16 h)
{
    signed char i;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    unsigned short tmp = 0;
    asm("{ .reg.b8 myreg;\n"
        "  cvt.rzi.s8.bf16 myreg, %1;\n"
        "  mov.b16 %0, {myreg, 0};\n}"
         :"=h"(tmp) : "h"(__BFLOAT16_TO_CUS(h)));
    const unsigned char u = static_cast<unsigned char>(tmp);
    i = static_cast<signed char>(u);
,
    const float f = __bfloat162float(h);
    const signed char max_val = (signed char)0x7fU;
    const signed char min_val = (signed char)0x80U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__nv_bfloat16_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xFF00U) {
        // NaN
        i = 0;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<signed char>(f);
    }
)
    return i;
}

__CUDA_HOSTDEVICE_BF16_DECL__ unsigned char __bfloat162uchar_rz(const __nv_bfloat16 h)
{
    unsigned char i;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    unsigned short tmp = 0;
    asm("{ .reg.b8 myreg;\n"
        "  cvt.rzi.u8.bf16 myreg, %1;\n"
        "  mov.b16 %0, {myreg, 0};\n}"
         :"=h"(tmp) : "h"(__BFLOAT16_TO_CUS(h)));
    i = static_cast<unsigned char>(tmp);
,
    const float f = __bfloat162float(h);
    const unsigned char max_val = 0xffU;
    const unsigned char min_val = 0U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__nv_bfloat16_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xFF00U) {
        // NaN
        i = 0U;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<unsigned char>(f);
    }
)
    return i;
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 __int2bfloat16_rz(const int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
     __nv_bfloat16 val;
    asm("cvt.rz.bf16.s32 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "r"(i));
    return val;
,
    return __float2bfloat16_rz(__int2float_rz(i));
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 __int2bfloat16_rd(const int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
     __nv_bfloat16 val;
    asm("cvt.rm.bf16.s32 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "r"(i));
    return val;
,
    return __float2bfloat16_rd(__int2float_rd(i));
)
}

__CUDA_BF16_DECL__ __nv_bfloat16 __int2bfloat16_ru(const int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
     __nv_bfloat16 val;
    asm("cvt.rp.bf16.s32 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "r"(i));
    return val;
,
    return __float2bfloat16_ru(__int2float_ru(i));
)
}

__CUDA_BF16_DECL__ short int __bfloat162short_rn(const __nv_bfloat16 h)
{
   short int val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm("cvt.rni.s16.bf16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
,
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rni.s16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
)
   return val;
}

__CUDA_BF16_DECL__ short int __internal_device_bfloat162short_rz(const __nv_bfloat16 h)
{
    short int val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("cvt.rzi.s16.bf16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
,
    asm("{ .reg.f32 f;\n"
        "  mov.b32 f, {0,%1};\n"
        "  cvt.rzi.s16.f32 %0,f;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
)
    return val;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ short int __bfloat162short_rz(const __nv_bfloat16 h)
{
    short int val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    val = __internal_device_bfloat162short_rz(h);
,
    const float f = __bfloat162float(h);
    const short int max_val = (short int)0x7fffU;
    const short int min_val = (short int)0x8000U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__nv_bfloat16_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xFF00U) {
        // NaN
        val = 0;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        val = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        val = min_val;
    } else {
        val = static_cast<short int>(f);
    }
)
   return val;
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ short int __bfloat162short_rd(const __nv_bfloat16 h)
{
   short int val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm("cvt.rmi.s16.bf16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
,
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rmi.s16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
)
   return val;
}
__CUDA_BF16_DECL__ short int __bfloat162short_ru(const __nv_bfloat16 h)
{
   short int val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm("cvt.rpi.s16.bf16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
,
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rpi.s16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
)
   return val;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __short2bfloat16_rn(const short int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 val;
    asm("cvt.rn.bf16.s16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(i));
    return val;
,
    const float f = static_cast<float>(i);
    return __float2bfloat16_rn(f);
)
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 __short2bfloat16_rz(const short int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 val;
    asm("cvt.rz.bf16.s16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(i));
    return val;
,
    return __float2bfloat16_rz(__int2float_rz(static_cast<int>(i)));
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 __short2bfloat16_rd(const short int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 val;
    asm("cvt.rm.bf16.s16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(i));
    return val;
,
    return __float2bfloat16_rd(__int2float_rd(static_cast<int>(i)));
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 __short2bfloat16_ru(const short int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 val;
    asm("cvt.rp.bf16.s16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(i));
    return val;
,
    return __float2bfloat16_ru(__int2float_ru(static_cast<int>(i)));
)
}

__CUDA_BF16_DECL__ unsigned int __bfloat162uint_rn(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    unsigned int val;
    asm("{  cvt.rni.u32.bf16 %0, %1;}\n" : "=r"(val) : "h"(__BFLOAT16_TO_CUS(h)));
    return val;
,
    return __float2uint_rn(__bfloat162float(h));
)
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __internal_bfloat162uint_rz(const __nv_bfloat16 h)
{
    const float f = __bfloat162float(h);
    unsigned int i;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    i = __float2uint_rz(f);
,
    const unsigned int max_val = 0xffffffffU;
    const unsigned int min_val = 0U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__nv_bfloat16_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xFF00U) {
        // NaN
        i = 0U;
    } else if (f >= static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        i = static_cast<unsigned int>(f);
    }
)
    return i;
}

__CUDA_HOSTDEVICE_BF16_DECL__ unsigned int __bfloat162uint_rz(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    unsigned int val;
    asm("{  cvt.rzi.u32.bf16 %0, %1;}\n" : "=r"(val) : "h"(__BFLOAT16_TO_CUS(h)));
    return val;
,
    return __internal_bfloat162uint_rz(h);
)
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ unsigned int __bfloat162uint_rd(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    unsigned int val;
    asm("{  cvt.rmi.u32.bf16 %0, %1;}\n" : "=r"(val) : "h"(__BFLOAT16_TO_CUS(h)));
    return val;
,
    return __float2uint_rd(__bfloat162float(h));
)
}
__CUDA_BF16_DECL__ unsigned int __bfloat162uint_ru(const __nv_bfloat16 h)
{
    unsigned int val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("{  cvt.rpi.u32.bf16 %0, %1;}\n" : "=r"(val) : "h"(__BFLOAT16_TO_CUS(h)));
,
    const float f = __bfloat162float(h);
    asm("cvt.rpi.u32.f32 %0, %1;" : "=r"(val) : "f"(f));
)
    return val;
}

__CUDA_BF16_DECL__ __nv_bfloat16 __internal_device_uint2bfloat16_rn(const unsigned int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 val;
    asm("cvt.rn.bf16.u32 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "r"(i));
    return val;
,
    const float ru = __uint2float_ru(i);
    const float rd = __uint2float_rd(i);
    float rz = __uint2float_rz(i);
    if (ru != rd) {
        rz = __uint_as_float(__float_as_uint(rz) | 1U);
    }
    return __float2bfloat16_rn(rz);
)
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __uint2bfloat16_rn(const unsigned int i)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_uint2bfloat16_rn(i);
,
    const double d = static_cast<double>(i);
    return __double2bfloat16(d);
)
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 __uint2bfloat16_rz(const unsigned int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
     __nv_bfloat16 val;
    asm("cvt.rz.bf16.u32 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "r"(i));
    return val;
,
    return __float2bfloat16_rz(__uint2float_rz(i));
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 __uint2bfloat16_rd(const unsigned int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
     __nv_bfloat16 val;
    asm("cvt.rm.bf16.u32 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "r"(i));
    return val;
,
    return __float2bfloat16_rd(__uint2float_rd(i));
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 __uint2bfloat16_ru(const unsigned int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
     __nv_bfloat16 val;
    asm("cvt.rp.bf16.u32 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "r"(i));
    return val;
,
    return __float2bfloat16_ru(__uint2float_ru(i));
)
}

__CUDA_BF16_DECL__ unsigned short int __bfloat162ushort_rn(const __nv_bfloat16 h)
{
   unsigned short int val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm("cvt.rni.u16.bf16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
,
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rni.u16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
)
   return val;
}

__CUDA_BF16_DECL__ unsigned short int __internal_device_bfloat162ushort_rz(const __nv_bfloat16 h)
{
   unsigned short int val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm("cvt.rzi.u16.bf16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
,
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rzi.u16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
)
   return val;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned short int __bfloat162ushort_rz(const __nv_bfloat16 h)
{
   unsigned short int val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
   val = __internal_device_bfloat162ushort_rz(h);
,
    const float f = __bfloat162float(h);
    const unsigned short int max_val = 0xffffU;
    const unsigned short int min_val = 0U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__nv_bfloat16_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xFF00U) {
        // NaN
        val = 0U;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        val = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        val = min_val;
    } else {
        val = static_cast<unsigned short int>(f);
    }
)
   return val;
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ unsigned short int __bfloat162ushort_rd(const __nv_bfloat16 h)
{
   unsigned short int val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm("cvt.rmi.u16.bf16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
,
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rmi.u16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
)
   return val;
}
__CUDA_BF16_DECL__ unsigned short int __bfloat162ushort_ru(const __nv_bfloat16 h)
{
   unsigned short int val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm("cvt.rpi.u16.bf16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
,
   asm("{ .reg.f32 f;\n"
       "  mov.b32 f, {0,%1};\n"
       "  cvt.rpi.u16.f32 %0,f;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(h)));
)
   return val;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __ushort2bfloat16_rn(const unsigned short int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 val;
    asm("cvt.rn.bf16.u16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(i));
    return val;
,
    const float f = static_cast<float>(i);
    return __float2bfloat16_rn(f);
)
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 __ushort2bfloat16_rz(const unsigned short int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 val;
    asm("cvt.rz.bf16.u16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(i));
    return val;
,
    return __float2bfloat16_rz(__uint2float_rz(static_cast<unsigned int>(i)));
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 __ushort2bfloat16_rd(const unsigned short int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 val;
    asm("cvt.rm.bf16.u16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(i));
    return val;
,
    return __float2bfloat16_rd(__uint2float_rd(static_cast<unsigned int>(i)));
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 __ushort2bfloat16_ru(const unsigned short int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 val;
    asm("cvt.rp.bf16.u16 %0, %1;" : "=h"(__BFLOAT16_TO_US(val)) : "h"(i));
    return val;
,
    return __float2bfloat16_ru(__uint2float_ru(static_cast<unsigned int>(i)));
)
}

__CUDA_BF16_DECL__ unsigned long long int __bfloat162ull_rn(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    unsigned long long int i;
    asm("cvt.rni.u64.bf16 %0, %1;" : "=l"(i) : "h"(__BFLOAT16_TO_CUS(h)));
    return i;
,
    return __float2ull_rn(__bfloat162float(h));
)
}

__CUDA_BF16_DECL__ unsigned long long int __internal_device_bfloat162ull_rz(const __nv_bfloat16 h)
{
    unsigned long long int i;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("cvt.rzi.u64.bf16 %0, %1;" : "=l"(i) : "h"(__BFLOAT16_TO_CUS(h)));
,
    const float f = __bfloat162float(h);
    i = __float2ull_rz(f);
)
    return i;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned long long int __bfloat162ull_rz(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_bfloat162ull_rz(h);
,
    const float f = __bfloat162float(h);
    unsigned long long int i;
    const unsigned long long int max_val = 0xffffffffffffffffULL;
    const unsigned long long int min_val = 0ULL;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__nv_bfloat16_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xFF00U) {
        // NaN
        i = 0x8000000000000000ULL;
    } else if (f >= static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        i = static_cast<unsigned long long int>(f);
    }
    return i;
)
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ unsigned long long int __bfloat162ull_rd(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    unsigned long long int i;
    asm("cvt.rmi.u64.bf16 %0, %1;" : "=l"(i) : "h"(__BFLOAT16_TO_CUS(h)));
    return i;
,
    return __float2ull_rd(__bfloat162float(h));
)
}
__CUDA_BF16_DECL__ unsigned long long int __bfloat162ull_ru(const __nv_bfloat16 h)
{
    unsigned long long int i;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("cvt.rpi.u64.bf16 %0, %1;" : "=l"(i) : "h"(__BFLOAT16_TO_CUS(h)));
,
    const float f = __bfloat162float(h);
    asm("cvt.rpi.u64.f32 %0, %1;" : "=l"(i) : "f"(f));
)
    return i;
}

__CUDA_BF16_DECL__ __nv_bfloat16 __internal_device_ull2bfloat16_rn(const unsigned long long int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 h;
    asm("cvt.rn.bf16.u64 %0, %1;" : "=h"(__BFLOAT16_TO_US(h)) : "l"(i));
    return h;
,
    const float ru = __ull2float_ru(i);
    const float rd = __ull2float_rd(i);
    float rz = __ull2float_rz(i);
    if (ru != rd) {
        rz = __uint_as_float(__float_as_uint(rz) | 1U);
    }
    return __float2bfloat16_rn(rz);
)
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __ull2bfloat16_rn(const unsigned long long int i)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_ull2bfloat16_rn(i);
,
    float f = static_cast<float>(i);
    const unsigned long long int uf = static_cast<unsigned long long int>(f);
    unsigned int u = __internal_float_as_uint(f);
    // round up happened here
    // note: no need to handle round up to f == 0x1.p64 specially
    if (uf > i) {
        u--;
    }
    if (uf != i) {
        u |= 1U;
    }
    f = __internal_uint_as_float(u);
    return __float2bfloat16_rn(f);
)
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 __ull2bfloat16_rz(const unsigned long long int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 h;
    asm("cvt.rz.bf16.u64 %0, %1;" : "=h"(__BFLOAT16_TO_US(h)) : "l"(i));
    return h;
,
    return __float2bfloat16_rz(__ull2float_rz(i));
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 __ull2bfloat16_rd(const unsigned long long int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 h;
    asm("cvt.rm.bf16.u64 %0, %1;" : "=h"(__BFLOAT16_TO_US(h)) : "l"(i));
    return h;
,
    return __float2bfloat16_rd(__ull2float_rd(i));
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 __ull2bfloat16_ru(const unsigned long long int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 h;
    asm("cvt.rp.bf16.u64 %0, %1;" : "=h"(__BFLOAT16_TO_US(h)) : "l"(i));
    return h;
,
    return __float2bfloat16_ru(__ull2float_ru(i));
)
}
__CUDA_BF16_DECL__ long long int __bfloat162ll_rn(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    long long int i;
    asm("cvt.rni.s64.bf16 %0, %1;" : "=l"(i) : "h"(__BFLOAT16_TO_CUS(h)));
    return i;
,
    return __float2ll_rn(__bfloat162float(h));
)
}

__CUDA_BF16_DECL__ long long int __internal_device_bfloat162ll_rz(const __nv_bfloat16 h)
{
    long long int i;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("cvt.rzi.s64.bf16 %0, %1;" : "=l"(i) : "h"(__BFLOAT16_TO_CUS(h)));
,
    const float f = __bfloat162float(h);
    i = __float2ll_rz(f);
)
    return i;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ long long int __bfloat162ll_rz(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_bfloat162ll_rz(h);
,
    long long int i;
    const float f = __bfloat162float(h);
    const long long int max_val = (long long int)0x7fffffffffffffffULL;
    const long long int min_val = (long long int)0x8000000000000000ULL;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__nv_bfloat16_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xFF00U) {
        // NaN
        i = min_val;
    } else if (f >= static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        i = static_cast<long long int>(f);
    }
    return i;
)
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ long long int __bfloat162ll_rd(const __nv_bfloat16 h)
{
    long long int i;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("cvt.rmi.s64.bf16 %0, %1;" : "=l"(i) : "h"(__BFLOAT16_TO_CUS(h)));
,
    const float f = __bfloat162float(h);
    asm("cvt.rmi.s64.f32 %0, %1;" : "=l"(i) : "f"(f));
)
    return i;
}
__CUDA_BF16_DECL__ long long int __bfloat162ll_ru(const __nv_bfloat16 h)
{
    long long int i;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("cvt.rpi.s64.bf16 %0, %1;" : "=l"(i) : "h"(__BFLOAT16_TO_CUS(h)));
,
    const float f = __bfloat162float(h);
    asm("cvt.rpi.s64.f32 %0, %1;" : "=l"(i) : "f"(f));
)
    return i;
}

__CUDA_BF16_DECL__ __nv_bfloat16 __internal_device_ll2bfloat16_rn(const long long int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 h;
    asm("cvt.rn.bf16.s64 %0, %1;" : "=h"(__BFLOAT16_TO_US(h)) : "l"(i));
    return h;
,
    const float ru = __ll2float_ru(i);
    const float rd = __ll2float_rd(i);
    float rz = __ll2float_rz(i);
    if (ru != rd) {
        rz = __uint_as_float(__float_as_uint(rz) | 1U);
    }
    return __float2bfloat16_rn(rz);
)
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __ll2bfloat16_rn(const long long int i)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_ll2bfloat16_rn(i);
,
    float f = static_cast<float>(i);
    const long long int lf = static_cast<long long int>(f);
    unsigned int u = __internal_float_as_uint(f);

    if ((f > 0.0f) && (lf > i)) {
        u--;
    }
    if ((f < 0.0f) && (lf < i)) {
        u--;
    }
    if (lf != i) {
        u |= 1U;
    }

    f = __internal_uint_as_float(u);
    return __float2bfloat16_rn(f);
)
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 __ll2bfloat16_rz(const long long int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 h;
    asm("cvt.rz.bf16.s64 %0, %1;" : "=h"(__BFLOAT16_TO_US(h)) : "l"(i));
    return h;
,
    return __float2bfloat16_rz(__ll2float_rz(i));
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 __ll2bfloat16_rd(const long long int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 h;
    asm("cvt.rm.bf16.s64 %0, %1;" : "=h"(__BFLOAT16_TO_US(h)) : "l"(i));
    return h;
,
    return __float2bfloat16_rd(__ll2float_rd(i));
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 __ll2bfloat16_ru(const long long int i)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 h;
    asm("cvt.rp.bf16.s64 %0, %1;" : "=h"(__BFLOAT16_TO_US(h)) : "l"(i));
    return h;
,
    return __float2bfloat16_ru(__ll2float_ru(i));
)
}

__CUDA_BF16_DECL__ __nv_bfloat16 htrunc(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 r;
    asm("cvt.rzi.bf16.bf16 %0, %1;" : "=h"(__BFLOAT16_TO_US(r)) : "h"(__BFLOAT16_TO_CUS(h)));
    return r;
,
    return __float2bfloat16_rz(truncf(__bfloat162float(h)));
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 hceil(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 r;
    asm("cvt.rpi.bf16.bf16 %0, %1;" : "=h"(__BFLOAT16_TO_US(r)) : "h"(__BFLOAT16_TO_CUS(h)));
    return r;
,
    float fh = __bfloat162float(h);
    asm( "{ cvt.rpi.f32.f32 %0, %0; }\n"
        :"+f"(fh));
    return __float2bfloat16_rz(fh);
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 hfloor(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 r;
    asm("cvt.rmi.bf16.bf16 %0, %1;" : "=h"(__BFLOAT16_TO_US(r)) : "h"(__BFLOAT16_TO_CUS(h)));
    return r;
,
    float fh = __bfloat162float(h);
    asm( "{ cvt.rmi.f32.f32 %0, %0; }\n"
        :"+f"(fh));
    return __float2bfloat16_rz(fh);
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 hrint(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 r;
    asm("cvt.rni.bf16.bf16 %0, %1;" : "=h"(__BFLOAT16_TO_US(r)) : "h"(__BFLOAT16_TO_CUS(h)));
    return r;
,
    return __float2bfloat16_rz(rintf(__bfloat162float(h)));
)
}

__CUDA_BF16_DECL__ __nv_bfloat162 h2trunc(const __nv_bfloat162 h)
{
    const __nv_bfloat16 low  = htrunc(h.x);
    const __nv_bfloat16 high = htrunc(h.y);
    return __nv_bfloat162(low, high);
}
__CUDA_BF16_DECL__ __nv_bfloat162 h2ceil(const __nv_bfloat162 h)
{
    const __nv_bfloat16 low  = hceil(h.x);
    const __nv_bfloat16 high = hceil(h.y);
    return __nv_bfloat162(low, high);
}
__CUDA_BF16_DECL__ __nv_bfloat162 h2floor(const __nv_bfloat162 h)
{
    const __nv_bfloat16 low  = hfloor(h.x);
    const __nv_bfloat16 high = hfloor(h.y);
    return __nv_bfloat162(low, high);
}

__CUDA_BF16_DECL__ __nv_bfloat162 h2rint(const __nv_bfloat162 h)
{
    return __halves2bfloat162(hrint(__low2bfloat16(h)), hrint(__high2bfloat16(h)));
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __lows2bfloat162(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .b16 alow,ahigh,blow,bhigh;\n"
        "  mov.b32 {alow,ahigh}, %1;\n"
        "  mov.b32 {blow,bhigh}, %2;\n"
        "  mov.b32 %0, {alow,blow};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)), "r"(__BFLOAT162_TO_CUI(b)));
,
    val.x = a.x;
    val.y = b.x;
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __highs2bfloat162(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .b16 alow,ahigh,blow,bhigh;\n"
        "  mov.b32 {alow,ahigh}, %1;\n"
        "  mov.b32 {blow,bhigh}, %2;\n"
        "  mov.b32 %0, {ahigh,bhigh};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)), "r"(__BFLOAT162_TO_CUI(b)));
,
    val.x = a.y;
    val.y = b.y;
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __low2bfloat16(const __nv_bfloat162 a)
{
    __nv_bfloat16 ret;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .b16 low,high;\n"
        " mov.b32 {low,high}, %1;\n"
        " mov.b16 %0, low;}" : "=h"(__BFLOAT16_TO_US(ret)) : "r"(__BFLOAT162_TO_CUI(a)));
,
    ret = a.x;
)
    return ret;
}
__CUDA_HOSTDEVICE_BF16_DECL__ int __hisinf(const __nv_bfloat16 a)
{
    int retval;
    const __nv_bfloat16_raw araw = __nv_bfloat16_raw(a);
    if (araw.x == 0xFF80U) {
        retval = -1;
    } else if (araw.x == 0x7F80U) {
        retval = 1;
    } else {
        retval = 0;
    }
    return retval;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __low2bfloat162(const __nv_bfloat162 a)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .b16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {low,low};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)));
,
    val.x = a.x;
    val.y = a.x;
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __high2bfloat162(const __nv_bfloat162 a)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .b16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {high,high};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)));
,
    val.x = a.y;
    val.y = a.y;
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __high2bfloat16(const __nv_bfloat162 a)
{
    __nv_bfloat16 ret;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .b16 low,high;\n"
        " mov.b32 {low,high}, %1;\n"
        " mov.b16 %0, high;}" : "=h"(__BFLOAT16_TO_US(ret)) : "r"(__BFLOAT162_TO_CUI(a)));
,
    ret = a.y;
)
    return ret;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __halves2bfloat162(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{  mov.b32 %0, {%1,%2};}\n"
        : "=r"(__BFLOAT162_TO_UI(val)) : "h"(__BFLOAT16_TO_CUS(a)), "h"(__BFLOAT16_TO_CUS(b)));
,
    val.x = a;
    val.y = b;
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __bfloat162bfloat162(const __nv_bfloat16 a)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{  mov.b32 %0, {%1,%1};}\n"
        : "=r"(__BFLOAT162_TO_UI(val)) : "h"(__BFLOAT16_TO_CUS(a)));
,
    val.x = a;
    val.y = a;
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __lowhigh2highlow(const __nv_bfloat162 a)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    asm("{.reg .b16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {high,low};}\n" : "=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)));
,
    val.x = a.y;
    val.y = a.x;
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ short int __bfloat16_as_short(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return static_cast<short int>(__BFLOAT16_TO_CUS(h));
,
    return static_cast<short int>(__nv_bfloat16_raw(h).x);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned short int __bfloat16_as_ushort(const __nv_bfloat16 h)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __BFLOAT16_TO_CUS(h);
,
    return __nv_bfloat16_raw(h).x;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __short_as_bfloat16(const short int i)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __nv_bfloat16 h;
    __BFLOAT16_TO_US(h) = static_cast<unsigned short int>(i);
    return h;
,
    __nv_bfloat16_raw hr;
    hr.x = static_cast<unsigned short int>(i);
    return __nv_bfloat16(hr);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __ushort_as_bfloat16(const unsigned short int i)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __nv_bfloat16 h;
    __BFLOAT16_TO_US(h) = i;
    return h;
,
    __nv_bfloat16_raw hr;
    hr.x = i;
    return __nv_bfloat16(hr);
)
}

#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300))) || defined(_NVHPC_CUDA)
/******************************************************************************
*                           __nv_bfloat16, __nv_bfloat162 warp shuffle        *
******************************************************************************/
#define __SHUFFLE_SYNC_BFLOAT162_MACRO(name) /* do */ {\
   __nv_bfloat162 r; \
   asm volatile ("{" __CUDA_BF16_STRINGIFY(name) " %0,%1,%2,%3,%4;\n}" \
       :"=r"(__BFLOAT162_TO_UI(r)): "r"(__BFLOAT162_TO_CUI(var)), "r"(delta), "r"(c), "r"(mask)); \
   return r; \
} /* while(0) */

__CUDA_BF16_DECL__ __nv_bfloat162 __shfl_sync(const unsigned mask, const __nv_bfloat162 var, const int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    __SHUFFLE_SYNC_BFLOAT162_MACRO(shfl.sync.idx.b32)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __shfl_up_sync(const unsigned mask, const __nv_bfloat162 var, const unsigned int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = (warp_size - static_cast<unsigned>(width)) << 8U;
    __SHUFFLE_SYNC_BFLOAT162_MACRO(shfl.sync.up.b32)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __shfl_down_sync(const unsigned mask, const __nv_bfloat162 var, const unsigned int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    __SHUFFLE_SYNC_BFLOAT162_MACRO(shfl.sync.down.b32)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __shfl_xor_sync(const unsigned mask, const __nv_bfloat162 var, const int delta, const int width)
{
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    __SHUFFLE_SYNC_BFLOAT162_MACRO(shfl.sync.bfly.b32)
}

#undef __SHUFFLE_SYNC_BFLOAT162_MACRO

__CUDA_BF16_DECL__ __nv_bfloat16 __shfl_sync(const unsigned mask, const __nv_bfloat16 var, const int delta, const int width)
{
    const __nv_bfloat162 temp1 = __halves2bfloat162(var, var);
    const __nv_bfloat162 temp2 = __shfl_sync(mask, temp1, delta, width);
    return __low2bfloat16(temp2);
}
__CUDA_BF16_DECL__ __nv_bfloat16 __shfl_up_sync(const unsigned mask, const __nv_bfloat16 var, const unsigned int delta, const int width)
{
    const __nv_bfloat162 temp1 = __halves2bfloat162(var, var);
    const __nv_bfloat162 temp2 = __shfl_up_sync(mask, temp1, delta, width);
    return __low2bfloat16(temp2);
}
__CUDA_BF16_DECL__ __nv_bfloat16 __shfl_down_sync(const unsigned mask, const __nv_bfloat16 var, const unsigned int delta, const int width)
{
    const __nv_bfloat162 temp1 = __halves2bfloat162(var, var);
    const __nv_bfloat162 temp2 = __shfl_down_sync(mask, temp1, delta, width);
    return __low2bfloat16(temp2);
}
__CUDA_BF16_DECL__ __nv_bfloat16 __shfl_xor_sync(const unsigned mask, const __nv_bfloat16 var, const int delta, const int width)
{
    const __nv_bfloat162 temp1 = __halves2bfloat162(var, var);
    const __nv_bfloat162 temp2 = __shfl_xor_sync(mask, temp1, delta, width);
    return __low2bfloat16(temp2);
}

/******************************************************************************
*               __nv_bfloat16 and __nv_bfloat162 __ldg,__ldcg,__ldca,__ldcs   *
******************************************************************************/

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/
__CUDA_BF16_DECL__ __nv_bfloat162 __ldg(const  __nv_bfloat162 *const ptr)
{
    __nv_bfloat162 ret;
    asm ("ld.global.nc.b32 %0, [%1];"  : "=r"(__BFLOAT162_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __ldg(const __nv_bfloat16 *const ptr)
{
    __nv_bfloat16 ret;
    asm ("ld.global.nc.b16 %0, [%1];"  : "=h"(__BFLOAT16_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_BF16_DECL__ __nv_bfloat162 __ldcg(const  __nv_bfloat162 *const ptr)
{
    __nv_bfloat162 ret;
    asm ("ld.global.cg.b32 %0, [%1];"  : "=r"(__BFLOAT162_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __ldcg(const __nv_bfloat16 *const ptr)
{
    __nv_bfloat16 ret;
    asm ("ld.global.cg.b16 %0, [%1];"  : "=h"(__BFLOAT16_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_BF16_DECL__ __nv_bfloat162 __ldca(const  __nv_bfloat162 *const ptr)
{
    __nv_bfloat162 ret;
    asm ("ld.global.ca.b32 %0, [%1];"  : "=r"(__BFLOAT162_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __ldca(const __nv_bfloat16 *const ptr)
{
    __nv_bfloat16 ret;
    asm ("ld.global.ca.b16 %0, [%1];"  : "=h"(__BFLOAT16_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_BF16_DECL__ __nv_bfloat162 __ldcs(const  __nv_bfloat162 *const ptr)
{
    __nv_bfloat162 ret;
    asm ("ld.global.cs.b32 %0, [%1];"  : "=r"(__BFLOAT162_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __ldcs(const __nv_bfloat16 *const ptr)
{
    __nv_bfloat16 ret;
    asm ("ld.global.cs.b16 %0, [%1];"  : "=h"(__BFLOAT16_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_BF16_DECL__ __nv_bfloat162 __ldlu(const  __nv_bfloat162 *const ptr)
{
    __nv_bfloat162 ret;
    asm ("ld.global.lu.b32 %0, [%1];"  : "=r"(__BFLOAT162_TO_UI(ret)) : __LDG_PTR(ptr) : "memory");
    return ret;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __ldlu(const __nv_bfloat16 *const ptr)
{
    __nv_bfloat16 ret;
    asm ("ld.global.lu.b16 %0, [%1];"  : "=h"(__BFLOAT16_TO_US(ret)) : __LDG_PTR(ptr) : "memory");
    return ret;
}
__CUDA_BF16_DECL__ __nv_bfloat162 __ldcv(const  __nv_bfloat162 *const ptr)
{
    __nv_bfloat162 ret;
    asm ("ld.global.cv.b32 %0, [%1];"  : "=r"(__BFLOAT162_TO_UI(ret)) : __LDG_PTR(ptr) : "memory");
    return ret;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __ldcv(const __nv_bfloat16 *const ptr)
{
    __nv_bfloat16 ret;
    asm ("ld.global.cv.b16 %0, [%1];"  : "=h"(__BFLOAT16_TO_US(ret)) : __LDG_PTR(ptr) : "memory");
    return ret;
}

__CUDA_BF16_DECL__ void __stwb(__nv_bfloat162 *const ptr, const __nv_bfloat162 value)
{
    asm ("st.global.wb.b32 [%0], %1;"  :: __LDG_PTR(ptr), "r"(__BFLOAT162_TO_CUI(value)) : "memory");
}
__CUDA_BF16_DECL__ void __stwb(__nv_bfloat16 *const ptr, const __nv_bfloat16 value)
{
    asm ("st.global.wb.b16 [%0], %1;"  :: __LDG_PTR(ptr),  "h"(__BFLOAT16_TO_CUS(value)) : "memory");
}
__CUDA_BF16_DECL__ void __stcg(__nv_bfloat162 *const ptr, const __nv_bfloat162 value)
{
    asm ("st.global.cg.b32 [%0], %1;"  :: __LDG_PTR(ptr), "r"(__BFLOAT162_TO_CUI(value)) : "memory");
}
__CUDA_BF16_DECL__ void __stcg(__nv_bfloat16 *const ptr, const __nv_bfloat16 value)
{
    asm ("st.global.cg.b16 [%0], %1;"  :: __LDG_PTR(ptr),  "h"(__BFLOAT16_TO_CUS(value)) : "memory");
}
__CUDA_BF16_DECL__ void __stcs(__nv_bfloat162 *const ptr, const __nv_bfloat162 value)
{
    asm ("st.global.cs.b32 [%0], %1;"  :: __LDG_PTR(ptr), "r"(__BFLOAT162_TO_CUI(value)) : "memory");
}
__CUDA_BF16_DECL__ void __stcs(__nv_bfloat16 *const ptr, const __nv_bfloat16 value)
{
    asm ("st.global.cs.b16 [%0], %1;"  :: __LDG_PTR(ptr),  "h"(__BFLOAT16_TO_CUS(value)) : "memory");
}
__CUDA_BF16_DECL__ void __stwt(__nv_bfloat162 *const ptr, const __nv_bfloat162 value)
{
    asm ("st.global.wt.b32 [%0], %1;"  :: __LDG_PTR(ptr), "r"(__BFLOAT162_TO_CUI(value)) : "memory");
}
__CUDA_BF16_DECL__ void __stwt(__nv_bfloat16 *const ptr, const __nv_bfloat16 value)
{
    asm ("st.global.wt.b16 [%0], %1;"  :: __LDG_PTR(ptr),  "h"(__BFLOAT16_TO_CUS(value)) : "memory");
}

#undef __LDG_PTR
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300))) || defined(_NVHPC_CUDA) */
/******************************************************************************
*                             __nv_bfloat162 comparison                       *
******************************************************************************/
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define __COMPARISON_OP_BFLOAT162_MACRO(name) {\
   __nv_bfloat162 val; \
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,\
   asm( "{ " __CUDA_BF16_STRINGIFY(name) ".bf16x2.bf16x2 %0,%1,%2;\n}" \
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b))); \
,\
   asm( "{.reg .b32 low_a,low_b,high_a,high_b,high_res,low_res;\n"\
        "  and.b32 high_a, %1, 0xffff0000U;\n"\
        "  and.b32 high_b, %2, 0xffff0000U;\n"\
        "  shl.b32 low_a, %1, 16;\n"\
        "  shl.b32 low_b, %2, 16;\n"\
        "  " __CUDA_BF16_STRINGIFY(name) ".f32.f32 low_res, low_a, low_b;\n"\
        "  " __CUDA_BF16_STRINGIFY(name) ".f32.f32 high_res, high_a, high_b;\n"\
        "  shr.u32 low_res, low_res, 16;\n"\
        "  or.b32  %0, high_res, low_res;}\n"\
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b))); \
)\
   return val; \
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_heq2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.eq)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hne2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.ne)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hle2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.le)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hge2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.ge)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hlt2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.lt)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hgt2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.gt)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hequ2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.equ)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hneu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.neu)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hleu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.leu)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hgeu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.geu)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hltu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.ltu)
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hgtu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __COMPARISON_OP_BFLOAT162_MACRO(set.gtu)
}
#undef __COMPARISON_OP_BFLOAT162_MACRO
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */

__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __heq2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_heq2(a, b);
,
    __nv_bfloat162_raw val;
    val.x = __heq(a.x, b.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __heq(a.y, b.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    return __nv_bfloat162(val);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hne2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hne2(a, b);
,
    __nv_bfloat162_raw val;
    val.x = __hne(a.x, b.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __hne(a.y, b.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    return __nv_bfloat162(val);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hle2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hle2(a, b);
,
    __nv_bfloat162_raw val;
    val.x = __hle(a.x, b.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __hle(a.y, b.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    return __nv_bfloat162(val);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hge2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hge2(a, b);
,
    __nv_bfloat162_raw val;
    val.x = __hge(a.x, b.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __hge(a.y, b.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    return __nv_bfloat162(val);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hlt2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hlt2(a, b);
,
    __nv_bfloat162_raw val;
    val.x = __hlt(a.x, b.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __hlt(a.y, b.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    return __nv_bfloat162(val);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hgt2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hgt2(a, b);
,
    __nv_bfloat162_raw val;
    val.x = __hgt(a.x, b.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __hgt(a.y, b.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    return __nv_bfloat162(val);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hequ2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hequ2(a, b);
,
    __nv_bfloat162_raw val;
    val.x = __hequ(a.x, b.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __hequ(a.y, b.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    return __nv_bfloat162(val);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hneu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hneu2(a, b);
,
    __nv_bfloat162_raw val;
    val.x = __hneu(a.x, b.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __hneu(a.y, b.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    return __nv_bfloat162(val);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hleu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hleu2(a, b);
,
    __nv_bfloat162_raw val;
    val.x = __hleu(a.x, b.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __hleu(a.y, b.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    return __nv_bfloat162(val);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hgeu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hgeu2(a, b);
,
    __nv_bfloat162_raw val;
    val.x = __hgeu(a.x, b.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __hgeu(a.y, b.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    return __nv_bfloat162(val);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hltu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hltu2(a, b);
,
    __nv_bfloat162_raw val;
    val.x = __hltu(a.x, b.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __hltu(a.y, b.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    return __nv_bfloat162(val);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hgtu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hgtu2(a, b);
,
    __nv_bfloat162_raw val;
    val.x = __hgtu(a.x, b.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __hgtu(a.y, b.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    return __nv_bfloat162(val);
)
}

/******************************************************************************
*                __nv_bfloat162 comparison with mask output                   *
******************************************************************************/
#define __COMPARISON_OP_BFLOAT162_MACRO_MASK(name) {\
   unsigned val; \
   asm( "{ " __CUDA_BF16_STRINGIFY(name) ".u32.bf16x2 %0,%1,%2;\n}" \
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b))); \
   return val; \
}

__CUDA_HOSTDEVICE_BF16_DECL__ unsigned __heq2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __COMPARISON_OP_BFLOAT162_MACRO_MASK(set.eq)
,
    const unsigned short px = __heq(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __heq(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned __hne2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __COMPARISON_OP_BFLOAT162_MACRO_MASK(set.ne)
,
    const unsigned short px = __hne(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hne(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned __hle2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __COMPARISON_OP_BFLOAT162_MACRO_MASK(set.le)
,
    const unsigned short px = __hle(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hle(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned __hge2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __COMPARISON_OP_BFLOAT162_MACRO_MASK(set.ge)
,
    const unsigned short px = __hge(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hge(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned __hlt2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __COMPARISON_OP_BFLOAT162_MACRO_MASK(set.lt)
,
    const unsigned short px = __hlt(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hlt(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned __hgt2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __COMPARISON_OP_BFLOAT162_MACRO_MASK(set.gt)
,
    const unsigned short px = __hgt(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hgt(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned __hequ2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __COMPARISON_OP_BFLOAT162_MACRO_MASK(set.equ)
,
    const unsigned short px = __hequ(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hequ(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned __hneu2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __COMPARISON_OP_BFLOAT162_MACRO_MASK(set.neu)
,
    const unsigned short px = __hneu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hneu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned __hleu2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __COMPARISON_OP_BFLOAT162_MACRO_MASK(set.leu)
,
    const unsigned short px = __hleu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hleu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned __hgeu2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __COMPARISON_OP_BFLOAT162_MACRO_MASK(set.geu)
,
    const unsigned short px = __hgeu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hgeu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned __hltu2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __COMPARISON_OP_BFLOAT162_MACRO_MASK(set.ltu)
,
    const unsigned short px = __hltu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hltu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ unsigned __hgtu2_mask(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __COMPARISON_OP_BFLOAT162_MACRO_MASK(set.gtu)
,
    const unsigned short px = __hgtu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    const unsigned short py = __hgtu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
    unsigned ur = (unsigned)py;
             ur <<= (unsigned)16U;
             ur |= (unsigned)px;
    return ur;
)
}
#undef __COMPARISON_OP_BFLOAT162_MACRO_MASK

#define __BOOL_COMPARISON_OP_BFLOAT162_MACRO(name) {\
   unsigned int val; \
   bool retval; \
   asm( "{ " __CUDA_BF16_STRINGIFY(name) ".bf16x2.bf16x2 %0,%1,%2;\n}" \
        :"=r"(val) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b))); \
   if (val == 0x3F803F80U) {\
      retval = true; \
   } else { \
      retval = false; \
   }\
   return retval;\
}

__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbeq2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.eq)
,
    return (__heq(a.x, b.x) && __heq(a.y, b.y));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbne2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.ne)
,
    return (__hne(a.x, b.x) && __hne(a.y, b.y));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hble2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.le)
,
    return (__hle(a.x, b.x) && __hle(a.y, b.y));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbge2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.ge)
,
    return (__hge(a.x, b.x) && __hge(a.y, b.y));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hblt2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.lt)
,
    return (__hlt(a.x, b.x) && __hlt(a.y, b.y));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbgt2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.gt)
,
    return (__hgt(a.x, b.x) && __hgt(a.y, b.y));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbequ2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.equ)
,
    return (__hequ(a.x, b.x) && __hequ(a.y, b.y));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbneu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.neu)
,
    return (__hneu(a.x, b.x) && __hneu(a.y, b.y));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbleu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.leu)
,
    return (__hleu(a.x, b.x) && __hleu(a.y, b.y));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbgeu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.geu)
,
    return (__hgeu(a.x, b.x) && __hgeu(a.y, b.y));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbltu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.ltu)
,
    return (__hltu(a.x, b.x) && __hltu(a.y, b.y));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hbgtu2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __BOOL_COMPARISON_OP_BFLOAT162_MACRO(set.gtu)
,
    return (__hgtu(a.x, b.x) && __hgtu(a.y, b.y));
)
}
#undef __BOOL_COMPARISON_OP_BFLOAT162_MACRO
/******************************************************************************
*                             __nv_bfloat16 comparison                              *
******************************************************************************/
#define __COMPARISON_OP_BFLOAT16_MACRO(name) {\
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,\
   unsigned short val; \
   asm( "{ .reg .pred __$temp3;\n" \
        "  setp." __CUDA_BF16_STRINGIFY(name) ".bf16  __$temp3, %1, %2;\n" \
        "  selp.u16 %0, 1, 0, __$temp3;}" \
        : "=h"(val) : "h"(__BFLOAT16_TO_CUS(a)), "h"(__BFLOAT16_TO_CUS(b))); \
   return (val != 0U) ? true : false; \
,\
   unsigned int val; \
   asm( "{.reg .b32 a,b;\n"\
        "  mov.b32 a, {0, %1};\n"\
        "  mov.b32 b, {0, %2};\n"\
        "  set." __CUDA_BF16_STRINGIFY(name) ".f32.f32 %0, a, b;}\n"\
        :"=r"(val) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b))); \
   return (val != 0U) ? true : false; \
)\
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __heq(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __COMPARISON_OP_BFLOAT16_MACRO(eq)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return (fa == fb);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hne(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __COMPARISON_OP_BFLOAT16_MACRO(ne)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return (fa != fb) && (!__hisnan(a)) && (!__hisnan(b));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hle(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __COMPARISON_OP_BFLOAT16_MACRO(le)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return (fa <= fb);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hge(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __COMPARISON_OP_BFLOAT16_MACRO(ge)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return (fa >= fb);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hlt(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __COMPARISON_OP_BFLOAT16_MACRO(lt)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return (fa < fb);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hgt(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __COMPARISON_OP_BFLOAT16_MACRO(gt)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return (fa > fb);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hequ(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __COMPARISON_OP_BFLOAT16_MACRO(equ)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return (fa == fb) || (__hisnan(a)) || (__hisnan(b));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hneu(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __COMPARISON_OP_BFLOAT16_MACRO(neu)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return (fa != fb);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hleu(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __COMPARISON_OP_BFLOAT16_MACRO(leu)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return (fa <= fb) || (__hisnan(a)) || (__hisnan(b));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hgeu(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __COMPARISON_OP_BFLOAT16_MACRO(geu)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return (fa >= fb) || (__hisnan(a)) || (__hisnan(b));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hltu(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __COMPARISON_OP_BFLOAT16_MACRO(ltu)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return (fa < fb) || (__hisnan(a)) || (__hisnan(b));
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hgtu(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    __COMPARISON_OP_BFLOAT16_MACRO(gtu)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return (fa > fb) || (__hisnan(a)) || (__hisnan(b));
)
}
#undef __COMPARISON_OP_BFLOAT16_MACRO
/******************************************************************************
*                            __nv_bfloat162 arithmetic                        *
******************************************************************************/
#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hadd2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
   __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm( "{ add.bf16x2 %0,%1,%2; }\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
,
   asm( "{.reg .b32 c;\n"
        "  mov.b32 c, 0x3f803f80U;\n"
        "  fma.rn.bf16x2 %0,%1,c,%2;}\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
)
   return val;
}

__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hsub2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
   __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm( "{ sub.bf16x2 %0,%1,%2; }\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
,
   asm( "{.reg .b32 c;\n"
        "  mov.b32 c, 0xbf80bf80U;\n"
        "  fma.rn.bf16x2 %0,%2,c,%1;}\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
)
   return val;
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hmul2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
   __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm( "{ mul.bf16x2 %0,%1,%2; }\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
,
   asm( "{.reg .b32 c;\n"
        "  mov.b32 c, 0x80008000U;\n"
        "  fma.rn.bf16x2 %0,%1,%2,c;}\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
)
   return val;
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hadd2_rn(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
   __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm( "{ add.rn.bf16x2 %0,%1,%2; }\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
,
   asm( "{.reg .b32 c;\n"
        "  mov.b32 c, 0x3f803f80U;\n"
        "  fma.rn.bf16x2 %0,%1,c,%2;}\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
)
   return val;
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hsub2_rn(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
   __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm( "{ sub.rn.bf16x2 %0,%1,%2; }\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
,
   asm( "{.reg .b32 c;\n"
        "  mov.b32 c, 0xbf80bf80U;\n"
        "  fma.rn.bf16x2 %0,%2,c,%1;}\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
)
   return val;
}
__CUDA_BF16_DECL__ __nv_bfloat162 __internal_device_hmul2_rn(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
   __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm( "{ mul.rn.bf16x2 %0,%1,%2; }\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
,
   asm( "{.reg .b32 c;\n"
        "  mov.b32 c, 0x80008000U;\n"
        "  fma.rn.bf16x2 %0,%1,%2,c;}\n"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
)
   return val;
}
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hadd2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    val = __internal_device_hadd2(a, b);
,
    val.x = __hadd(a.x, b.x);
    val.y = __hadd(a.y, b.y);
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hsub2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    val = __internal_device_hsub2(a, b);
,
    val.x = __hsub(a.x, b.x);
    val.y = __hsub(a.y, b.y);
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmul2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    val = __internal_device_hmul2(a, b);
,
    val.x = __hmul(a.x, b.x);
    val.y = __hmul(a.y, b.y);
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hadd2_rn(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    val = __internal_device_hadd2_rn(a, b);
,
    val.x = __hadd_rn(a.x, b.x);
    val.y = __hadd_rn(a.y, b.y);
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hsub2_rn(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    val = __internal_device_hsub2_rn(a, b);
,
    val.x = __hsub_rn(a.x, b.x);
    val.y = __hsub_rn(a.y, b.y);
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmul2_rn(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    val = __internal_device_hmul2_rn(a, b);
,
    val.x = __hmul_rn(a.x, b.x);
    val.y = __hmul_rn(a.y, b.y);
)
    return val;
}

__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hadd2_sat(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm( "{.reg .b32 f, one, zero;\n"
        "  mov.b32 one, 0x3f803f80U;\n"
        "  mov.b32 zero, 0;\n"
        "  fma.rn.bf16x2 f,%1,one,%2;\n"
        "  max.bf16x2 f, f, zero;\n"
        "  min.bf16x2 %0, f, one;\n}"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
,
    val.x = __hadd_sat(a.x, b.x);
    val.y = __hadd_sat(a.y, b.y);
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hsub2_sat(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm( "{.reg .b32 f, one, zero, mone;\n"
        "  mov.b32 one, 0x3f803f80U;\n"
        "  mov.b32 zero, 0;\n"
        "  mov.b32 mone, 0xbf80bf80U;\n"
        "  fma.rn.bf16x2 f,%2,mone,%1;\n"
        "  max.bf16x2 f, f, zero;\n"
        "  min.bf16x2 %0, f, one;\n}"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
,
    val.x = __hsub_sat(a.x, b.x);
    val.y = __hsub_sat(a.y, b.y);
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmul2_sat(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
    __nv_bfloat162 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm( "{.reg .b32 f, one, zero, mzero;\n"
        "  mov.b32 one, 0x3f803f80U;\n"
        "  mov.b32 zero, 0;\n"
        "  mov.b32 mzero, 0x80008000U;\n"
        "  fma.rn.bf16x2 f,%1,%2,mzero;\n"
        "  max.bf16x2 f, f, zero;\n"
        "  min.bf16x2 %0, f, one;\n}"
        :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
,
    val.x = __hmul_sat(a.x, b.x);
    val.y = __hmul_sat(a.y, b.y);
)
    return val;
}
#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat162 __hfma2(const __nv_bfloat162 a, const __nv_bfloat162 b, const __nv_bfloat162 c)
{
    __nv_bfloat162 val;
    asm( "{fma.rn.bf16x2 %0,%1,%2,%3;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)),"r"(__BFLOAT162_TO_CUI(c)));
    return val;
}
__CUDA_BF16_DECL__ __nv_bfloat162 __hfma2_sat(const __nv_bfloat162 a, const __nv_bfloat162 b, const __nv_bfloat162 c)
{
    __nv_bfloat162 val;
    asm( "{ .reg .b32 f, one, zero;\n"
         "  mov.b32 one, 0x3f803f80U;\n"
         "  mov.b32 zero, 0;\n"
         "  fma.rn.bf16x2 f, %1, %2, %3;\n"
         "  max.bf16x2 f, f, zero;\n"
         "  min.bf16x2 %0, f, one;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)),"r"(__BFLOAT162_TO_CUI(c)));
    return val;
}
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __h2div(const __nv_bfloat162 a, const __nv_bfloat162 b) {
    __nv_bfloat16 ha, hb;

    ha = __low2bfloat16(a);
    hb = __low2bfloat16(b);

    const __nv_bfloat16 v1 = __hdiv(ha, hb);

    ha = __high2bfloat16(a);
    hb = __high2bfloat16(b);

    const __nv_bfloat16 v2 = __hdiv(ha, hb);

    return __halves2bfloat162(v1, v2);
}
/******************************************************************************
*                             __nv_bfloat16 arithmetic                        *
******************************************************************************/
#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 __internal_sm80_device_hadd(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
    __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm( "{ add.bf16 %0,%1,%2; }\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
,
    asm( "{.reg .b16 c;\n"
        "  mov.b16 c, 0x3f80U;\n"
        "  fma.rn.bf16 %0,%1,c,%2;}\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
)
    return val;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __internal_sm80_device_hsub(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
   __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm( "{ sub.bf16 %0,%1,%2; }\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
,
   asm( "{.reg .b16 c;\n"
        "  mov.b16 c, 0xbf80U;\n"
        "  fma.rn.bf16 %0,%2,c,%1;}\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
)
   return val;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __internal_sm80_device_hmul(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
   __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm( "{ mul.bf16 %0,%1,%2; }\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
,
   asm( "{.reg .b16 c;\n"
        "  mov.b16 c, 0x8000U;\n"
        "  fma.rn.bf16 %0,%1,%2,c;}\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
)
   return val;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __internal_sm80_device_hadd_rn(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
   __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm( "{ add.rn.bf16 %0,%1,%2; }\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
,
   asm( "{.reg .b16 c;\n"
        "  mov.b16 c, 0x3f80U;\n"
        "  fma.rn.bf16 %0,%1,c,%2;}\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
)
   return val;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __internal_sm80_device_hsub_rn(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
   __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm( "{ sub.rn.bf16 %0,%1,%2; }\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
,
   asm( "{.reg .b16 c;\n"
        "  mov.b16 c, 0xbf80U;\n"
        "  fma.rn.bf16 %0,%2,c,%1;}\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
)
   return val;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __internal_sm80_device_hmul_rn(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
   __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
   asm( "{ mul.rn.bf16 %0,%1,%2; }\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
,
   asm( "{.reg .b16 c;\n"
        "  mov.b16 c, 0x8000U;\n"
        "  fma.rn.bf16 %0,%1,%2,c;}\n"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
)
   return val;
}
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA) */
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 __internal_device_hadd(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
    __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    val = __internal_sm80_device_hadd(a, b);
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    // avoid ftz in device code
    val = __float2bfloat16(__fmaf_ieee_rn(fa, 1.0f, fb));
)
    return val;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __internal_device_hsub(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
    __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    val = __internal_sm80_device_hsub(a, b);
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    // avoid ftz in device code
    val = __float2bfloat16(__fmaf_ieee_rn(fb, -1.0f, fa));
)
    return val;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __internal_device_hmul(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
    __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    val = __internal_sm80_device_hmul(a, b);
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    // avoid ftz in device code
    val = __float2bfloat16(__fmaf_ieee_rn(fa, fb, -0.0f));
)
    return val;
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hadd(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hadd(a, b);
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return __float2bfloat16(fa + fb);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hsub(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hsub(a, b);
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return __float2bfloat16(fa - fb);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmul(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hmul(a, b);
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return __float2bfloat16(fa * fb);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hadd_rn(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    return __internal_sm80_device_hadd_rn(a, b);
,
    return __hadd(a, b);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hsub_rn(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    return __internal_sm80_device_hsub_rn(a, b);
,
    return __hsub(a, b);

)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmul_rn(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    return __internal_sm80_device_hmul_rn(a, b);
,
    return __hmul(a, b);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hadd_sat(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
    __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm( "{ .reg .b16 f, one, zero;\n"
         "  mov.b16 one, 0x3f80U;\n"
         "  mov.b16 zero, 0;\n"
         "  fma.rn.bf16 f, %1, one, %2;\n"
         "  max.bf16 f, f, zero;\n"
         "  min.bf16 %0, f, one;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
,
    val = __hmin(__hmax(__hadd(a, b), CUDART_ZERO_BF16), CUDART_ONE_BF16);
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hsub_sat(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
    __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm( "{ .reg .b16 f, one, zero, mone;\n"
         "  mov.b16 one, 0x3f80U;\n"
         "  mov.b16 zero, 0;\n"
         "  mov.b16 mone, 0xbf80U;\n"
         "  fma.rn.bf16 f, %2, mone, %1;\n"
         "  max.bf16 f, f, zero;\n"
         "  min.bf16 %0, f, one;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
,
    val = __hmin(__hmax(__hsub(a, b), CUDART_ZERO_BF16), CUDART_ONE_BF16);
)
    return val;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmul_sat(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
    __nv_bfloat16 val;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm( "{ .reg .b16 f, one, zero, mzero;\n"
         "  mov.b16 one, 0x3f80U;\n"
         "  mov.b16 zero, 0;\n"
         "  mov.b16 mzero, 0x8000U;\n"
         "  fma.rn.bf16 f, %1, %2, mzero;\n"
         "  max.bf16 f, f, zero;\n"
         "  min.bf16 %0, f, one;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
,
    val = __hmin(__hmax(__hmul(a, b), CUDART_ZERO_BF16), CUDART_ONE_BF16);
)
    return val;
}
#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 __hfma(const __nv_bfloat16 a, const __nv_bfloat16 b, const __nv_bfloat16 c)
{
    __nv_bfloat16 val;
    asm( "{fma.rn.bf16 %0,%1,%2,%3;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)),"h"(__BFLOAT16_TO_CUS(c)));
    return val;
}
__CUDA_BF16_DECL__ __nv_bfloat16 __hfma_sat(const __nv_bfloat16 a, const __nv_bfloat16 b, const __nv_bfloat16 c)
{
    __nv_bfloat16 val;
    asm( "{ .reg .b16 f, one, zero;\n"
         "  mov.b16 one, 0x3f80U;\n"
         "  mov.b16 zero, 0;\n"
         "  fma.rn.bf16 f, %1, %2, %3;\n"
         "  max.bf16 f, f, zero;\n"
         "  min.bf16 %0, f, one;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)),"h"(__BFLOAT16_TO_CUS(c)));
    return val;
}
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA) */
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define __BINARY_OP_BFLOAT16_MACRO(name) /* do */ {\
   __nv_bfloat16 val; \
   asm( "{.reg .b32 a,b,res;\n"\
        "  mov.b32 a, {0,%1};\n"\
        "  mov.b32 b, {0,%2};\n"\
        "  " __CUDA_BF16_STRINGIFY(name) ".f32 res, a, b;\n"\
        "  cvt.rn.bf16.f32 %0, res;}\n"\
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b))); \
   return val; \
} /* while(0) */
__CUDA_BF16_DECL__ __nv_bfloat16 __internal_device_hdiv(const __nv_bfloat16 a, const __nv_bfloat16 b) {
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __BINARY_OP_BFLOAT16_MACRO(div.rn)
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    float fr;
    asm( "{ div.rn.f32 %0, %1, %2; }\n"
         :"=f"(fr) : "f"(fa),"f"(fb));
    return __float2bfloat16(fr);
)
}
#undef __BINARY_OP_BFLOAT16_MACRO
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hdiv(const __nv_bfloat16 a, const __nv_bfloat16 b) {
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hdiv(a, b);
,
    const float fa = __bfloat162float(a);
    const float fb = __bfloat162float(b);
    return __float2bfloat16(fa / fb);
)
}

/******************************************************************************
*                             __nv_bfloat162 functions                        *
******************************************************************************/
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 __hsin_internal(const __nv_bfloat16 a) {
    float f = __bfloat162float(a);
    float r = sinf(f);
    // Detect compile-time FTZ setting:
    // if subnormal constant is not flushed to zero at compile-time, then
    // ftz=off, and it is safe to return result of sinf()
    // Otherwise, ftz=on, then sinf() result is valid for non-flushed
    // values, and subnormal input is returned unchanged via else
    // branch.
    if ((__uint_as_float(0x00000001U) > 0.0f) || (f != 0.0f))
    {
        f = r;
    }
    return __float2bfloat16_rn(f);
}
__CUDA_BF16_DECL__ __nv_bfloat16 hsin(const __nv_bfloat16 a) {
    return __hsin_internal(a);
}
__CUDA_BF16_DECL__ __nv_bfloat162 h2sin(const __nv_bfloat162 a) {
    const __nv_bfloat16 l = __low2bfloat16(a);
    const __nv_bfloat16 h = __high2bfloat16(a);
    return __halves2bfloat162(__hsin_internal(l), __hsin_internal(h));
}
__CUDA_BF16_DECL__ __nv_bfloat16 __hcos_internal(const __nv_bfloat16 a) {
    float f = __bfloat162float(a);
    f = cosf(f);
    return __float2bfloat16_rn(f);
}
__CUDA_BF16_DECL__ __nv_bfloat16 hcos(const __nv_bfloat16 a) {
    return __hcos_internal(a);
}
__CUDA_BF16_DECL__ __nv_bfloat162 h2cos(const __nv_bfloat162 a) {
    const __nv_bfloat16 l = __low2bfloat16(a);
    const __nv_bfloat16 h = __high2bfloat16(a);
    return __halves2bfloat162(__hcos_internal(l), __hcos_internal(h));
}

__CUDA_BF16_DECL__ float __internal_device_fast_bf16exp(const float x)
{
    const float log2e_up = __uint_as_float(0x3FB8AA3CU);
    float fa = x * log2e_up;
    asm("{ ex2.approx.f32 %0, %0; }" : "+f"(fa));
    return fa;
}

__CUDA_BF16_DECL__ __nv_bfloat16 hexp(const __nv_bfloat16 a) {
    float fa = __bfloat162float(a);
    fa = __internal_device_fast_bf16exp(fa);
    return __float2bfloat16_rn(fa);
}

#define __APPROX_FCAST2(fun) /* do */ {\
   __nv_bfloat162 val;\
   asm("{.reg.b16         hl, hu;         \n"\
                " .reg.b32         fl, fu;         \n"\
                "  mov.b32         {hl, hu}, %1;   \n"\
                "  mov.b32         fl, {0,hl};     \n"\
                "  mov.b32         fu, {0,hu};     \n"\
                "  " __CUDA_BF16_STRINGIFY(fun) ".approx.f32   fl, fl;     \n"\
                "  " __CUDA_BF16_STRINGIFY(fun) ".approx.f32   fu, fu;     \n"\
                "  cvt.rn.bf16.f32    hl, fl;     \n"\
                "  cvt.rn.bf16.f32    hu, fu;     \n"\
                "  mov.b32         %0, {hl, hu};   \n"\
                "}":"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)));       \
   return val;\
} /* while(0) */
#define __BF16_SPEC_CASE2(i,r, spc, ulp) \
   "{.reg.b32 spc, ulp, p;\n"\
   "  mov.b32 spc," __CUDA_BF16_STRINGIFY(spc) ";\n"\
   "  mov.b32 ulp," __CUDA_BF16_STRINGIFY(ulp) ";\n"\
   "  set.eq.f16x2.f16x2 p," __CUDA_BF16_STRINGIFY(i) ", spc;\n"\
   "  fma.rn.bf16x2 " __CUDA_BF16_STRINGIFY(r) ",p,ulp," __CUDA_BF16_STRINGIFY(r) ";\n}\n"

__CUDA_BF16_DECL__ __nv_bfloat162 h2exp(const __nv_bfloat162 a) {
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat162 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         h,r,fl,fu, C;   \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  mov.b32         h, %1;          \n"
        "  mov.b32         fl, {0,hl};     \n"
        "  mov.b32         fu, {0,hu};     \n"
        "  mov.b32         C, 0x3FB8AA3CU;  \n"
        "  mul.f32         fl,fl,C;        \n"
        "  mul.f32         fu,fu,C;        \n"
        "  ex2.approx.f32      fl, fl;     \n"
        "  ex2.approx.f32      fu, fu;     \n"
        "  cvt.rn.bf16.f32    hl, fl;     \n"
        "  cvt.rn.bf16.f32    hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        "  mov.b32         %0, r;  \n"
        "}":"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)));
    return val;
,
    return __floats2bfloat162_rn( __internal_device_fast_bf16exp(__low2float(a)), __internal_device_fast_bf16exp(__high2float(a)) );
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 hexp2(const __nv_bfloat16 a) {
    float fa = __bfloat162float(a);
    asm("{ ex2.approx.f32 %0, %0; }" : "+f"(fa));
    return __float2bfloat16_rn(fa);
}
__CUDA_BF16_DECL__ __nv_bfloat162 h2exp2(const __nv_bfloat162 a) {
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __APPROX_FCAST2(ex2)
,
    float fl = __low2float(a);
    asm("{ ex2.approx.f32 %0, %0; }" : "+f"(fl));
    float fh = __high2float(a);
    asm("{ ex2.approx.f32 %0, %0; }" : "+f"(fh));
    return __floats2bfloat162_rn( fl, fh );
)
}

__CUDA_BF16_DECL__ __nv_bfloat16 hexp10(const __nv_bfloat16 a) {
    const float log10_2 = __uint_as_float(0x40549A78U);
    float fa = __bfloat162float(a) * log10_2;
    asm("{ ex2.approx.f32 %0, %0; }" : "+f"(fa));
    __nv_bfloat16 r = __float2bfloat16_rn(fa);
    __nv_bfloat16_raw araw = static_cast<__nv_bfloat16_raw>(a);
    if (araw.x == (unsigned short)0xBC95U)
    {
        araw.x = 0x3f75U;
        r = static_cast<__nv_bfloat16>(araw);
    }
    return r;
}
__CUDA_BF16_DECL__ __nv_bfloat162 h2exp10(const __nv_bfloat162 a) {
    __nv_bfloat162 r;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         h,r,fl,fu, C;   \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  mov.b32         fl, {0,hl};     \n"
        "  mov.b32         fu, {0,hu};     \n"
        "  mov.b32         C, 0x40549A78U;  \n"
        "  mul.f32         fl,fl,C;        \n"
        "  mul.f32         fu,fu,C;        \n"
        "  ex2.approx.f32      fl, fl;     \n"
        "  ex2.approx.f32      fu, fu;     \n"
        "  cvt.rn.bf16.f32    hl, fl;     \n"
        "  cvt.rn.bf16.f32    hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        __BF16_SPEC_CASE2(%1, r, 0xBC95BC95U,0xBF00BF00U)
        "  mov.b32         %0, r;  \n"
        "}":"=r"(__BFLOAT162_TO_UI(r)) : "r"(__BFLOAT162_TO_CUI(a)));
,
    const float log10_2 = __uint_as_float(0x40549A78U);
    float fl = __low2float(a) * log10_2;
    asm("{ ex2.approx.f32 %0, %0; }" : "+f"(fl));

    float fh = __high2float(a) * log10_2;
    asm("{ ex2.approx.f32 %0, %0; }" : "+f"(fh));

    r = __floats2bfloat162_rn( fl, fh );

    const __nv_bfloat162_raw araw = static_cast<__nv_bfloat162_raw>(a);
    if (araw.x == (unsigned short)0xBC95U)
    {
        __nv_bfloat16_raw raw_fix;
        raw_fix.x = (unsigned short)0x3f75U;
        r.x = static_cast<__nv_bfloat16>(raw_fix);
    }
    if (araw.y == (unsigned short)0xBC95U)
    {
        __nv_bfloat16_raw raw_fix;
        raw_fix.x = (unsigned short)0x3f75U;
        r.y = static_cast<__nv_bfloat16>(raw_fix);
    }
)
    return r;
}

__CUDA_BF16_DECL__ float __internal_device_fast_bf16log2(float x)
{
    asm("{ lg2.approx.f32 %0, %0; }" : "+f"(x));
    return x;
}

__CUDA_BF16_DECL__ __nv_bfloat16 hlog2(const __nv_bfloat16 a) {
    float fa = __bfloat162float(a);
    fa = __internal_device_fast_bf16log2(fa);
    return __float2bfloat16_rn(fa);
}

__CUDA_BF16_DECL__ __nv_bfloat162 h2log2(const __nv_bfloat162 a) {
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __APPROX_FCAST2(lg2)
,
    float fl = __low2float(a);
    fl = __internal_device_fast_bf16log2(fl);
    float fh = __high2float(a);
    fh = __internal_device_fast_bf16log2(fh);
    return __floats2bfloat162_rn( fl, fh );
)
}

__CUDA_BF16_DECL__ __nv_bfloat16 hlog(const __nv_bfloat16 a) {
    const float flt_ln2 = __uint_as_float(0x3f317218U);
    float fa = __bfloat162float(a);
    fa = __internal_device_fast_bf16log2(fa);
    fa = fa * flt_ln2;
    return __float2bfloat16_rn(fa);
}
__CUDA_BF16_DECL__ __nv_bfloat162 h2log(const __nv_bfloat162 a) {
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat162 val;
    asm("{.reg.b16         hl, hu;             \n"
        " .reg.b32         r, fl, fu, C, h;    \n"
        "  mov.b32         {hl, hu}, %1;       \n"
        "  mov.b32         h, %1;              \n"
        "  mov.b32         fl, {0,hl};         \n"
        "  mov.b32         fu, {0,hu};         \n"
        "  lg2.approx.f32      fl, fl;         \n"
        "  lg2.approx.f32      fu, fu;         \n"
        "  mov.b32         C, 0x3f317218U;     \n"
        "  mul.f32         fl,fl,C;            \n"
        "  mul.f32         fu,fu,C;            \n"
        "  cvt.rn.bf16.f32    hl, fl;         \n"
        "  cvt.rn.bf16.f32    hu, fu;         \n"
        "  mov.b32         r, {hl, hu};        \n"
        "  mov.b32         %0, r;              \n"
        "}":"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)));
    return val;
,
    const float flt_ln2 = __uint_as_float(0x3f317218U);

    float fl = __low2float(a);
    fl = __internal_device_fast_bf16log2(fl);
    fl = fl * flt_ln2;

    float fh = __high2float(a);
    fh = __internal_device_fast_bf16log2(fh);
    fh = fh * flt_ln2;

    return __floats2bfloat162_rn( fl, fh );
)
}

__CUDA_BF16_DECL__ __nv_bfloat16 hlog10(const __nv_bfloat16 a) {
    const float flt_log10_2 = __uint_as_float(0x3E9A209BU);
    float fa = __bfloat162float(a);
    fa = __internal_device_fast_bf16log2(fa);
    fa = fa * flt_log10_2;
    return __float2bfloat16_rn(fa);
}
__CUDA_BF16_DECL__ __nv_bfloat162 h2log10(const __nv_bfloat162 a) {
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat162 val;
    asm("{.reg.b16         hl, hu;             \n"
        " .reg.b32         r, fl, fu, C, h;    \n"
        "  mov.b32         {hl, hu}, %1;       \n"
        "  mov.b32         h, %1;              \n"
        "  mov.b32         fl, {0,hl};         \n"
        "  mov.b32         fu, {0,hu};         \n"
        "  lg2.approx.f32      fl, fl;         \n"
        "  lg2.approx.f32      fu, fu;         \n"
        "  mov.b32         C, 0x3E9A209BU;      \n"
        "  mul.f32         fl,fl,C;            \n"
        "  mul.f32         fu,fu,C;            \n"
        "  cvt.rn.bf16.f32    hl, fl;         \n"
        "  cvt.rn.bf16.f32    hu, fu;         \n"
        "  mov.b32         r, {hl, hu};        \n"
        "  mov.b32         %0, r;              \n"
        "}":"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)));
    return val;
,
    const float flt_log10_2 = __uint_as_float(0x3E9A209BU);

    float fl = __low2float(a);
    fl = __internal_device_fast_bf16log2(fl);
    fl = fl * flt_log10_2;

    float fh = __high2float(a);
    fh = __internal_device_fast_bf16log2(fh);
    fh = fh * flt_log10_2;

    return __floats2bfloat162_rn( fl, fh );
)
}

__CUDA_BF16_DECL__ __nv_bfloat162 h2rcp(const __nv_bfloat162 a) {
    float fl = __low2float(a);
    asm("{ rcp.approx.f32 %0, %0; }" : "+f"(fl));
    float fh = __high2float(a);
    asm("{ rcp.approx.f32 %0, %0; }" : "+f"(fh));
    return __floats2bfloat162_rn( fl, fh );
}
__CUDA_BF16_DECL__ __nv_bfloat16 hrcp(const __nv_bfloat16 a) {
    float fa = __bfloat162float(a);
    asm("{ rcp.approx.f32 %0, %0; }" : "+f"(fa));
    return __float2bfloat16_rn(fa);
}
__CUDA_BF16_DECL__ __nv_bfloat162 h2rsqrt(const __nv_bfloat162 a) {
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __APPROX_FCAST2(rsqrt)
,
    float fl = __low2float(a);
    asm("{ rsqrt.approx.f32 %0, %0; }" : "+f"(fl));
    float fh = __high2float(a);
    asm("{ rsqrt.approx.f32 %0, %0; }" : "+f"(fh));
    return __floats2bfloat162_rn( fl, fh );
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 hrsqrt(const __nv_bfloat16 a) {
    float fa = __bfloat162float(a);
    asm("{ rsqrt.approx.f32 %0, %0; }" : "+f"(fa));
    return __float2bfloat16_rn(fa);
}
__CUDA_BF16_DECL__ __nv_bfloat162 h2sqrt(const __nv_bfloat162 a) {
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __APPROX_FCAST2(sqrt)
,
    float fl = __low2float(a);
    asm("{ sqrt.approx.f32 %0, %0; }" : "+f"(fl));
    float fh = __high2float(a);
    asm("{ sqrt.approx.f32 %0, %0; }" : "+f"(fh));
    return __floats2bfloat162_rn( fl, fh );
)
}
__CUDA_BF16_DECL__ __nv_bfloat16 hsqrt(const __nv_bfloat16 a) {
    float fa = __bfloat162float(a);
    asm("{ sqrt.approx.f32 %0, %0; }" : "+f"(fa));
    return __float2bfloat16_rn(fa);
}
#undef __APPROX_FCAST2
#undef __BF16_SPEC_CASE2

__CUDA_BF16_DECL__ bool __internal_device_hisnan(const __nv_bfloat16 a)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 r;
    asm("{set.nan.bf16.bf16 %0,%1,%1;\n}"
        :"=h"(__BFLOAT16_TO_US(r)) : "h"(__BFLOAT16_TO_CUS(a)));
    return __BFLOAT16_TO_CUS(r) != 0U;
,
    unsigned int r;
    asm( "{.reg .b32 a;\n"
         "  mov.b32 a, {0,%1};\n"
         "  set.nan.f32.f32 %0, a, a;}\n"
         :"=r"(r) : "h"(__BFLOAT16_TO_CUS(a)));
    return r != 0U;
)
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hisnan2(const __nv_bfloat162 a)
{
    __nv_bfloat162 r;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    asm("{set.nan.bf16x2.bf16x2 %0,%1,%1;\n}"
        :"=r"(__BFLOAT162_TO_UI(r)) : "r"(__BFLOAT162_TO_CUI(a)));
,
    __nv_bfloat162_raw val;
    val.x = __hisnan(a.x) ? (unsigned short)0x3F80U : (unsigned short)0U;
    val.y = __hisnan(a.y) ? (unsigned short)0x3F80U : (unsigned short)0U;
    r = __nv_bfloat162(val);
)
    return r;
}
__CUDA_HOSTDEVICE_BF16_DECL__ bool __hisnan(const __nv_bfloat16 a)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hisnan(a);
,
    const __nv_bfloat16_raw hr = static_cast<__nv_bfloat16_raw>(a);
    return ((hr.x & 0x7FFFU) > 0x7F80U);
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hneg2(const __nv_bfloat162 a)
{
    __nv_bfloat162 r;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm("{neg.bf16x2 %0,%1;\n}"
        :"=r"(__BFLOAT162_TO_UI(r)) : "r"(__BFLOAT162_TO_CUI(a)));
,
    r.x = __hneg(a.x);
    r.y = __hneg(a.y);
)
    return r;
}
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 __internal_device_hneg(const __nv_bfloat16 a)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat16 r;
    asm("{neg.bf16 %0,%1;\n}"
        :"=h"(__BFLOAT16_TO_US(r)) : "h"(__BFLOAT16_TO_CUS(a)));
    return r;
,
    const float fa = __bfloat162float(a);
    return __float2bfloat16(__fmaf_ieee_rn(fa, -1.0f, -0.0f));
)
}
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hneg(const __nv_bfloat16 a)
{
NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    return __internal_device_hneg(a);
,
    const float fa = __bfloat162float(a);
    return __float2bfloat16(-fa);
)
}

__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __habs2(const __nv_bfloat162 a)
{
    __nv_bfloat162 r;
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    asm("{abs.bf16x2 %0,%1;\n}"
        :"=r"(__BFLOAT162_TO_UI(r)) : "r"(__BFLOAT162_TO_CUI(a)));
,
    r.x = __habs(a.x);
    r.y = __habs(a.y);
)
    return r;
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __habs(const __nv_bfloat16 a)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat16 r;
    asm("{abs.bf16 %0,%1;\n}"
        :"=h"(__BFLOAT16_TO_US(r)) : "h"(__BFLOAT16_TO_CUS(a)));
    return r;
,
    __nv_bfloat16_raw abs_a_raw = static_cast<__nv_bfloat16_raw>(a);
    abs_a_raw.x &= (unsigned short)0x7FFFU;
    if (abs_a_raw.x > (unsigned short)0x7F80U)
    {
        // return canonical NaN
        abs_a_raw.x = (unsigned short)0x7FFFU;
    }
    return static_cast<__nv_bfloat16>(abs_a_raw);
)
}

/******************************************************************************
*                             __nv_bfloat16 arithmetic                             *
******************************************************************************/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmax(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat16 val;
    asm( "{ max.bf16 %0,%1,%2;\n}"
        :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
    return val;
,
    __nv_bfloat16 maxval;

    maxval = (__hge(a, b) || __hisnan(b)) ? a : b;

    if (__hisnan(maxval))
    {
        // if both inputs are NaN, return canonical NaN
        maxval = CUDART_NAN_BF16;
    }
    else if (__heq(a, b))
    {
        // hmax(+0.0, -0.0) = +0.0
        // unsigned compare 0x8000U > 0x0000U
        __nv_bfloat16_raw ra = __nv_bfloat16_raw(a);
        __nv_bfloat16_raw rb = __nv_bfloat16_raw(b);
        maxval = (ra.x > rb.x) ? b : a;
    }

    return maxval;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmin(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat16 val;
    asm( "{ min.bf16 %0,%1,%2;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
    return val;
,
    __nv_bfloat16 minval;

    minval = (__hle(a, b) || __hisnan(b)) ? a : b;

    if (__hisnan(minval))
    {
        // if both inputs are NaN, return canonical NaN
        minval = CUDART_NAN_BF16;
    }
    else if (__heq(a, b))
    {
        // hmin(+0.0, -0.0) = -0.0
        // unsigned compare 0x8000U > 0x0000U
        __nv_bfloat16_raw ra = __nv_bfloat16_raw(a);
        __nv_bfloat16_raw rb = __nv_bfloat16_raw(b);
        minval = (ra.x > rb.x) ? a : b;
    }

    return minval;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmax_nan(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat16 val;
    asm( "{ max.NaN.bf16 %0,%1,%2;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
    return val;
,
    __nv_bfloat16 maxval;

    if (__hisnan(a) || __hisnan(b))
    {
        // if either input is NaN, return canonical NaN
        maxval = CUDART_NAN_BF16;
    }
    else
    {
        maxval = __hge(a, b) ? a : b;
    }

    return maxval;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __hmin_nan(const __nv_bfloat16 a, const __nv_bfloat16 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat16 val;
    asm( "{ min.NaN.bf16 %0,%1,%2;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)));
    return val;
,
    __nv_bfloat16 minval;

    if (__hisnan(a) || __hisnan(b))
    {
        // if either input is NaN, return canonical NaN
        minval = CUDART_NAN_BF16;
    }
    else
    {
        minval = __hle(a, b) ? a : b;
    }

    return minval;
)
}
#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 __hfma_relu(const __nv_bfloat16 a, const __nv_bfloat16 b, const __nv_bfloat16 c)
{
    __nv_bfloat16 val;
    asm( "{ fma.rn.relu.bf16 %0,%1,%2,%3;\n}"
         :"=h"(__BFLOAT16_TO_US(val)) : "h"(__BFLOAT16_TO_CUS(a)),"h"(__BFLOAT16_TO_CUS(b)),"h"(__BFLOAT16_TO_CUS(c)));
    return val;
}
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA) */
/******************************************************************************
*                            __nv_bfloat162 arithmetic                             *
******************************************************************************/
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmax2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat162 val;
    asm( "{ max.bf16x2 %0,%1,%2;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
    return val;
,
    __nv_bfloat162 val;
    val.x = __hmax(a.x, b.x);
    val.y = __hmax(a.y, b.y);
    return val;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmin2(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat162 val;
    asm( "{ min.bf16x2 %0,%1,%2;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
    return val;
,
    __nv_bfloat162 val;
    val.x = __hmin(a.x, b.x);
    val.y = __hmin(a.y, b.y);
    return val;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmax2_nan(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat162 val;
    asm( "{ max.NaN.bf16x2 %0,%1,%2;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
    return val;
,
    __nv_bfloat162 val;
    val.x = __hmax_nan(a.x, b.x);
    val.y = __hmax_nan(a.y, b.y);
    return val;
)
}
__CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __hmin2_nan(const __nv_bfloat162 a, const __nv_bfloat162 b)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
    __nv_bfloat162 val;
    asm( "{ min.NaN.bf16x2 %0,%1,%2;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)));
    return val;
,
    __nv_bfloat162 val;
    val.x = __hmin_nan(a.x, b.x);
    val.y = __hmin_nan(a.y, b.y);
    return val;
)
}
#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat162 __hfma2_relu(const __nv_bfloat162 a, const __nv_bfloat162 b, const __nv_bfloat162 c)
{
    __nv_bfloat162 val;
    asm( "{ fma.rn.relu.bf16x2 %0,%1,%2,%3;\n}"
         :"=r"(__BFLOAT162_TO_UI(val)) : "r"(__BFLOAT162_TO_CUI(a)),"r"(__BFLOAT162_TO_CUI(b)),"r"(__BFLOAT162_TO_CUI(c)));
    return val;
}

__CUDA_BF16_DECL__ __nv_bfloat162 __hcmadd(const __nv_bfloat162 a, const __nv_bfloat162 b, const __nv_bfloat162 c)
{
    // fast version of complex multiply-accumulate
    // (a.re, a.im) * (b.re, b.im) + (c.re, c.im)
    // acc.re = (c.re + a.re*b.re) - a.im*b.im
    // acc.im = (c.im + a.re*b.im) + a.im*b.re
    __nv_bfloat16 real_tmp = __hfma(a.x, b.x, c.x);
    __nv_bfloat16 img_tmp  = __hfma(a.x, b.y, c.y);
    real_tmp = __hfma(__hneg(a.y), b.y, real_tmp);
    img_tmp  = __hfma(a.y,         b.x, img_tmp);
    return make_bfloat162(real_tmp, img_tmp);
}
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))) || defined(_NVHPC_CUDA) */

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
/* Define __PTR for atomicAdd prototypes below, undef after done */
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __PTR   "l"
#else
#define __PTR   "r"
#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/

__CUDA_BF16_DECL__ __nv_bfloat162 atomicAdd(__nv_bfloat162 *const address, const __nv_bfloat162 val)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat162 r;
    asm volatile ("{ atom.add.noftz.bf16x2 %0,[%1],%2; }\n"
                  : "=r"(__BFLOAT162_TO_UI(r)) : __PTR(address), "r"(__BFLOAT162_TO_CUI(val))
                  : "memory");
    return r;
,
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        __nv_bfloat162 new_val = __hadd2(val, *(__nv_bfloat162*)&assumed);
        old = atomicCAS(address_as_uint, assumed, *(unsigned int*)&new_val);
    } while (assumed != old);
    return *(__nv_bfloat162*)&old;
)
}

#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA)
__CUDA_BF16_DECL__ __nv_bfloat16 atomicAdd(__nv_bfloat16 *const address, const __nv_bfloat16 val)
{
NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    __nv_bfloat16 r;
    asm volatile ("{ atom.add.noftz.bf16 %0,[%1],%2; }\n"
                  : "=h"(__BFLOAT16_TO_US(r))
                  : __PTR(address), "h"(__BFLOAT16_TO_CUS(val))
                  : "memory");
    return r;
,
    unsigned short int* address_as_us = (unsigned short int*)address;
    unsigned short int old = *address_as_us;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_us, assumed,
            __bfloat16_as_ushort(__hadd(val, __ushort_as_bfloat16(assumed))));
    } while (assumed != old);
    return __ushort_as_bfloat16(old);
)
}
#endif /* (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA) */

#undef __PTR
#endif /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
#endif /* defined(__cplusplus) */

#undef __CUDA_HOSTDEVICE_BF16_DECL__
#undef __CUDA_BF16_DECL__
#undef __CUDA_BF16_CONSTEXPR__

/* Define first-class types "nv_bfloat16" and "nv_bfloat162", unless user specifies otherwise via "#define CUDA_NO_BFLOAT16" */
/* C cannot ever have these types defined here, because __nv_bfloat16 and __nv_bfloat162 are C++ classes */
#if defined(__cplusplus) && !defined(CUDA_NO_BFLOAT16)
/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief This datatype is meant to be the first-class or fundamental
 * implementation of the bfloat16 numbers format.
 * 
 * \details Should be implemented in the compiler in the future.
 * Current implementation is a simple typedef to a respective
 * user-level type with underscores.
 */
typedef __nv_bfloat16  nv_bfloat16;

/**
 * \ingroup CUDA_MATH_INTRINSIC_BFLOAT16
 * \brief This datatype is meant to be the first-class or fundamental
 * implementation of type for pairs of bfloat16 numbers.
 * 
 * \details Should be implemented in the compiler in the future.
 * Current implementation is a simple typedef to a respective
 * user-level type with underscores.
 */
typedef __nv_bfloat162 nv_bfloat162;

#endif /* defined(__cplusplus) && !defined(CUDA_NO_BFLOAT16) */
 
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
#undef __CPP_VERSION_AT_LEAST_11_BF16
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */

#endif /* end of include guard: __CUDA_BF16_HPP__ */
