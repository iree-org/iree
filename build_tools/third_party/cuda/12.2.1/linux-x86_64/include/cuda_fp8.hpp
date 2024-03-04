/*
 * Copyright 2022-2023 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__CUDA_FP8_HPP__)
#define __CUDA_FP8_HPP__

#if !defined(__CUDA_FP8_H__)
#error "Do not include this file directly. Instead, include cuda_fp8.h."
#endif

/* C++ header for std::memcpy (used for type punning in host-side
 * implementations). When compiling as a CUDA source file memcpy is provided
 * implicitly. !defined(__CUDACC__) implies !defined(__CUDACC_RTC__).
 */
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <cstring>
#elif !defined(__cplusplus) && !defined(__CUDACC__)
#include <string.h>
#endif /* defined(__cplusplus) && !defined(__CUDACC__) */

/* Set up structure-alignment attribute */
#if !(defined __CUDA_ALIGN__)
#if defined(__CUDACC__)
#define __CUDA_ALIGN__(align) __align__(align)
#else
/* Define alignment macro based on compiler type (cannot assume C11 "_Alignas"
 * is available) */
#if __cplusplus >= 201103L
#define __CUDA_ALIGN__(n)                                                      \
    alignas(n) /* C++11 kindly gives us a keyword for this */
#else          /* !defined(__CPP_VERSION_AT_LEAST_11_FP8)*/
#if defined(__GNUC__)
#define __CUDA_ALIGN__(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#define __CUDA_ALIGN__(n) __declspec(align(n))
#else
#define __CUDA_ALIGN__(n)
#endif /* defined(__GNUC__) */
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP8) */
#endif /* defined(__CUDACC__) */
#endif /* !(defined __CUDA_ALIGN__) */

#if !(defined __CPP_VERSION_AT_LEAST_11_FP8)
/* need c++11 for explicit operators */
#define __CUDA_NO_FP8_CONVERSION_OPERATORS__
#endif

__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t
__nv_cvt_double_to_fp8(const double x, const __nv_saturation_t saturate,
                       const __nv_fp8_interpretation_t fp8_interpretation) {
    unsigned char res;
    unsigned long long int xbits;

#if defined(__CUDACC__) || (!defined __cplusplus)
    (void)memcpy(&xbits, &x, sizeof(x));
#else
    (void)std::memcpy(&xbits, &x, sizeof(x));
#endif
    unsigned char FP8_MAXNORM;
    unsigned char FP8_MANTISSA_MASK;
    unsigned short int FP8_EXP_BIAS;
    unsigned long long int FP8_SIGNIFICAND_BITS;
    const unsigned long long int DP_INF_BITS = 0x7FF0000000000000ULL;
    unsigned long long int FP8_MINDENORM_O2;
    unsigned long long int FP8_OVERFLOW_THRESHOLD;
    unsigned long long int FP8_MINNORM;

    if (fp8_interpretation == __NV_E4M3) {
        FP8_EXP_BIAS = 7U;
        FP8_SIGNIFICAND_BITS = 4ULL;
        FP8_MANTISSA_MASK = 0x7U;
        FP8_MINDENORM_O2 = 0x3F50000000000000ULL; // mindenorm/2 = 2^-10
        FP8_OVERFLOW_THRESHOLD =
            0x407D000000000000ULL; // maxnorm + 1/2ulp = 0x1.Cp+8 + 0x1p+4
        FP8_MAXNORM = 0x7EU;
        FP8_MINNORM = 0x3F90000000000000ULL; // minnorm = 2^-6
    } else {                                 //__NV_E5M2
        FP8_EXP_BIAS = 15U;
        FP8_SIGNIFICAND_BITS = 3ULL;
        FP8_MANTISSA_MASK = 0x3U;
        FP8_MINDENORM_O2 = 0x3EE0000000000000ULL; // mindenorm/2 = 2^-17
        FP8_OVERFLOW_THRESHOLD =
            0x40EE000000000000ULL -
            1ULL; // maxnorm + 1/2ulp = 0x1.Ep+15, and -1 to have common code
        FP8_MAXNORM = 0x7BU;
        FP8_MINNORM = 0x3F10000000000000ULL; // minnorm = 2^-14
    }

    // 1/2 LSB of the target format, positioned in double precision mantissa
    // helpful in midpoints detection during round-to-nearest-even step
    const unsigned long long int FP8_DP_HALF_ULP =
        (unsigned long long int)1ULL << (53ULL - FP8_SIGNIFICAND_BITS - 1ULL);
    // prepare sign bit in target format
    unsigned char sign = (unsigned char)((xbits >> 63ULL) << 7U);
    // prepare exponent field in target format
    unsigned char exp =
        (unsigned char)((((unsigned short int)(xbits >> 52ULL)) & 0x7FFU) -
                        1023U + FP8_EXP_BIAS);
    // round mantissa to target format width, rounding towards zero
    unsigned char mantissa =
        (unsigned char)(xbits >> (53ULL - FP8_SIGNIFICAND_BITS)) &
        FP8_MANTISSA_MASK;
    unsigned long long int absx = xbits & 0x7FFFFFFFFFFFFFFFULL;

    if (absx <= FP8_MINDENORM_O2) {
        // zero or underflow
        res = 0U;
    } else if (absx > DP_INF_BITS) {
        // NaN
        if (fp8_interpretation == __NV_E4M3) {
            res = 0x7FU;
        } else {
            // NaN --> QNaN
            res = 0x7EU | mantissa;
        }
    } else if (absx > FP8_OVERFLOW_THRESHOLD) {
        if (saturate == __NV_SATFINITE) {
            res = FP8_MAXNORM;
        } else {
            // __NV_NOSAT
            if (fp8_interpretation == __NV_E4M3) {
                // no Inf in E4M3
                res = 0x7FU; // NaN
            } else {
                res = 0x7CU; // Inf in E5M2
            }
        }
    } else if (absx >= FP8_MINNORM) {
        res = (unsigned char)((exp << (FP8_SIGNIFICAND_BITS - 1U)) | mantissa);
        // rounded-off bits
        unsigned long long int round =
            xbits & ((FP8_DP_HALF_ULP << 1ULL) - 1ULL);
        // round-to-nearest-even adjustment
        if ((round > FP8_DP_HALF_ULP) ||
            ((round == FP8_DP_HALF_ULP) && (mantissa & 1U))) {
            res = (unsigned char)(res + 1U);
        }
    } else // Denormal range
    {
        unsigned char shift = (unsigned char)(1U - exp);
        // add implicit leading bit
        mantissa |= (unsigned char)(1U << (FP8_SIGNIFICAND_BITS - 1U));
        // additional round-off due to denormalization
        res = (unsigned char)(mantissa >> shift);

        // rounded-off bits, including implicit leading bit
        unsigned long long int round =
            (xbits | ((unsigned long long int)1ULL << (53ULL - 1ULL))) &
            ((FP8_DP_HALF_ULP << (shift + 1ULL)) - 1ULL);
        // round-to-nearest-even adjustment
        if ((round > (FP8_DP_HALF_ULP << shift)) ||
            ((round == (FP8_DP_HALF_ULP << shift)) && (res & 1U))) {
            res = (unsigned char)(res + 1U);
        }
    }

    res |= sign;

    return (__nv_fp8_storage_t)res;
}

__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t
__nv_cvt_double2_to_fp8x2(const double2 x, const __nv_saturation_t saturate,
                          const __nv_fp8_interpretation_t fp8_interpretation) {
    __nv_fp8x2_storage_t storage = (__nv_fp8x2_storage_t)__nv_cvt_double_to_fp8(
        x.y, saturate, fp8_interpretation);
    storage = (__nv_fp8x2_storage_t)(storage << 8U);
    storage = (__nv_fp8x2_storage_t)(storage |
                                     __nv_cvt_double_to_fp8(
                                         x.x, saturate, fp8_interpretation));
    return storage;
}

__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t
__nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate,
                      const __nv_fp8_interpretation_t fp8_interpretation) {
    __nv_fp8_storage_t res = 0U;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    if (saturate == __NV_SATFINITE) {
        __nv_fp8x2_storage_t storage;
        if (fp8_interpretation == __NV_E5M2) {
            asm("{cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;}\n"
                : "=h"(storage)
                : "f"(x), "f"(0.0f));
        } else {
            asm("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}\n"
                : "=h"(storage)
                : "f"(x), "f"(0.0f));
        }
        res = (__nv_fp8_storage_t)storage;
    } else
#endif
    {
        unsigned int xbits;
#if defined(__CUDACC__) || (!defined __cplusplus)
        (void)memcpy(&xbits, &x, sizeof(x));
#else
        (void)std::memcpy(&xbits, &x, sizeof(x));
#endif

        // isnan
        if ((xbits & 0x7FFFFFFFU) > 0x7F800000U) {
            // Canonical NaN
            xbits = 0x7FFFFFFFU;
        }

        float fx;
#if defined(__CUDACC__) || (!defined __cplusplus)
        (void)memcpy(&fx, &xbits, sizeof(xbits));
#else
        (void)std::memcpy(&fx, &xbits, sizeof(xbits));
#endif

        const double dx = (double)fx;
        res = __nv_cvt_double_to_fp8(dx, saturate, fp8_interpretation);
    }
    return res;
}

__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t
__nv_cvt_float2_to_fp8x2(const float2 x, const __nv_saturation_t saturate,
                         const __nv_fp8_interpretation_t fp8_interpretation) {
    __nv_fp8x2_storage_t storage;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    if (saturate == __NV_SATFINITE) {
        if (fp8_interpretation == __NV_E5M2) {
            asm("{cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;}\n"
                : "=h"(storage)
                : "f"(x.x), "f"(x.y));
        } else {
            asm("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}\n"
                : "=h"(storage)
                : "f"(x.x), "f"(x.y));
        }
    } else
#endif
    {
        storage = (__nv_fp8x2_storage_t)__nv_cvt_float_to_fp8(
            x.y, saturate, fp8_interpretation);
        storage = (__nv_fp8x2_storage_t)(storage << 8U);
        storage = (__nv_fp8x2_storage_t)(storage | __nv_cvt_float_to_fp8(
                                                       x.x, saturate,
                                                       fp8_interpretation));
    }
    return storage;
}

__CUDA_HOSTDEVICE_FP8_DECL__ float
__internal_halfraw_to_float(const __half_raw x) {
    float f;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    asm("{cvt.f32.f16 %0, %1;}\n" : "=f"(f) : "h"(x.x));
#else
    const unsigned int ux = (unsigned int)x.x;
    unsigned int sign = (ux >> 15U) & 1U;
    unsigned int exponent = (ux >> 10U) & 0x1fU;
    unsigned int mantissa = (ux & 0x3ffU) << 13U;
    if (exponent == 0x1fU) { /* NaN or Inf */
        /* discard sign of a NaN */
        sign = ((mantissa != 0U) ? (sign >> 1U) : sign);
        mantissa = ((mantissa != 0U) ? 0x7fffffU : 0U);
        exponent = 0xffU;
    } else if (exponent == 0U) { /* Denorm or Zero */
        if (mantissa != 0U) {
            unsigned int msb;
            exponent = 0x71U;
            do {
                msb = (mantissa & 0x400000U);
                mantissa <<= 1U; /* normalize */
                --exponent;
            } while (msb == 0U);
            mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70U;
    }
    const unsigned int u = ((sign << 31U) | (exponent << 23U) | mantissa);
#if defined(__CUDACC__) || (!defined __cplusplus)
    (void)memcpy(&f, &u, sizeof(u));
#else
    (void)std::memcpy(&f, &u, sizeof(u));
#endif
#endif /* (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 530) */
    return f;
}

__CUDA_HOSTDEVICE_FP8_DECL__ float2
__internal_halfraw2_to_float2(const __half2_raw x) {
    __half_raw raw;
    float2 res;
    raw.x = x.x;
    res.x = __internal_halfraw_to_float(raw);
    raw.x = x.y;
    res.y = __internal_halfraw_to_float(raw);
    return res;
}

__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t
__nv_cvt_halfraw_to_fp8(const __half_raw x, const __nv_saturation_t saturate,
                        const __nv_fp8_interpretation_t fp8_interpretation) {
    __nv_fp8_storage_t res = 0U;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    if (saturate == __NV_SATFINITE) {
        unsigned int half2_storage = (unsigned int)(x.x);
        __nv_fp8x2_storage_t tmp;
        if (fp8_interpretation == __NV_E5M2) {
            asm("{cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;}\n"
                : "=h"(tmp)
                : "r"(half2_storage));
        } else {
            asm("{cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;}\n"
                : "=h"(tmp)
                : "r"(half2_storage));
        }
        res = (__nv_fp8_storage_t)tmp;
    } else
#endif
    {
        float fx = __internal_halfraw_to_float(x);
        res = __nv_cvt_float_to_fp8(fx, saturate, fp8_interpretation);
    }
    return res;
}

__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(
    const __half2_raw x, const __nv_saturation_t saturate,
    const __nv_fp8_interpretation_t fp8_interpretation) {
    __nv_fp8x2_storage_t tmp;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    if (saturate == __NV_SATFINITE) {
        unsigned int half2_storage;
        (void)memcpy(&half2_storage, &x, sizeof(x));

        if (fp8_interpretation == __NV_E5M2) {
            asm("{cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;}\n"
                : "=h"(tmp)
                : "r"(half2_storage));
        } else {
            asm("{cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;}\n"
                : "=h"(tmp)
                : "r"(half2_storage));
        }
    } else
#endif
    {
        __half_raw raw;
        raw.x = x.x;
        __nv_fp8_storage_t lo =
            __nv_cvt_halfraw_to_fp8(raw, saturate, fp8_interpretation);
        raw.x = x.y;
        __nv_fp8_storage_t hi =
            __nv_cvt_halfraw_to_fp8(raw, saturate, fp8_interpretation);
        tmp = hi;
        tmp = (__nv_fp8x2_storage_t)(tmp << 8U);
        tmp = (__nv_fp8x2_storage_t)(tmp | lo);
    }
    return tmp;
}

__CUDA_HOSTDEVICE_FP8_DECL__ float
__internal_bf16raw_to_float(const __nv_bfloat16_raw x) {
    const unsigned int ux = ((unsigned int)x.x) << 16U;
    float fx;
#if defined(__CUDACC__) || (!defined __cplusplus)
    (void)memcpy(&fx, &ux, sizeof(ux));
#else
    (void)std::memcpy(&fx, &ux, sizeof(ux));
#endif
    return fx;
}

__CUDA_HOSTDEVICE_FP8_DECL__ __nv_bfloat16_raw
__internal_float_to_bf16raw_rz(const float x) {
    unsigned int ux;
    __nv_bfloat16_raw r;
#if defined(__CUDACC__) || (!defined __cplusplus)
    (void)memcpy(&ux, &x, sizeof(x));
#else
    (void)std::memcpy(&ux, &x, sizeof(x));
#endif
    r.x = (unsigned short int)(ux >> 16U);
    return r;
}

__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_fp8(
    const __nv_bfloat16_raw x, const __nv_saturation_t saturate,
    const __nv_fp8_interpretation_t fp8_interpretation) {
    const float fx = __internal_bf16raw_to_float(x);
    const __nv_fp8_storage_t res =
        __nv_cvt_float_to_fp8(fx, saturate, fp8_interpretation);
    return res;
}

__CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t
__nv_cvt_bfloat16raw2_to_fp8x2(
    const __nv_bfloat162_raw x, const __nv_saturation_t saturate,
    const __nv_fp8_interpretation_t fp8_interpretation) {
    __nv_bfloat16_raw raw;
    raw.x = x.y;
    __nv_fp8x2_storage_t storage =
        (__nv_fp8x2_storage_t)__nv_cvt_bfloat16raw_to_fp8(raw, saturate,
                                                          fp8_interpretation);
    storage = (__nv_fp8x2_storage_t)(storage << 8U);
    raw.x = x.x;
    storage = (__nv_fp8x2_storage_t)(storage |
                                     __nv_cvt_bfloat16raw_to_fp8(
                                         raw, saturate, fp8_interpretation));
    return storage;
}

__CUDA_HOSTDEVICE_FP8_DECL__ __half2_raw
__nv_cvt_fp8x2_to_halfraw2(const __nv_fp8x2_storage_t x,
                           const __nv_fp8_interpretation_t fp8_interpretation);
__CUDA_HOSTDEVICE_FP8_DECL__ __half_raw
__nv_cvt_fp8_to_halfraw(const __nv_fp8_storage_t x,
                        const __nv_fp8_interpretation_t fp8_interpretation) {
    __half_raw res;
    res.x = 0U;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    res.x =
        __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)x, fp8_interpretation)
            .x;
#else
    unsigned short int ur = (unsigned short int)x;
    ur = (unsigned short int)(ur << 8U);

    if (fp8_interpretation == __NV_E5M2) {
        if ((ur & 0x7FFFU) > 0x7C00U) {
            /* If NaN, return canonical NaN */
            ur = 0x7FFFU;
        }
    } else { // __NV_E4M3
        unsigned short int sign = ur & 0x8000U;
        unsigned short int exponent =
            (unsigned short int)(((ur & 0x7800U) >> 1U) + 0x2000U);
        unsigned short int mantissa = (ur & 0x0700U) >> 1U;
        unsigned char absx = 0x7FU & (unsigned char)x;

        if (absx == 0x7FU) // NaN
        {
            ur = 0x7FFFU; // fp16 canonical NaN, discard sign
        } else if (exponent == 0x2000U) {
            // zero or denormal
            if (mantissa != 0U) {
                // normalize
                mantissa = (unsigned short int)(mantissa << 1U);
                while ((mantissa & 0x0400U) == 0U) {
                    mantissa = (unsigned short int)(mantissa << 1U);
                    exponent = (unsigned short int)(exponent - 0x0400U);
                }
                // discard implicit leading bit
                mantissa &= 0x03FFU;
            } else { // Zero
                exponent = 0U;
            }

            ur = (sign | exponent) | mantissa;
        } else {
            ur = (sign | exponent) | mantissa;
        }
    }
    res.x = ur;
#endif
    return res;
}

__CUDA_HOSTDEVICE_FP8_DECL__ __half2_raw
__nv_cvt_fp8x2_to_halfraw2(const __nv_fp8x2_storage_t x,
                           const __nv_fp8_interpretation_t fp8_interpretation) {
    __half2_raw res;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    unsigned int half2_storage;
    if (fp8_interpretation == __NV_E5M2) {
        asm("{cvt.rn.f16x2.e5m2x2 %0, %1;}\n" : "=r"(half2_storage) : "h"(x));
    } else {
        asm("{cvt.rn.f16x2.e4m3x2 %0, %1;}\n" : "=r"(half2_storage) : "h"(x));
    }
    (void)memcpy(&res, &half2_storage, sizeof(half2_storage));
#else
    res.x =
        __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)x, fp8_interpretation).x;
    res.y = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(x >> 8U),
                                    fp8_interpretation)
                .x;
#endif
    return res;
}

/* All other definitions in this file are only visible to C++ compilers */
#if defined(__cplusplus)

/**
 * \defgroup CUDA_MATH_FP8_E5M2_STRUCT C++ struct for handling fp8 data type of e5m2 kind.
 * \ingroup CUDA_MATH_INTRINSIC_FP8
 */

/**
 * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
 * \brief __nv_fp8_e5m2 datatype
 *
 * \details This structure implements the datatype for handling
 * \p fp8 floating-point numbers of \p e5m2 kind:
 * with 1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits.
 *
 * The structure implements converting constructors and operators.
 */
struct __CUDA_ALIGN__(1) __nv_fp8_e5m2 {
  public:
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Storage variable contains the \p fp8 floating-point data.
     */
    __nv_fp8_storage_t __x;

    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP8)
    __nv_fp8_e5m2() = default;
#else
    __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e5m2() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP8) */

#if !defined(__CUDA_NO_FP8_CONVERSIONS__)

    /* Construct from wider FP types */
    /* Note we do avoid constructor init-list because of special host/device
     * compilation rules */

    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor from \p __half data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e5m2(const __half f) {
        __x = __nv_cvt_halfraw_to_fp8(static_cast<__half_raw>(f),
                                      __NV_SATFINITE, __NV_E5M2);
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor from \p __nv_bfloat16 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e5m2(const __nv_bfloat16 f) {
        __x = __nv_cvt_bfloat16raw_to_fp8(static_cast<__nv_bfloat16_raw>(f),
                                          __NV_SATFINITE, __NV_E5M2);
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor from \p float data type, relies on \p __NV_SATFINITE behavior
     * for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e5m2(const float f) {
        __x = __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E5M2);
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor from \p double data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e5m2(const double f) {
        __x = __nv_cvt_double_to_fp8(f, __NV_SATFINITE, __NV_E5M2);
    }

    /* Converts from integral */

    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor from \p unsigned \p short \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__
    __nv_fp8_e5m2(const unsigned short int val) {
        __x = static_cast<__nv_fp8_e5m2>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor from \p unsigned \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e5m2(const unsigned int val) {
        __x = static_cast<__nv_fp8_e5m2>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor from \p unsigned \p long \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e5m2(const unsigned long int val) {
        __x = static_cast<__nv_fp8_e5m2>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor from \p unsigned \p long \p long \p int data type, relies on
     * \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__
    __nv_fp8_e5m2(const unsigned long long int val) {
        __x = static_cast<__nv_fp8_e5m2>(static_cast<float>(val)).__x;
    }

    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor from \p short \p int data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e5m2(const short int val) {
        __x = static_cast<__nv_fp8_e5m2>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor from \p int data type, relies on \p __NV_SATFINITE behavior
     * for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e5m2(const int val) {
        __x = static_cast<__nv_fp8_e5m2>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor from \p long \p int data type, relies on \p __NV_SATFINITE behavior
     * for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e5m2(const long int val) {
        __x = static_cast<__nv_fp8_e5m2>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Constructor from \p long \p long \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e5m2(const long long int val) {
        __x = static_cast<__nv_fp8_e5m2>(static_cast<float>(val)).__x;
    }

#if !defined(__CUDA_NO_FP8_CONVERSION_OPERATORS__)
    /* Widening FP converts */
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p __half data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator __half() const {
        return static_cast<__half>(__nv_cvt_fp8_to_halfraw(__x, __NV_E5M2));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p float data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator float() const {
        return __internal_halfraw_to_float(
            __nv_cvt_fp8_to_halfraw(__x, __NV_E5M2));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p __nv_bfloat16 data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator __nv_bfloat16() const {
        return static_cast<__nv_bfloat16>(
            __internal_float_to_bf16raw_rz(float(*this)));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p double data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator double() const {
        return static_cast<double>(float(*this));
    }

    /* Convert to integral */

    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p unsigned \p char data type.
     * Clamps negative and too large inputs to the output range.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator unsigned char() const {
        unsigned char i;
        const float f = float(*this);
        const unsigned char max_val = 0xFFU;
        const unsigned char min_val = 0U;
        const unsigned char bits = (*this).__x;
        // saturation fixup
        if ((bits & 0x7FU) > 0x7CU) {
            // NaN
            i = 0;
        } else if (f > static_cast<float>(max_val)) {
            // saturate maximum
            i = max_val;
        } else if (f < static_cast<float>(min_val)) {
            // saturate minimum
            i = min_val;
        } else {
            // normal value
            i = static_cast<unsigned char>(f);
        }
        return i;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p unsigned \p short \p int data type.
     * Clamps negative and too large inputs to the output range.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator unsigned short int() const {
        return __half2ushort_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p unsigned \p int data type.
     * Clamps negative and too large inputs to the output range.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator unsigned int() const {
        return __half2uint_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p unsigned \p long \p int data type.
     * Clamps negative and too large inputs to the output range.
     * \p NaN inputs convert to \p zero if output type is 32-bit.
     * \p NaN inputs convert to \p 0x8000000000000000ULL if output type is 64-bit.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator unsigned long int() const {
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
            retval = static_cast<unsigned long>(__half2ull_rz(__half(*this)));
        }
        else
        {
            retval = static_cast<unsigned long>(__half2uint_rz(__half(*this)));
        }
        return retval;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p unsigned \p long \p long \p int data type.
     * Clamps negative and too large inputs to the output range.
     * \p NaN inputs convert to \p 0x8000000000000000ULL.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator unsigned long long int() const {
        return __half2ull_rz(__half(*this));
    }

    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p signed \p char data type.
     * Clamps too large inputs to the output range.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator signed char() const {
        signed char i;
        const float f = float(*this);
        const signed char max_val = (signed char)0x7FU;
        const signed char min_val = (signed char)0x80U;
        const unsigned char bits = (*this).__x;
        // saturation fixup
        if ((bits & 0x7FU) > 0x7CU) {
            // NaN
            i = 0;
        } else if (f > static_cast<float>(max_val)) {
            // saturate maximum
            i = max_val;
        } else if (f < static_cast<float>(min_val)) {
            // saturate minimum
            i = min_val;
        } else {
            // normal value
            i = static_cast<signed char>(f);
        }
        return i;
    }

    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to an implementation defined \p char data type.
     * 
     * Detects signedness of the \p char type and proceeds accordingly, see
     * further details in signed and unsigned char operators.

     * Clamps inputs to the output range.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator char() const {
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
            value = static_cast<char>(static_cast<signed char>(*this));
        }
        else
        {
            value = static_cast<char>(static_cast<unsigned char>(*this));
        }
        return value;
    }

    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p short \p int data type.
     * Clamps too large inputs to the output range.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator short int() const {
        return __half2short_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p int data type.
     * Clamps too large inputs to the output range.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator int() const {
        return __half2int_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p long \p int data type.
     * Clamps too large inputs to the output range.
     * \p NaN inputs convert to \p zero if output type is 32-bit.
     * \p NaN inputs convert to \p 0x8000000000000000ULL if output type is 64-bit.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator long int() const {
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
            retval = static_cast<long>(__half2ll_rz(__half(*this)));
        }
        else
        {
            retval = static_cast<long>(__half2int_rz(__half(*this)));
        }
        return retval;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p long \p long \p int data type.
     * Clamps too large inputs to the output range.
     * \p NaN inputs convert to \p 0x8000000000000000LL.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator long long int() const {
        return __half2ll_rz(__half(*this));
    }

    /**
     * \ingroup CUDA_MATH_FP8_E5M2_STRUCT
     * Conversion operator to \p bool data type.
     * +0 and -0 inputs convert to \p false.
     * Non-zero inputs convert to \p true.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator bool() const {
        return (__x & 0x7FU) != 0U;
    }
#endif /* !defined(__CUDA_NO_FP8_CONVERSION_OPERATORS__) */
#endif /* !defined(__CUDA_NO_FP8_CONVERSIONS__) */
};

/**
 * \defgroup CUDA_MATH_FP8X2_E5M2_STRUCT C++ struct for handling vector type of two fp8 values of e5m2 kind.
 * \ingroup CUDA_MATH_INTRINSIC_FP8
 */

/**
 * \ingroup CUDA_MATH_FP8X2_E5M2_STRUCT
 * \brief __nv_fp8x2_e5m2 datatype
 *
 * \details This structure implements the datatype for handling two
 * \p fp8 floating-point numbers of \p e5m2 kind each:
 * with 1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits.
 *
 * The structure implements converting constructors and operators.
 */
struct __CUDA_ALIGN__(2) __nv_fp8x2_e5m2 {
  public:
    /**
     * \ingroup CUDA_MATH_FP8X2_E5M2_STRUCT
     * Storage variable contains the vector of two \p fp8 floating-point data
     * values.
     */
    __nv_fp8x2_storage_t __x;

    /**
     * \ingroup CUDA_MATH_FP8X2_E5M2_STRUCT
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP8)
    __nv_fp8x2_e5m2() = default;
#else
    __CUDA_HOSTDEVICE_FP8__ __nv_fp8x2_e5m2() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP8) */

#if !defined(__CUDA_NO_FP8_CONVERSIONS__)

    /* Construct from wider types */

    /**
     * \ingroup CUDA_MATH_FP8X2_E5M2_STRUCT
     * Constructor from \p __half2 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x2_e5m2(const __half2 f) {
        __x = __nv_cvt_halfraw2_to_fp8x2(static_cast<__half2_raw>(f),
                                         __NV_SATFINITE, __NV_E5M2);
    }
    /**
     * \ingroup CUDA_MATH_FP8X2_E5M2_STRUCT
     * Constructor from \p __nv_bfloat162 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x2_e5m2(const __nv_bfloat162 f) {
        __x = __nv_cvt_bfloat16raw2_to_fp8x2(static_cast<__nv_bfloat162_raw>(f),
                                             __NV_SATFINITE, __NV_E5M2);
    }
    /**
     * \ingroup CUDA_MATH_FP8X2_E5M2_STRUCT
     * Constructor from \p float2 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x2_e5m2(const float2 f) {
        __x = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E5M2);
    }
    /**
     * \ingroup CUDA_MATH_FP8X2_E5M2_STRUCT
     * Constructor from \p double2 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x2_e5m2(const double2 f) {
        __x = __nv_cvt_double2_to_fp8x2(f, __NV_SATFINITE, __NV_E5M2);
    }

#if !defined(__CUDA_NO_FP8_CONVERSION_OPERATORS__)
    /* Widening converts */
    /**
     * \ingroup CUDA_MATH_FP8X2_E5M2_STRUCT
     * Conversion operator to \p __half2 data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator __half2() const {
        return static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(__x, __NV_E5M2));
    }
    /**
     * \ingroup CUDA_MATH_FP8X2_E5M2_STRUCT
     * Conversion operator to \p float2 data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator float2() const {
        return __internal_halfraw2_to_float2(
            __nv_cvt_fp8x2_to_halfraw2(__x, __NV_E5M2));
    }
#endif /* !defined(__CUDA_NO_FP8_CONVERSION_OPERATORS__) */
#endif /* !defined(__CUDA_NO_FP8_CONVERSIONS__) */
};

__CUDA_HOSTDEVICE_FP8_DECL__ unsigned int
__internal_pack_u16x2_to_u32(const unsigned short int src_lo,
                             const unsigned short int src_hi) {
    unsigned int dst;
#if (defined __CUDACC__) && (defined __CUDA_ARCH__)
    asm("{  mov.b32 %0, {%1,%2};}\n" : "=r"(dst) : "h"(src_lo), "h"(src_hi));
#else
    dst = (static_cast<unsigned int>(src_hi) << 16U) |
          static_cast<unsigned int>(src_lo);
#endif
    return dst;
}

/**
 * \defgroup CUDA_MATH_FP8X4_E5M2_STRUCT C++ struct for handling vector type of four fp8 values of e5m2 kind.
 * \ingroup CUDA_MATH_INTRINSIC_FP8
 */

/**
 * \ingroup CUDA_MATH_FP8X4_E5M2_STRUCT
 * \brief __nv_fp8x4_e5m2 datatype
 *
 * \details This structure implements the datatype for handling four
 * \p fp8 floating-point numbers of \p e5m2 kind each:
 * with 1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits.
 *
 * The structure implements converting constructors and operators.
 */
struct __CUDA_ALIGN__(4) __nv_fp8x4_e5m2 {
  public:
    /**
     * \ingroup CUDA_MATH_FP8X4_E5M2_STRUCT
     * Storage variable contains the vector of four \p fp8 floating-point data
     * values.
     */
    __nv_fp8x4_storage_t __x;

    /**
     * \ingroup CUDA_MATH_FP8X4_E5M2_STRUCT
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP8)
    __nv_fp8x4_e5m2() = default;
#else
    __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e5m2() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP8) */

#if !defined(__CUDA_NO_FP8_CONVERSIONS__)

    /* Construct from wider types */

    /**
     * \ingroup CUDA_MATH_FP8X4_E5M2_STRUCT
     * Constructor from a pair of \p __half2 data type values,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e5m2(const __half2 flo,
                                                     const __half2 fhi) {
        const __nv_fp8x2_storage_t rlo = __nv_cvt_halfraw2_to_fp8x2(
            static_cast<__half2_raw>(flo), __NV_SATFINITE, __NV_E5M2);
        const __nv_fp8x2_storage_t rhi = __nv_cvt_halfraw2_to_fp8x2(
            static_cast<__half2_raw>(fhi), __NV_SATFINITE, __NV_E5M2);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    /**
     * \ingroup CUDA_MATH_FP8X4_E5M2_STRUCT
     * Constructor from a pair of \p __nv_bfloat162 data type values,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e5m2(const __nv_bfloat162 flo,
                                                     const __nv_bfloat162 fhi) {
        const __nv_fp8x2_storage_t rlo = __nv_cvt_bfloat16raw2_to_fp8x2(
            static_cast<__nv_bfloat162_raw>(flo), __NV_SATFINITE, __NV_E5M2);
        const __nv_fp8x2_storage_t rhi = __nv_cvt_bfloat16raw2_to_fp8x2(
            static_cast<__nv_bfloat162_raw>(fhi), __NV_SATFINITE, __NV_E5M2);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    /**
     * \ingroup CUDA_MATH_FP8X4_E5M2_STRUCT
     * Constructor from \p float4 vector data type,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e5m2(const float4 f) {
        const float2 flo = {f.x, f.y};
        const float2 fhi = {f.z, f.w};
        const __nv_fp8x2_storage_t rlo =
            __nv_cvt_float2_to_fp8x2(flo, __NV_SATFINITE, __NV_E5M2);
        const __nv_fp8x2_storage_t rhi =
            __nv_cvt_float2_to_fp8x2(fhi, __NV_SATFINITE, __NV_E5M2);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    /**
     * \ingroup CUDA_MATH_FP8X4_E5M2_STRUCT
     * Constructor from \p double4 vector data type,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e5m2(const double4 f) {
        const double2 flo = {f.x, f.y};
        const double2 fhi = {f.z, f.w};
        const __nv_fp8x2_storage_t rlo =
            __nv_cvt_double2_to_fp8x2(flo, __NV_SATFINITE, __NV_E5M2);
        const __nv_fp8x2_storage_t rhi =
            __nv_cvt_double2_to_fp8x2(fhi, __NV_SATFINITE, __NV_E5M2);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }

#if !defined(__CUDA_NO_FP8_CONVERSION_OPERATORS__)
    /* Widening converts */

    /**
     * \ingroup CUDA_MATH_FP8X4_E5M2_STRUCT
     * Conversion operator to \p float4 vector data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator float4() const {
        const __nv_fp8x2_storage_t slo = static_cast<__nv_fp8x2_storage_t>(__x);
        const __nv_fp8x2_storage_t shi =
            static_cast<__nv_fp8x2_storage_t>(__x >> 16U);
        float2 rlo = __internal_halfraw2_to_float2(
            __nv_cvt_fp8x2_to_halfraw2(slo, __NV_E5M2));
        float2 rhi = __internal_halfraw2_to_float2(
            __nv_cvt_fp8x2_to_halfraw2(shi, __NV_E5M2));
        float4 res = {rlo.x, rlo.y, rhi.x, rhi.y};
        return res;
    }
#endif /* !defined(__CUDA_NO_FP8_CONVERSION_OPERATORS__) */
#endif /* !defined(__CUDA_NO_FP8_CONVERSIONS__) */
};

/**
 * \defgroup CUDA_MATH_FP8_E4M3_STRUCT C++ struct for handling fp8 data type of e4m3 kind.
 * \ingroup CUDA_MATH_INTRINSIC_FP8
 */

/**
 * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
 * \brief __nv_fp8_e4m3 datatype
 *
 * \details This structure implements the datatype for storing
 * \p fp8 floating-point numbers of \p e4m3 kind:
 * with 1 sign, 4 exponent, 1 implicit and 3 explicit mantissa bits.
 * The encoding doesn't support Infinity.
 * NaNs are limited to 0x7F and 0xFF values.
 *
 * The structure implements converting constructors and operators.
 */
struct __CUDA_ALIGN__(1) __nv_fp8_e4m3 {
  public:
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Storage variable contains the \p fp8 floating-point data.
     */
    __nv_fp8_storage_t __x;

    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP8)
    __nv_fp8_e4m3() = default;
#else
    __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e4m3() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP8) */

#if !defined(__CUDA_NO_FP8_CONVERSIONS__)

    /* Construct from wider FP types */
    /* Note we do avoid constructor init-list because of special host/device
     * compilation rules */

    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor from \p __half data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e4m3(const __half f) {
        __x = __nv_cvt_halfraw_to_fp8(static_cast<__half_raw>(f),
                                      __NV_SATFINITE, __NV_E4M3);
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor from \p __nv_bfloat16 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e4m3(const __nv_bfloat16 f) {
        __x = __nv_cvt_bfloat16raw_to_fp8(static_cast<__nv_bfloat16_raw>(f),
                                          __NV_SATFINITE, __NV_E4M3);
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor from \p float data type, relies on \p __NV_SATFINITE behavior
     * for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e4m3(const float f) {
        __x = __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor from \p double data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e4m3(const double f) {
        __x = __nv_cvt_double_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
    }

    /* Converts from integral */

    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor from \p unsigned \p short \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__
    __nv_fp8_e4m3(const unsigned short int val) {
        __x = static_cast<__nv_fp8_e4m3>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor from \p unsigned \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e4m3(const unsigned int val) {
        __x = static_cast<__nv_fp8_e4m3>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor from \p unsigned \p long \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e4m3(const unsigned long int val) {
        __x = static_cast<__nv_fp8_e4m3>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor from \p unsigned \p long \p long \p int data type, relies on
     * \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__
    __nv_fp8_e4m3(const unsigned long long int val) {
        __x = static_cast<__nv_fp8_e4m3>(static_cast<float>(val)).__x;
    }

    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor from \p short \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e4m3(const short int val) {
        __x = static_cast<__nv_fp8_e4m3>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor from \p int data type, relies on \p __NV_SATFINITE behavior
     * for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e4m3(const int val) {
        __x = static_cast<__nv_fp8_e4m3>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor from \p long \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e4m3(const long int val) {
        __x = static_cast<__nv_fp8_e4m3>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Constructor from \p long \p long \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8_e4m3(const long long int val) {
        __x = static_cast<__nv_fp8_e4m3>(static_cast<float>(val)).__x;
    }

#if !defined(__CUDA_NO_FP8_CONVERSION_OPERATORS__)
    /* Widening FP converts */
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p __half data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator __half() const {
        return static_cast<__half>(__nv_cvt_fp8_to_halfraw(__x, __NV_E4M3));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p float data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator float() const {
        return __internal_halfraw_to_float(
            __nv_cvt_fp8_to_halfraw(__x, __NV_E4M3));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p __nv_bfloat16 data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator __nv_bfloat16() const {
        return static_cast<__nv_bfloat16>(
            __internal_float_to_bf16raw_rz(float(*this)));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p double data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator double() const {
        return static_cast<double>(float(*this));
    }

    /* Convert to integral */

    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p unsigned \p char data type.
     * Clamps negative and too large inputs to the output range.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator unsigned char() const {
        unsigned char i;
        const float f = float(*this);
        const unsigned char max_val = 0xFFU;
        const unsigned char min_val = 0U;
        const unsigned char bits = (*this).__x;
        // saturation fixup
        if ((bits & 0x7FU) == 0x7FU) {
            // NaN
            i = 0;
        } else if (f > static_cast<float>(max_val)) {
            // saturate maximum
            i = max_val;
        } else if (f < static_cast<float>(min_val)) {
            // saturate minimum
            i = min_val;
        } else {
            // normal value
            i = static_cast<unsigned char>(f);
        }
        return i;
    }

    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p unsigned \p short \p int data type.
     * Clamps negative inputs to zero.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator unsigned short int() const {
        return __half2ushort_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p unsigned \p int data type.
     * Clamps negative inputs to zero.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator unsigned int() const {
        return __half2uint_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p unsigned \p long \p int data type.
     * Clamps negative and too large inputs to the output range.
     * \p NaN inputs convert to \p zero if output type is 32-bit.
     * \p NaN inputs convert to \p 0x8000000000000000ULL if output type is 64-bit.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator unsigned long int() const {
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
            retval = static_cast<unsigned long>(__half2ull_rz(__half(*this)));
        }
        else
        {
            retval = static_cast<unsigned long>(__half2uint_rz(__half(*this)));
        }
        return retval;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p unsigned \p long \p long \p int data type.
     * Clamps negative inputs to zero.
     * \p NaN inputs convert to \p 0x8000000000000000ULL.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator unsigned long long int() const {
        return __half2ull_rz(__half(*this));
    }

    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p signed \p char data type.
     * Clamps too large inputs to the output range.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator signed char() const {
        signed char i;
        const float f = float(*this);
        const signed char max_val = (signed char)0x7FU;
        const signed char min_val = (signed char)0x80U;
        const unsigned char bits = (*this).__x;
        // saturation fixup
        if ((bits & 0x7FU) == 0x7FU) {
            // NaN
            i = 0;
        } else if (f > static_cast<float>(max_val)) {
            // saturate maximum
            i = max_val;
        } else if (f < static_cast<float>(min_val)) {
            // saturate minimum
            i = min_val;
        } else {
            // normal value
            i = static_cast<signed char>(f);
        }
        return i;
    }

    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to an implementation defined \p char data type.
     * 
     * Detects signedness of the \p char type and proceeds accordingly, see
     * further details in signed and unsigned char operators.

     * Clamps inputs to the output range.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator char() const {
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
            value = static_cast<char>(static_cast<signed char>(*this));
        }
        else
        {
            value = static_cast<char>(static_cast<unsigned char>(*this));
        }
        return value;
    }

    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p short \p int data type.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator short int() const {
        return __half2short_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p int data type.
     * \p NaN inputs convert to \p zero.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator int() const {
        return __half2int_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p long \p int data type.
     * Clamps too large inputs to the output range.
     * \p NaN inputs convert to \p zero if output type is 32-bit.
     * \p NaN inputs convert to \p 0x8000000000000000ULL if output type is 64-bit.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator long int() const {
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
            retval = static_cast<long>(__half2ll_rz(__half(*this)));
        }
        else
        {
            retval = static_cast<long>(__half2int_rz(__half(*this)));
        }
        return retval;
    }
    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p long \p long \p int data type.
     * \p NaN inputs convert to \p 0x8000000000000000LL.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator long long int() const {
        return __half2ll_rz(__half(*this));
    }

    /**
     * \ingroup CUDA_MATH_FP8_E4M3_STRUCT
     * Conversion operator to \p bool data type.
     * +0 and -0 inputs convert to \p false.
     * Non-zero inputs convert to \p true.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator bool() const {
        return (__x & 0x7FU) != 0U;
    }
#endif /* !defined(__CUDA_NO_FP8_CONVERSION_OPERATORS__) */
#endif /* !defined(__CUDA_NO_FP8_CONVERSIONS__) */
};

/**
 * \defgroup CUDA_MATH_FP8X2_E4M3_STRUCT C++ struct for handling vector type of two fp8 values of e4m3 kind.
 * \ingroup CUDA_MATH_INTRINSIC_FP8
 */

/**
 * \ingroup CUDA_MATH_FP8X2_E4M3_STRUCT
 * \brief __nv_fp8x2_e4m3 datatype
 *
 * \details This structure implements the datatype for storage
 * and operations on the vector of two \p fp8 values of \p e4m3 kind each:
 * with 1 sign, 4 exponent, 1 implicit and 3 explicit mantissa bits.
 * The encoding doesn't support Infinity.
 * NaNs are limited to 0x7F and 0xFF values.
 */
struct __CUDA_ALIGN__(2) __nv_fp8x2_e4m3 {
  public:
    /**
     * \ingroup CUDA_MATH_FP8X2_E4M3_STRUCT
     * Storage variable contains the vector of two \p fp8 floating-point data
     * values.
     */
    __nv_fp8x2_storage_t __x;

    /**
     * \ingroup CUDA_MATH_FP8X2_E4M3_STRUCT
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP8)
    __nv_fp8x2_e4m3() = default;
#else
    __CUDA_HOSTDEVICE_FP8__ __nv_fp8x2_e4m3() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP8) */

#if !defined(__CUDA_NO_FP8_CONVERSIONS__)

    /* Construct from wider types */

    /**
     * \ingroup CUDA_MATH_FP8X2_E4M3_STRUCT
     * Constructor from \p __half2 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x2_e4m3(const __half2 f) {
        __x = __nv_cvt_halfraw2_to_fp8x2(static_cast<__half2_raw>(f),
                                         __NV_SATFINITE, __NV_E4M3);
    }
    /**
     * \ingroup CUDA_MATH_FP8X2_E4M3_STRUCT
     * Constructor from \p __nv_bfloat162 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x2_e4m3(const __nv_bfloat162 f) {
        __x = __nv_cvt_bfloat16raw2_to_fp8x2(static_cast<__nv_bfloat162_raw>(f),
                                             __NV_SATFINITE, __NV_E4M3);
    }
    /**
     * \ingroup CUDA_MATH_FP8X2_E4M3_STRUCT
     * Constructor from \p float2 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x2_e4m3(const float2 f) {
        __x = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
    }
    /**
     * \ingroup CUDA_MATH_FP8X2_E4M3_STRUCT
     * Constructor from \p double2 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x2_e4m3(const double2 f) {
        __x = __nv_cvt_double2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
    }

#if !defined(__CUDA_NO_FP8_CONVERSION_OPERATORS__)
    /* Widening converts */
    /**
     * \ingroup CUDA_MATH_FP8X2_E4M3_STRUCT
     * Conversion operator to \p __half2 data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator __half2() const {
        return static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(__x, __NV_E4M3));
    }
    /**
     * \ingroup CUDA_MATH_FP8X2_E4M3_STRUCT
     * Conversion operator to \p float2 data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator float2() const {
        return __internal_halfraw2_to_float2(
            __nv_cvt_fp8x2_to_halfraw2(__x, __NV_E4M3));
    }
#endif /* !defined(__CUDA_NO_FP8_CONVERSION_OPERATORS__) */
#endif /* !defined(__CUDA_NO_FP8_CONVERSIONS__) */
};

/**
 * \defgroup CUDA_MATH_FP8X4_E4M3_STRUCT C++ struct for handling vector type of four fp8 values of e4m3 kind.
 * \ingroup CUDA_MATH_INTRINSIC_FP8
 */

/**
 * \ingroup CUDA_MATH_FP8X4_E4M3_STRUCT
 * \brief __nv_fp8x4_e4m3 datatype
 *
 * \details This structure implements the datatype for storage
 * and operations on the vector of four \p fp8 values of \p e4m3 kind each:
 * with 1 sign, 4 exponent, 1 implicit and 3 explicit mantissa bits.
 * The encoding doesn't support Infinity.
 * NaNs are limited to 0x7F and 0xFF values.
 */
struct __CUDA_ALIGN__(4) __nv_fp8x4_e4m3 {
  public:
    /**
     * \ingroup CUDA_MATH_FP8X4_E4M3_STRUCT
     * Storage variable contains the vector of four \p fp8 floating-point data
     * values.
     */
    __nv_fp8x4_storage_t __x;

    /**
     * \ingroup CUDA_MATH_FP8X4_E4M3_STRUCT
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP8)
    __nv_fp8x4_e4m3() = default;
#else
    __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e4m3() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP8) */

#if !defined(__CUDA_NO_FP8_CONVERSIONS__)

    /* Construct from wider types */

    /**
     * \ingroup CUDA_MATH_FP8X4_E4M3_STRUCT
     * Constructor from a pair of \p __half2 data type values,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e4m3(const __half2 flo,
                                                     const __half2 fhi) {
        const __nv_fp8x2_storage_t rlo = __nv_cvt_halfraw2_to_fp8x2(
            static_cast<__half2_raw>(flo), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t rhi = __nv_cvt_halfraw2_to_fp8x2(
            static_cast<__half2_raw>(fhi), __NV_SATFINITE, __NV_E4M3);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    /**
     * \ingroup CUDA_MATH_FP8X4_E4M3_STRUCT
     * Constructor from a pair of \p __nv_bfloat162 data type values,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e4m3(const __nv_bfloat162 flo,
                                                     const __nv_bfloat162 fhi) {
        const __nv_fp8x2_storage_t rlo = __nv_cvt_bfloat16raw2_to_fp8x2(
            static_cast<__nv_bfloat162_raw>(flo), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t rhi = __nv_cvt_bfloat16raw2_to_fp8x2(
            static_cast<__nv_bfloat162_raw>(fhi), __NV_SATFINITE, __NV_E4M3);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    /**
     * \ingroup CUDA_MATH_FP8X4_E4M3_STRUCT
     * Constructor from \p float4 vector data type,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e4m3(const float4 f) {
        const float2 flo = {f.x, f.y};
        const float2 fhi = {f.z, f.w};
        const __nv_fp8x2_storage_t rlo =
            __nv_cvt_float2_to_fp8x2(flo, __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t rhi =
            __nv_cvt_float2_to_fp8x2(fhi, __NV_SATFINITE, __NV_E4M3);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    /**
     * \ingroup CUDA_MATH_FP8X4_E4M3_STRUCT
     * Constructor from \p double4 vector data type,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ __nv_fp8x4_e4m3(const double4 f) {
        const double2 flo = {f.x, f.y};
        const double2 fhi = {f.z, f.w};
        const __nv_fp8x2_storage_t rlo =
            __nv_cvt_double2_to_fp8x2(flo, __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t rhi =
            __nv_cvt_double2_to_fp8x2(fhi, __NV_SATFINITE, __NV_E4M3);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }

#if !defined(__CUDA_NO_FP8_CONVERSION_OPERATORS__)
    /* Widening converts */

    /**
     * \ingroup CUDA_MATH_FP8X4_E4M3_STRUCT
     * Conversion operator to \p float4 vector data type.
     */
    explicit __CUDA_HOSTDEVICE_FP8__ operator float4() const {
        const __nv_fp8x2_storage_t slo = static_cast<__nv_fp8x2_storage_t>(__x);
        const __nv_fp8x2_storage_t shi =
            static_cast<__nv_fp8x2_storage_t>(__x >> 16U);
        float2 rlo = __internal_halfraw2_to_float2(
            __nv_cvt_fp8x2_to_halfraw2(slo, __NV_E4M3));
        float2 rhi = __internal_halfraw2_to_float2(
            __nv_cvt_fp8x2_to_halfraw2(shi, __NV_E4M3));
        float4 res = {rlo.x, rlo.y, rhi.x, rhi.y};
        return res;
    }
#endif /* !defined(__CUDA_NO_FP8_CONVERSION_OPERATORS__) */
#endif /* !defined(__CUDA_NO_FP8_CONVERSIONS__) */
};

#endif /* defined(__cplusplus) */

#endif /* end of include guard: __CUDA_FP8_HPP__ */
