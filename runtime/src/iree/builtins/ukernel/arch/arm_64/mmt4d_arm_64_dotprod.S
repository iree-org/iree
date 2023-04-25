#include "iree/builtins/ukernel/arch/arm_64/assembly.h"
#include "iree/builtins/ukernel/exported_bits.h"

// Parameters:
//   x0: iree_uk_int32_t* out_tile
//   x1: const iree_uk_int8_t* lhs_panel
//   x2: const iree_uk_int8_t* rhs_panel
//   w3: iree_uk_int32_t K. Note: K>=1, as the K==0 case was handled as an early-return.
//   w4: iree_uk_uint32_t flags
//   x5: (UNUSED) params - relevant params K and flags already passed above.

BEGIN_FUNCTION iree_uk_mmt4d_tile_i8i8i32_8x8x4_arm_64_dotprod

        // Do we accumulate into or clear the accumulator tile?
        tbnz w4, IREE_UK_FLAG_MMT4D_ACCUMULATE_BIT_POS, 1f

    0:
        // No-accumulate case. Clear the 8x8 accumulator tile.
        movi v16.16b, 0
        movi v17.16b, 0
        movi v18.16b, 0
        movi v19.16b, 0
        movi v20.16b, 0
        movi v21.16b, 0
        movi v22.16b, 0
        movi v23.16b, 0
        movi v24.16b, 0
        movi v25.16b, 0
        movi v26.16b, 0
        movi v27.16b, 0
        movi v28.16b, 0
        movi v29.16b, 0
        movi v30.16b, 0
        movi v31.16b, 0
        b 2f
  
    1:
        // Accumulate case. Load the 8x8 accumulator tile from row-major
        // out_tile, into temporary registers v16--v31.
        ldp q16, q17, [x0, 0]
        ldp q18, q19, [x0, 32]
        ldp q20, q21, [x0, 64]
        ldp q22, q23, [x0, 96]
        ldp q24, q25, [x0, 128]
        ldp q26, q27, [x0, 160]
        ldp q28, q29, [x0, 192]
        ldp q30, q31, [x0, 224]
        
    2:
        // Loop body. Decrement the loop counter K.
        subs w3, w3, 1
        // Load 8x4 LHS tile
        ldp q0, q1, [x1, 0]
        add x1, x1, 32
        // Load 8x4 RHS tile
        ldp q4, q5, [x2, 0]
        add x2, x2, 32
        // Multiply-accumulate, row 0.
        sdot v16.4s, v4.16b, v0.4b[0]
        sdot v17.4s, v5.16b, v0.4b[0]
        // Multiply-accumulate, row 1.
        sdot v18.4s, v4.16b, v0.4b[1]
        sdot v19.4s, v5.16b, v0.4b[1]
        // Multiply-accumulate, row 2.
        sdot v20.4s, v4.16b, v0.4b[2]
        sdot v21.4s, v5.16b, v0.4b[2]
        // Multiply-accumulate, row 3.
        sdot v22.4s, v4.16b, v0.4b[3]
        sdot v23.4s, v5.16b, v0.4b[3]
        // Multiply-accumulate, row 4.
        sdot v24.4s, v4.16b, v1.4b[0]
        sdot v25.4s, v5.16b, v1.4b[0]
        // Multiply-accumulate, row 5.
        sdot v26.4s, v4.16b, v1.4b[1]
        sdot v27.4s, v5.16b, v1.4b[1]
        // Multiply-accumulate, row 6.
        sdot v28.4s, v4.16b, v1.4b[2]
        sdot v29.4s, v5.16b, v1.4b[2]
        // Multiply-accumulate, row 7.
        sdot v30.4s, v4.16b, v1.4b[3]
        sdot v31.4s, v5.16b, v1.4b[3]
        // Loop if K != 0.
        b.ne 2b

    3:
        // Store the accumulator tile to the destination.
        stp q16, q17, [x0, 0]
        stp q18, q19, [x0, 32]
        stp q20, q21, [x0, 64]
        stp q22, q23, [x0, 96]
        stp q24, q25, [x0, 128]
        stp q26, q27, [x0, 160]
        stp q28, q29, [x0, 192]
        stp q30, q31, [x0, 224]
        ret

END_FUNCTION iree_uk_mmt4d_tile_i8i8i32_8x8x4_arm_64_dotprod

ALLOW_NON_EXECUTABLE_STACK
