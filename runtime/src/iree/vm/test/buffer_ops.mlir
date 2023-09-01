vm.module @buffer_ops {

  vm.rodata private @rodata_3xi32 dense<[1, 2, 3]> : tensor<3xi32>

  //===--------------------------------------------------------------------===//
  // Compare
  //===--------------------------------------------------------------------===//
  // NOTE: we test this first because all of the other tests rely on it and we
  // can do it with rodata.

  vm.rodata private @rodata_cmp_3xi32_a dense<[100, 200, 300]> : tensor<3xi32>
  vm.rodata private @rodata_cmp_3xi32_b dense<[100, 201, 300]> : tensor<3xi32>

  // Compares some multi-element buffers. Note that comparisons are bytewise.
  vm.export @test_compare
  vm.func @test_compare() {
    %rodata_a = vm.const.ref.rodata @rodata_cmp_3xi32_a : !vm.buffer
    %rodata_b = vm.const.ref.rodata @rodata_cmp_3xi32_b : !vm.buffer
    %rodata_a_dno = util.optimization_barrier %rodata_a : !vm.buffer
    %rodata_b_dno = util.optimization_barrier %rodata_b : !vm.buffer

    %c0 = vm.const.i64 0
    %length = vm.buffer.length %rodata_a_dno : !vm.buffer -> i64

    %cmp0 = vm.buffer.compare %rodata_a_dno, %c0, %rodata_a_dno, %c0, %length : !vm.buffer, !vm.buffer
    vm.check.nz %cmp0, "buffer a == a" : i32

    %cmp1 = vm.buffer.compare %rodata_a_dno, %c0, %rodata_b_dno, %c0, %length : !vm.buffer, !vm.buffer
    %c0_i32 = vm.const.i32 0
    vm.check.eq %cmp1, %c0_i32, "buffer a != b" : i32

    vm.return
  }

  // Tests comparing an empty range, which should always be equal.
  vm.export @test_compare_empty
  vm.func @test_compare_empty() {
    %rodata_a = vm.const.ref.rodata @rodata_cmp_3xi32_a : !vm.buffer
    %rodata_b = vm.const.ref.rodata @rodata_cmp_3xi32_b : !vm.buffer
    %rodata_a_dno = util.optimization_barrier %rodata_a : !vm.buffer
    %rodata_b_dno = util.optimization_barrier %rodata_b : !vm.buffer

    %c0 = vm.const.i64 0
    %c2 = vm.const.i64 2

    %cmp = vm.buffer.compare %rodata_a_dno, %c2, %rodata_a_dno, %c2, %c0 : !vm.buffer, !vm.buffer
    vm.check.nz %cmp, "empty buffer ranges are always equal" : i32

    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Allocation
  //===--------------------------------------------------------------------===//

  // Tests allocating a buffer.
  vm.export @test_alloc
  vm.func @test_alloc() {
    %c128 = vm.const.i64 128
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %c128, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    %buf_length = vm.buffer.length %buf_dno : !vm.buffer -> i64
    vm.check.eq %c128, %buf_length, "buffer length == 128" : i64

    vm.return
  }

  // Tests that zero-length buffers can be allocated.
  vm.export @test_alloc_empty
  vm.func @test_alloc_empty() {
    %c0 = vm.const.i64 0
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %c0, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    %buf_length = vm.buffer.length %buf_dno : !vm.buffer -> i64
    vm.check.eq %c0, %buf_length, "buffer length == 0" : i64

    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Cloning
  //===--------------------------------------------------------------------===//

  // Tests cloning a subrange of a buffer.
  vm.export @test_clone
  vm.func @test_clone() {
    // Fetch source .rodata blob.
    %rodata = vm.const.ref.rodata @rodata_3xi32 : !vm.buffer

    // Clone the last two 32-bit elements.
    %c4 = vm.const.i64 4
    %c8 = vm.const.i64 8
    %alignment = vm.const.i32 16
    %buf = vm.buffer.clone %rodata, %c4, %c8, %alignment : !vm.buffer -> !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    // Compare the cloned range to the original.
    %c0 = vm.const.i64 0
    %cmp = vm.buffer.compare %rodata, %c4, %buf_dno, %c0, %c8 : !vm.buffer, !vm.buffer
    vm.check.nz %cmp, "buffer subspans are equal" : i32

    vm.return
  }

  // Tests cloning a zero-length buffer.
  vm.export @test_clone_empty
  vm.func @test_clone_empty() {
    // Allocate source zero-length buffer.
    %c0 = vm.const.i64 0
    %alignment = vm.const.i32 16
    %buf0 = vm.buffer.alloc %c0, %alignment : !vm.buffer
    %buf0_dno = util.optimization_barrier %buf0 : !vm.buffer
    vm.check.nz %buf0_dno, "!null" : !vm.buffer
    %buf0_length = vm.buffer.length %buf0_dno : !vm.buffer -> i64
    vm.check.eq %c0, %buf0_length, "buffer length == 0" : i64

    // Clone it all (or, clone nothing?).
    %buf1 = vm.buffer.clone %buf0_dno, %c0, %c0, %alignment : !vm.buffer -> !vm.buffer
    %buf1_dno = util.optimization_barrier %buf1 : !vm.buffer
    vm.check.nz %buf1_dno, "!null" : !vm.buffer
    %buf1_length = vm.buffer.length %buf1_dno : !vm.buffer -> i64
    vm.check.eq %c0, %buf1_length, "buffer length == 0" : i64

    vm.return
  }

  // Tests an out-of-bounds cloning subrange.
  vm.export @fail_clone_out_of_range
  vm.func @fail_clone_out_of_range() {
    // Fetch source .rodata blob.
    %rodata = vm.const.ref.rodata @rodata_3xi32 : !vm.buffer
    %rodata_dno = util.optimization_barrier %rodata : !vm.buffer
    vm.check.nz %rodata_dno, "!null" : !vm.buffer

    // Try to clone off the end of the buffer.
    %c8 = vm.const.i64 8
    %alignment = vm.const.i32 16
    %buf = vm.buffer.clone %rodata, %c8, %c8, %alignment : !vm.buffer -> !vm.buffer

    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Copy
  //===--------------------------------------------------------------------===//

  // Tests copying an entire buffer from one buffer to another.
  vm.export @test_copy_full
  vm.func @test_copy_full() {
    // Fetch source .rodata blob.
    %rodata = vm.const.ref.rodata @rodata_3xi32 : !vm.buffer
    %rodata_length = vm.buffer.length %rodata : !vm.buffer -> i64
    vm.check.nz %rodata, "!null" : !vm.buffer

    // Allocate target buffer.
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %rodata_length, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    // Copy the entire contents.
    %c0 = vm.const.i64 0
    vm.buffer.copy %rodata, %c0, %buf_dno, %c0, %rodata_length : !vm.buffer -> !vm.buffer

    // Compare to source.
    %cmp = vm.buffer.compare %rodata, %c0, %buf_dno, %c0, %rodata_length : !vm.buffer, !vm.buffer
    vm.check.nz %cmp, "source and target match" : i32

    vm.return
  }

  vm.rodata private @test_copy_partial_ref dense<[2]> : tensor<1xi32>

  // Tests copying a range of bytes from one buffer to another.
  vm.export @test_copy_partial
  vm.func @test_copy_partial() {
    // Allocate target buffer.
    %c4 = vm.const.i64 4
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %c4, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    // Copy the middle 4-byte element.
    %rodata = vm.const.ref.rodata @rodata_3xi32 : !vm.buffer
    %c0 = vm.const.i64 0
    vm.buffer.copy %rodata, %c4, %buf_dno, %c0, %c4 : !vm.buffer -> !vm.buffer

    // Compare to reference.
    %ref = vm.const.ref.rodata @test_copy_partial_ref : !vm.buffer
    %cmp = vm.buffer.compare %ref, %c0, %buf_dno, %c0, %c4 : !vm.buffer, !vm.buffer
    vm.check.nz %cmp, "source and target match" : i32

    vm.return
  }

  // Tests an out-of-bounds copy source.
  vm.export @fail_copy_out_of_range_source_offset
  vm.func @fail_copy_out_of_range_source_offset() {
    %rodata = vm.const.ref.rodata @rodata_3xi32 : !vm.buffer
    %c128 = vm.const.i64 128
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %c128, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    // Try to clone off the end of the source buffer.
    %c0 = vm.const.i64 0
    vm.buffer.copy %rodata, %c0, %buf_dno, %c0, %c128 : !vm.buffer -> !vm.buffer

    vm.return
  }

  // Tests an out-of-bounds copy source.
  vm.export @fail_copy_out_of_range_source_length
  vm.func @fail_copy_out_of_range_source_length() {
    %rodata = vm.const.ref.rodata @rodata_3xi32 : !vm.buffer
    %c128 = vm.const.i64 128
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %c128, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    // Try to clone off the end of the source buffer.
    %c0 = vm.const.i64 0
    %c8 = vm.const.i64 8
    vm.buffer.copy %rodata, %c8, %buf_dno, %c0, %c8 : !vm.buffer -> !vm.buffer

    vm.return
  }

  // Tests an out-of-bounds copy target.
  vm.export @fail_copy_out_of_range_target_offset
  vm.func @fail_copy_out_of_range_target_offset() {
    %rodata = vm.const.ref.rodata @rodata_3xi32 : !vm.buffer
    %rodata_length = vm.buffer.length %rodata : !vm.buffer -> i64
    %c8 = vm.const.i64 8
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %c8, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    // Try to clone off the end of the target buffer.
    %c0 = vm.const.i64 0
    vm.buffer.copy %rodata, %c0, %buf_dno, %c0, %rodata_length : !vm.buffer -> !vm.buffer

    vm.return
  }

  // Tests an out-of-bounds copy target.
  vm.export @fail_copy_out_of_range_target_length
  vm.func @fail_copy_out_of_range_target_length() {
    %rodata = vm.const.ref.rodata @rodata_3xi32 : !vm.buffer
    %c8 = vm.const.i64 8
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %c8, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    // Try to clone off the end of the target buffer.
    %c0 = vm.const.i64 0
    vm.buffer.copy %rodata, %c0, %buf_dno, %c8, %c8 : !vm.buffer -> !vm.buffer

    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Fill
  //===--------------------------------------------------------------------===//

  vm.rodata private @test_fill_f32_ref dense<[0.0, 42.0, 42.0, 0.0]> : tensor<4xf32>

  // Tests filling a buffer with 32-bit floating-point values.
  vm.export @test_fill_f32
  vm.func @test_fill_f32() {
    // Allocate zeroed buffer.
    %element_size = vm.const.i64 4
    %num_elements = vm.const.i64 4
    %buffer_size = vm.mul.i64 %num_elements, %element_size : i64
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %buffer_size, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    // Fill the middle two elements.
    %c2 = vm.const.i64 1
    %c4 = vm.const.i64 2
    %value = vm.const.f32 42.0
    vm.buffer.fill.f32 %buf_dno, %c2, %c4, %value : f32 -> !vm.buffer

    // Compare to reference.
    %c0 = vm.const.i64 0
    %rodata_ref = vm.const.ref.rodata @test_fill_f32_ref : !vm.buffer
    %cmp = vm.buffer.compare %rodata_ref, %c0, %buf_dno, %c0, %buffer_size : !vm.buffer, !vm.buffer
    vm.check.nz %cmp, "buffer should match reference" : i32

    vm.return
  }

  vm.rodata private @test_fill_i8_ref  dense<[0, 102, 102, 0]> : tensor<4xi8>

  // Tests filling a buffer with 8-bit values.
  vm.export @test_fill_i8
  vm.func @test_fill_i8() {
    // Allocate zeroed buffer.
    %element_size = vm.const.i64 1
    %num_elements = vm.const.i64 4
    %buffer_size = vm.mul.i64 %num_elements, %element_size : i64
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %buffer_size, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    // Fill the middle two elements.
    %c2 = vm.const.i64 1
    %c4 = vm.const.i64 2
    %value = vm.const.i32 102
    vm.buffer.fill.i8 %buf_dno, %c2, %c4, %value : i32 -> !vm.buffer

    // Compare to reference.
    %c0 = vm.const.i64 0
    %rodata_ref = vm.const.ref.rodata @test_fill_i8_ref : !vm.buffer
    %cmp = vm.buffer.compare %rodata_ref, %c0, %buf_dno, %c0, %buffer_size : !vm.buffer, !vm.buffer
    vm.check.nz %cmp, "buffer should match reference" : i32

    vm.return
  }

  vm.rodata private @test_fill_i16_ref dense<[0, 51966, 51966, 0]> : tensor<4xi16>

  // Tests filling a buffer with 16-bit values.
  vm.export @test_fill_i16
  vm.func @test_fill_i16() {
    // Allocate zeroed buffer.
    %element_size = vm.const.i64 2
    %num_elements = vm.const.i64 4
    %buffer_size = vm.mul.i64 %num_elements, %element_size : i64
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %buffer_size, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    // Fill the middle two elements.
    %c2 = vm.const.i64 1
    %c4 = vm.const.i64 2
    %value = vm.const.i32 0xCAFE
    vm.buffer.fill.i16 %buf_dno, %c2, %c4, %value : i32 -> !vm.buffer

    // Compare to reference.
    %c0 = vm.const.i64 0
    %rodata_ref = vm.const.ref.rodata @test_fill_i16_ref : !vm.buffer
    %cmp = vm.buffer.compare %rodata_ref, %c0, %buf_dno, %c0, %buffer_size : !vm.buffer, !vm.buffer
    vm.check.nz %cmp, "buffer should match reference" : i32

    vm.return
  }

  vm.rodata private @test_fill_i32_ref dense<[0, 0xFFFF0000, 0xFFFF0000, 0]> : tensor<4xi32>

  // Tests filling a buffer with 32-bit values.
  vm.export @test_fill_i32
  vm.func @test_fill_i32() {
    // Allocate zeroed buffer.
    %element_size = vm.const.i64 4
    %num_elements = vm.const.i64 4
    %buffer_size = vm.mul.i64 %num_elements, %element_size : i64
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %buffer_size, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    // Fill the middle two elements.
    %c2 = vm.const.i64 1
    %c4 = vm.const.i64 2
    %value = vm.const.i32 0xFFFF0000
    vm.buffer.fill.i32 %buf_dno, %c2, %c4, %value : i32 -> !vm.buffer

    // Compare to reference.
    %c0 = vm.const.i64 0
    %rodata_ref = vm.const.ref.rodata @test_fill_i32_ref : !vm.buffer
    %cmp = vm.buffer.compare %rodata_ref, %c0, %buf_dno, %c0, %buffer_size : !vm.buffer, !vm.buffer
    vm.check.nz %cmp, "buffer should match reference" : i32

    vm.return
  }

  vm.rodata private @test_fill_i64_ref dense<[0, 0x100000000, 0x100000000, 0]> : tensor<4xi64>

  // Tests filling a buffer with 64-bit values.
  vm.export @test_fill_i64
  vm.func @test_fill_i64() {
    // Allocate zeroed buffer.
    %element_size = vm.const.i64 8
    %num_elements = vm.const.i64 4
    %buffer_size = vm.mul.i64 %num_elements, %element_size : i64
    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %buffer_size, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer
    vm.check.nz %buf_dno, "!null" : !vm.buffer

    // Fill the middle two elements.
    %c2 = vm.const.i64 1
    %c4 = vm.const.i64 2
    %value = vm.const.i64 0x100000000
    vm.buffer.fill.i64 %buf_dno, %c2, %c4, %value : i64 -> !vm.buffer

    // Compare to reference.
    %c0 = vm.const.i64 0
    %rodata_ref = vm.const.ref.rodata @test_fill_i64_ref : !vm.buffer
    %cmp = vm.buffer.compare %rodata_ref, %c0, %buf_dno, %c0, %buffer_size : !vm.buffer, !vm.buffer
    vm.check.nz %cmp, "buffer should match reference" : i32

    vm.return
  }

  // Tests that trying to fill .rodata will fail.
  vm.export @fail_fill_i16_rodata
  vm.func @fail_fill_i16_rodata() {
    %rodata = vm.const.ref.rodata @rodata_3xi32 : !vm.buffer

    // Permission denied:
    %c0 = vm.const.i64 0
    %c2 = vm.const.i64 2
    %cafe = vm.const.i32 0xCAFE
    vm.buffer.fill.i16 %rodata, %c0, %c2, %cafe : i32 -> !vm.buffer

    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Load
  //===--------------------------------------------------------------------===//

  vm.rodata private @test_load_i8_data dense<[0x00, 0x01, 0x7F, 0x80, 0xFF]> : tensor<5xui8>

  vm.export @test_load_i8u
  vm.func @test_load_i8u() {
    %c0 = vm.const.i64 0
    %c1 = vm.const.i64 1
    %c2 = vm.const.i64 2
    %c3 = vm.const.i64 3
    %c4 = vm.const.i64 4
    %rodata = vm.const.ref.rodata @test_load_i8_data : !vm.buffer
    %v0 = vm.buffer.load.i8.u %rodata[%c0] : !vm.buffer -> i32
    %e0 = vm.const.i32 0
    vm.check.eq %v0, %e0, "0" : i32
    %v1 = vm.buffer.load.i8.u %rodata[%c1] : !vm.buffer -> i32
    %e1 = vm.const.i32 1
    vm.check.eq %v1, %e1, "1" : i32
    %v2 = vm.buffer.load.i8.u %rodata[%c2] : !vm.buffer -> i32
    %e2 = vm.const.i32 0x7F
    vm.check.eq %v2, %e2, "0x7F" : i32
    %v3 = vm.buffer.load.i8.u %rodata[%c3] : !vm.buffer -> i32
    %e3 = vm.const.i32 0x80
    vm.check.eq %v3, %e3, "0x80" : i32
    %v4 = vm.buffer.load.i8.u %rodata[%c4] : !vm.buffer -> i32
    %e4 = vm.const.i32 0xFF
    vm.check.eq %v4, %e4, "0xFF" : i32
    vm.return
  }

  vm.export @test_load_i8s
  vm.func @test_load_i8s() {
    %c0 = vm.const.i64 0
    %c1 = vm.const.i64 1
    %c2 = vm.const.i64 2
    %c3 = vm.const.i64 3
    %c4 = vm.const.i64 4
    %rodata = vm.const.ref.rodata @test_load_i8_data : !vm.buffer
    %v0 = vm.buffer.load.i8.s %rodata[%c0] : !vm.buffer -> i32
    %e0 = vm.const.i32 0
    vm.check.eq %v0, %e0, "0" : i32
    %v1 = vm.buffer.load.i8.s %rodata[%c1] : !vm.buffer -> i32
    %e1 = vm.const.i32 1
    vm.check.eq %v1, %e1, "1" : i32
    %v2 = vm.buffer.load.i8.s %rodata[%c2] : !vm.buffer -> i32
    %e2 = vm.const.i32 0x7F
    vm.check.eq %v2, %e2, "0x7F" : i32
    %v3 = vm.buffer.load.i8.s %rodata[%c3] : !vm.buffer -> i32
    %e3 = vm.const.i32 -128
    vm.check.eq %v3, %e3, "-128" : i32
    %v4 = vm.buffer.load.i8.s %rodata[%c4] : !vm.buffer -> i32
    %e4 = vm.const.i32 -1
    vm.check.eq %v4, %e4, "-1" : i32
    vm.return
  }

  vm.rodata private @test_load_i16_data dense<[0x0000, 0x0001, 0x7FFF, 0x8000, 0xFFFF]> : tensor<5xui16>

  vm.export @test_load_i16u
  vm.func @test_load_i16u() {
    %c0 = vm.const.i64 0
    %c1 = vm.const.i64 1
    %c2 = vm.const.i64 2
    %c3 = vm.const.i64 3
    %c4 = vm.const.i64 4
    %rodata = vm.const.ref.rodata @test_load_i16_data : !vm.buffer
    %v0 = vm.buffer.load.i16.u %rodata[%c0] : !vm.buffer -> i32
    %e0 = vm.const.i32 0
    vm.check.eq %v0, %e0, "0" : i32
    %v1 = vm.buffer.load.i16.u %rodata[%c1] : !vm.buffer -> i32
    %e1 = vm.const.i32 1
    vm.check.eq %v1, %e1, "1" : i32
    %v2 = vm.buffer.load.i16.u %rodata[%c2] : !vm.buffer -> i32
    %e2 = vm.const.i32 0x7FFF
    vm.check.eq %v2, %e2, "0x7FFF" : i32
    %v3 = vm.buffer.load.i16.u %rodata[%c3] : !vm.buffer -> i32
    %e3 = vm.const.i32 0x8000
    vm.check.eq %v3, %e3, "0x8000" : i32
    %v4 = vm.buffer.load.i16.u %rodata[%c4] : !vm.buffer -> i32
    %e4 = vm.const.i32 0xFFFF
    vm.check.eq %v4, %e4, "0xFFFF" : i32
    vm.return
  }

  vm.export @test_load_i16s
  vm.func @test_load_i16s() {
    %c0 = vm.const.i64 0
    %c1 = vm.const.i64 1
    %c2 = vm.const.i64 2
    %c3 = vm.const.i64 3
    %c4 = vm.const.i64 4
    %rodata = vm.const.ref.rodata @test_load_i16_data : !vm.buffer
    %v0 = vm.buffer.load.i16.s %rodata[%c0] : !vm.buffer -> i32
    %e0 = vm.const.i32 0
    vm.check.eq %v0, %e0, "0" : i32
    %v1 = vm.buffer.load.i16.s %rodata[%c1] : !vm.buffer -> i32
    %e1 = vm.const.i32 1
    vm.check.eq %v1, %e1, "1" : i32
    %v2 = vm.buffer.load.i16.s %rodata[%c2] : !vm.buffer -> i32
    %e2 = vm.const.i32 0x7FFF
    vm.check.eq %v2, %e2, "0x7FFF" : i32
    %v3 = vm.buffer.load.i16.s %rodata[%c3] : !vm.buffer -> i32
    %e3 = vm.const.i32 -32768
    vm.check.eq %v3, %e3, "-32768" : i32
    %v4 = vm.buffer.load.i16.s %rodata[%c4] : !vm.buffer -> i32
    %e4 = vm.const.i32 -1
    vm.check.eq %v4, %e4, "-1" : i32
    vm.return
  }

  vm.rodata private @test_load_i32_data dense<[0x00000000, 0x00000001, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]> : tensor<5xui32>

  vm.export @test_load_i32
  vm.func @test_load_i32() {
    %c0 = vm.const.i64 0
    %c1 = vm.const.i64 1
    %c2 = vm.const.i64 2
    %c3 = vm.const.i64 3
    %c4 = vm.const.i64 4
    %rodata = vm.const.ref.rodata @test_load_i32_data : !vm.buffer
    %v0 = vm.buffer.load.i32 %rodata[%c0] : !vm.buffer -> i32
    %e0 = vm.const.i32 0
    vm.check.eq %v0, %e0, "0" : i32
    %v1 = vm.buffer.load.i32 %rodata[%c1] : !vm.buffer -> i32
    %e1 = vm.const.i32 1
    vm.check.eq %v1, %e1, "1" : i32
    %v2 = vm.buffer.load.i32 %rodata[%c2] : !vm.buffer -> i32
    %e2 = vm.const.i32 0x7FFFFFFF
    vm.check.eq %v2, %e2, "0x7FFFFFFF" : i32
    %v3 = vm.buffer.load.i32 %rodata[%c3] : !vm.buffer -> i32
    %e3 = vm.const.i32 0x80000000
    vm.check.eq %v3, %e3, "0x80000000" : i32
    %v4 = vm.buffer.load.i32 %rodata[%c4] : !vm.buffer -> i32
    %e4 = vm.const.i32 0xFFFFFFFF
    vm.check.eq %v4, %e4, "0xFFFFFFFF" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Store
  //===--------------------------------------------------------------------===//

  vm.rodata private @test_store_i8_ref dense<[0x00, 0x01, 0x7F, 0x80, 0xFF]> : tensor<5xui8>

  vm.export @test_store_i8
  vm.func @test_store_i8() {
    %ref = vm.const.ref.rodata @test_store_i8_ref : !vm.buffer
    %ref_dno = util.optimization_barrier %ref : !vm.buffer
    %ref_length = vm.buffer.length %ref_dno : !vm.buffer -> i64

    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %ref_length, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer

    %c0 = vm.const.i64 0
    %e0 = vm.const.i32 0
    vm.buffer.store.i8 %e0, %buf_dno[%c0] : i32 -> !vm.buffer
    %c1 = vm.const.i64 1
    %e1 = vm.const.i32 1
    vm.buffer.store.i8 %e1, %buf_dno[%c1] : i32 -> !vm.buffer
    %c2 = vm.const.i64 2
    %e2 = vm.const.i32 0x7F
    vm.buffer.store.i8 %e2, %buf_dno[%c2] : i32 -> !vm.buffer
    %c3 = vm.const.i64 3
    %e3 = vm.const.i32 0x80
    vm.buffer.store.i8 %e3, %buf_dno[%c3] : i32 -> !vm.buffer
    %c4 = vm.const.i64 4
    %e4 = vm.const.i32 0xFF
    vm.buffer.store.i8 %e4, %buf_dno[%c4] : i32 -> !vm.buffer

    %cmp = vm.buffer.compare %ref_dno, %c0, %buf_dno, %c0, %ref_length : !vm.buffer, !vm.buffer
    vm.check.nz %cmp, "source and target match" : i32

    vm.return
  }

  vm.rodata private @test_store_i16_ref dense<[0x0000, 0x0001, 0x7FFF, 0x8000, 0xFFFF]> : tensor<5xui16>

  vm.export @test_store_i16
  vm.func @test_store_i16() {
    %ref = vm.const.ref.rodata @test_store_i16_ref : !vm.buffer
    %ref_dno = util.optimization_barrier %ref : !vm.buffer
    %ref_length = vm.buffer.length %ref_dno : !vm.buffer -> i64

    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %ref_length, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer

    %c0 = vm.const.i64 0
    %e0 = vm.const.i32 0
    vm.buffer.store.i16 %e0, %buf_dno[%c0] : i32 -> !vm.buffer
    %c1 = vm.const.i64 1
    %e1 = vm.const.i32 1
    vm.buffer.store.i16 %e1, %buf_dno[%c1] : i32 -> !vm.buffer
    %c2 = vm.const.i64 2
    %e2 = vm.const.i32 0x7FFF
    vm.buffer.store.i16 %e2, %buf_dno[%c2] : i32 -> !vm.buffer
    %c3 = vm.const.i64 3
    %e3 = vm.const.i32 0x8000
    vm.buffer.store.i16 %e3, %buf_dno[%c3] : i32 -> !vm.buffer
    %c4 = vm.const.i64 4
    %e4 = vm.const.i32 0xFFFF
    vm.buffer.store.i16 %e4, %buf_dno[%c4] : i32 -> !vm.buffer

    %cmp = vm.buffer.compare %ref_dno, %c0, %buf_dno, %c0, %ref_length : !vm.buffer, !vm.buffer
    vm.check.nz %cmp, "source and target match" : i32

    vm.return
  }

  vm.rodata private @test_store_i32_ref dense<[0x00000000, 0x00000001, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]> : tensor<5xui32>

  vm.export @test_store_i32
  vm.func @test_store_i32() {
    %ref = vm.const.ref.rodata @test_store_i32_ref : !vm.buffer
    %ref_dno = util.optimization_barrier %ref : !vm.buffer
    %ref_length = vm.buffer.length %ref_dno : !vm.buffer -> i64

    %alignment = vm.const.i32 16
    %buf = vm.buffer.alloc %ref_length, %alignment : !vm.buffer
    %buf_dno = util.optimization_barrier %buf : !vm.buffer

    %c0 = vm.const.i64 0
    %e0 = vm.const.i32 0
    vm.buffer.store.i32 %e0, %buf_dno[%c0] : i32 -> !vm.buffer
    %c1 = vm.const.i64 1
    %e1 = vm.const.i32 1
    vm.buffer.store.i32 %e1, %buf_dno[%c1] : i32 -> !vm.buffer
    %c2 = vm.const.i64 2
    %e2 = vm.const.i32 0x7FFFFFFF
    vm.buffer.store.i32 %e2, %buf_dno[%c2] : i32 -> !vm.buffer
    %c3 = vm.const.i64 3
    %e3 = vm.const.i32 0x80000000
    vm.buffer.store.i32 %e3, %buf_dno[%c3] : i32 -> !vm.buffer
    %c4 = vm.const.i64 4
    %e4 = vm.const.i32 0xFFFFFFFF
    vm.buffer.store.i32 %e4, %buf_dno[%c4] : i32 -> !vm.buffer

    %cmp = vm.buffer.compare %ref_dno, %c0, %buf_dno, %c0, %ref_length : !vm.buffer, !vm.buffer
    vm.check.nz %cmp, "source and target match" : i32

    vm.return
  }

}
