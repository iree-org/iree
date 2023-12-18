// RUN: (echo -e "hello\nworld\nlines" | iree-run-mlir --Xcompiler,iree-hal-target-backends=vmvx %s --function=echo) | FileCheck %s --check-prefix=CHECK-ECHO

// CHECK-ECHO-LABEL: EXEC @echo
// CHECK-ECHO-NEXT: type a line
// CHECK-ECHO-SAME: hello
// CHECK-ECHO-NEXT: type a line
// CHECK-ECHO-SAME: world
// CHECK-ECHO-NEXT: type a line
// CHECK-ECHO-SAME: lines
// CHECK-ECHO-NEXT: type a line
func.func @echo() -> () {
  %stdin = io_stream.console.stdin : !io_stream.handle
  %stdout = io_stream.console.stdout : !io_stream.handle

  %c0 = arith.constant 0 : index
  %true = arith.constant 1 : i1
  %null = util.null : !util.buffer

  // Prompt printed each time we wait for input.
  %prompt = util.buffer.constant : !util.buffer = "type a line, ctrl-c to exit > "

  // Loop until the end-of-stream is reached.
  scf.while(%not_eos = %true) : (i1) -> i1 {
    scf.condition(%not_eos) %not_eos : i1
  } do {
  ^bb0(%_: i1):
    // Write prompt and read input until newline/ctrl-c is reached.
    io_stream.write.bytes(%stdout, %prompt) : (!io_stream.handle, !util.buffer) -> ()
    %line = io_stream.read.line(%stdin) : (!io_stream.handle) -> !util.buffer
    // A null return indicates end-of-stream.
    %not_eos = util.cmp.ne %line, %null : !util.buffer
    scf.if %not_eos {
      // If not yet at the end-of-stream check the returned line. If it's empty
      // (user just pressed enter) we skip it. stdin piped from files/echo/etc
      // will usually have a trailing newline.
      %line_length = util.buffer.size %line : !util.buffer
      %not_line_empty = arith.cmpi ne, %line_length, %c0 : index
      scf.if %not_line_empty {
        // Echo the line and then add a newline as what we read excludes it.
        io_stream.write.line(%stdout, %line) : (!io_stream.handle, !util.buffer) -> ()
        scf.yield
      }
      scf.yield
    }
    // Continue so long as we are not at the stdin end-of-stream.
    scf.yield %not_eos : i1
  }

  return
}
