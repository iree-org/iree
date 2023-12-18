vm.module @io_stream {

// Returns a handle to the console `stdin` stream.
vm.import private @console.stdin() -> !vm.ref<!io_stream.handle>
attributes {nosideeffects}

// Returns a handle to the console `stdout` stream.
vm.import private @console.stdout() -> !vm.ref<!io_stream.handle>
attributes {nosideeffects}

// Returns a handle to the console `stderr` stream.
vm.import private @console.stderr() -> !vm.ref<!io_stream.handle>
attributes {nosideeffects}

// Returns the current offset of the stream.
vm.import private @offset(
  %handle: !vm.ref<!io_stream.handle>
) -> i64
attributes {nosideeffects}

// Returns the current length of the stream or -1 if infinite.
vm.import private @length(
  %handle: !vm.ref<!io_stream.handle>
) -> i64
attributes {nosideeffects}

// Reads a single byte from the stream.
// Returns the byte or -1 if the end-of-stream was reached.
vm.import private @read.byte(
  %handle: !vm.ref<!io_stream.handle>
) -> i32

// Reads up to |length| bytes from the stream into |buffer| at |offset|.
// Returns the total number of bytes read which may be fewer than the requested
// if the end-of-stream is reached.
vm.import private @read.bytes(
  %handle: !vm.ref<!io_stream.handle>,
  %buffer: !vm.buffer,
  %offset: i64,
  %length: i64
) -> i64

// Reads characters until |delimiter| is reached and returns all characters read
// excluding the delimiter.
vm.import private @read.delimiter(
  %handle: !vm.ref<!io_stream.handle>,
  %delimiter: i32
) -> !vm.buffer

// Writes a byte to the stream.
vm.import private @write.byte(
  %handle: !vm.ref<!io_stream.handle>,
  %byte: i32
)

// Writes |length| bytes to the stream from |buffer| at |offset|.
vm.import private @write.bytes(
  %handle: !vm.ref<!io_stream.handle>,
  %buffer: !vm.buffer,
  %offset: i64,
  %length: i64
)

}  // module
