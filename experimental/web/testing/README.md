# Web Testing Infrastructure

Testing support for IREE's web platform port:

* Scripts to build tests with CMake/Emscripten
* HTML/JS for displaying individual tests and test suites
* Scripting to parse the list of tests provided by `ctest` into the format
  expected by the HTML/JS test suite runner

## Quickstart

1. Install IREE's host tools (e.g. by building the `install` target with CMake)
2. Install the Emscripten SDK by
   [following these directions](https://emscripten.org/docs/getting_started/downloads.html)
3. Initialize your Emscripten environment (e.g. run `emsdk_env.bat`)
4. From this directory, run `bash ./build_sample.sh [path to install] && bash ./serve_sample.sh`
5. Open the localhost address linked in the script output

## Implementation Details

* Our tests (and benchmarks) generate binary files that when executed run tests
  then return an exit code
* When compiling for the web using Emscripten, each binary target produces a
  `.js` and a `.wasm` file
* Running an Emscripten-produced binary on a webpage requires just defining the
  `Module` object and importing the `.js`
* The [`test-runner.html`](test-runner.html) webpage runs an individual test,
  based on URL parameters
* `ctest --show-only=json-v1` outputs a JSON file enumerating all tests and
  their properties (working directory, required files, arguments, etc.). The
  [`parse_test_list.py`](parse_test_list.py) script parses this file into a
  list of HTML elements linking to the test runner webpage with properties set
* The [`index_template.html`](index_template.html) webpage absorbs that list
  of hyperlinks and runs them in an `<iframe>`
