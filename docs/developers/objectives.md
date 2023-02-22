# 2021 Q2 Objectives (OKRs)

## This Quarter's High-level Themes

**IREE Production Use Cases**

1. **ASR.** Continue to improve IREE capabilities and performance for targeting lingvo-like ASR models.
1. **TFLite/TOSA.** Demonstrate a complete user journey with an e2e quantized TFLite model running in a sample Android app using TOSA.

**IREE Codegen**

1. **Code Health.** As we deprecate the legacy compilation of Linalg on buffers, revisit all the passes/ordering of transformations to see if there are still relevant. Better structure of the codegeneration pipeline to address more use cases.
1. **Performance.** Focus on performance improvements that are likely to benefit the ASR models, with a mix of near-term and 1-2 quarter long investments. Two main buckets here
    1. **Architecture-independent optimizations.** Make sure that lowering of model is not introducing unnecessary overheads.
    1. **Architecture-specific optimizations.** Last-mile tuning for a particular hardware.

## Specific OKRs

### P0 O: Demonstrate e2e compilation and execution of a TFLite model on Android using IREE

+ P1 KR: TOSA supports all non-MobileNet floating point models in tracking list (internal)
+ P1 KR: Android demo running IREE runtime through Java and/or C APIs
+ P1 KR: Documented workflows for compiling TFLite models to an IREE-compatible format

### P1 O: Improve IREE code health and extensibility

+ P0 KR: Removed legacy MHLO based dispatch region creation and associated code in the backends to support that path
+ P1 KR: Made the backend less monolithic and easily extensible.
    + Note: All operations go through the same compilation steps making changes hard. Instead have multiple codegen strategies that could define their own scope but still usable within IREE
+ P2 KR: Removed VMLA backend

#### P1 O: Track and improve kernel performance of code-generation backends for ASR-like models

+ P1 KR: Created microbenchmarks for matmuls/convolutions based on configurations used in models like ASR and track the performance on CPU/GPU backends
+ P1 KR: Improved performance of depthwise convolution on CPU backend
+ P1 KR: Improved performance of different transpose variants of matmuls on CPU and GPU
+ P1 KR: Improved performance of convolution/matmul with padding on CPU and GPU backends
+ P1 KR: Address inefficiencies in the lowering of models through IREE compilation pipeline
+ P1 KR: Find and improve inefficient dispatch region creation on the Linalg on tensors path.
+ P1 KR: Overhead of buffer allocation on MobileBERT e2e test < 10ms

### P1 O: Improve IREE infrastructure for targeting different CPU/GPU hardware

+ P1 KR: Define proper class abstractions and utilities for GPU target triples
+ P1 KR: Flesh out target feature queries (fp16, etc.) at runtime and be consistent with capabilities used in shaders
+ P1 KR: Integrate with CI to build/test models according to target environment
+ P2 KR: Support dynamically select kernel flavors at runtime

### P1 O: Improve e2e testing, benchmarking, and build infrastructure

+ P0 KR: Benchmark results can be pulled into a presentation without consultation
    + Note: This implies they should be accurate, well-documented, and readily interpretable.
+ P1 KR: Defined a roadmap for robust multi-level testing
+ P1 KR: Made significant step toward initial e2e testing milestone
+ P1 KR: Handed off maintenance of LLVM Bazel BUILD


# 2020 Q4 Objectives (OKRs)

## This Quarter's High-level themes

1.  **CPU perf burndown.** Bring up the perf burndown process and turn the crank a few times on CPU codegen. The initial workload we'll be targeting is the MobileBERT encoder. In parallel we will assess potential alternative workloads for next quarter.
1.  **GPU back-end.** Continue to land critical infrastructure, take a moment to pause and evaluate where we stand on performance, and pursue any low-hanging perf fruit in anticipation of a GPU perf burndown in Q1. To the extent it makes sense, prioritize work that benefits both CPU and GPU perf.
1.  **Infrastructure.** Continue to make critical improvements to build infrastructure that improve development velocity. Support the CPU perf burndown effort by ensuring that benchmarking can be performed easily and its results are meaningful and reliable.

## Specific OKRs

### P1 O: [Perf burndown] Bring up and execute perf burndown process for MobileBERT workload

+   P1 KR: IREE CPU codegen achieves near peak throughput on all of the MobileBERT matmul shapes
    + At least the ones with > 4% weight in the above profile.
+   P1 KR: Ensure that IREE CPU codegen achieves decent performance on softmax. That means matching TFLite performance on the softmax layers in the MobileBERT model.
+   P1 KR: Able to benchmark and profile the whole MobileBERT workload in both TFLite and IREE and compare results.
+   P1 KR: Performed at least 2 cycles of: (a) assess whole workload performance (b) identify key source of performance delta between TFLite and IREE (c) fix the issue (d) repeat.
+   P1 KR: Prioritized list of key sources of performance delta between TFLite and IREE
    + This list should be largely composed of issues that have non-trivial resolutions, like a re-design of a key IREE component. It should be updated on an ongoing basis.
+   P2 KR: MobileBERT end-to-end benchmark matches 80% of TFLite performance.
+   P2 KR: Sources of remaining perf gap in TFLite vs IREE on MobileBERT end-to-end benchmark characterized.

### P1 O: [Perf burndown] Improve benchmarking and profiling tooling to support perf burndown

+   P1 KR: Able to micro-benchmark kernels using a shared, documented tool
    +   Improve the dump of input and output for each dispatch function.
    +   Dump dispatch functions to files
    +   Improve diffing tools
    +   Report performance for each dispatch function?

+   P1 KR: Able to perform IREE profiling using Tracy

    + See [https://github.com/openxla/iree/issues/1886](https://github.com/openxla/iree/issues/1886), [https://github.com/wolfpld/tracy](https://github.com/wolfpld/tracy)

+   P1 KR: Able to map time spent in execution to back to source using Tracy
    +   See https://github.com/openxla/iree/issues/1199
    +   Source layer (source python, HLO, HAL, etc) is configurable at compile time.

+   P1 KR: Able to track compile-time performance-related statistics
    + See [https://github.com/openxla/iree/issues/1409](https://github.com/openxla/iree/issues/1409)
    + Initial stats to track: number of executables, the serialized size of constant data, the serialized size of the executables, the number of host readbacks (flow.tensor.load), backend specific stats like the number of split dispatches in the SPIR-V backend, dynamic shape info like the number of tensors with dynamic shapes that survive after shape propagation

+   P1 KR: Internal and external contributors able to confidently assess performance impact of a change.
    +   Like correctness, have tests to guard against performance regressions
    +   Either some kind of presubmit using a consistent environment or instructions for running something manually that we believe will offer a useful before/after signal.
    +   Include TFLite as a baseline.
    +   Build test suite and mechanism for tracking performance

### P1 O: [Perf burndown] Identify target workload for IREE perf credibility burndown

Note: This is in preparation for a 2021Q1 objective: Establish IREE's credibility at delivering competitive production levels of performance on a realistic use case.

+   P1 KR: Defined criteria for evaluating workloads for the burndown.
+   P1 KR: Evaluated criteria (including performance analysis) for all candidate workloads.
+   P1 KR: Selected target workload for the CPU burndown.
+   P1 KR: Selected target workload for the GPU burndown in Q1.
    +  Representative for GPU. what to compare. What we can achieve in 1Q

### P1 O: [Perf burndown] Add initial support for multi-threaded workloads

+   P1 KR: Selected a target multi-threaded workload for development and initial benchmarking
+   P1 KR: Able to benchmark a multi-threaded workload on CPU.
+   P1 KR: Use GPU tiling pass on CPU, as well as treating CPU as another device
+   P2 KR: Documentation of known issues / architectural challenges with current approach to multi-threading.

### P1 O: [Infra] Improve OSS build infrastructure to support continued development

+   P1 KR: Minimal-effort merging process for integrating new LLVM commits with no dependency on TF

    + Continuous build for OSS LLVM build files. Propose upstreaming to LLVM community.

+   P1 KR: IREE Core build doesn't depend on TF

    + Requires integration with separate MLIR-HLO repo.

+   P1 KR: Extended build bot / lint coverage catches issues in OSS

    + asan, clang-tidy, windows iff volunteer with windows machine, yapf, bazel android, unibeautify for formatting more generally(?), binary size.

+   P1 KR: General build health

    + A bunch of small things that require attention. RBE warnings on diagnostics that are disabled, remove use of globs in cmake, keep up to date with Bazel versions, see if we can speedup Bazel bot builds by moving sandbox root to tmpfs, remove common include directories.

+   P2 KR: Check tests are fast in dbg mode

    + Currently we've got quadratic growth for a slow path in swiftshader dbg. Potential solutions: IREE multi-module -> archive compilation support, conditional sharing of instances in tests.

### P1 O: [User-facing] Prepare to support real-world use cases

Notes: Keep a pulse on deployment user journeys, continue to gather requirements from interested users, set ourselves on a path to production use on at least one platform.

+   P1 KR: A new sample application showing high-level IREE behavior

    +  Android/Java speech demo and/or desktop Vulkan image-to-image pipeline

+   P1 KR: A PRD that captures deployment requirements (such as platform and device support, async model download, etc.)
+   P2 KR: Identify and track list of potential customers, their use cases, and deployment requirements
+   P2 KR: Proof of concept deployment with at least one other team

    + Focus on requirements gathering and prototyping, find what features we're missing (e.g. Android or Vulkan memory sharing, build configurations compatible with Stadia, etc.)

### P1 O: [User-facing] Expand Java API to support a useful sample Android app

+   P1 KR: Support for different input and output types
+   P1 KR: Support for driver creation
+   P1 KR: Support for different module types, including: HAL, bytecode, tensorlist, and string
+   P2 KR: Design for supporting custom modules

### P1 O: [Model support] Achieve target Tensorflow front-end fidelity

+   P1 KR: Full ASR Decoder to HLO
+   P1 KR: Investigate TF to HLO lowering without TF folding (e.g. linspace)
+   P2 KR: Transformer ASR to HLO
+   P2 KR: Documentation detailing unlowerable operations to HLO
+   P3 KR: Stretch: SASP rewrite for better shape inference

### P1 O: [Model support] Implement remaining VMLA ops

+   P1 KR: FFT operations
+   P1 KR: Sort operation
+   P1 KR: Full ASR Decoder executing on VMLA
+   P2 KR: Transformer ASR executing on VMLA

### P1 O: [CPU Codegen] Improve CPU codegen infrastructure

+   P1 KR: Lowering path MHLO ops --> linalg named Ops (library calls).

    +   Sort, TopK and similar indexing stye ops

+   P1 KR: Support Linalg fusion on buffers using stack allocations.
+   P1 KR: Improve AOT linking and support automatic toolchain discovery

    +   Link all executables in a single dylib.
    +   Support exporting/loading dylib to standalone binary.

### P1 O: [MLIR codegen] Retargetable codegeneration (Vector dialect-based approach)

+   P1 KR: Develop mechanisms to distribute vector operation at workgroup level to vector operation at subgroup level / work item level
+   P1 KR: Handle distribution of producer-consumer vector operations to implement fusion

### P1 O: [GPU Codegen] Improve generic GPU codegen performance

+   P1 KR: Using Linalg fusion on buffers

    + Fuse operations like matmul/conv with its producers/consumers, using workgroup memory as intermediate tile storage. Examples: Elementwise -> Matmul -> Elementwise, Padding -> Conv/Pool operations

+   P1 KR: Fast matmul for mobile GPUs that don't have tensor core units
    +   - This applies work from Q2 to more architectures (e.g., ARM and Qualcomm GPUs)
    +   - Match handwritten vulkan kernel (On Pixel4: current iree matmul for a 1K matrix runs in 128ms and handwritten kernel runs in 19ms)

+   P1 KR: Using subgroup operations for reduction
+   P1 KR: Achieve reasonable performance for one model on one mobile GPU
    +   - Decide on a mobile GPU
    +   - Match TFLite MobileNetV2 performance (f32, imagenet)

### P1 O: [GPU Codegen] Add support for targeting different GPU hardware

+   P1 KR: Define representation for Vulkan/SPIR-V targets

    + Define at least one target for NVIDIA, AMD, Qualcomm, ARM

+   P1 KR: Use Vulkan/SPIR-V targets to guide GPU CodeGen

    + Enable promotion when workgroup memory is available
    + Choose proper tile parameters
    + Choose proper cooperative matrix parameters
    + Choose proper workgroup/subgroup size
    + Goal is to have a prototype supporting the limited cases we currently have and have a plan to scale. Ex: Support both Mali/Adreno best parameters for tile/workgroup size. Potentially also support Nvidia Turing GPU.

### P1 O: [GPU Codegen] Build infrastructure for performance improvement

+   P1 KR: Document clear profiling tools/flows
    +   - Mostly covering vendor/platform specific tools
    +   - For both desktop NVIDIA/AMD and Android AGI

+   P1 KR: Collect performance metrics of mobile GPUs

    +   - Get empirical data over data movement performance
    +   - Get empirical data over subgroup ops performance
    +   - Get empirical data over best tiling/workgroup size
    +   - Get empirical data over different matmul impls
    +   - Produce a repo for hosting such benchmarks
    +   - Produce a doc listing such results

### P1 O: [Strategy] Define high-level strategy for quantization support in IREE

+   P1 KR: Strategy document describing 2021 roadmap, resourcing, and approach
+   P1 KR: Two initial quantization projects described and queued up for 2021 Q1
