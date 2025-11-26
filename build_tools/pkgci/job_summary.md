## Benchmark Comparison Report

HEAD a77a1e57f4168313174e41d80f1ed8cea2e6859f vs PR 17f639737b9e624c4630ef0fa801e1e45a7d6748

### Benchmark Comparison for cpu

| Benchmark | Main Time (ms) | PR Time (ms) | Change (%) |
|-----------|----------------|---------------|--------|
| sdxl/clip_benchmark_cpu.json | 393.46 | 400.63 | 1.82% |
### Benchmark Comparison for mi325

| Benchmark | Main Time (ms) | PR Time (ms) | Change (%) |
|-----------|----------------|---------------|--------|
| llama_8b_fp16/decode_benchmark_seq128_mi325.json | 6.19 | 6.45 | 4.20% |
| llama_8b_fp16/decode_benchmark_seq2048_mi325.json | 7.48 | 7.63 | 1.98% |
| llama_8b_fp16/prefill_benchmark_seq128_mi325.json | 29.56 | 29.45 | -0.37% |
| llama_8b_fp16/prefill_benchmark_seq2048_mi325.json | 430.70 | 431.13 | 0.10% |
| llama_8b_fp8/decode_benchmark_seq128_mi325.json | 7.93 | 8.28 | 4.40% |
| llama_8b_fp8/decode_benchmark_seq2048_mi325.json | 10.34 | 10.87 | 5.10% |
| llama_8b_fp8/prefill_benchmark_seq128_mi325.json | 19.25 | 19.59 | 1.78% |
| llama_8b_fp8/prefill_benchmark_seq2048_mi325.json | 256.71 | 259.35 | 1.03% |
| sdxl/clip_benchmark_mi325.json | 7.78 | 7.64 | -1.88% |
