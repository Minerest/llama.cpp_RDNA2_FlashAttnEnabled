

Title
RDNA2: cudaOccupancyMaxActiveBlocksPerMultiprocessor returns 0 for 256-thread fattn tile (ncols1=4), causing assertion failure during prefill

Post
Hi friends,
looks like I found the happy assertion path referenced in #16643 running on my dual RDNA2 gpu setup.
I bypassed this case and I got a massive speed boost compared to the modern vulkan binaries from less than 30 tok/s to over 50tok/s generation for the Qwen3.6-35B-A3B Q4_K_M!
I had it writing code edits straight through the qwen code CLI no problems.
Super excited this finally feels usable.
I thought I would try to answer the open ended question about the compilation details in #16633 and maybe provide some additional datapoints related to this issue.

Hardware
Radeon RX 6800 (gfx1030, nsm=30)
RX 6700 XT (gfx1031 reporting as gfx1030, nsm=20)
Fedora 44, kernel 7.0.6 with `CONFIG_HSA_AMD=y`
ROCm  7.1.1 (distro package, `/usr` prefix — not TheRock)
HIP  7.1.52802-9999
HIP compiler  `/usr/lib64/rocm/llvm/bin/clang++` (clang 20.0.0.rocm)
llama.cpp commit `cc7200bf1` (version 9166), upstream master with and without patch


crash output
`fattn-common.cuh` had the upstream `GGML_ASSERT(max_blocks_per_sm > 0)` left intact; only the diagnostic prints were added under `#ifdef GGML_FATTN_TRACE`.

```
[fattn-path] tile (DKQ=256, DV=256, ncols2=8, Q->ne[1]=512)
[fattn-trace] void launch_fattn(...) [DV = 256, ncols1 = 4, ncols2 = 8]
[fattn-trace]   device=0 cc=16781360 nsm=30  warp_size=32 nwarps=8  threads/block=256
[fattn-trace]   nbytes_shared=0  nbatch_fa=64  stream_k=0  need_f16_K=1 need_f16_V=1
[fattn-trace]   cudaOccupancyMaxActiveBlocksPerMultiprocessor -> max_blocks_per_sm=0
/home/ediaz/llama/rocm/llama.cpp/ggml/src/ggml-cuda/template-instances/../fattn-common.cuh:1068: GGML_ASSERT(max_blocks_per_sm > 0) failed
```

Backtrace (relevant frames):

```
ggml_abort
launch_fattn<256, 4, 8>(...)                              libggml-hip.so
ggml_cuda_flash_attn_ext_tile_case<256, 256>(...)         libggml-hip.so
ggml_cuda_graph_evaluate_and_capture(...)                 libggml-hip.so
ggml_backend_cuda_graph_compute(...)                      libggml-hip.so
```

Exit code 134 (SIGABRT).

---

Local workaround (what's running now)

`ggml/src/ggml-cuda/fattn-common.cuh` around the assertion site:

```diff
- GGML_ASSERT(max_blocks_per_sm > 0);
+ if (max_blocks_per_sm <= 0) {
+     GGML_LOG_WARN("cudaOccupancyMaxActiveBlocksPerMultiprocessor returned %d, falling back to 1\n", max_blocks_per_sm);
+     max_blocks_per_sm = 1;
+ }
```

Patch and build:

```
cmake -S . -B build-instrumented \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx1030;gfx1031" \
  -DROCM_PATH=/usr \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_HIP_FLAGS="-DGGML_FATTN_TRACE"
cmake --build build-instrumented --target llama-bench -j4
```

| Workload                       | Prefill (t/s) | Decode (t/s) |
| ------------------------------ | ------------: | -----------: |
| -fa off (no flash attention)   |         ~203  |        ~54   |
| -fa on (with fallback patch)   |        ~1314  |        ~60   |


flash_attn_tile

┌───────────────────────────────────────────────────────┬───────────────────────┬───────────────────────┐
│                       Resource                        │ <256,256,4,8> (fails) │ <256,256,1,8> (works) │
├───────────────────────────────────────────────────────┼───────────────────────┼───────────────────────┤
│ Threads/block                                         │         256 (8 waves) │         128 (4 waves) │
├───────────────────────────────────────────────────────┼───────────────────────┼───────────────────────┤
│ VGPRs/thread                                          │                   203 │                   102 │
├───────────────────────────────────────────────────────┼───────────────────────┼───────────────────────┤
│ TotalSGPRs                                            │                    42 │                    44 │
├───────────────────────────────────────────────────────┼───────────────────────┼───────────────────────┤
│ VGPR/SGPR spills                                      │                 0 / 0 │                 0 / 0 │
├───────────────────────────────────────────────────────┼───────────────────────┼───────────────────────┤
│ Static LDS/block                                      │              37,888 B │              21,504 B │
├───────────────────────────────────────────────────────┼───────────────────────┼───────────────────────┤
│ Dynamic shared (runtime)                              │                     0 │                     0 │
├───────────────────────────────────────────────────────┼───────────────────────┼───────────────────────┤
│ Compiler-reported occupancy                           │          4 waves/SIMD │          6 waves/SIMD │
├───────────────────────────────────────────────────────┼───────────────────────┼───────────────────────┤
│ Runtime cudaOccupancyMaxActiveBlocksPerMultiprocessor │                     0 │                     2 │
└───────────────────────────────────────────────────────┴───────────────────────┴───────────────────────┘


Same kernel family (fattn-tile), same hardware, same binary. The occupancy API answers correctly for one and incorrectly for the other.

| Field | tg8 (decode, works) | pp512 (prefill, fails) |
|---|---:|---:|
| Dispatcher | `ggml_cuda_flash_attn_ext_tile_case<256, 256>` | (same) |
| `launch_fattn` template | `<DV=256, ncols1=1, ncols2=8>` | `<DV=256, ncols1=4, ncols2=8>` |
| `Q->ne[1]` | 1 | 512 |
| `nwarps` | 4 | **8** |
| threads/block | 128 | **256** |
| `nbatch_fa` | 32 | 64 |
| `nbytes_shared` (dynamic) | 0 | **0** |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` return | **2** | **0** |
| Outcome | runs, ~54 t/s | assert fires, SIGABRT |

The only kernel-launch difference between working and failing case is **threads/block (128 → 256)** and the register pressure implied by `ncols1=4`. There is **no dynamic shared memory request** in either case.

