// Backward kernel templates for LayerNorm and RMSNorm (plain and fused-add).
// Extracted verbatim from norm_bwd.cu so the launchers, and later additional
// instantiation TUs, can share them.
//
// Notation: xz is the tensor the forward normalised - the input for plain
// norms, the rounded sum z = round(x + residual) for fused-add (which the
// forward already returns, so saving it costs no extra memory). mean/rstd are
// the forward's saved per-row statistics (acc dtype). gamma is the forward
// weight (null => 1). All math in acc_t.
//
// dx (one block per row, scalar strided):
//   LayerNorm: xhat = (xz - mean) * rstd;  g = dy * gamma
//     dx = rstd * (g - sum(g)/N - xhat * sum(g*xhat)/N)        [+ dz_extra]
//   RMSNorm:   xhat = xz * rstd;           g = dy * gamma
//     dx = rstd * (g - xhat * sum(g*xhat)/N)                   [+ dz_extra]
//   The two row sums reduce together (Sum2). dz_extra is the downstream
//   cotangent of the fused-add op's second output (new_residual): because
//   z = x + residual, dx = dresidual = norm_dx + dz_extra, so the launcher
//   adds it elementwise here and the same tensor serves both input grads.
//
// dgamma/dbeta (two-stage, DETERMINISTIC - no atomics):
//   Stage 1 (here): grid (ceil(N/32), num_chunks), 32x32 blocks. Each block
//   owns a 32-column slice and a fixed chunk of rows; per-thread fp32
//   accumulation over its rows, a shared-memory tree reduce over the row
//   axis, one write of partials[chunk, col]:
//     dgamma_p = sum(dy * xhat), dbeta_p = sum(dy).
//   Stage 2 (bindings): partials.sum(0).to(param dtype) - a fixed-shape aten
//   reduction, so parameter grads are bitwise run-to-run reproducible.
//   Atomics would be faster to write and nondeterministic; determinism wins.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "norm_reduce.cuh"
#include "norm_vec.cuh"

namespace fused_norm {

template <typename scalar_t, typename acc_t, bool kRMS>
__global__ void norm_bwd_dx_kernel(const scalar_t* __restrict__ dy,
                                   const scalar_t* __restrict__ dz_extra,  // null => 0
                                   const scalar_t* __restrict__ xz,
                                   const acc_t* __restrict__ mean,         // null iff kRMS
                                   const acc_t* __restrict__ rstd,
                                   const scalar_t* __restrict__ gamma,     // null => 1
                                   scalar_t* __restrict__ dx,
                                   int64_t N) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  const scalar_t* DY = dy + row * N;
  const scalar_t* XZ = xz + row * N;
  const scalar_t* DZ = dz_extra != nullptr ? dz_extra + row * N : nullptr;
  scalar_t* DX = dx + row * N;
  const acc_t mu = kRMS ? static_cast<acc_t>(0) : mean[row];
  const acc_t rs = rstd[row];

  __shared__ acc_t s_c1;  // sum(g) / N          (LayerNorm only; 0 for RMS)
  __shared__ acc_t s_c2;  // sum(g * xhat) / N

  Sum2<acc_t> sums;
  for (int64_t i = tid; i < N; i += stride) {
    const acc_t g = static_cast<acc_t>(DY[i]) *
                    (gamma != nullptr ? static_cast<acc_t>(gamma[i]) : static_cast<acc_t>(1));
    const acc_t xhat = (static_cast<acc_t>(XZ[i]) - mu) * rs;
    if constexpr (!kRMS) sums.a += g;
    sums.b += g * xhat;
  }
  sums = blockReduceSum2<acc_t>(sums);
  if (tid == 0) {
    s_c1 = kRMS ? static_cast<acc_t>(0) : sums.a / static_cast<acc_t>(N);
    s_c2 = sums.b / static_cast<acc_t>(N);
  }
  __syncthreads();
  const acc_t c1 = s_c1;
  const acc_t c2 = s_c2;

  for (int64_t i = tid; i < N; i += stride) {
    const acc_t g = static_cast<acc_t>(DY[i]) *
                    (gamma != nullptr ? static_cast<acc_t>(gamma[i]) : static_cast<acc_t>(1));
    const acc_t xhat = (static_cast<acc_t>(XZ[i]) - mu) * rs;
    acc_t v = rs * (g - c1 - xhat * c2);
    if (DZ != nullptr) v += static_cast<acc_t>(DZ[i]);
    DX[i] = static_cast<scalar_t>(v);
  }
}

// Vectorised dx kernel: identical math and reduction to the scalar flavour
// above, with 16-byte Vec loads/stores (dy, xz, gamma, dz_extra, dx). Keeps
// the two-pass structure deliberately: the row a block owns stays resident
// in L1/L2 between the passes (the same argument as the forward's store-pass
// re-read), so caching it in registers would cost occupancy for no HBM
// saving — a measured non-choice, not an oversight.
template <typename scalar_t, typename acc_t, bool kRMS>
__global__ void norm_bwd_dx_vec_kernel(const scalar_t* __restrict__ dy,
                                       const scalar_t* __restrict__ dz_extra,  // null => 0
                                       const scalar_t* __restrict__ xz,
                                       const acc_t* __restrict__ mean,         // null iff kRMS
                                       const acc_t* __restrict__ rstd,
                                       const scalar_t* __restrict__ gamma,     // null => 1
                                       scalar_t* __restrict__ dx,
                                       int64_t N) {
  using V = Vec<scalar_t>;
  constexpr int kW = kVecWidth<scalar_t>;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t nvec = N / kW;

  const V* DY = reinterpret_cast<const V*>(dy + row * N);
  const V* XZ = reinterpret_cast<const V*>(xz + row * N);
  const V* DZ =
      dz_extra != nullptr ? reinterpret_cast<const V*>(dz_extra + row * N) : nullptr;
  const V* G = gamma != nullptr ? reinterpret_cast<const V*>(gamma) : nullptr;
  V* DX = reinterpret_cast<V*>(dx + row * N);
  const acc_t mu = kRMS ? static_cast<acc_t>(0) : mean[row];
  const acc_t rs = rstd[row];

  __shared__ acc_t s_c1;  // sum(g) / N          (LayerNorm only; 0 for RMS)
  __shared__ acc_t s_c2;  // sum(g * xhat) / N

  Sum2<acc_t> sums;
  for (int64_t i = tid; i < nvec; i += stride) {
    const V dyv = DY[i];
    const V xzv = XZ[i];
    V gv;
    if (G) gv = G[i];
#pragma unroll
    for (int k = 0; k < kW; ++k) {
      const acc_t g = static_cast<acc_t>(dyv.v[k]) *
                      (G ? static_cast<acc_t>(gv.v[k]) : static_cast<acc_t>(1));
      const acc_t xhat = (static_cast<acc_t>(xzv.v[k]) - mu) * rs;
      if constexpr (!kRMS) sums.a += g;
      sums.b += g * xhat;
    }
  }
  sums = blockReduceSum2<acc_t>(sums);
  if (tid == 0) {
    s_c1 = kRMS ? static_cast<acc_t>(0) : sums.a / static_cast<acc_t>(N);
    s_c2 = sums.b / static_cast<acc_t>(N);
  }
  __syncthreads();
  const acc_t c1 = s_c1;
  const acc_t c2 = s_c2;

  for (int64_t i = tid; i < nvec; i += stride) {
    const V dyv = DY[i];
    const V xzv = XZ[i];
    V gv, dzv;
    if (G) gv = G[i];
    if (DZ) dzv = DZ[i];
    V out;
#pragma unroll
    for (int k = 0; k < kW; ++k) {
      const acc_t g = static_cast<acc_t>(dyv.v[k]) *
                      (G ? static_cast<acc_t>(gv.v[k]) : static_cast<acc_t>(1));
      const acc_t xhat = (static_cast<acc_t>(xzv.v[k]) - mu) * rs;
      acc_t v = rs * (g - c1 - xhat * c2);
      if (DZ) v += static_cast<acc_t>(dzv.v[k]);
      out.v[k] = static_cast<scalar_t>(v);
    }
    DX[i] = out;
  }
}

// Stage 1 of the parameter gradients. kTile = 32.
constexpr int kBwdTile = 32;

template <typename scalar_t, typename acc_t, bool kRMS, bool kBeta>
__global__ void norm_bwd_param_partials_kernel(const scalar_t* __restrict__ dy,
                                               const scalar_t* __restrict__ xz,
                                               const acc_t* __restrict__ mean,  // null iff kRMS
                                               const acc_t* __restrict__ rstd,
                                               acc_t* __restrict__ dgamma_partials,  // [chunks, N]
                                               acc_t* __restrict__ dbeta_partials,   // [chunks, N] or null
                                               int64_t M,
                                               int64_t N,
                                               int64_t rows_per_chunk) {
  constexpr int kTile = kBwdTile;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * kTile + threadIdx.x;
  const int64_t chunk = blockIdx.y;
  const int64_t row_begin = chunk * rows_per_chunk;
  const int64_t row_end = row_begin + rows_per_chunk < M ? row_begin + rows_per_chunk : M;

  acc_t dg = 0;
  acc_t db = 0;
  if (col < N) {
    for (int64_t row = row_begin + threadIdx.y; row < row_end; row += kTile) {
      const acc_t d = static_cast<acc_t>(dy[row * N + col]);
      const acc_t mu = kRMS ? static_cast<acc_t>(0) : mean[row];
      const acc_t xhat = (static_cast<acc_t>(xz[row * N + col]) - mu) * rstd[row];
      dg += d * xhat;
      if constexpr (kBeta) db += d;
    }
  }

  // Transpose-free tree reduce over threadIdx.y (the row dimension of the
  // block): +33 padding keeps the column accesses bank-conflict-free.
  __shared__ acc_t s_dg[kTile][kTile + 1];
  __shared__ acc_t s_db[kTile][kTile + 1];
  s_dg[threadIdx.y][threadIdx.x] = dg;
  if constexpr (kBeta) s_db[threadIdx.y][threadIdx.x] = db;
  __syncthreads();
  for (int offset = kTile / 2; offset > 0; offset /= 2) {
    if (threadIdx.y < offset) {
      s_dg[threadIdx.y][threadIdx.x] += s_dg[threadIdx.y + offset][threadIdx.x];
      if constexpr (kBeta) s_db[threadIdx.y][threadIdx.x] += s_db[threadIdx.y + offset][threadIdx.x];
    }
    __syncthreads();
  }

  if (threadIdx.y == 0 && col < N) {
    dgamma_partials[chunk * N + col] = s_dg[0][threadIdx.x];
    if constexpr (kBeta) dbeta_partials[chunk * N + col] = s_db[0][threadIdx.x];
  }
}

// Vectorised stage 1: block (kPTx, kPTy) = 256 threads; each thread owns kW
// consecutive columns (one 16-byte Vec of dy and of xz per visited row) with
// register accumulators, so a warp issues 4 full 128-byte transactions per
// load where the scalar kernel issued 64 useful bytes. mean/rstd hoist to
// registers once per row (L1 broadcast serves the warp). The shared reduce
// runs over y only — 3 levels instead of 5 — with layout s[y][k][x]: x is the
// fastest index, so lanes touch consecutive words at every k and the tiles
// are bank-conflict-free without padding.
constexpr int kPTx = 32;
constexpr int kPTy = 8;

template <typename scalar_t, typename acc_t, bool kRMS, bool kBeta>
__global__ void norm_bwd_param_partials_vec_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ xz,
    const acc_t* __restrict__ mean,  // null iff kRMS
    const acc_t* __restrict__ rstd,
    acc_t* __restrict__ dgamma_partials,  // [chunks, N]
    acc_t* __restrict__ dbeta_partials,   // [chunks, N] or null
    int64_t M,
    int64_t N,
    int64_t rows_per_chunk) {
  using V = Vec<scalar_t>;
  constexpr int kW = kVecWidth<scalar_t>;
  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const int64_t nvec = N / kW;  // this path requires N % kW == 0
  const int64_t vcol = static_cast<int64_t>(blockIdx.x) * kPTx + x;
  const int64_t chunk = blockIdx.y;
  const int64_t row_begin = chunk * rows_per_chunk;
  const int64_t row_end = row_begin + rows_per_chunk < M ? row_begin + rows_per_chunk : M;

  acc_t dg[kW] = {};
  acc_t db[kW] = {};
  if (vcol < nvec) {
    for (int64_t row = row_begin + y; row < row_end; row += kPTy) {
      const V dyv = reinterpret_cast<const V*>(dy + row * N)[vcol];
      const V xzv = reinterpret_cast<const V*>(xz + row * N)[vcol];
      const acc_t mu = kRMS ? static_cast<acc_t>(0) : mean[row];
      const acc_t rs = rstd[row];
#pragma unroll
      for (int k = 0; k < kW; ++k) {
        const acc_t d = static_cast<acc_t>(dyv.v[k]);
        const acc_t xhat = (static_cast<acc_t>(xzv.v[k]) - mu) * rs;
        dg[k] += d * xhat;
        if constexpr (kBeta) db[k] += d;
      }
    }
  }

  // kBeta sizes the beta tile so the no-beta instantiations don't pay shared
  // memory for a buffer they never touch.
  __shared__ acc_t s_dg[kPTy][kW][kPTx];
  __shared__ acc_t s_db[kBeta ? kPTy : 1][kBeta ? kW : 1][kPTx];
#pragma unroll
  for (int k = 0; k < kW; ++k) s_dg[y][k][x] = dg[k];
  if constexpr (kBeta) {
#pragma unroll
    for (int k = 0; k < kW; ++k) s_db[y][k][x] = db[k];
  }
  __syncthreads();
  for (int offset = kPTy / 2; offset > 0; offset /= 2) {
    if (y < offset) {
#pragma unroll
      for (int k = 0; k < kW; ++k) {
        s_dg[y][k][x] += s_dg[y + offset][k][x];
        if constexpr (kBeta) s_db[y][k][x] += s_db[y + offset][k][x];
      }
    }
    __syncthreads();
  }

  if (y == 0 && vcol < nvec) {
#pragma unroll
    for (int k = 0; k < kW; ++k) {
      dgamma_partials[chunk * N + vcol * kW + k] = s_dg[0][k][x];
      if constexpr (kBeta) dbeta_partials[chunk * N + vcol * kW + k] = s_db[0][k][x];
    }
  }
}

// Stage 2: one launch finalises both parameter grads. Block (kFTx, kFTy):
// each x-lane owns a column; the kFTy y-threads split the chunk axis in a
// fixed strided pattern and a fixed 3-level shared tree combines them — the
// order is a pure function of the shapes, so the result stays bitwise
// run-to-run deterministic. The y-split matters: N alone is only ~N threads
// of parallelism (a 4-block grid at N=1024 left the kernel latency-bound at
// ~10 us); splitting chunks across y turns it into ~2 us. Replaces aten's
// partials.sum(0).to(dtype) x2 (up to four launches plus two temporaries).
constexpr int kFTx = 32;
constexpr int kFTy = 8;

template <typename scalar_t, typename acc_t>
__global__ void norm_bwd_param_finalize_kernel(
    const acc_t* __restrict__ dgamma_partials,  // [chunks, N] (always present)
    const acc_t* __restrict__ dbeta_partials,   // null when beta absent/unrequested
    scalar_t* __restrict__ dgamma,              // null when unrequested
    scalar_t* __restrict__ dbeta,               // null when unrequested
    int64_t chunks,
    int64_t N) {
  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * kFTx + x;

  acc_t sg = 0;
  acc_t sb = 0;
  if (col < N) {
    for (int64_t c = y; c < chunks; c += kFTy) {
      if (dgamma != nullptr) sg += dgamma_partials[c * N + col];
      if (dbeta != nullptr) sb += dbeta_partials[c * N + col];
    }
  }

  __shared__ acc_t s_g[kFTy][kFTx];
  __shared__ acc_t s_b[kFTy][kFTx];
  s_g[y][x] = sg;
  s_b[y][x] = sb;
  __syncthreads();
  for (int offset = kFTy / 2; offset > 0; offset /= 2) {
    if (y < offset) {
      s_g[y][x] += s_g[y + offset][x];
      s_b[y][x] += s_b[y + offset][x];
    }
    __syncthreads();
  }

  if (y == 0 && col < N) {
    if (dgamma != nullptr) dgamma[col] = static_cast<scalar_t>(s_g[0][x]);
    if (dbeta != nullptr) dbeta[col] = static_cast<scalar_t>(s_b[0][x]);
  }
}

}  // namespace fused_norm
