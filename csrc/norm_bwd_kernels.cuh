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

}  // namespace fused_norm
