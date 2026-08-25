// Generic forward kernels for the op family:
//   {LayerNorm, RMSNorm} x {plain, fused residual-add} x epilogue functor,
// in a scalar (strided) flavour and a 16-byte-vectorised flavour, one thread
// block per row, following the same structure as the verified v0.3.0
// LayerNorm kernels in layernorm_cuda_kernel.cu (which remain in use for
// plain LayerNorm inference and are NOT replaced by these templates).
//
// Semantics:
//   * kRMS=false: y = (x - mean(x)) * rsqrt(var(x) + eps) [* gamma] [+ beta]
//   * kRMS=true : y = x * rsqrt(mean(x^2) + eps) [* gamma]   (beta must be null)
//   * kFusedAdd : z = scalar_t(acc(x) + acc(residual_in)) is written to
//     residual_out ONCE (rounded exactly once), and the statistics and the
//     normalise pass both consume the ROUNDED z. This makes
//     out == plain_norm(residual_out) hold bitwise (composite equivalence:
//     the fused op agrees exactly with "add, round, then norm"), and it is
//     what vLLM's fused_add_rms_norm does. residual_out may alias
//     residual_in (in-place): each element is read and written exactly once,
//     by the same thread, before any element of it is re-read after the
//     block-wide barrier that follows the stats reduction.
//   * mean_out / rstd_out (when non-null): per-row statistics in acc_t,
//     written by thread 0 - the values autograd needs (mean only exists for
//     LayerNorm).
//
// Accumulation is acc_t (float for fp16/bf16/fp32, double for fp64)
// throughout; eps is added to the biased variance / mean-square inside
// rsqrt, matching F.layer_norm / F.rms_norm.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "norm_reduce.cuh"
#include "norm_vec.cuh"

namespace fused_norm {

// ---------------------------------------------------------------------------
// Scalar (strided) flavour - the universal fallback, any N.
// LayerNorm uses the two-pass form (mean, then centred sum of squares);
// RMSNorm needs a single stats pass (sum of squares around 0).
// ---------------------------------------------------------------------------
template <typename scalar_t, typename acc_t, bool kRMS, bool kFusedAdd, typename Epi>
__global__ void norm_fwd_kernel(const scalar_t* __restrict__ input,
                                // No __restrict__ on the residual pair: residual_out may alias
                                // residual_in (the in-place variant), and promising no-alias
                                // there would be UB.
                                const scalar_t* residual_in,   // null unless kFusedAdd
                                scalar_t* residual_out,        // null unless kFusedAdd
                                typename Epi::out_t* __restrict__ output,
                                const scalar_t* __restrict__ gamma,         // null ok
                                const scalar_t* __restrict__ beta,          // null ok; null if kRMS
                                acc_t* __restrict__ mean_out,               // null ok; [M]
                                acc_t* __restrict__ rstd_out,               // null ok; [M]
                                int64_t N,
                                acc_t eps,
                                Epi epi) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  const scalar_t* X = input + row * N;
  typename Epi::out_t* Y = output + row * N;
  const scalar_t* R = kFusedAdd ? residual_in + row * N : nullptr;
  scalar_t* Z = kFusedAdd ? residual_out + row * N : nullptr;
  // Every pass after the first reads the row from here: the rounded sum for
  // fused-add, the input otherwise.
  const scalar_t* SRC = kFusedAdd ? Z : X;

  __shared__ acc_t s_mean;
  __shared__ acc_t s_rstd;

  // Stats pass 1. Fused-add also materialises z = round(x + r) exactly here.
  acc_t accum = 0;  // sum(z) for LayerNorm, sum(z^2) for RMSNorm
  for (int64_t i = tid; i < N; i += stride) {
    acc_t v;
    if constexpr (kFusedAdd) {
      const acc_t z_acc = static_cast<acc_t>(X[i]) + static_cast<acc_t>(R[i]);
      const scalar_t z = static_cast<scalar_t>(z_acc);
      Z[i] = z;
      v = static_cast<acc_t>(z);  // stats over the ROUNDED value
    } else {
      v = static_cast<acc_t>(X[i]);
    }
    if constexpr (kRMS) {
      accum += v * v;
    } else {
      accum += v;
    }
  }
  accum = blockReduceSum<acc_t>(accum);

  acc_t mean = 0;
  if constexpr (!kRMS) {
    if (tid == 0) s_mean = accum / static_cast<acc_t>(N);
    __syncthreads();  // publishes s_mean AND separates the two block reductions
    mean = s_mean;

    // Stats pass 2 (LayerNorm only): centred sum of squares.
    acc_t sq = 0;
    for (int64_t i = tid; i < N; i += stride) {
      const acc_t d = static_cast<acc_t>(SRC[i]) - mean;
      sq += d * d;
    }
    accum = blockReduceSum<acc_t>(sq);
  }

  if (tid == 0) {
    s_rstd = rsqrt(accum / static_cast<acc_t>(N) + eps);
    if (rstd_out != nullptr) rstd_out[row] = s_rstd;
    if constexpr (!kRMS) {
      if (mean_out != nullptr) mean_out[row] = s_mean;
    }
  }
  __syncthreads();
  const acc_t rstd = s_rstd;

  // Quantising epilogues: inv_scale is per-thread (static scale) or comes
  // from an extra per-row amax pass over the would-be scalar_t outputs
  // (dynamic scale). The barrier above separates this block reduction from
  // the stats one, as blockReduce's contract requires.
  float inv_scale = epi.load_inv_scale();
  if constexpr (Epi::kNeedsRowMax) {
    __shared__ float s_inv_scale;
    // |y| >= 0, so IEEE ordering equals integer ordering on the raw bits, and
    // every NaN pattern compares above +inf: integer max over the bit
    // patterns is a branch-free NaN-PROPAGATING amax (fmaxf would drop NaN,
    // giving a poisoned row a tiny finite scale; torch.amax propagates).
    int amax_bits = 0;
    for (int64_t i = tid; i < N; i += stride) {
      acc_t v = static_cast<acc_t>(SRC[i]);
      if constexpr (kRMS) {
        v *= rstd;
      } else {
        v = (v - mean) * rstd;
      }
      if (gamma) v *= static_cast<acc_t>(gamma[i]);
      if constexpr (!kRMS) {
        if (beta) v += static_cast<acc_t>(beta[i]);
      }
      const float y_s = fabsf(static_cast<float>(static_cast<scalar_t>(v)));
      amax_bits = max(amax_bits, __float_as_int(y_s));
    }
    float amax = __int_as_float(amax_bits);
    amax = blockReduceMax<float>(amax);  // maxNanPropagate keeps NaN across threads
    if (tid == 0) s_inv_scale = epi.finalize_scale(amax, row);
    __syncthreads();
    inv_scale = s_inv_scale;
  }

  // Store pass: normalise, optional affine, epilogue.
  for (int64_t i = tid; i < N; i += stride) {
    acc_t v = static_cast<acc_t>(SRC[i]);
    if constexpr (kRMS) {
      v *= rstd;
    } else {
      v = (v - mean) * rstd;
    }
    if (gamma) v *= static_cast<acc_t>(gamma[i]);
    if constexpr (!kRMS) {
      if (beta) v += static_cast<acc_t>(beta[i]);
    }
    Y[i] = epi.store(v, inv_scale);
  }
}

// ---------------------------------------------------------------------------
// Vectorised flavour: 16-byte loads/stores. The launcher guarantees N is a
// multiple of kVecWidth<scalar_t> and every pointer is 16-byte aligned (see
// aligned16 / choose_vec). LayerNorm statistics use single-pass Welford (the
// same machinery as the verified layernorm_vec_kernel); RMSNorm uses a plain
// sum of squares. The store pass re-reads the row (mostly an L1/L2 hit).
// ---------------------------------------------------------------------------
template <typename scalar_t, typename acc_t, bool kRMS, bool kFusedAdd, typename Epi>
__global__ void norm_fwd_vec_kernel(const scalar_t* __restrict__ input,
                                    const scalar_t* residual_in,   // aliasable pair: see above
                                    scalar_t* residual_out,
                                    typename Epi::out_t* __restrict__ output,
                                    const scalar_t* __restrict__ gamma,
                                    const scalar_t* __restrict__ beta,
                                    acc_t* __restrict__ mean_out,
                                    acc_t* __restrict__ rstd_out,
                                    int64_t N,  // multiple of kVecWidth<scalar_t>
                                    acc_t eps,
                                    Epi epi) {
  using V = Vec<scalar_t>;
  // Same-size out_t (the None epilogue) stores whole output vectors; a
  // narrower out_t (fp8) stores element-wise below - Vec<out_t> would pack
  // the wrong element count for the input-vector-indexed loop.
  constexpr bool kSameSizeOut = sizeof(typename Epi::out_t) == sizeof(scalar_t);
  constexpr int kW = kVecWidth<scalar_t>;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t nvec = N / kW;

  const V* X = reinterpret_cast<const V*>(input + row * N);
  const V* R = kFusedAdd ? reinterpret_cast<const V*>(residual_in + row * N) : nullptr;
  V* Z = kFusedAdd ? reinterpret_cast<V*>(residual_out + row * N) : nullptr;
  const V* SRC = kFusedAdd ? Z : X;
  const V* G = reinterpret_cast<const V*>(gamma);  // may be null
  const V* B = reinterpret_cast<const V*>(beta);   // may be null

  __shared__ acc_t s_mean;
  __shared__ acc_t s_rstd;

  // Stats pass. LayerNorm: per-thread Welford merged across the block;
  // RMSNorm: sum of squares (a single blockReduceSum).
  Welford<acc_t> w;
  acc_t sq = 0;
  for (int64_t i = tid; i < nvec; i += stride) {
    V z;
    if constexpr (kFusedAdd) {
      const V x = X[i];
      const V r = R[i];
#pragma unroll
      for (int k = 0; k < kW; ++k) {
        z.v[k] = static_cast<scalar_t>(static_cast<acc_t>(x.v[k]) +
                                       static_cast<acc_t>(r.v[k]));
      }
      Z[i] = z;
    } else {
      z = X[i];
    }
#pragma unroll
    for (int k = 0; k < kW; ++k) {
      const acc_t v = static_cast<acc_t>(z.v[k]);  // fused-add: the ROUNDED value
      if constexpr (kRMS) {
        sq += v * v;
      } else {
        w.n += 1;
        const acc_t delta = v - w.mean;
        w.mean += delta / w.n;
        w.m2 += delta * (v - w.mean);
      }
    }
  }

  if constexpr (kRMS) {
    sq = blockReduceSum<acc_t>(sq);
    if (tid == 0) {
      s_rstd = rsqrt(sq / static_cast<acc_t>(N) + eps);
      if (rstd_out != nullptr) rstd_out[row] = s_rstd;
    }
  } else {
    w = blockReduceWelford<acc_t>(w);
    if (tid == 0) {
      s_mean = w.mean;
      s_rstd = rsqrt(w.m2 / static_cast<acc_t>(N) + eps);
      if (mean_out != nullptr) mean_out[row] = s_mean;
      if (rstd_out != nullptr) rstd_out[row] = s_rstd;
    }
  }
  __syncthreads();
  const acc_t mean = kRMS ? static_cast<acc_t>(0) : s_mean;
  const acc_t rstd = s_rstd;

  // Quantising epilogues: see the scalar kernel; identical structure with
  // vector loads.
  float inv_scale = epi.load_inv_scale();
  if constexpr (Epi::kNeedsRowMax) {
    __shared__ float s_inv_scale;
    // |y| >= 0, so IEEE ordering equals integer ordering on the raw bits, and
    // every NaN pattern compares above +inf: integer max over the bit
    // patterns is a branch-free NaN-PROPAGATING amax (fmaxf would drop NaN,
    // giving a poisoned row a tiny finite scale; torch.amax propagates).
    int amax_bits = 0;
    for (int64_t i = tid; i < nvec; i += stride) {
      const V z = SRC[i];
      V gv, bv;
      if (gamma) gv = G[i];
      if constexpr (!kRMS) {
        if (beta) bv = B[i];
      }
#pragma unroll
      for (int k = 0; k < kW; ++k) {
        acc_t v = static_cast<acc_t>(z.v[k]);
        if constexpr (kRMS) {
          v *= rstd;
        } else {
          v = (v - mean) * rstd;
        }
        if (gamma) v *= static_cast<acc_t>(gv.v[k]);
        if constexpr (!kRMS) {
          if (beta) v += static_cast<acc_t>(bv.v[k]);
        }
        const float y_s = fabsf(static_cast<float>(static_cast<scalar_t>(v)));
        amax_bits = max(amax_bits, __float_as_int(y_s));
      }
    }
    float amax = __int_as_float(amax_bits);
    amax = blockReduceMax<float>(amax);  // maxNanPropagate keeps NaN across threads
    if (tid == 0) s_inv_scale = epi.finalize_scale(amax, row);
    __syncthreads();
    inv_scale = s_inv_scale;
  }

  // Store pass.
  for (int64_t i = tid; i < nvec; i += stride) {
    const V z = SRC[i];
    V gv, bv;
    if (gamma) gv = G[i];
    if constexpr (!kRMS) {
      if (beta) bv = B[i];
    }
    typename Epi::out_t y_narrow[kW];
    V y_wide;
#pragma unroll
    for (int k = 0; k < kW; ++k) {
      acc_t v = static_cast<acc_t>(z.v[k]);
      if constexpr (kRMS) {
        v *= rstd;
      } else {
        v = (v - mean) * rstd;
      }
      if (gamma) v *= static_cast<acc_t>(gv.v[k]);
      if constexpr (!kRMS) {
        if (beta) v += static_cast<acc_t>(bv.v[k]);
      }
      if constexpr (kSameSizeOut) {
        // Same size implies same type in this library (EpiNone).
        y_wide.v[k] = epi.store(v, inv_scale);
      } else {
        y_narrow[k] = epi.store(v, inv_scale);
      }
    }
    if constexpr (kSameSizeOut) {
      reinterpret_cast<V*>(output + row * N)[i] = y_wide;
    } else {
#pragma unroll
      for (int k = 0; k < kW; ++k) {
        output[row * N + i * kW + k] = y_narrow[k];
      }
    }
  }
}

}  // namespace fused_norm
