// 16-byte vector types and the vectorisability gate shared by every kernel.
// Extracted verbatim from layernorm_cuda_kernel.cu (v0.3.0); only the
// namespace and the named helper functions are new.
#pragma once

#include <cstdint>

namespace fused_norm {

// Elements per 16-byte vector load for each dtype.
template <typename scalar_t>
constexpr int kVecWidth = 16 / sizeof(scalar_t);

template <typename scalar_t>
struct alignas(16) Vec {
  scalar_t v[kVecWidth<scalar_t>];
};

// Contiguity alone does not guarantee alignment: a contiguous 1-D slice like
// base[1:] keeps its storage offset, so its data_ptr can sit 4 bytes into an
// allocation; PyTorch's own vectorised kernel makes the same runtime check.
inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

}  // namespace fused_norm
