#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {
constexpr float scale = 0x1p39f;
constexpr float low_base = 0x1p16f;
constexpr float high_unit = 0x1p-23f;
constexpr float low_unit = 0x1p-39f;

struct Chebyshev {
    float value;
    float next;
};

template <int degree>
inline Chebyshev chebyshev(float x) {
    if constexpr (degree == 1) {
        return {x, 2.f*x*x - 1.f};
    } else {
        const auto [a, b] = chebyshev<degree/2>(x);
        const float mixed = 2.f*a*b - x;
        if constexpr (degree % 2) return {mixed, 2.f*b*b - 1.f};
        return {2.f*a*a - 1.f, mixed};
    }
}

template <int degree>
inline float gate(float x) {
    return std::clamp((1.f + chebyshev<degree>(x).value)*0.5f, 0.f, 1.f);
}

inline void accumulate(float term, float& high, float& low) {
    const float units = std::nearbyint(term*scale);
    const float upper = static_cast<float>(static_cast<int>(units/low_base));
    high += upper;
    low += units - upper*low_base;
}
}

extern "C" void gates(int n, const float* pairs, const float* state,
                       float* a, float* b, float* c, float* d) {
    const std::size_t cube = static_cast<std::size_t>(n)*n*n;
    #pragma omp parallel for schedule(static)
    for (std::size_t row = 0; row < cube; ++row) {
        const int u = row/(n*n), v = (row/n)%n, w = row%n;
        const std::size_t offset = row*n;
        const float first = (pairs[u*n+v] + 3.f*pairs[u*n+w]) + 9.f*pairs[v*n+w];
        #pragma omp simd
        for (int z = 0; z < n; ++z) {
            const float base = (((first + 27.f*pairs[u*n+z]) + 81.f*pairs[v*n+z])
                               + 243.f*pairs[w*n+z])/364.f;
            const float current = state ? state[offset+z] : base;
            const float x = 2.f*(base + (1.f-base)*current) - 1.f;
            a[offset+z] = gate<769>(x);
            b[offset+z] = gate<997>(x);
            c[offset+z] = gate<1237>(x);
            d[offset+z] = gate<1597>(x);
        }
    }
}

extern "C" void contract(int n, const int* offsets, const int* neighbors,
                          const float* first_walk, const float* second_walk,
                          const float* a, const float* b, const float* c,
                          const float* d, float* result) {
    const std::size_t square = static_cast<std::size_t>(n)*n;
    const std::size_t cube = square*n;
    #pragma omp parallel
    {
        std::vector<float> high(n), low(n);
        #pragma omp for schedule(dynamic, 8)
        for (std::size_t pair = 0; pair < square; ++pair) {
            const int u = pair/n, v = pair%n;
            for (int w = 0; w < n; ++w) {
                const std::size_t row = pair*square + w*n;
                std::fill(high.begin(), high.end(), 0.f);
                std::fill(low.begin(), low.end(), 0.f);
                for (int k = offsets[u]; k < offsets[u+1]; ++k) {
                    const int x = neighbors[k];
                    const float wx = first_walk[u*n+x], vx = second_walk[v*n+x];
                    if (vx == 0.f) continue;
                    const float* ar = a + x*cube + v*square + w*n;
                    const float* br = b + u*cube + x*square + w*n;
                    const float* cr = c + pair*square + x*n;
                    const float dr = d[row+x];
                    #pragma omp simd
                    for (int z = 0; z < n; ++z) {
                        const float joint = (((ar[z]*br[z])*cr[z])*dr);
                        accumulate((joint*wx)*vx, high[z], low[z]);
                    }
                }
                #pragma omp simd
                for (int z = 0; z < n; ++z)
                    result[row+z] = high[z]*high_unit + low[z]*low_unit;
            }
        }
    }
}

extern "C" void static_joint(int n, const int* offsets, const int* neighbors,
                              const float* walk, float* result) {
    const std::size_t square = static_cast<std::size_t>(n)*n;
    #pragma omp parallel
    {
        std::vector<float> high(n), low(n);
        #pragma omp for schedule(static)
        for (std::size_t pair = 0; pair < square; ++pair) {
            const int u = pair/n, v = pair%n;
            for (int w = 0; w < n; ++w) {
                const std::size_t row = pair*square + w*n;
                std::fill(high.begin(), high.end(), 0.f);
                std::fill(low.begin(), low.end(), 0.f);
                for (int k = offsets[u]; k < offsets[u+1]; ++k) {
                    const int x = neighbors[k];
                    const float term = (walk[u*n+x]*walk[v*n+x])*walk[w*n+x];
                    if (term == 0.f) continue;
                    #pragma omp simd
                    for (int z = 0; z < n; ++z)
                        accumulate(term*walk[z*n+x], high[z], low[z]);
                }
                #pragma omp simd
                for (int z = 0; z < n; ++z)
                    result[row+z] = high[z]*high_unit + low[z]*low_unit;
            }
        }
    }
}
