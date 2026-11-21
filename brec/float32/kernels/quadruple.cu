using Index = unsigned long long;

struct Chebyshev { float value, next; };

template <int degree>
__device__ __forceinline__ Chebyshev chebyshev(float x) {
    if constexpr (degree == 1) {
        return {x, 2.f*x*x - 1.f};
    } else {
        const auto prev = chebyshev<degree/2>(x);
        const float a = prev.value, b = prev.next;
        const float mixed = 2.f*a*b - x;
        if constexpr (degree % 2) return {mixed, 2.f*b*b - 1.f};
        return {2.f*a*a - 1.f, mixed};
    }
}

template <int degree>
__device__ __forceinline__ float gate(float x) {
    return fminf(1.f, fmaxf(0.f, (1.f + chebyshev<degree>(x).value)*0.5f));
}

__device__ __forceinline__ void accumulate(float term, float& high, float& low) {
    const float units = rintf(term*0x1p39f);
    const float upper = floorf(units*0x1p-16f);
    high += upper;
    low += units - upper*0x1p16f;
}

extern "C" __global__ void gates(const float* pairs, const float* state, int n,
                                 Index size, int initialized, float* a, float* b,
                                 float* c, float* d) {
    const Index stride = blockDim.x*gridDim.x;
    for (Index i = blockIdx.x*blockDim.x+threadIdx.x; i < size; i += stride) {
        const int z = i%n, w = (i/n)%n, v = (i/(n*n))%n, u = i/(n*n*n);
        const float head = (pairs[u*n+v] + 3.f*pairs[u*n+w]) + 9.f*pairs[v*n+w];
        const float base = (((head + 27.f*pairs[u*n+z]) + 81.f*pairs[v*n+z])
                            + 243.f*pairs[w*n+z])/364.f;
        const float previous = initialized ? state[i] : base;
        const float x = 2.f*(base + (1.f-base)*previous) - 1.f;
        a[i] = gate<769>(x);
        b[i] = gate<997>(x);
        c[i] = gate<1237>(x);
        d[i] = gate<1597>(x);
    }
}

extern "C" __global__ void contract(const int* offsets, const int* neighbors,
                                    const float* first, const float* second,
                                    const float* a, const float* b, const float* c,
                                    const float* d, int n, Index size, float* state) {
    const int square = n*n, cube = n*square;
    const Index stride = blockDim.x*gridDim.x;
    for (Index i = blockIdx.x*blockDim.x+threadIdx.x; i < size; i += stride) {
        const int z = i%n, w = (i/n)%n, v = (i/square)%n, u = i/cube;
        const Index ai = v*square + w*n + z, bi = u*cube + w*n + z;
        const Index ci = u*cube + v*square + z, di = i-z;
        float high = 0.f, low = 0.f;
        for (int k = offsets[u]; k < offsets[u+1]; ++k) {
            const int x = neighbors[k];
            const float joint = ((a[ai+x*cube]*b[bi+x*square])*c[ci+x*n])*d[di+x];
            accumulate((joint*first[u*n+x])*second[v*n+x], high, low);
        }
        state[i] = high*0x1p-23f + low*0x1p-39f;
    }
}

extern "C" __global__ void static_joint(const int* offsets, const int* neighbors,
                                       const float* walk, int n, Index size, float* state) {
    const Index stride = blockDim.x*gridDim.x;
    for (Index i = blockIdx.x*blockDim.x+threadIdx.x; i < size; i += stride) {
        const int z = i%n, w = (i/n)%n, v = (i/(n*n))%n, u = i/(n*n*n);
        float high = 0.f, low = 0.f;
        for (int k = offsets[u]; k < offsets[u+1]; ++k) {
            const int x = neighbors[k];
            const float term = ((walk[u*n+x]*walk[v*n+x])*walk[w*n+x])*walk[z*n+x];
            accumulate(term, high, low);
        }
        state[i] = high*0x1p-23f + low*0x1p-39f;
    }
}
