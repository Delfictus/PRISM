// double_double.cu - GPU kernels for double-double (106-bit) arithmetic
// Implements Bailey's algorithms for extended precision on GPU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

// Double-double representation: value = hi + lo
// Maintains ~106 bits of precision (2 * 53 bits)
struct dd_real {
    double hi;
    double lo;
};

struct dd_complex {
    dd_real real;
    dd_real imag;
};

// Constants for double-double arithmetic
__constant__ double DD_EPS = 4.93038065763132e-32;  // 2^-106
__constant__ double DD_SPLIT_THRESH = 6.69692879491417e+299;  // 2^996

// ============================================================================
// Basic Double-Double Operations
// ============================================================================

// Knuth's 2Sum algorithm: Exact sum of two floating-point numbers
__device__ __forceinline__ dd_real two_sum(double a, double b) {
    dd_real result;
    result.hi = a + b;
    double v = result.hi - a;
    result.lo = (a - (result.hi - v)) + (b - v);
    return result;
}

// Quick-Two-Sum: Faster when |a| >= |b|
__device__ __forceinline__ dd_real quick_two_sum(double a, double b) {
    dd_real result;
    result.hi = a + b;
    result.lo = b - (result.hi - a);
    return result;
}

// Two-Product: Exact product of two floating-point numbers
__device__ __forceinline__ dd_real two_prod(double a, double b) {
    dd_real result;
    result.hi = a * b;

    // Veltkamp's splitting
    const double SPLIT = 134217729.0;  // 2^27 + 1
    double a_hi = a * SPLIT;
    a_hi = a_hi - (a_hi - a);
    double a_lo = a - a_hi;

    double b_hi = b * SPLIT;
    b_hi = b_hi - (b_hi - b);
    double b_lo = b - b_hi;

    result.lo = ((a_hi * b_hi - result.hi) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    return result;
}

// ============================================================================
// Double-Double Arithmetic Operations
// ============================================================================

// Addition: (a_hi + a_lo) + (b_hi + b_lo)
__device__ dd_real dd_add(dd_real a, dd_real b) {
    dd_real s = two_sum(a.hi, b.hi);
    double e = a.lo + b.lo;
    double v = s.hi + e;
    s.lo += (e - (v - s.hi));
    s.hi = v;

    // Renormalize
    dd_real result = quick_two_sum(s.hi, s.lo);
    return result;
}

// Subtraction: (a_hi + a_lo) - (b_hi + b_lo)
__device__ dd_real dd_sub(dd_real a, dd_real b) {
    dd_real s = two_sum(a.hi, -b.hi);
    double e = a.lo - b.lo;
    double v = s.hi + e;
    s.lo += (e - (v - s.hi));
    s.hi = v;

    // Renormalize
    dd_real result = quick_two_sum(s.hi, s.lo);
    return result;
}

// Multiplication: Bailey's algorithm
__device__ dd_real dd_mul(dd_real a, dd_real b) {
    dd_real p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo;
    p.lo += a.lo * b.hi;

    // Renormalize
    dd_real result = quick_two_sum(p.hi, p.lo);
    return result;
}

// Division: Newton-Raphson iteration
__device__ dd_real dd_div(dd_real a, dd_real b) {
    // Initial approximation
    double q0 = a.hi / b.hi;

    // First correction
    dd_real r = dd_sub(a, dd_mul(b, (dd_real){q0, 0.0}));
    double q1 = r.hi / b.hi;

    // Second correction
    r = dd_sub(r, dd_mul(b, (dd_real){q1, 0.0}));
    double q2 = r.hi / b.hi;

    // Combine corrections
    dd_real q = quick_two_sum(q0, q1);
    q = dd_add(q, (dd_real){q2, 0.0});

    return q;
}

// Square root: Newton-Raphson iteration
__device__ dd_real dd_sqrt(dd_real a) {
    if (a.hi == 0.0) return (dd_real){0.0, 0.0};
    if (a.hi < 0.0) return (dd_real){NAN, NAN};

    // Initial approximation
    double x0 = sqrt(a.hi);
    dd_real x = (dd_real){x0, 0.0};

    // Newton iterations: x = (x + a/x) / 2
    for (int i = 0; i < 3; i++) {
        dd_real ax = dd_div(a, x);
        x = dd_add(x, ax);
        x.hi *= 0.5;
        x.lo *= 0.5;
    }

    return x;
}

// ============================================================================
// Complex Double-Double Operations
// ============================================================================

__device__ dd_complex dd_complex_add(dd_complex a, dd_complex b) {
    dd_complex result;
    result.real = dd_add(a.real, b.real);
    result.imag = dd_add(a.imag, b.imag);
    return result;
}

__device__ dd_complex dd_complex_sub(dd_complex a, dd_complex b) {
    dd_complex result;
    result.real = dd_sub(a.real, b.real);
    result.imag = dd_sub(a.imag, b.imag);
    return result;
}

__device__ dd_complex dd_complex_mul(dd_complex a, dd_complex b) {
    dd_complex result;
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    dd_real ac = dd_mul(a.real, b.real);
    dd_real bd = dd_mul(a.imag, b.imag);
    dd_real ad = dd_mul(a.real, b.imag);
    dd_real bc = dd_mul(a.imag, b.real);

    result.real = dd_sub(ac, bd);
    result.imag = dd_add(ad, bc);
    return result;
}

__device__ dd_complex dd_complex_div(dd_complex a, dd_complex b) {
    // (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
    dd_real c2 = dd_mul(b.real, b.real);
    dd_real d2 = dd_mul(b.imag, b.imag);
    dd_real denominator = dd_add(c2, d2);

    dd_real ac = dd_mul(a.real, b.real);
    dd_real bd = dd_mul(a.imag, b.imag);
    dd_real bc = dd_mul(a.imag, b.real);
    dd_real ad = dd_mul(a.real, b.imag);

    dd_complex result;
    result.real = dd_div(dd_add(ac, bd), denominator);
    result.imag = dd_div(dd_sub(bc, ad), denominator);
    return result;
}

// Magnitude squared: |z|² = real² + imag²
__device__ dd_real dd_complex_abs2(dd_complex z) {
    dd_real r2 = dd_mul(z.real, z.real);
    dd_real i2 = dd_mul(z.imag, z.imag);
    return dd_add(r2, i2);
}

// Complex exponential: e^(a+bi) = e^a * (cos(b) + i*sin(b))
__device__ dd_complex dd_complex_exp(dd_complex z) {
    // For now, use double precision for transcendental functions
    // TODO: Implement dd_exp, dd_cos, dd_sin for full precision
    double exp_real = exp(z.real.hi);
    double cos_imag = cos(z.imag.hi);
    double sin_imag = sin(z.imag.hi);

    dd_complex result;
    result.real = (dd_real){exp_real * cos_imag, 0.0};
    result.imag = (dd_real){exp_real * sin_imag, 0.0};
    return result;
}

// ============================================================================
// Kernel Functions for Array Operations
// ============================================================================

// Element-wise addition of double-double arrays
__global__ void dd_array_add(
    dd_real* __restrict__ result,
    const dd_real* __restrict__ a,
    const dd_real* __restrict__ b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = dd_add(a[idx], b[idx]);
    }
}

// Matrix-vector multiplication with double-double precision
__global__ void dd_matrix_vector_mul(
    dd_complex* __restrict__ y,
    const dd_complex* __restrict__ A,
    const dd_complex* __restrict__ x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dd_complex sum = {0.0, 0.0, 0.0, 0.0};

        for (int j = 0; j < n; j++) {
            dd_complex aij = A[i * n + j];
            dd_complex xj = x[j];
            dd_complex prod = dd_complex_mul(aij, xj);
            sum = dd_complex_add(sum, prod);
        }

        y[i] = sum;
    }
}

// Kahan summation for improved accuracy in reductions
__device__ double kahan_sum(const double* data, int n) {
    double sum = 0.0;
    double c = 0.0;  // Compensation for lost digits

    for (int i = 0; i < n; i++) {
        double y = data[i] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

// Deterministic parallel reduction with double-double precision
__global__ void dd_deterministic_reduce(
    dd_real* __restrict__ output,
    const dd_real* __restrict__ input,
    int n
) {
    extern __shared__ dd_real sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    // Load and perform first reduction
    dd_real mySum = {0.0, 0.0};
    if (i < n) {
        mySum = input[i];
        if (i + blockDim.x < n) {
            mySum = dd_add(mySum, input[i + blockDim.x]);
        }
    }

    sdata[tid] = mySum;
    __syncthreads();

    // Tree reduction with deterministic ordering
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata[tid] = dd_add(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

// Convert double to double-double
__device__ __host__ dd_real double_to_dd(double x) {
    return (dd_real){x, 0.0};
}

// Convert double-double to double (with rounding)
__device__ __host__ double dd_to_double(dd_real x) {
    return x.hi + x.lo;
}

// Print double-double value (for debugging)
__device__ void dd_print(const char* name, dd_real x) {
    printf("%s: %.17e + %.17e (total: %.32e)\n",
           name, x.hi, x.lo, x.hi + x.lo);
}

// ============================================================================
// Test Kernel
// ============================================================================

__global__ void test_dd_arithmetic() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("=== Double-Double Arithmetic Test ===\n");

        // Test addition: 1/3 + 1/7
        dd_real a = dd_div(double_to_dd(1.0), double_to_dd(3.0));
        dd_real b = dd_div(double_to_dd(1.0), double_to_dd(7.0));
        dd_real sum = dd_add(a, b);
        dd_print("1/3 + 1/7", sum);

        // Test multiplication: π * e (approximately)
        dd_real pi = {3.141592653589793, 1.2246467991473532e-16};
        dd_real e = {2.718281828459045, 1.4456468917292502e-16};
        dd_real product = dd_mul(pi, e);
        dd_print("π * e", product);

        // Test complex operations
        dd_complex z1 = {{1.0, 0.0}, {1.0, 0.0}};  // 1 + i
        dd_complex z2 = {{2.0, 0.0}, {-1.0, 0.0}}; // 2 - i
        dd_complex z_product = dd_complex_mul(z1, z2);
        printf("(1+i) * (2-i) = %.17e + %.17e i\n",
               z_product.real.hi, z_product.imag.hi);
    }
}

// Host wrapper for testing
extern "C" void run_dd_test() {
    test_dd_arithmetic<<<1, 1>>>();
    cudaDeviceSynchronize();
}