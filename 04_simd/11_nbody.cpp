#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
    const int N = 8;
    float x[N], y[N], m[N], fx[N], fy[N];
    for (int i = 0; i < N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
        fx[i] = fy[i] = 0;
    }
    for (int i = 0; i < N; i += 8) {
        __m512 xi = _mm512_load_ps(x + i);
        __m512 yi = _mm512_load_ps(y + i);
        __m512 mi = _mm512_load_ps(m + i);
        __m512 fxi = _mm512_setzero_ps();
        __m512 fyi = _mm512_setzero_ps();

        for (int j = 0; j < N; j += 8) {
            __m512 xj = _mm512_load_ps(x + j);
            __m512 yj = _mm512_load_ps(y + j);
            __m512 mj = _mm512_load_ps(m + j);

            __m512 rx = _mm512_sub_ps(xi, xj);
            __m512 ry = _mm512_sub_ps(yi, yj);
            __m512 r2 = _mm512_fmadd_ps(rx, rx, _mm512_mul_ps(ry, ry));
            __m512 r = _mm512_rsqrt14_ps(r2);

            __mmask16 mask = _mm512_cmp_ps_mask(r2, _mm512_set1_ps(1e-6), _CMP_GT_OQ);
            __m512 r3 = _mm512_maskz_mul_ps(mask, _mm512_mul_ps(r, _mm512_mul_ps(r, r)), mj);

            fxi = _mm512_fmadd_ps(rx, r3, fxi);
            fyi = _mm512_fmadd_ps(ry, r3, fyi);
        }
        _mm512_store_ps(fx + i, fxi);
        _mm512_store_ps(fy + i, fyi);
    }
    for (int i = 0; i < N; i++) {
        printf("%d %g %g\n", i, fx[i], fy[i]);
    }
    return 0;
}
