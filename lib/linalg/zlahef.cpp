#ifdef __cplusplus
extern "C" {
#endif
#include "lmp_f2c.h"
static doublecomplex c_b1 = {1., 0.};
static integer c__1 = 1;
int zlahef_(char *uplo, integer *n, integer *nb, integer *kb, doublecomplex *a, integer *lda,
            integer *ipiv, doublecomplex *w, integer *ldw, integer *info, ftnlen uplo_len)
{
    integer a_dim1, a_offset, w_dim1, w_offset, i__1, i__2, i__3, i__4, i__5;
    doublereal d__1, d__2, d__3, d__4;
    doublecomplex z__1, z__2, z__3, z__4;
    double sqrt(doublereal), d_lmp_imag(doublecomplex *);
    void d_lmp_cnjg(doublecomplex *, doublecomplex *),
        z_lmp_div(doublecomplex *, doublecomplex *, doublecomplex *);
    integer j, k;
    doublereal t, r1;
    doublecomplex d11, d21, d22;
    integer jb, jj, kk, jp, kp, kw, kkw, imax, jmax;
    doublereal alpha;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    extern int zgemm_(char *, char *, integer *, integer *, integer *, doublecomplex *,
                      doublecomplex *, integer *, doublecomplex *, integer *, doublecomplex *,
                      doublecomplex *, integer *, ftnlen, ftnlen);
    integer kstep;
    extern int zgemv_(char *, integer *, integer *, doublecomplex *, doublecomplex *, integer *,
                      doublecomplex *, integer *, doublecomplex *, doublecomplex *, integer *,
                      ftnlen),
        zcopy_(integer *, doublecomplex *, integer *, doublecomplex *, integer *),
        zswap_(integer *, doublecomplex *, integer *, doublecomplex *, integer *);
    doublereal absakk;
    extern int zdscal_(integer *, doublereal *, doublecomplex *, integer *);
    doublereal colmax;
    extern int zlacgv_(integer *, doublecomplex *, integer *);
    extern integer izamax_(integer *, doublecomplex *, integer *);
    doublereal rowmax;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    w_dim1 = *ldw;
    w_offset = 1 + w_dim1;
    w -= w_offset;
    *info = 0;
    alpha = (sqrt(17.) + 1.) / 8.;
    if (lsame_(uplo, (char *)"U", (ftnlen)1, (ftnlen)1)) {
        k = *n;
    L10:
        kw = *nb + k - *n;
        if (k <= *n - *nb + 1 && *nb < *n || k < 1) {
            goto L30;
        }
        kstep = 1;
        i__1 = k - 1;
        zcopy_(&i__1, &a[k * a_dim1 + 1], &c__1, &w[kw * w_dim1 + 1], &c__1);
        i__1 = k + kw * w_dim1;
        i__2 = k + k * a_dim1;
        d__1 = a[i__2].r;
        w[i__1].r = d__1, w[i__1].i = 0.;
        if (k < *n) {
            i__1 = *n - k;
            z__1.r = -1., z__1.i = -0.;
            zgemv_((char *)"No transpose", &k, &i__1, &z__1, &a[(k + 1) * a_dim1 + 1], lda,
                   &w[k + (kw + 1) * w_dim1], ldw, &c_b1, &w[kw * w_dim1 + 1], &c__1, (ftnlen)12);
            i__1 = k + kw * w_dim1;
            i__2 = k + kw * w_dim1;
            d__1 = w[i__2].r;
            w[i__1].r = d__1, w[i__1].i = 0.;
        }
        i__1 = k + kw * w_dim1;
        absakk = (d__1 = w[i__1].r, abs(d__1));
        if (k > 1) {
            i__1 = k - 1;
            imax = izamax_(&i__1, &w[kw * w_dim1 + 1], &c__1);
            i__1 = imax + kw * w_dim1;
            colmax =
                (d__1 = w[i__1].r, abs(d__1)) + (d__2 = d_lmp_imag(&w[imax + kw * w_dim1]), abs(d__2));
        } else {
            colmax = 0.;
        }
        if (max(absakk, colmax) == 0.) {
            if (*info == 0) {
                *info = k;
            }
            kp = k;
            i__1 = k + k * a_dim1;
            i__2 = k + k * a_dim1;
            d__1 = a[i__2].r;
            a[i__1].r = d__1, a[i__1].i = 0.;
        } else {
            if (absakk >= alpha * colmax) {
                kp = k;
            } else {
                i__1 = imax - 1;
                zcopy_(&i__1, &a[imax * a_dim1 + 1], &c__1, &w[(kw - 1) * w_dim1 + 1], &c__1);
                i__1 = imax + (kw - 1) * w_dim1;
                i__2 = imax + imax * a_dim1;
                d__1 = a[i__2].r;
                w[i__1].r = d__1, w[i__1].i = 0.;
                i__1 = k - imax;
                zcopy_(&i__1, &a[imax + (imax + 1) * a_dim1], lda, &w[imax + 1 + (kw - 1) * w_dim1],
                       &c__1);
                i__1 = k - imax;
                zlacgv_(&i__1, &w[imax + 1 + (kw - 1) * w_dim1], &c__1);
                if (k < *n) {
                    i__1 = *n - k;
                    z__1.r = -1., z__1.i = -0.;
                    zgemv_((char *)"No transpose", &k, &i__1, &z__1, &a[(k + 1) * a_dim1 + 1], lda,
                           &w[imax + (kw + 1) * w_dim1], ldw, &c_b1, &w[(kw - 1) * w_dim1 + 1],
                           &c__1, (ftnlen)12);
                    i__1 = imax + (kw - 1) * w_dim1;
                    i__2 = imax + (kw - 1) * w_dim1;
                    d__1 = w[i__2].r;
                    w[i__1].r = d__1, w[i__1].i = 0.;
                }
                i__1 = k - imax;
                jmax = imax + izamax_(&i__1, &w[imax + 1 + (kw - 1) * w_dim1], &c__1);
                i__1 = jmax + (kw - 1) * w_dim1;
                rowmax = (d__1 = w[i__1].r, abs(d__1)) +
                         (d__2 = d_lmp_imag(&w[jmax + (kw - 1) * w_dim1]), abs(d__2));
                if (imax > 1) {
                    i__1 = imax - 1;
                    jmax = izamax_(&i__1, &w[(kw - 1) * w_dim1 + 1], &c__1);
                    i__1 = jmax + (kw - 1) * w_dim1;
                    d__3 = rowmax, d__4 = (d__1 = w[i__1].r, abs(d__1)) +
                                          (d__2 = d_lmp_imag(&w[jmax + (kw - 1) * w_dim1]), abs(d__2));
                    rowmax = max(d__3, d__4);
                }
                if (absakk >= alpha * colmax * (colmax / rowmax)) {
                    kp = k;
                } else {
                    i__1 = imax + (kw - 1) * w_dim1;
                    if ((d__1 = w[i__1].r, abs(d__1)) >= alpha * rowmax) {
                        kp = imax;
                        zcopy_(&k, &w[(kw - 1) * w_dim1 + 1], &c__1, &w[kw * w_dim1 + 1], &c__1);
                    } else {
                        kp = imax;
                        kstep = 2;
                    }
                }
            }
            kk = k - kstep + 1;
            kkw = *nb + kk - *n;
            if (kp != kk) {
                i__1 = kp + kp * a_dim1;
                i__2 = kk + kk * a_dim1;
                d__1 = a[i__2].r;
                a[i__1].r = d__1, a[i__1].i = 0.;
                i__1 = kk - 1 - kp;
                zcopy_(&i__1, &a[kp + 1 + kk * a_dim1], &c__1, &a[kp + (kp + 1) * a_dim1], lda);
                i__1 = kk - 1 - kp;
                zlacgv_(&i__1, &a[kp + (kp + 1) * a_dim1], lda);
                if (kp > 1) {
                    i__1 = kp - 1;
                    zcopy_(&i__1, &a[kk * a_dim1 + 1], &c__1, &a[kp * a_dim1 + 1], &c__1);
                }
                if (k < *n) {
                    i__1 = *n - k;
                    zswap_(&i__1, &a[kk + (k + 1) * a_dim1], lda, &a[kp + (k + 1) * a_dim1], lda);
                }
                i__1 = *n - kk + 1;
                zswap_(&i__1, &w[kk + kkw * w_dim1], ldw, &w[kp + kkw * w_dim1], ldw);
            }
            if (kstep == 1) {
                zcopy_(&k, &w[kw * w_dim1 + 1], &c__1, &a[k * a_dim1 + 1], &c__1);
                if (k > 1) {
                    i__1 = k + k * a_dim1;
                    r1 = 1. / a[i__1].r;
                    i__1 = k - 1;
                    zdscal_(&i__1, &r1, &a[k * a_dim1 + 1], &c__1);
                    i__1 = k - 1;
                    zlacgv_(&i__1, &w[kw * w_dim1 + 1], &c__1);
                }
            } else {
                if (k > 2) {
                    i__1 = k - 1 + kw * w_dim1;
                    d21.r = w[i__1].r, d21.i = w[i__1].i;
                    d_lmp_cnjg(&z__2, &d21);
                    z_lmp_div(&z__1, &w[k + kw * w_dim1], &z__2);
                    d11.r = z__1.r, d11.i = z__1.i;
                    z_lmp_div(&z__1, &w[k - 1 + (kw - 1) * w_dim1], &d21);
                    d22.r = z__1.r, d22.i = z__1.i;
                    z__1.r = d11.r * d22.r - d11.i * d22.i, z__1.i = d11.r * d22.i + d11.i * d22.r;
                    t = 1. / (z__1.r - 1.);
                    z__2.r = t, z__2.i = 0.;
                    z_lmp_div(&z__1, &z__2, &d21);
                    d21.r = z__1.r, d21.i = z__1.i;
                    i__1 = k - 2;
                    for (j = 1; j <= i__1; ++j) {
                        i__2 = j + (k - 1) * a_dim1;
                        i__3 = j + (kw - 1) * w_dim1;
                        z__3.r = d11.r * w[i__3].r - d11.i * w[i__3].i,
                        z__3.i = d11.r * w[i__3].i + d11.i * w[i__3].r;
                        i__4 = j + kw * w_dim1;
                        z__2.r = z__3.r - w[i__4].r, z__2.i = z__3.i - w[i__4].i;
                        z__1.r = d21.r * z__2.r - d21.i * z__2.i,
                        z__1.i = d21.r * z__2.i + d21.i * z__2.r;
                        a[i__2].r = z__1.r, a[i__2].i = z__1.i;
                        i__2 = j + k * a_dim1;
                        d_lmp_cnjg(&z__2, &d21);
                        i__3 = j + kw * w_dim1;
                        z__4.r = d22.r * w[i__3].r - d22.i * w[i__3].i,
                        z__4.i = d22.r * w[i__3].i + d22.i * w[i__3].r;
                        i__4 = j + (kw - 1) * w_dim1;
                        z__3.r = z__4.r - w[i__4].r, z__3.i = z__4.i - w[i__4].i;
                        z__1.r = z__2.r * z__3.r - z__2.i * z__3.i,
                        z__1.i = z__2.r * z__3.i + z__2.i * z__3.r;
                        a[i__2].r = z__1.r, a[i__2].i = z__1.i;
                    }
                }
                i__1 = k - 1 + (k - 1) * a_dim1;
                i__2 = k - 1 + (kw - 1) * w_dim1;
                a[i__1].r = w[i__2].r, a[i__1].i = w[i__2].i;
                i__1 = k - 1 + k * a_dim1;
                i__2 = k - 1 + kw * w_dim1;
                a[i__1].r = w[i__2].r, a[i__1].i = w[i__2].i;
                i__1 = k + k * a_dim1;
                i__2 = k + kw * w_dim1;
                a[i__1].r = w[i__2].r, a[i__1].i = w[i__2].i;
                i__1 = k - 1;
                zlacgv_(&i__1, &w[kw * w_dim1 + 1], &c__1);
                i__1 = k - 2;
                zlacgv_(&i__1, &w[(kw - 1) * w_dim1 + 1], &c__1);
            }
        }
        if (kstep == 1) {
            ipiv[k] = kp;
        } else {
            ipiv[k] = -kp;
            ipiv[k - 1] = -kp;
        }
        k -= kstep;
        goto L10;
    L30:
        i__1 = -(*nb);
        for (j = (k - 1) / *nb * *nb + 1; i__1 < 0 ? j >= 1 : j <= 1; j += i__1) {
            i__2 = *nb, i__3 = k - j + 1;
            jb = min(i__2, i__3);
            i__2 = j + jb - 1;
            for (jj = j; jj <= i__2; ++jj) {
                i__3 = jj + jj * a_dim1;
                i__4 = jj + jj * a_dim1;
                d__1 = a[i__4].r;
                a[i__3].r = d__1, a[i__3].i = 0.;
                i__3 = jj - j + 1;
                i__4 = *n - k;
                z__1.r = -1., z__1.i = -0.;
                zgemv_((char *)"No transpose", &i__3, &i__4, &z__1, &a[j + (k + 1) * a_dim1], lda,
                       &w[jj + (kw + 1) * w_dim1], ldw, &c_b1, &a[j + jj * a_dim1], &c__1,
                       (ftnlen)12);
                i__3 = jj + jj * a_dim1;
                i__4 = jj + jj * a_dim1;
                d__1 = a[i__4].r;
                a[i__3].r = d__1, a[i__3].i = 0.;
            }
            i__2 = j - 1;
            i__3 = *n - k;
            z__1.r = -1., z__1.i = -0.;
            zgemm_((char *)"No transpose", (char *)"Transpose", &i__2, &jb, &i__3, &z__1, &a[(k + 1) * a_dim1 + 1],
                   lda, &w[j + (kw + 1) * w_dim1], ldw, &c_b1, &a[j * a_dim1 + 1], lda, (ftnlen)12,
                   (ftnlen)9);
        }
        j = k + 1;
    L60:
        jj = j;
        jp = ipiv[j];
        if (jp < 0) {
            jp = -jp;
            ++j;
        }
        ++j;
        if (jp != jj && j <= *n) {
            i__1 = *n - j + 1;
            zswap_(&i__1, &a[jp + j * a_dim1], lda, &a[jj + j * a_dim1], lda);
        }
        if (j < *n) {
            goto L60;
        }
        *kb = *n - k;
    } else {
        k = 1;
    L70:
        if (k >= *nb && *nb < *n || k > *n) {
            goto L90;
        }
        kstep = 1;
        i__1 = k + k * w_dim1;
        i__2 = k + k * a_dim1;
        d__1 = a[i__2].r;
        w[i__1].r = d__1, w[i__1].i = 0.;
        if (k < *n) {
            i__1 = *n - k;
            zcopy_(&i__1, &a[k + 1 + k * a_dim1], &c__1, &w[k + 1 + k * w_dim1], &c__1);
        }
        i__1 = *n - k + 1;
        i__2 = k - 1;
        z__1.r = -1., z__1.i = -0.;
        zgemv_((char *)"No transpose", &i__1, &i__2, &z__1, &a[k + a_dim1], lda, &w[k + w_dim1], ldw, &c_b1,
               &w[k + k * w_dim1], &c__1, (ftnlen)12);
        i__1 = k + k * w_dim1;
        i__2 = k + k * w_dim1;
        d__1 = w[i__2].r;
        w[i__1].r = d__1, w[i__1].i = 0.;
        i__1 = k + k * w_dim1;
        absakk = (d__1 = w[i__1].r, abs(d__1));
        if (k < *n) {
            i__1 = *n - k;
            imax = k + izamax_(&i__1, &w[k + 1 + k * w_dim1], &c__1);
            i__1 = imax + k * w_dim1;
            colmax =
                (d__1 = w[i__1].r, abs(d__1)) + (d__2 = d_lmp_imag(&w[imax + k * w_dim1]), abs(d__2));
        } else {
            colmax = 0.;
        }
        if (max(absakk, colmax) == 0.) {
            if (*info == 0) {
                *info = k;
            }
            kp = k;
            i__1 = k + k * a_dim1;
            i__2 = k + k * a_dim1;
            d__1 = a[i__2].r;
            a[i__1].r = d__1, a[i__1].i = 0.;
        } else {
            if (absakk >= alpha * colmax) {
                kp = k;
            } else {
                i__1 = imax - k;
                zcopy_(&i__1, &a[imax + k * a_dim1], lda, &w[k + (k + 1) * w_dim1], &c__1);
                i__1 = imax - k;
                zlacgv_(&i__1, &w[k + (k + 1) * w_dim1], &c__1);
                i__1 = imax + (k + 1) * w_dim1;
                i__2 = imax + imax * a_dim1;
                d__1 = a[i__2].r;
                w[i__1].r = d__1, w[i__1].i = 0.;
                if (imax < *n) {
                    i__1 = *n - imax;
                    zcopy_(&i__1, &a[imax + 1 + imax * a_dim1], &c__1,
                           &w[imax + 1 + (k + 1) * w_dim1], &c__1);
                }
                i__1 = *n - k + 1;
                i__2 = k - 1;
                z__1.r = -1., z__1.i = -0.;
                zgemv_((char *)"No transpose", &i__1, &i__2, &z__1, &a[k + a_dim1], lda, &w[imax + w_dim1],
                       ldw, &c_b1, &w[k + (k + 1) * w_dim1], &c__1, (ftnlen)12);
                i__1 = imax + (k + 1) * w_dim1;
                i__2 = imax + (k + 1) * w_dim1;
                d__1 = w[i__2].r;
                w[i__1].r = d__1, w[i__1].i = 0.;
                i__1 = imax - k;
                jmax = k - 1 + izamax_(&i__1, &w[k + (k + 1) * w_dim1], &c__1);
                i__1 = jmax + (k + 1) * w_dim1;
                rowmax = (d__1 = w[i__1].r, abs(d__1)) +
                         (d__2 = d_lmp_imag(&w[jmax + (k + 1) * w_dim1]), abs(d__2));
                if (imax < *n) {
                    i__1 = *n - imax;
                    jmax = imax + izamax_(&i__1, &w[imax + 1 + (k + 1) * w_dim1], &c__1);
                    i__1 = jmax + (k + 1) * w_dim1;
                    d__3 = rowmax, d__4 = (d__1 = w[i__1].r, abs(d__1)) +
                                          (d__2 = d_lmp_imag(&w[jmax + (k + 1) * w_dim1]), abs(d__2));
                    rowmax = max(d__3, d__4);
                }
                if (absakk >= alpha * colmax * (colmax / rowmax)) {
                    kp = k;
                } else {
                    i__1 = imax + (k + 1) * w_dim1;
                    if ((d__1 = w[i__1].r, abs(d__1)) >= alpha * rowmax) {
                        kp = imax;
                        i__1 = *n - k + 1;
                        zcopy_(&i__1, &w[k + (k + 1) * w_dim1], &c__1, &w[k + k * w_dim1], &c__1);
                    } else {
                        kp = imax;
                        kstep = 2;
                    }
                }
            }
            kk = k + kstep - 1;
            if (kp != kk) {
                i__1 = kp + kp * a_dim1;
                i__2 = kk + kk * a_dim1;
                d__1 = a[i__2].r;
                a[i__1].r = d__1, a[i__1].i = 0.;
                i__1 = kp - kk - 1;
                zcopy_(&i__1, &a[kk + 1 + kk * a_dim1], &c__1, &a[kp + (kk + 1) * a_dim1], lda);
                i__1 = kp - kk - 1;
                zlacgv_(&i__1, &a[kp + (kk + 1) * a_dim1], lda);
                if (kp < *n) {
                    i__1 = *n - kp;
                    zcopy_(&i__1, &a[kp + 1 + kk * a_dim1], &c__1, &a[kp + 1 + kp * a_dim1], &c__1);
                }
                if (k > 1) {
                    i__1 = k - 1;
                    zswap_(&i__1, &a[kk + a_dim1], lda, &a[kp + a_dim1], lda);
                }
                zswap_(&kk, &w[kk + w_dim1], ldw, &w[kp + w_dim1], ldw);
            }
            if (kstep == 1) {
                i__1 = *n - k + 1;
                zcopy_(&i__1, &w[k + k * w_dim1], &c__1, &a[k + k * a_dim1], &c__1);
                if (k < *n) {
                    i__1 = k + k * a_dim1;
                    r1 = 1. / a[i__1].r;
                    i__1 = *n - k;
                    zdscal_(&i__1, &r1, &a[k + 1 + k * a_dim1], &c__1);
                    i__1 = *n - k;
                    zlacgv_(&i__1, &w[k + 1 + k * w_dim1], &c__1);
                }
            } else {
                if (k < *n - 1) {
                    i__1 = k + 1 + k * w_dim1;
                    d21.r = w[i__1].r, d21.i = w[i__1].i;
                    z_lmp_div(&z__1, &w[k + 1 + (k + 1) * w_dim1], &d21);
                    d11.r = z__1.r, d11.i = z__1.i;
                    d_lmp_cnjg(&z__2, &d21);
                    z_lmp_div(&z__1, &w[k + k * w_dim1], &z__2);
                    d22.r = z__1.r, d22.i = z__1.i;
                    z__1.r = d11.r * d22.r - d11.i * d22.i, z__1.i = d11.r * d22.i + d11.i * d22.r;
                    t = 1. / (z__1.r - 1.);
                    z__2.r = t, z__2.i = 0.;
                    z_lmp_div(&z__1, &z__2, &d21);
                    d21.r = z__1.r, d21.i = z__1.i;
                    i__1 = *n;
                    for (j = k + 2; j <= i__1; ++j) {
                        i__2 = j + k * a_dim1;
                        d_lmp_cnjg(&z__2, &d21);
                        i__3 = j + k * w_dim1;
                        z__4.r = d11.r * w[i__3].r - d11.i * w[i__3].i,
                        z__4.i = d11.r * w[i__3].i + d11.i * w[i__3].r;
                        i__4 = j + (k + 1) * w_dim1;
                        z__3.r = z__4.r - w[i__4].r, z__3.i = z__4.i - w[i__4].i;
                        z__1.r = z__2.r * z__3.r - z__2.i * z__3.i,
                        z__1.i = z__2.r * z__3.i + z__2.i * z__3.r;
                        a[i__2].r = z__1.r, a[i__2].i = z__1.i;
                        i__2 = j + (k + 1) * a_dim1;
                        i__3 = j + (k + 1) * w_dim1;
                        z__3.r = d22.r * w[i__3].r - d22.i * w[i__3].i,
                        z__3.i = d22.r * w[i__3].i + d22.i * w[i__3].r;
                        i__4 = j + k * w_dim1;
                        z__2.r = z__3.r - w[i__4].r, z__2.i = z__3.i - w[i__4].i;
                        z__1.r = d21.r * z__2.r - d21.i * z__2.i,
                        z__1.i = d21.r * z__2.i + d21.i * z__2.r;
                        a[i__2].r = z__1.r, a[i__2].i = z__1.i;
                    }
                }
                i__1 = k + k * a_dim1;
                i__2 = k + k * w_dim1;
                a[i__1].r = w[i__2].r, a[i__1].i = w[i__2].i;
                i__1 = k + 1 + k * a_dim1;
                i__2 = k + 1 + k * w_dim1;
                a[i__1].r = w[i__2].r, a[i__1].i = w[i__2].i;
                i__1 = k + 1 + (k + 1) * a_dim1;
                i__2 = k + 1 + (k + 1) * w_dim1;
                a[i__1].r = w[i__2].r, a[i__1].i = w[i__2].i;
                i__1 = *n - k;
                zlacgv_(&i__1, &w[k + 1 + k * w_dim1], &c__1);
                i__1 = *n - k - 1;
                zlacgv_(&i__1, &w[k + 2 + (k + 1) * w_dim1], &c__1);
            }
        }
        if (kstep == 1) {
            ipiv[k] = kp;
        } else {
            ipiv[k] = -kp;
            ipiv[k + 1] = -kp;
        }
        k += kstep;
        goto L70;
    L90:
        i__1 = *n;
        i__2 = *nb;
        for (j = k; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
            i__3 = *nb, i__4 = *n - j + 1;
            jb = min(i__3, i__4);
            i__3 = j + jb - 1;
            for (jj = j; jj <= i__3; ++jj) {
                i__4 = jj + jj * a_dim1;
                i__5 = jj + jj * a_dim1;
                d__1 = a[i__5].r;
                a[i__4].r = d__1, a[i__4].i = 0.;
                i__4 = j + jb - jj;
                i__5 = k - 1;
                z__1.r = -1., z__1.i = -0.;
                zgemv_((char *)"No transpose", &i__4, &i__5, &z__1, &a[jj + a_dim1], lda, &w[jj + w_dim1],
                       ldw, &c_b1, &a[jj + jj * a_dim1], &c__1, (ftnlen)12);
                i__4 = jj + jj * a_dim1;
                i__5 = jj + jj * a_dim1;
                d__1 = a[i__5].r;
                a[i__4].r = d__1, a[i__4].i = 0.;
            }
            if (j + jb <= *n) {
                i__3 = *n - j - jb + 1;
                i__4 = k - 1;
                z__1.r = -1., z__1.i = -0.;
                zgemm_((char *)"No transpose", (char *)"Transpose", &i__3, &jb, &i__4, &z__1, &a[j + jb + a_dim1],
                       lda, &w[j + w_dim1], ldw, &c_b1, &a[j + jb + j * a_dim1], lda, (ftnlen)12,
                       (ftnlen)9);
            }
        }
        j = k - 1;
    L120:
        jj = j;
        jp = ipiv[j];
        if (jp < 0) {
            jp = -jp;
            --j;
        }
        --j;
        if (jp != jj && j >= 1) {
            zswap_(&j, &a[jp + a_dim1], lda, &a[jj + a_dim1], lda);
        }
        if (j > 1) {
            goto L120;
        }
        *kb = k - 1;
    }
    return 0;
}
#ifdef __cplusplus
}
#endif
