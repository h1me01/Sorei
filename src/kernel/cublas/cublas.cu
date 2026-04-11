#include <cstdlib>

#include "cublas.h"

#define CUBLAS_CHECK(expr)                                                                         \
    do {                                                                                           \
        cublasStatus_t status = (expr);                                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                     \
            printf("CUBLAS error calling %s\n", #expr);                                            \
            printf("    file:  %s\n", __FILE__);                                                   \
            printf("    line:  %d\n", __LINE__);                                                   \
            printf("    error: %s\n", cublasGetStatusString(status));                              \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

namespace kernel::cublas {

namespace {
cublasHandle_t handle = nullptr;
}

void create() {
    if (!handle)
        CUBLAS_CHECK(cublasCreate(&handle));
}

void destroy() {
    if (handle) {
        CUBLAS_CHECK(cublasDestroy(handle));
        handle = nullptr;
    }
}

void sgemm(
    bool trans_a,
    bool trans_b,
    float alpha,
    const tensor::GPUMatrix<float>& a,
    const tensor::GPUMatrix<float>& b,
    float beta,
    tensor::GPUMatrix<float>& c
) {
    int m = trans_a ? a.cols() : a.rows();
    int k = trans_a ? a.rows() : a.cols();
    int n = trans_b ? b.rows() : b.cols();
    int k_b = trans_b ? b.cols() : b.rows();

    CHECK(k == k_b);
    CHECK(c.rows() == m);
    CHECK(c.cols() == n);

    CHECK(a.data());
    CHECK(b.data());
    CHECK(c.data());

    CUBLAS_CHECK(cublasSgemm(
        handle,
        trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
        trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        a.data(),
        a.rows(),
        b.data(),
        b.rows(),
        &beta,
        c.data(),
        c.rows()
    ));
}

} // namespace kernel::cublas
