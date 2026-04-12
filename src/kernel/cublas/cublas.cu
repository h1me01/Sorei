#include <cstdlib>

#include "cublas.h"

#define SOREI_CUBLAS_CHECK(expr)                                                                   \
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

namespace sorei::kernel::cublas {

namespace {
cublasHandle_t get_handle() {
    struct HandleOwner {
        cublasHandle_t handle;
        HandleOwner() { SOREI_CUBLAS_CHECK(cublasCreate(&handle)); }
        ~HandleOwner() { cublasDestroy(handle); }
    };
    static HandleOwner owner;
    return owner.handle;
}
} // namespace

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

    SOREI_CHECK(k == k_b);
    SOREI_CHECK(c.rows() == m);
    SOREI_CHECK(c.cols() == n);

    SOREI_CHECK(a.data());
    SOREI_CHECK(b.data());
    SOREI_CHECK(c.data());

    SOREI_CUBLAS_CHECK(cublasSgemm(
        get_handle(),
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

} // namespace sorei::kernel::cublas
