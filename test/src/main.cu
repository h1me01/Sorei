#include "framework.h"

#include "test_grad.h"
#include "test_kernels.h"
#include "test_layers.h"
#include "test_lr_sched.h"
#include "test_matrix.h"
#include "test_model.h"
#include "test_optim.h"

int main() { return sorei::test::run_all_tests(); }
