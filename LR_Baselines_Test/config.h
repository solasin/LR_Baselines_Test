#ifndef CONFIG_H
#define CONFIG_H

#define INIT_EPS 1e-4  // gradient check  -- 1e-5 ~ 1e-7   hessian check -- 1e-4
#define RAN_SEED 1

#define RAN_MU 0.0
#define RAN_SIGMA 0.01

#define CHECK_EPS 1e-6
#define OPT_LOSS 0.267217680662898

#define QUASI_EPS 1e-10
#define ARMIJOR_EPS 1e-4

#define OVER_SAMP 5
#define APRX_RAN_MU 0.0
#define APRX_RAN_SIGMA 1e-5
#define REG_EPS 1e-7
#endif