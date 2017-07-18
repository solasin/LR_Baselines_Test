#include"check.h"
#include"config.h"
#include<omp.h>
#include<mkl.h>
#include<cstring>
#include<math.h>
#include<cstdio>
double lr_loss(int exp_num, int fea_num, double* wi, double** xi, double* yi, double lambda){
	double lr_func[OMP_THREADS];
	memset(lr_func, 0, sizeof(lr_func));
#pragma omp parallel
	{
		int th_now = omp_get_thread_num();
		for (int i = th_now; i < exp_num; i += OMP_THREADS){
			double z_now = cblas_ddot(fea_num, wi, 1, xi[i], 1);
			double g_now = (1.0 + exp(-1 * (yi[i] * z_now)));
			lr_func[th_now] += log(g_now);
		}
	}
	double lr_ret = 0;
	for (int i = 0; i < OMP_THREADS; i++)
		lr_ret += lr_func[i];
	lr_ret = double(lr_ret) / double(exp_num) + 0.5*lambda*cblas_ddot(fea_num, wi, 1, wi, 1);
	return (lr_ret);
}