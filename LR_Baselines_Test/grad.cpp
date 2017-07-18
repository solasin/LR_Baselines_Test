#include"config.h"
#include<omp.h>
#include<cstring>
#include<mkl.h>
#include<math.h>
#include<cstdio>
int lr_grad(int exp_num, int fea_num, double* wi, double** xi, double* yi, double lambda, double* delta_wi){
	double** tmp_wi;
	memset(delta_wi, 0, sizeof(double)*fea_num);
	tmp_wi = (double**)malloc(sizeof(double*)*OMP_THREADS);
	tmp_wi[0] = (double*)malloc(sizeof(double)*OMP_THREADS*fea_num);
	for (int i = 1; i < OMP_THREADS; i++)
		tmp_wi[i] = tmp_wi[0] + i*fea_num;
#pragma omp parallel
	{
		int th_now = omp_get_thread_num();
		memset(tmp_wi[th_now], 0, sizeof(double)*fea_num);
		for (int i = th_now; i < exp_num; i+=OMP_THREADS){
			double z_now = cblas_ddot(fea_num, wi, 1, xi[i], 1);
			double g_now = 1.0 / (1.0 + exp(yi[i] * z_now));
			cblas_daxpy(fea_num, -1.0* yi[i] * g_now, xi[i], 1, tmp_wi[th_now], 1);
		}
	}
	for (int i = 0; i < OMP_THREADS; i++)
		cblas_daxpy(fea_num, 1.0, tmp_wi[i], 1, delta_wi, 1);
	cblas_dscal(fea_num, double(1.0) / double(exp_num), delta_wi, 1);
	cblas_daxpy(fea_num, lambda, wi, 1, delta_wi, 1);
	free(tmp_wi[0]);
	free(tmp_wi);
	return 0;
}
int lr_sto_grad(int delta_exp, int fea_num, double* wi, double** xi, double* yi, double lambda, double* delta_wi){
	memset(delta_wi, 0, sizeof(double)*fea_num);
	double z_now = cblas_ddot(fea_num, xi[delta_exp], 1, wi, 1);
	double g_now = 1.0 / (1.0 + exp(yi[delta_exp] * z_now));
	cblas_daxpy(fea_num, -1.0*yi[delta_exp] * g_now, xi[delta_exp], 1, delta_wi, 1);
	cblas_daxpy(fea_num, lambda, wi, 1, delta_wi, 1);
	return 0;
}
int lr_mini_gra(int batch_len, int* batch, int fea_num, double* wi, double** xi, double* yi, double lambda, double* delta_wi){
	double** tmp_wi;
	memset(delta_wi, 0, sizeof(double)*fea_num);
	tmp_wi = (double**)malloc(sizeof(double*)*OMP_THREADS);
	tmp_wi[0] = (double*)malloc(sizeof(double)*OMP_THREADS*fea_num);
	for (int i = 1; i < OMP_THREADS; i++)
		tmp_wi[i] = tmp_wi[0] + i*fea_num;
#pragma omp parallel
	{
		int th_now = omp_get_thread_num();
		memset(tmp_wi[th_now], 0, sizeof(double)*fea_num);
		for (int i = th_now; i < batch_len; i += OMP_THREADS){
			int exp_id = batch[i];
			double z_now = cblas_ddot(fea_num, wi, 1, xi[exp_id], 1);
			double g_now = 1.0 / (1.0 + exp(yi[exp_id] * z_now));
			cblas_daxpy(fea_num, -1.0*yi[exp_id] * g_now, xi[exp_id], 1, tmp_wi[th_now], 1);
		}
	}
	for (int i = 0; i < OMP_THREADS; i++)
		cblas_daxpy(fea_num, 1.0, tmp_wi[i], 1, delta_wi, 1);
	cblas_dscal(fea_num, double(1.0) / double(batch_len), delta_wi, 1);
	cblas_daxpy(fea_num, lambda, wi, 1, delta_wi, 1);
	free(tmp_wi[0]);
	free(tmp_wi);
	return 0;
}