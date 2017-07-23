#include"config.h"
#include<omp.h>
#include<cstring>
#include<mkl.h>
#include<math.h>
#include<cstdio>
int lr_grad(int exp_num, int fea_num, double* wi, double** xi, double* yi, double lambda, double* delta_wi){
	memset(delta_wi, 0, sizeof(double)*fea_num);
	int th_now = omp_get_thread_num();
	for (int i = 0; i < exp_num; i++){
		double z_now = cblas_ddot(fea_num, wi, 1, xi[i], 1);
		double g_now = 1.0 / (1.0 + exp(yi[i] * z_now));
		cblas_daxpy(fea_num, -1.0* yi[i] * g_now, xi[i], 1, delta_wi, 1);
	}
	cblas_dscal(fea_num, double(1.0) / double(exp_num), delta_wi, 1);
	cblas_daxpy(fea_num, lambda, wi, 1, delta_wi, 1);
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
	memset(delta_wi, 0, sizeof(double)*fea_num);
	int th_now = omp_get_thread_num();
	for (int i = 0; i < batch_len; i++){
		int exp_id = batch[i];
		double z_now = cblas_ddot(fea_num, wi, 1, xi[exp_id], 1);
		double g_now = 1.0 / (1.0 + exp(yi[exp_id] * z_now));
		cblas_daxpy(fea_num, -1.0*yi[exp_id] * g_now, xi[exp_id], 1, delta_wi, 1);
	}
	cblas_dscal(fea_num, double(1.0) / double(batch_len), delta_wi, 1);
	cblas_daxpy(fea_num, lambda, wi, 1, delta_wi, 1);
	return 0;
}
int softmax_grad(int exp_num, int cate, int fea_num, double** wi, double** xi, int* yi, double lambda, double** delta_wi) {
	memset(delta_wi[0], 0, sizeof(double)*cate*fea_num);
	for (int k = 0; k < cate; k++) {
		memset(delta_wi[k], 0, sizeof(double)*fea_num);
		for (int i = 0; i < exp_num; i++) {
			double tmp_cof = 0;
			for (int j = 0; j < cate; j++) 
				tmp_cof += exp(cblas_ddot(fea_num, wi[j], 1, xi[i], 1));
			tmp_cof = exp(cblas_ddot(fea_num, wi[k], 1, xi[i], 1)) / tmp_cof;
			if (yi[i] == k)
				tmp_cof -= 1;
			cblas_daxpy(fea_num, tmp_cof / double(1.0*exp_num), xi[i], 1, delta_wi[k], 1);
		}
		cblas_daxpy(fea_num, lambda, wi[k], 1, delta_wi[k], 1);
	}
	return 0;
}
int softmax_sto_grad(int delta_exp, int cate, int fea_num, double** wi, double** xi, int* yi, double lambda, double** delta_wi) {
	for (int k = 0; k < cate; k++) {
		memset(delta_wi[k], 0, sizeof(double)*fea_num);
		double tmp_cof = 0;
		for (int j = 0; j < cate; j++)
			tmp_cof += exp(cblas_ddot(fea_num, wi[j], 1, xi[delta_exp], 1));
		tmp_cof = exp(cblas_ddot(fea_num, wi[k], 1, xi[delta_exp], 1)) / tmp_cof;
		if (yi[delta_exp] == k)
			tmp_cof -= 1;
		cblas_daxpy(fea_num, tmp_cof, xi[delta_exp], 1, delta_wi[k] , 1);
		cblas_daxpy(fea_num, lambda, wi[k], 1, delta_wi[k], 1);
	}
	return 0;
}