#include"check.h"
#include"config.h"
#include<mkl.h>
#include<cstring>
#include<math.h>
#include<cstdio>
double lr_loss(int exp_num, int fea_num, double* wi, double** xi, double* yi, double lambda){
	double lr_func;
	lr_func = 0;
	for (int i = 0; i < exp_num; i ++){
		double z_now = cblas_ddot(fea_num, wi, 1, xi[i], 1);
		double g_now = (1.0 + exp(-1 * (yi[i] * z_now)));
		lr_func += log(g_now);
	}
	lr_func = double(lr_func) / double(exp_num) + 0.5*lambda*cblas_ddot(fea_num, wi, 1, wi, 1);
	return (lr_func);
}
double softmax_loss(int exp_num, int cate, int fea_num, double** wi, double** xi, int* yi, double lambda) {
	double softmax_func;
	softmax_func = 0;
	for (int i = 0; i < exp_num; i++) {
		double tmp_term = 0;
		for (int j = 0; j < cate; j++)
			tmp_term += exp(cblas_ddot(fea_num, wi[j], 1, xi[i], 1));
		tmp_term = log(tmp_term);
		tmp_term -= cblas_ddot(fea_num, wi[yi[i]], 1, xi[i], 1);
		softmax_func += tmp_term;
	}
	softmax_func /= (double)(exp_num*1.0);
	for (int i = 0; i < cate; i++)
		softmax_func = softmax_func + 0.5*lambda*cblas_ddot(fea_num, wi[i], 1, wi[i], 1);
	return (softmax_func);
}
double softmax_sto_loss(int delta_exp, int cate, int fea_num, double** wi, double** xi, int* yi, double lambda) {
	double softmax_func;
	softmax_func = 0;
	double tmp_term = 0;
	for (int j = 0; j < cate; j++)
		tmp_term += exp(cblas_ddot(fea_num, wi[j], 1, xi[delta_exp], 1));
	tmp_term = log(tmp_term);
	tmp_term -= cblas_ddot(fea_num, wi[yi[delta_exp]], 1, xi[delta_exp], 1);
	softmax_func += tmp_term;
	for (int i = 0; i < cate; i++)
		softmax_func = softmax_func + 0.5*lambda*cblas_ddot(fea_num, wi[i], 1, wi[i], 1);
	return (softmax_func);
}