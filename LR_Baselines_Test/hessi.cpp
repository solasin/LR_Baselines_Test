#include"hessi.h"
#include"config.h"
#include<mkl.h>
#include<math.h>
#include<cstdio>
#include<cstring>
int softmax_hessi(int var_idx, int exp_num, int cate, int fea_num, double** wi, double** xi, int* yi, double lambda, double** delta_hessi) {
	memset(delta_hessi[0], 0, sizeof(double)*fea_num*fea_num);
	for (int i = 0; i < exp_num; i++) {
		double tmp_sum,tmp_now;
		tmp_now = exp(cblas_ddot(fea_num, wi[var_idx], 1, xi[i], 1));
		tmp_sum = 0;
		for (int j = 0; j < cate; j++)
			tmp_sum += exp(cblas_ddot(fea_num, wi[j], 1, xi[i], 1));
		tmp_now = (tmp_now)*(tmp_sum - tmp_now) / (tmp_sum*tmp_sum);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, fea_num, fea_num, 1, tmp_now, xi[i], 1, xi[i], fea_num, 1.0, delta_hessi[0], fea_num);
	}
	cblas_dscal(fea_num*fea_num, 1.0 / double(exp_num*1.0), delta_hessi[0], 1);
	for (int i = 0; i < fea_num; i++)
		delta_hessi[i][i] += lambda;
	return 0;
}