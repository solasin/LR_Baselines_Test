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
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, fea_num, fea_num, 1, tmp_now / double(exp_num*1.0) , xi[i], 1, xi[i], fea_num, 1.0, delta_hessi[0], fea_num);
	}
	//cblas_dscal(fea_num*fea_num, 1.0 / double(exp_num*1.0), delta_hessi[0], 1);
	for (int i = 0; i < fea_num; i++)
		delta_hessi[i][i] += lambda;
	return 0;
}


int softmax_hessi(int exp_num, int cate, int fea_num, double** wi, double** xi, int* yi, double lambda, double** delta_hessi) {
	memset(delta_hessi[0], 0, sizeof(double)*cate*fea_num*cate*fea_num);
	double** tmp_now;
	tmp_now = (double**)malloc(sizeof(double*)*cate);
	tmp_now[0] = (double*)malloc(sizeof(double)*cate*exp_num);
	for (int i = 1; i < cate; i++)
		tmp_now[i] = tmp_now[0] + i*exp_num;
	for (int i = 0; i < exp_num; i++) {
		double tmp_sum = 0;
		for (int j = 0; j < cate; j++)
			tmp_sum+=exp(cblas_ddot(fea_num, wi[j], 1, xi[i], 1));
		for (int j = 0; j < cate; j++) {
			tmp_now[j][i] = exp(cblas_ddot(fea_num, wi[j], 1, xi[i], 1))/tmp_sum;
		}
	}

	double** tmp_hessi;
	tmp_hessi = (double**)malloc(sizeof(double*)*fea_num);
	tmp_hessi[0] = (double*)malloc(sizeof(double)*fea_num*fea_num);
	for (int i = 1; i < fea_num; i++)
		tmp_hessi[i] = tmp_hessi[0] + i*fea_num;


	for (int p = 0; p < cate; p++) {
		printf("%d block is calculating\n", p);
		for (int q = 0; q < cate; q++) {
			cblas_dscal(fea_num*fea_num, 0, tmp_hessi[0], 1);
			if (p == q) {
				for (int i = 0; i < fea_num; i++)
					tmp_hessi[i][i] = lambda;
				for (int i = 0; i < exp_num; i++)
					cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, fea_num, fea_num, 1, tmp_now[p][i] * (1 - tmp_now[p][i]) / double(exp_num), xi[i], 1, xi[i], fea_num, 1.0, tmp_hessi[0], fea_num);			
			}
			else {
				for (int i = 0; i < exp_num; i++) 
					cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, fea_num, fea_num, 1, -tmp_now[p][i] * tmp_now[q][i] / double(exp_num), xi[i], 1, xi[i], fea_num, 1.0, tmp_hessi[0], fea_num);
			}
			
			for (int i = 0; i < fea_num; i++)
				cblas_dcopy(fea_num, tmp_hessi[i], 1, &delta_hessi[p*fea_num + i][q*fea_num], 1);
		}
	}
	free(tmp_now[0]);
	free(tmp_now);
	free(tmp_hessi[0]);
	free(tmp_hessi);
	return 0;
}