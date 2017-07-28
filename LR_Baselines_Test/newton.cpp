#include"newton.h"
#include"config.h"
#include"grad.h"
#include"hessi.h"
#include"check.h"
#include<mkl.h>
#include<cstring>
#include<cstdio>
int newton::init_newton(char* fea_file, char* label_file, double lambda, int iter_num) {
	freopen(fea_file, "r", stdin);
	scanf("%d%d%d", &this->exp_num, &this->fea_num, &this->cate);
	this->wi = (double**)malloc(sizeof(double*)*this->cate);
	this->wi[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		this->wi[i] = this->wi[0] + i*this->fea_num;

	this->xi = (double**)malloc(sizeof(double*)*this->exp_num);
	this->xi[0] = (double*)malloc(sizeof(double)*(this->exp_num*this->fea_num));
	for (int i = 1; i < this->exp_num; i++)
		this->xi[i] = this->xi[0] + i*this->fea_num;

	this->yi = (int*)malloc(sizeof(int)*this->exp_num);

	for (int i = 0; i < this->exp_num; i++)
		for (int j = 0; j < this->fea_num; j++)
			scanf("%lf", &this->xi[i][j]);
	freopen(label_file, "r", stdin);
	for (int i = 0; i < this->exp_num; i++)
		scanf("%d", &this->yi[i]);

	this->lambda = lambda;
	this->iter_num = iter_num;
	return 0;
}

int newton::find_opt() {
	VSLStreamStatePtr stream;
	double** gd_grad;
	double** delta_var;
	double** var_hessi;
	gd_grad = (double**)malloc(sizeof(double*)*this->cate);
	gd_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		gd_grad[i] = gd_grad[0] + i*this->fea_num;

	delta_var = (double**)malloc(sizeof(double*)*this->cate);
	delta_var[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		delta_var[i] = delta_var[0] + i*this->fea_num;

	var_hessi = (double**)malloc(sizeof(double*)*this->fea_num);
	var_hessi[0] = (double*)malloc(sizeof(double)*this->fea_num*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		var_hessi[i] = var_hessi[0] + i*this->fea_num;

	vslNewStream(&stream, VSL_BRNG_MCG31, RAN_SEED);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, this->cate*this->fea_num, this->wi[0], RAN_MU, RAN_SIGMA);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, this->cate*this->fea_num, delta_var[0], RAN_MU, RAN_SIGMA*(1e-12));
	cblas_daxpy(this->cate*this->fea_num, 1.0, this->wi[0], 1, delta_var[0], 1);
	check_hessian(delta_var);


	vslDeleteStream(&stream);
	free(var_hessi[0]);
	free(var_hessi);
	free(delta_var[0]);
	free(delta_var);
	free(gd_grad[0]);
	free(gd_grad);
	return 0;
}
bool newton::check_hessian(double** delta_wi) {
	int flag = true;
	double** check_hessi;
	check_hessi = (double**)malloc(sizeof(double*)*this->fea_num);
	check_hessi[0] = (double*)malloc(sizeof(double)*this->fea_num*this->fea_num);
	for (int i = 1; i < this->fea_num; i++)
		check_hessi[i] = check_hessi[0] + i*this->fea_num;
	for (int i = 0; i < this->cate; i++) {
		softmax_hessi(i, this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda, check_hessi);
		
		// xi = 180, xj = 180 check_hessi[xi][xj]!=0
		for (int xi = 0; xi < this->fea_num; xi++)
			for (int xj = 0; xj < this->fea_num; xj++) {
				double fo_delta, la_delta;
				this->wi[i][xi] += INIT_EPS;
				this->wi[i][xj] += INIT_EPS;
				fo_delta = softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
				this->wi[i][xj] -= INIT_EPS;
				fo_delta = fo_delta - softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
				fo_delta /= INIT_EPS;

				this->wi[i][xi] -= INIT_EPS;
				this->wi[i][xj] += INIT_EPS;
				la_delta = softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
				this->wi[i][xj] -= INIT_EPS;
				la_delta = la_delta - softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
				la_delta /= INIT_EPS;
				printf("%d %d %d ----------------: %.15lf    %.15lf\n", i, xi, xj, (fo_delta - la_delta) / INIT_EPS, check_hessi[xi][xj]);
			}
	}

	free(check_hessi[0]);
	free(check_hessi);
	return (flag);
}