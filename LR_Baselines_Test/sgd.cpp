#include"sgd.h"
#include"config.h"
#include"grad.h"
#include"check.h"
#include<mkl.h>
#include<math.h>
#include<cstdio>
#include<cstring>
#include<omp.h>
int sgd::init_sgd(char* fea_file, char* label_file, double lambda, double eta, bool step_dimin, int iter_num){
	freopen(fea_file, "r", stdin);
	scanf("%d%d", &this->exp_num, &this->fea_num);
	this->wi = (double*)malloc(sizeof(double)*this->fea_num);
	this->xi = (double**)malloc(sizeof(double*)*this->exp_num);
	this->xi[0] = (double*)malloc(sizeof(double)*(this->exp_num*this->fea_num));
	for (int i = 1; i < this->exp_num; i++)
		this->xi[i] = this->xi[0] + i*fea_num;
	this->yi = (double*)malloc(sizeof(double)*this->exp_num);
	for (int i = 0; i < this->exp_num; i++)
	for (int j = 0; j < this->fea_num; j++)
		scanf("%lf", &this->xi[i][j]);
	freopen(label_file, "r", stdin);
	for (int i = 0; i < this->exp_num; i++)
		scanf("%lf", &this->yi[i]);

	this->lambda = lambda;
	this->eta = eta;
	this->step_dimin = step_dimin;
	this->iter_num = iter_num;
	omp_set_num_threads(OMP_THREADS);
	return 0;
}
int sgd::find_opt(){
	VSLStreamStatePtr stream;
	double* sgd_grad;
	sgd_grad = (double*)malloc(sizeof(double)*(this->fea_num));
	vslNewStream(&stream, VSL_BRNG_MCG31, RAN_SEED);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, this->fea_num, this->wi, RAN_MU, RAN_SIGMA);
	vslNewStream(&stream, VSL_BRNG_MT19937, RAN_SEED);
	for (int t = 0; t < this->iter_num; t++){
		int exp_now;
		viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &exp_now, 0, this->exp_num);
		lr_sto_grad(exp_now, this->fea_num, this->wi, this->xi, this->yi, this->lambda, sgd_grad);
		// modification -- sometimes step-size should be decreased.
		cblas_daxpy(this->fea_num, -1.0*(this->eta), sgd_grad, 1, this->wi, 1);
	}
	vslDeleteStream(&stream);
	free(sgd_grad);
	return 0;
}
bool sgd::check_grad(){
	int flag = true;
	double* check_grad;
	check_grad = (double*)malloc(sizeof(double)*this->fea_num);
	lr_grad(this->exp_num, this->fea_num, this->wi, this->xi, this->yi, this->lambda, check_grad);

	double loss_wi, loss_dwi;
	loss_wi = lr_loss(this->exp_num, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
	for (int i = 0; i < fea_num; i++){
		this->wi[i] += INIT_EPS;
		loss_dwi = lr_loss(this->exp_num, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
		printf("%.8lf ---------- %.8lf\n", (loss_dwi - loss_wi) / INIT_EPS, check_grad[i]);
		if (fabs((loss_dwi - loss_wi) / INIT_EPS - check_grad[i]) >= CHECK_EPS)
			flag = false;
		this->wi[i] -= INIT_EPS;
	}
	return (flag);
}