#include"saga.h"
#include"config.h"
#include"grad.h"
#include"check.h"
#include<mkl.h>
#include<math.h>
#include<cstdio>
#include<cstring>
#include<omp.h>
int saga::init_saga(char* fea_file, char* label_file, double lambda, double eta, int epoch){
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
	this->iter_num = iter_num;

	//grad set init
	this->wi_set = (double**)malloc(sizeof(double*)*this->exp_num);
	this->wi_set[0] = (double*)malloc(sizeof(double)*this->fea_num*this->exp_num);
	memset(wi_set[0], 0, sizeof(double)*this->fea_num*this->exp_num);
	for (int i = 1; i < this->exp_num; i++)
		this->wi_set[i] = this->wi_set[0] + i*this->fea_num;
	omp_set_num_threads(OMP_THREADS);
	return 0;
}
int saga::find_opt(){
	//grad init
	double* mu_grad; // average grad
	double* now_grad; // var = grad of wi
	double* pre_grad; // var = grad of wi_set[j]
	double* delta_grad; // var = grad of wi - grad of wi_set[j]
	mu_grad = (double*)malloc(sizeof(double)*this->fea_num);
	memset(mu_grad, 0, sizeof(double)*this->fea_num);
	now_grad = (double*)malloc(sizeof(double)*this->fea_num);
	memset(now_grad, 0, sizeof(double)*this->fea_num);
	pre_grad = (double*)malloc(sizeof(double)*this->fea_num);
	memset(pre_grad, 0, sizeof(double)*this->fea_num);
	delta_grad = (double*)malloc(sizeof(double)*this->fea_num);
	memset(delta_grad, 0, sizeof(double)*this->fea_num);
	//var init
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MCG31, RAN_SEED);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, this->fea_num, this->wi, RAN_MU, RAN_SIGMA);

	//grad set and mu_grad init
#pragma omp parallel
	{
		int th_now = omp_get_thread_num();
		for (int i = th_now; i < this->exp_num; i += OMP_THREADS)
			cblas_dcopy(this->fea_num, this->wi, 1, this->wi_set[i], 1);
	}
	lr_grad(this->exp_num, this->fea_num, this->wi, this->xi, this->yi, this->lambda, mu_grad);

	//main iteration
	vslNewStream(&stream, VSL_BRNG_MT19937, RAN_SEED);
	for (int t = 0; t < this->iter_num; t++){
		int exp_now;
		viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &exp_now, 0, this->exp_num);
		lr_sto_grad(exp_now, this->fea_num, this->wi, this->xi, this->yi, this->lambda, now_grad);
		lr_sto_grad(exp_now, this->fea_num, this->wi_set[exp_now],this->xi, this->yi, this->lambda, pre_grad);
		
		cblas_dcopy(this->fea_num, now_grad, 1, delta_grad, 1);
		cblas_daxpy(this->fea_num, -1.0, pre_grad, 1, delta_grad, 1);
		cblas_dscal(this->fea_num, double(1.0) / double(this->exp_num), delta_grad, 1);
		cblas_dcopy(this->fea_num, this->wi, 1, this->wi_set[exp_now], 1);

		cblas_daxpy(this->fea_num, -1.0, pre_grad, 1, now_grad, 1);
		cblas_daxpy(this->fea_num, 1.0, mu_grad, 1, now_grad, 1);
		cblas_daxpy(this->fea_num, -1.0*this->eta, mu_grad, 1, this->wi, 1);

		cblas_daxpy(this->fea_num, 1.0, delta_grad, 1, mu_grad, 1);
	}
	vslDeleteStream(&stream);
	free(delta_grad);
	free(pre_grad);
	free(now_grad);
	free(mu_grad);
	return 0;
}