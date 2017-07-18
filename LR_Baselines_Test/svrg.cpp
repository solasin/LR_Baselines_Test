#include"svrg.h"
#include"config.h"
#include"grad.h"
#include"check.h"
#include<mkl.h>
#include<math.h>
#include<cstdio>
#include<cstring>
#include<omp.h>
int svrg::init_svrg(char* fea_file, char* label_file, double lambda, double eta, int epoch){
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
	this->epoch = epoch;
	omp_set_num_threads(OMP_THREADS);
	return 0;
}
int svrg::find_opt(){
	//grad init
	double* mu_grad; // average grad
	double* now_grad; // var = grad of w_{t-1}
	double* pre_grad; // var = grad of wi
	mu_grad = (double*)malloc(sizeof(double)*this->fea_num);
	now_grad = (double*)malloc(sizeof(double)*(this->fea_num));
	pre_grad = (double*)malloc(sizeof(double)*(this->fea_num));

	//var init
	double* in_wi;
	in_wi = (double*)malloc(sizeof(double)*this->fea_num);
	
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MCG31, RAN_SEED);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, this->fea_num, this->wi, RAN_MU, RAN_SIGMA);

	vslNewStream(&stream, VSL_BRNG_MT19937, RAN_SEED);
	for (int t = 0; t < this->epoch; t++){
		lr_grad(this->exp_num, this->fea_num, this->wi, this->xi, this->yi, this->lambda, mu_grad);
		cblas_dcopy(this->fea_num, this->wi, 1, in_wi, 1);
		for (int i = 0; i < this->exp_num;i++){
			int exp_now;
			viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &exp_now, 0, this->exp_num);
			lr_sto_grad(exp_now, this->fea_num, in_wi, this->xi, this->yi, this->lambda, now_grad);
			lr_sto_grad(exp_now, this->fea_num, this->wi, this->xi, this->yi, this->lambda, pre_grad);
			cblas_daxpy(this->fea_num, -1.0, pre_grad, 1, now_grad, 1);
			cblas_daxpy(this->fea_num, 1.0, mu_grad, 1, now_grad, 1);
			cblas_daxpy(this->fea_num, -1.0*this->eta, now_grad, 1, in_wi, 1);
		}
		cblas_dcopy(this->fea_num, in_wi, 1, this->wi, 1);
	}
	vslDeleteStream(&stream);
	free(in_wi);
	free(pre_grad);
	free(now_grad);
	free(mu_grad);
	return 0;
}
/*bool sgd::check_grad(){
int flag = true;
double* check_grad;
check_grad = (double*)malloc(sizeof(double)*this->fea_num);
lr_grad(this->exp_num, this->fea_num, this->wi, this->xi, this->yi, check_grad);

double loss_wi, loss_dwi;
loss_wi = lr_loss(this->exp_num, this->fea_num, this->wi, this->xi, this->yi);
for (int i = 0; i < fea_num; i++){
this->wi[i] += INIT_EPS;
loss_dwi = lr_loss(this->exp_num, this->fea_num, this->wi, this->xi, this->yi);
if (fabs((loss_dwi - loss_wi) / INIT_EPS - check_grad[i]) >= CHECK_EPS)
flag = false;
this->wi[i] -= INIT_EPS;
}
return (flag);
}*/