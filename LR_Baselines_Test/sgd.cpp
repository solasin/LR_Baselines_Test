#include"sgd.h"
#include"config.h"
#include"grad.h"
#include"check.h"
#include<mkl.h>
#include<math.h>
#include<cstdio>
#include<cstring>
int sgd::init_sgd(char* fea_file, char* label_file, double lambda, double eta, bool step_dimin, int iter_num){
	freopen(fea_file, "r", stdin);
	scanf("%d%d%d", &this->exp_num, &this->fea_num,&this->cate);
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
	this->eta = eta;
	this->step_dimin = step_dimin;
	this->iter_num = iter_num;
	return 0;
}
int sgd::find_opt(){
	VSLStreamStatePtr stream;
	double** sgd_grad;
	sgd_grad = (double**)malloc(sizeof(double*)*this->cate);
	sgd_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		sgd_grad[i] = sgd_grad[0] + i*this->fea_num;
	vslNewStream(&stream, VSL_BRNG_MCG31, RAN_SEED);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, this->cate*this->fea_num, this->wi[0], RAN_MU, RAN_SIGMA);
	vslNewStream(&stream, VSL_BRNG_MT19937, RAN_SEED);
	for (int t = 0; t < this->iter_num; t++){
		int exp_now;
		//display the loss
		double loss_now = 0;
		if (t % this->exp_num == 0) {
			loss_now = softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
			printf("EPOCH %d\nLOSS %.8lf\n--------------------------------------------------\n", t / (this->exp_num), loss_now);
		}
		//iterate optimization variable
		viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &exp_now, 0, this->exp_num);
		softmax_sto_grad(exp_now, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda, sgd_grad);
		cblas_daxpy(this->cate*this->fea_num, -1.0*(this->eta), sgd_grad[0], 1, this->wi[0], 1);
	}
	vslDeleteStream(&stream);
	free(sgd_grad[0]);
	free(sgd_grad);
	return 0;
}
bool sgd::check_grad(int delta_exp){
	int flag = true;
	double** check_grad;
	check_grad = (double**)malloc(sizeof(double*)*this->cate);
	check_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		check_grad[i] = check_grad[0] + i*this->fea_num;
	softmax_sto_grad(delta_exp, this->cate, this->fea_num, this->wi, this->xi, this->yi,this->lambda,check_grad);
	double loss_wi, loss_dwi;
	loss_wi = softmax_sto_loss(delta_exp, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
	for (int i=0;i<this->cate;i++)
		for (int j = 0; j < fea_num; j++){
			this->wi[i][j] += INIT_EPS;
			loss_dwi = softmax_sto_loss(delta_exp, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
			printf("%.8lf ---------- %.8lf\n", (loss_dwi - loss_wi) / INIT_EPS, check_grad[i][j]);
			if (fabs((loss_dwi - loss_wi) / INIT_EPS - check_grad[i][j]) >= CHECK_EPS)
				flag = false;
			this->wi[i][j] -= INIT_EPS;
		}
	free(check_grad[0]);
	free(check_grad);
	return (flag);
}