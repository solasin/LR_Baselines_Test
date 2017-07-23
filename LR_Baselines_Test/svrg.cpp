#include"svrg.h"
#include"config.h"
#include"grad.h"
#include"check.h"
#include<mkl.h>
#include<math.h>
#include<cstdio>
#include<cstring>
int svrg::init_svrg(char* fea_file, char* label_file, double lambda, double eta, int epoch){
	freopen(fea_file, "r", stdin);
	scanf("%d%d%d", &this->exp_num, &this->fea_num,&this->cate);
	this->wi = (double**)malloc(sizeof(double*)*this->cate);
	this->wi[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		this->wi[i] = this->wi[0] + i*this->fea_num;
	this->xi = (double**)malloc(sizeof(double*)*this->exp_num);
	this->xi[0] = (double*)malloc(sizeof(double)*(this->exp_num*this->fea_num));
	for (int i = 1; i < this->exp_num; i++)
		this->xi[i] = this->xi[0] + i*fea_num;
	this->yi = (int*)malloc(sizeof(int)*this->exp_num);
	for (int i = 0; i < this->exp_num; i++)
	for (int j = 0; j < this->fea_num; j++)
		scanf("%lf", &this->xi[i][j]);
	freopen(label_file, "r", stdin);
	for (int i = 0; i < this->exp_num; i++)
		scanf("%d", &this->yi[i]);

	this->lambda = lambda;
	this->eta = eta;
	this->epoch = epoch;
	return 0;
}
int svrg::find_opt(){
	//grad init
	double** mu_grad; // average grad
	double** now_grad; // var = grad of w_{t-1}
	double** pre_grad; // var = grad of wi

	mu_grad = (double**)malloc(sizeof(double*)*this->cate);
	mu_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	now_grad = (double**)malloc(sizeof(double*)*this->cate);
	now_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	pre_grad = (double**)malloc(sizeof(double*)*this->cate);
	pre_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);

	//var init
	double** in_wi;
	in_wi = (double**)malloc(sizeof(double)*this->cate);
	in_wi[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);

	for (int i = 1; i < this->cate; i++) {
		mu_grad[i] = mu_grad[0] + i*this->fea_num;
		now_grad[i] = now_grad[0] + i*this->fea_num;
		pre_grad[i] = pre_grad[0] + i*this->fea_num;
		in_wi[i] = in_wi[0] + i*this->fea_num;
	}

	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MCG31, RAN_SEED);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, this->cate*this->fea_num, this->wi[0], RAN_MU, RAN_SIGMA);
	vslNewStream(&stream, VSL_BRNG_MT19937, RAN_SEED);

	for (int t = 0; t < this->epoch; t++){
		double loss_now = 0;
		loss_now = softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
		printf("EPOCH %d\nLOSS %.8lf\n--------------------------------------------------\n", t, loss_now);
		softmax_grad(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda, mu_grad);
		cblas_dcopy(this->cate*this->fea_num, this->wi[0], 1, in_wi[0], 1);
		for (int i = 0; i < this->exp_num;i++){
			int exp_now;
			viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &exp_now, 0, this->exp_num);
			softmax_sto_grad(exp_now, this->cate, this->fea_num, in_wi, this->xi, this->yi, this->lambda, now_grad);
			softmax_sto_grad(exp_now, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda, pre_grad);
			cblas_daxpy(this->cate*this->fea_num, -1.0, pre_grad[0], 1, now_grad[0], 1);
			cblas_daxpy(this->cate*this->fea_num, 1.0, mu_grad[0], 1, now_grad[0], 1);
			cblas_daxpy(this->cate*this->fea_num, -1.0*this->eta, now_grad[0], 1, in_wi[0], 1);
		}
		cblas_dcopy(this->cate*this->fea_num, in_wi[0], 1, this->wi[0], 1);
	}
	vslDeleteStream(&stream);
	free(in_wi[0]);
	free(in_wi);
	free(pre_grad[0]);
	free(pre_grad);
	free(now_grad[0]);
	free(now_grad);
	free(mu_grad[0]);
	free(mu_grad);
	return 0;
}