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
	double* gd_tmp;
	double** gd_grad;
	double** var_hessi;
	double** inv_hessi;
	int* ipv_hessi;
	gd_tmp = (double*)malloc(sizeof(double)*this->fea_num);

	gd_grad = (double**)malloc(sizeof(double*)*this->cate);
	gd_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		gd_grad[i] = gd_grad[0] + i*this->fea_num;

	var_hessi = (double**)malloc(sizeof(double*)*this->fea_num);
	var_hessi[0] = (double*)malloc(sizeof(double)*this->fea_num*this->fea_num);
	for (int i = 1; i < this->fea_num; i++)
		var_hessi[i] = var_hessi[0] + i*this->fea_num;

	inv_hessi = (double**)malloc(sizeof(double*)*this->fea_num);
	inv_hessi[0] = (double*)malloc(sizeof(double)*this->fea_num*this->fea_num);
	for (int i = 1; i < this->fea_num; i++)
		inv_hessi[i] = inv_hessi[0] + i*this->fea_num;
		

	ipv_hessi = (int*)malloc(sizeof(int)*this->fea_num);

	// initialize opt_var
	vslNewStream(&stream, VSL_BRNG_MCG31, RAN_SEED);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, this->cate*this->fea_num, this->wi[0], RAN_MU, RAN_SIGMA);
	//check_hessian();
	for (int t = 0; t < this->iter_num; t++) {
		double loss_now = 0;
		loss_now = softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
		printf("EPOCH %d\nLOSS %.8lf\n--------------------------------------------------\n", t, loss_now);
		softmax_grad(this->exp_num, this->cate, this->fea_num, this->wi,this->xi, this->yi, this->lambda, gd_grad);
		for (int k = 0; k < this->cate; k++) {
			softmax_hessi(k, this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda, var_hessi);
			memset(inv_hessi[0], 0, sizeof(double)*this->fea_num*this->fea_num);
			for (int j = 0; j < this->fea_num; j++)
				inv_hessi[j][j] = 1.0;
			LAPACKE_dgesv(LAPACK_ROW_MAJOR,this->fea_num,this->fea_num,var_hessi[0],this->fea_num,ipv_hessi,inv_hessi[0],this->fea_num);
			//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, fea_num, fea_num, 1, tmp_now, xi[i], 1, xi[i], fea_num, 1.0, delta_hessi[0], fea_num);
			//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, this->fea_num, 1, this->fea_num, 1.0, inv_hessi[0], this->fea_num, gd_grad[k], 1, 0.0, gd_tmp, 1);
			cblas_dcopy(this->fea_num, gd_tmp, 1, gd_grad[k], 1);
		}
		cblas_daxpy(this->cate*this->fea_num, -1.0, gd_grad[0], 1, this->wi[0], 1);
	}
	vslDeleteStream(&stream);
	free(ipv_hessi);
	free(inv_hessi[0]);
	free(inv_hessi);
	free(var_hessi[0]);
	free(var_hessi);
	free(gd_grad[0]);
	free(gd_grad);
	return 0;
}
bool newton::check_hessian() {
	int flag = true;
	double** check_hessi;
	check_hessi = (double**)malloc(sizeof(double*)*this->fea_num);
	check_hessi[0] = (double*)malloc(sizeof(double)*this->fea_num*this->fea_num);
	for (int i = 1; i < this->fea_num; i++)
		check_hessi[i] = check_hessi[0] + i*this->fea_num;
	for (int i = 0; i < this->cate; i++) {
		softmax_hessi(i, this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda, check_hessi);
		
		// xi = 180, xj = 180 check_hessi[xi][xj]!=0
		for (int xi = 180; xi < this->fea_num; xi++)
			for (int xj = 180; xj < this->fea_num; xj++) {
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