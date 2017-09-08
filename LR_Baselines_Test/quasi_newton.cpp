#include"quasi_newton.h"
#include"config.h"
#include"grad.h"
#include"hessi.h"
#include"check.h"
#include"svrg.h"
#include<mkl.h>
#include<cstring>
#include<cstdio>
#include<cmath>

int quasi_newton::init_quasi_newton(char* fea_file, char* label_file, double lambda, int iter_num) {
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

int quasi_newton::init_quasi_newton(svrg* opt_pre, int iter_num) {
	this->exp_num = opt_pre->exp_num;
	this->fea_num = opt_pre->fea_num;
	this->cate = opt_pre->cate;
	this->wi = (double**)malloc(sizeof(double*)*this->cate);
	this->wi[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		this->wi[i] = this->wi[0] + i*this->fea_num;
	cblas_dcopy(this->cate*this->fea_num, opt_pre->wi[0], 1, this->wi[0], 1);

	this->xi = (double**)malloc(sizeof(double*)*this->exp_num);
	this->xi[0] = (double*)malloc(sizeof(double)*(this->exp_num*this->fea_num));
	for (int i = 1; i < this->exp_num; i++)
		this->xi[i] = this->xi[0] + i*this->fea_num;
	cblas_dcopy(this->fea_num*this->exp_num, opt_pre->xi[0], 1, this->xi[0], 1);

	this->yi = (int*)malloc(sizeof(int)*this->exp_num);
	memcpy(this->yi, opt_pre->yi, sizeof(int)*this->exp_num);

	this->lambda = opt_pre->lambda;
	this->iter_num = iter_num;

	return 0;
}

//the newton_step here is equal to -H*detf
double quasi_newton::backtracking_armijo(double init_step, double** newton_step, double** g_grad) {
	double det_step = init_step;
	double** tmp_var;
	tmp_var = (double**)malloc(sizeof(double*)*this->cate);

	tmp_var[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		tmp_var[i] = tmp_var[0] + i*this->fea_num;
	cblas_dcopy(this->cate*this->fea_num, this->wi[0], 1, tmp_var[0], 1);
	double inequ_fk = softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
    double print_now=cblas_ddot(this->cate*this->fea_num,g_grad[0],1,newton_step[0],1);
    printf("Show Print Now = %.8lf\n",print_now);
	while (1) {
		cblas_daxpy(this->cate*this->fea_num, 1.0*det_step, newton_step[0], 1, tmp_var[0], 1);
		double inequ_left = softmax_loss(this->exp_num, this->cate, this->fea_num, tmp_var, this->xi, this->yi, this->lambda);
		if (inequ_left <= inequ_fk + det_step*ARMIJOR_EPS*cblas_ddot(this->cate*this->fea_num, g_grad[0], 1, newton_step[0], 1))
			break;
		det_step = det_step / 2.0;
        cblas_dcopy(this->cate*this->fea_num,this->wi[0],1,tmp_var[0],1);
		if (det_step < QUASI_EPS)
			break;
        printf("%.8lf\t%.8lf\n%.8lf\n",inequ_left,inequ_fk,det_step);
	}
	printf("----------------------The step-size in this epoch is %.8lf\n", det_step);
    free(tmp_var[0]);
    free(tmp_var);
	return det_step;
}

int quasi_newton::find_opt(bool pre_opted) {
	VSLStreamStatePtr stream;
	double** gd_grad;
	double** inv_hessi;
	double** newton_step;
	double** det_variable;
	double** det_grad;

	gd_grad = (double**)malloc(sizeof(double*)*this->cate);
	gd_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		gd_grad[i] = gd_grad[0] + i*this->fea_num;

	newton_step = (double**)malloc(sizeof(double*)*this->cate);
	newton_step[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		newton_step[i] = newton_step[0] + i*this->fea_num;

	det_variable = (double**)malloc(sizeof(double*)*this->cate);
	det_variable[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		det_variable[i] = det_variable[0] + i*this->fea_num;

	det_grad = (double**)malloc(sizeof(double*)*this->cate);
	det_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		det_grad[i] = det_grad[0] + i*this->fea_num;


	inv_hessi = (double**)malloc(sizeof(double*)*this->cate*this->fea_num);
	inv_hessi[0] = (double*)malloc(sizeof(double)*this->cate*this->cate*this->fea_num*this->fea_num);
	cblas_dscal(this->cate*this->fea_num, 0.0, inv_hessi[0], 1);
	inv_hessi[0][0] = 1.0;
	for (int i = 1; i < this->cate*this->fea_num; i++) {
		inv_hessi[i] = inv_hessi[0] + i*this->fea_num*this->cate;
		cblas_dscal(this->cate*this->fea_num, 0.0, inv_hessi[i], 1);
		inv_hessi[i][i] = 1.0;
	}
		

	// initialize opt_var
	if (pre_opted == false) {
		vslNewStream(&stream, VSL_BRNG_MCG31, RAN_SEED);
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, this->cate*this->fea_num, this->wi[0], RAN_MU, RAN_SIGMA);
	}
    
	for (int t = 0; t < this->iter_num; t++) {
		double loss_now = 0;
		loss_now = softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
		printf("EPOCH %d\nLOSS %.8lf %.8lf\n--------------------------------------------------\n", t, loss_now, log(loss_now - OPT_LOSS));
		softmax_grad(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda, gd_grad);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, this->cate*this->fea_num, 1, this->cate*this->fea_num, -1.0, inv_hessi[0], this->cate*this->fea_num, gd_grad[0], 1, 0.0, newton_step[0], 1);
		double dec_step = backtracking_armijo(100.0, newton_step,gd_grad);

		cblas_dscal(this->cate*this->fea_num, 0.0, det_variable[0], 1);
		cblas_daxpy(this->cate*this->fea_num, dec_step, newton_step[0], 1, det_variable[0], 1);

		cblas_daxpy(this->cate*this->fea_num, dec_step, newton_step[0], 1, this->wi[0], 1);
		softmax_grad(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda, det_grad);
		cblas_daxpy(this->cate*this->fea_num, -1.0, gd_grad[0], 1, det_grad[0], 1);

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, this->cate*this->fea_num, 1, this->cate*this->fea_num, 1.0, inv_hessi[0], this->cate*this->fea_num, det_grad[0], 1, 0.0, gd_grad[0], 1);
		double cof_hessi = cblas_ddot(this->cate*this->fea_num, det_variable[0], 1, det_grad[0], 1);
		cof_hessi = (cof_hessi + cblas_ddot(this->cate*this->fea_num, det_grad[0], 1, gd_grad[0], 1)) / (cof_hessi*cof_hessi);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, this->cate*this->fea_num, this->cate*this->fea_num, 1, cof_hessi, det_variable[0], 1, det_variable[0], this->cate*this->fea_num, 1.0, inv_hessi[0], this->cate*this->fea_num);
		
		cof_hessi = -1.0/cblas_ddot(this->cate*this->fea_num, det_variable[0], 1, det_grad[0], 1);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, this->cate*this->fea_num, this->cate*this->fea_num, 1, cof_hessi, gd_grad[0], 1, det_variable[0], this->cate*this->fea_num, 1.0, inv_hessi[0], this->cate*this->fea_num);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, this->cate*this->fea_num, this->cate*this->fea_num, 1, cof_hessi, det_variable[0], 1, gd_grad[0], this->cate*this->fea_num, 1.0, inv_hessi[0], this->cate*this->fea_num);
	}
	vslDeleteStream(&stream);
	free(inv_hessi[0]);
	free(inv_hessi);
	free(det_grad[0]);
	free(det_grad);
	free(det_variable[0]);
	free(det_variable);
	free(newton_step[0]);
	free(newton_step);
	free(gd_grad[0]);
	free(gd_grad);
	return 0;
}
