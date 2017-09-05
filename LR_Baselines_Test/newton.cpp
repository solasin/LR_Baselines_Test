#include"newton.h"
#include"config.h"
#include"grad.h"
#include"hessi.h"
#include"check.h"
#include"svrg.h"
#include<mkl.h>
#include<cstring>
#include<cstdio>
#include<cmath>

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

int newton::init_newton(svrg* opt_pre, int iter_num) {
	this->exp_num = opt_pre->exp_num;
	this->fea_num = opt_pre->fea_num;
	this->cate = opt_pre->cate;
	this->wi = (double**)malloc(sizeof(double*)*this->cate);
	this->wi[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		this->wi[i] = this->wi[0] + i*this->fea_num;
	cblas_dcopy(this->cate*this->fea_num,opt_pre->wi[0], 1,this->wi[0],1);

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

int newton::find_opt(bool pre_opted) {
	VSLStreamStatePtr stream;
	double** gd_grad;
	double** var_hessi;
	//double** inv_hessi;
	int* ipv_hessi;

	gd_grad = (double**)malloc(sizeof(double*)*this->cate);
	gd_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		gd_grad[i] = gd_grad[0] + i*this->fea_num;

	var_hessi = (double**)malloc(sizeof(double*)*this->cate*this->fea_num);
	var_hessi[0] = (double*)malloc(sizeof(double)*this->cate*this->cate*this->fea_num*this->fea_num);
	for (int i = 1; i < this->cate*this->fea_num; i++)
		var_hessi[i] = var_hessi[0] + i*this->fea_num*this->cate;

	/*inv_hessi = (double**)malloc(sizeof(double*)*this->cate*this->fea_num);
	inv_hessi[0] = (double*)malloc(sizeof(double)*this->cate*this->cate*this->fea_num*this->fea_num);
	for (int i = 1; i < this->cate*this->fea_num; i++)
		inv_hessi[i] = inv_hessi[0] + i*this->fea_num*this->cate;
	*/

	ipv_hessi = (int*)malloc(sizeof(int)*this->fea_num*this->cate);

	// initialize opt_var
	if (pre_opted == false) {
		vslNewStream(&stream, VSL_BRNG_MCG31, RAN_SEED);
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, this->cate*this->fea_num, this->wi[0], RAN_MU, RAN_SIGMA);
	}
	for (int t = 0; t < this->iter_num; t++) {
		double loss_now = 0;
		loss_now = softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
		printf("EPOCH %d\nLOSS %.8lf %.8lf\n--------------------------------------------------\n", t, loss_now, log(loss_now-OPT_LOSS));
		softmax_grad(this->exp_num, this->cate, this->fea_num, this->wi,this->xi, this->yi, this->lambda, gd_grad);
		softmax_hessi(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda, var_hessi);
		//check_hessian(var_hessi,1000);
        //cblas_dcopy(this->cate*this->fea_num*this->cate*this->fea_num, var_hessi[0], 1, inv_hessi[0], 1);
        int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, this->cate*this->fea_num, this->cate*this->fea_num, var_hessi[0], this->cate*this->fea_num, ipv_hessi);
		printf("---------------- info = %d\n",info);
        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, this->cate*this->fea_num, var_hessi[0], this->cate*this->fea_num, ipv_hessi);
        printf("---------------- info = %d\n",info);		
        //check_inversion(inv_hessi,var_hessi,1000);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, this->cate*this->fea_num, 1, this->cate*this->fea_num, -1.0, var_hessi[0], this->cate*this->fea_num, gd_grad[0], 1, 1.0, this->wi[0], 1);
		/*for (int k = 0; k < this->cate; k++) {
			softmax_hessi(k, this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda, var_hessi);
			cblas_dcopy(this->fea_num*this->fea_num, var_hessi[0], 1, inv_hessi[0], 1);
			//LAPACKE_dgesv(LAPACK_ROW_MAJOR,this->fea_num,this->fea_num,var_hessi[0],this->fea_num,ipv_hessi,inv_hessi[0],this->fea_num);
			int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, this->fea_num, this->fea_num, var_hessi[0], this->fea_num, ipv_hessi);
			info=LAPACKE_dgetri(LAPACK_ROW_MAJOR,this->fea_num,var_hessi[0],this->fea_num,ipv_hessi);

			//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, fea_num, fea_num, 1, tmp_now, xi[i], 1, xi[i], fea_num, 1.0, delta_hessi[0], fea_num);
			//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, this->fea_num, 1, this->fea_num,1.0, var_hessi[0], this->fea_num, gd_grad[k], 1, 0, gd_tmp, 1);
			cblas_dcopy(this->fea_num, gd_tmp, 1, gd_grad[k], 1);
		}*/
		//cblas_daxpy(this->cate*this->fea_num, -1.0, gd_grad[0], 1, this->wi[0], 1);
	}
	vslDeleteStream(&stream);
	free(ipv_hessi);
	free(var_hessi[0]);
	free(var_hessi);
	free(gd_grad[0]);
	free(gd_grad);
	return 0;
}
int newton::check_hessian(double** now_hessi, int sam_num) {
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MT19937, RAN_SEED);
	for (int t = 0; t < sam_num; t++) {
		int cate_i,cate_j,fea_i,fea_j;
		viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &fea_j, 0, this->fea_num);
		viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &cate_j, 0, this->cate);
		viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &fea_i, 0, this->fea_num);
		viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &cate_i, 0, this->cate);
		double fo_delta, la_delta;
				this->wi[cate_i][fea_i] += INIT_EPS;
				this->wi[cate_j][fea_j] += INIT_EPS;
				fo_delta = softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
				this->wi[cate_j][fea_j] -= INIT_EPS;
				fo_delta = fo_delta - softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
				fo_delta /= INIT_EPS;

				this->wi[cate_i][fea_i] -= INIT_EPS;
				this->wi[cate_j][fea_j] += INIT_EPS;
				la_delta = softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
				this->wi[cate_j][fea_j] -= INIT_EPS;
				la_delta = la_delta - softmax_loss(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda);
				la_delta /= INIT_EPS;
				printf("%d %d, %d %d ----------------: %.15lf    %.15lf\n", cate_i, fea_i, cate_j, fea_j, (fo_delta - la_delta) / INIT_EPS, now_hessi[cate_i*this->fea_num+fea_i][cate_j*this->fea_num+fea_j]);
			
		
	}
	vslDeleteStream(&stream);
	return 0;
}

int newton::check_inversion(double** ori_mtx, double** inv_mtx, int sam_num) {
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MT19937, RAN_SEED);
	for (int t = 0; t < sam_num; t++) {
		int now_i, now_j;
		viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &now_i, 0, this->cate*this->fea_num);
		if (t % 5 == 0)
			now_j = now_i;
		else
			viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &now_j, 0, this->cate*this->fea_num);
		double outp_now = cblas_ddot(this->cate*this->fea_num, ori_mtx[now_i], 1, &inv_mtx[0][now_j], this->fea_num*this->cate);
		printf("%d %d ----------------- %.15lf\n", now_i, now_j, outp_now);
	}
	vslDeleteStream(&stream);
	return 0;
}
