#include"lant.h"
#include"config.h"
#include"grad.h"
#include"hessi.h"
#include"check.h"
#include"svrg.h"
#include<mkl.h>
#include<cstring>
#include<cstdio>
#include<cmath>

int lant::init_lant(char* fea_file, char* label_file, double lambda, int iter_num, int opt_rank, double gua_sigma) {
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
	this->fea_rank = opt_rank;
	this->gau_sigma = gua_sigma;
	return 0;
}

int lant::init_lant(svrg* opt_pre, int iter_num, int opt_rank, double gua_sigma) {
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
	this->fea_rank = opt_rank;
	this->gau_sigma = gua_sigma;

	return 0;
}

int lant::find_aprx(double** half_hessi, double &tuc_lam) {
	double** tmp_halhessi;
	double** muti_halhessi;
	double** ker_squ;
	double** delta_var;
	double** tmp_grad;
	double** sta_grad;
	double** tmp_u;
	double** tmp_v;
	double** tmp_ure;
	double* tmp_sigma;
	double* tmp_superb;

	int now_rank = 0;
	VSLStreamStatePtr aprx_stream;
	vslNewStream(&aprx_stream, VSL_BRNG_MCG31, RAN_SEED);

	tmp_halhessi = (double**)malloc(sizeof(double*)*this->cate*this->fea_num);
	tmp_halhessi[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num*(this->fea_rank + OVER_SAMP));
	for (int i = 1; i < this->cate*this->fea_num; i++)
		tmp_halhessi[i] = tmp_halhessi[0] + i*(this->fea_rank + OVER_SAMP);

	muti_halhessi = (double**)malloc(sizeof(double*)*this->cate*this->fea_num);
	muti_halhessi[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num*(this->fea_rank + OVER_SAMP));
	for (int i = 1; i < this->cate*this->fea_num; i++)
		muti_halhessi[i] = muti_halhessi[0] + i*(this->fea_rank + OVER_SAMP);

	ker_squ = (double**)malloc(sizeof(double*)*(this->fea_rank + OVER_SAMP));
	ker_squ[0] = (double*)malloc(sizeof(double)*(this->fea_rank + OVER_SAMP)*(this->fea_rank + OVER_SAMP));
	for (int i = 1; i < this->fea_rank + OVER_SAMP; i++)
		ker_squ[i] = ker_squ[0] + i*(this->fea_rank + OVER_SAMP);

	delta_var = (double**)malloc(sizeof(double)*this->cate);
	delta_var[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		delta_var[i] = delta_var[0] + i*this->fea_num;

	tmp_grad = (double**)malloc(sizeof(double)*this->cate);
	tmp_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		tmp_grad[i] = tmp_grad[0] + i*this->fea_num;

	sta_grad = (double**)malloc(sizeof(double)*this->cate);
	sta_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		sta_grad[i] = sta_grad[0] + i*this->fea_num;

	softmax_grad(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda, sta_grad);

	for (int i = 0; i < (this->fea_rank + OVER_SAMP); i++) {
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, aprx_stream, this->cate*this->fea_num, delta_var[0], APRX_RAN_MU, APRX_RAN_SIGMA);
		cblas_daxpy(this->cate*this->fea_num, 1.0, this->wi[0], 1, delta_var[0], 1);
		softmax_grad(this->exp_num, this->cate, this->fea_num, delta_var, this->xi, this->yi, this->lambda, tmp_grad);
		cblas_dcopy(this->cate*this->fea_num, tmp_grad[0], 1, &tmp_halhessi[0][i], (this->fea_rank + OVER_SAMP));
		cblas_daxpy(this->cate*this->fea_num, -1.0, sta_grad[0], 1, &tmp_halhessi[0][i], (this->fea_rank + OVER_SAMP));
		cblas_daxpy(this->cate*this->fea_num, -1.0, this->wi[0], 1, delta_var[0], 1);
	}

	//-----QR for tmp_halhessi-----//
	double* tau;
	tau = (double*)malloc(sizeof(double)*(this->fea_rank + OVER_SAMP));
	LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (this->cate*this->fea_num), (this->fea_rank+OVER_SAMP), tmp_halhessi[0], (this->fea_rank+OVER_SAMP), tau);
	LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (this->cate*this->fea_num), (this->fea_rank+OVER_SAMP), (this->fea_rank + OVER_SAMP), tmp_halhessi[0], (this->fea_rank+OVER_SAMP), tau);

	cblas_dscal(this->cate*this->fea_num*(this->fea_rank+OVER_SAMP), APRX_RAN_SIGMA, tmp_halhessi[0], 1);

	for (int i = 0; i < (this->fea_rank + OVER_SAMP); i++) {
		cblas_dcopy(this->cate*this->fea_num, &tmp_halhessi[0][i], (this->fea_rank + OVER_SAMP), delta_var[0], 1);
		cblas_daxpy(this->cate*this->fea_num, 1.0, this->wi[0], 1, delta_var[0], 1);
		softmax_grad(this->exp_num, this->cate, this->fea_num, delta_var, this->xi, this->yi, this->lambda, tmp_grad);
		cblas_dcopy(this->cate*this->fea_num, tmp_grad[0], 1, &muti_halhessi[0][i], (this->fea_rank + OVER_SAMP));
		cblas_daxpy(this->cate*this->fea_num, -1.0, sta_grad[0], 1, &muti_halhessi[0][i], (this->fea_rank + OVER_SAMP));
		cblas_daxpy(this->cate*this->fea_num, -1.0, this->wi[0], 1, delta_var[0], 1);
	}

	cblas_dscal(this->cate*this->fea_num*(this->fea_rank + OVER_SAMP), 1/APRX_RAN_SIGMA, tmp_halhessi[0], 1);
	cblas_dscal(this->cate*this->fea_num*(this->fea_rank + OVER_SAMP), 1 / APRX_RAN_SIGMA, muti_halhessi[0], 1);

	//need modify
	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, this->fea_rank+OVER_SAMP, this->fea_rank+OVER_SAMP, this->cate*this->fea_num, 1.0, muti_halhessi[0], this->cate*this->fea_num, muti_halhessi[0], this->cate*this->fea_num, 0.0, ker_squ[0], this->fea_rank);
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, this->fea_rank+OVER_SAMP, this->fea_rank+OVER_SAMP, this->cate*this->fea_num, 1.0, muti_halhessi[0], this->fea_rank+OVER_SAMP, muti_halhessi[0], this->fea_rank+OVER_SAMP, 0, ker_squ[0], this->fea_rank+OVER_SAMP);

	tmp_u = (double**)malloc(sizeof(double*)*(this->fea_rank + OVER_SAMP));
	tmp_u[0] = (double*)malloc(sizeof(double)*(this->fea_rank + OVER_SAMP)*(this->fea_rank+OVER_SAMP));
	for (int i = 1; i < this->fea_rank + OVER_SAMP; i++)
		tmp_u[i] = tmp_u[0] + i*(this->fea_rank + OVER_SAMP);

	tmp_v = (double**)malloc(sizeof(double*)*(this->fea_rank + OVER_SAMP));
	tmp_v[0] = (double*)malloc(sizeof(double)*(this->fea_rank + OVER_SAMP)*(this->fea_rank + OVER_SAMP));
	for (int i = 1; i < this->fea_rank + OVER_SAMP; i++)
		tmp_v[i] = tmp_v[0] + i*(this->fea_rank + OVER_SAMP);

	tmp_sigma = (double*)malloc(sizeof(double)*(this->fea_rank + OVER_SAMP));
	tmp_superb = (double*)malloc(sizeof(double)*(this->fea_rank + OVER_SAMP));

	LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', this->fea_rank + OVER_SAMP, this->fea_rank + OVER_SAMP, ker_squ[0], this->fea_rank + OVER_SAMP, tmp_sigma, tmp_u[0], (this->fea_rank + OVER_SAMP), tmp_v[0], (this->fea_rank + OVER_SAMP), tmp_superb);
	
	tuc_lam = 1/sqrt(tmp_sigma[this->fea_rank+1]);

	tmp_ure = (double**)malloc(sizeof(double*)*(this->fea_rank + OVER_SAMP));
	tmp_ure[0] = (double*)malloc(sizeof(double)*(this->fea_rank + OVER_SAMP)*this->fea_rank);
	for (int i = 1; i < this->fea_rank + OVER_SAMP; i++)
		tmp_ure[i] = tmp_ure[0] + i*this->fea_rank;
	for (int i = 0; i < this->fea_rank; i++) {
		cblas_dcopy(this->fea_rank + OVER_SAMP, &tmp_u[0][i], this->fea_rank + OVER_SAMP, &tmp_ure[0][i], this->fea_rank);
		cblas_dscal(this->fea_rank + OVER_SAMP, 1 / (sqrt(tmp_sigma[i])), &tmp_ure[0][i], this->fea_rank);
	}
		

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, this->cate*this->fea_num, this->fea_rank, this->fea_rank + OVER_SAMP, 1.0, tmp_halhessi[0], this->fea_rank + OVER_SAMP, tmp_ure[0], this->fea_rank, 0, half_hessi[0], this->fea_rank);

	// release
	free(tmp_superb);
	free(tmp_sigma);
	free(tmp_ure[0]);
	free(tmp_ure);
	free(tmp_v[0]);
	free(tmp_v);
	free(tmp_u[0]);
	free(tmp_u);
	free(tmp_grad[0]);
	free(tmp_grad);
	free(sta_grad[0]);
	free(sta_grad);
	free(delta_var[0]);
	free(delta_var);
	free(ker_squ[0]);
	free(ker_squ);
	free(muti_halhessi[0]);
	free(muti_halhessi);
	free(tmp_halhessi[0]);
	free(tmp_halhessi);
	return 0;
}

int lant::find_opt(bool pre_opted) {
	VSLStreamStatePtr stream;
	double** half_hessi;
	double tuc_lam;
	double** gd_grad;
	double* mid_muti;

	half_hessi = (double**)malloc(sizeof(double)*this->cate*this->fea_num);
	half_hessi[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num*this->fea_rank);
	for (int i = 1; i < this->cate*this->fea_rank; i++)
		half_hessi[i] = half_hessi[0] + i*this->fea_rank;

	gd_grad = (double**)malloc(sizeof(double)*this->cate);
	gd_grad[0] = (double*)malloc(sizeof(double)*this->cate*this->fea_num);
	for (int i = 1; i < this->cate; i++)
		gd_grad[i] = gd_grad[0] + i*this->fea_num;

	mid_muti = (double*)malloc(sizeof(double)*this->fea_rank);

	if (pre_opted == false) {
		vslNewStream(&stream, VSL_BRNG_MCG31, RAN_SEED);
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, this->cate*this->fea_num, this->wi[0], RAN_MU, RAN_SIGMA);
	}




	for (int t = 0; t < this->iter_num; t++) {
		softmax_grad(this->exp_num, this->cate, this->fea_num, this->wi, this->xi, this->yi, this->lambda, gd_grad);
		find_aprx(half_hessi, tuc_lam);
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, this->fea_rank, 1, this->cate*this->fea_num, 1.0, half_hessi[0], this->fea_rank, gd_grad[0], 1, 0, mid_muti, 1);
		//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, this->cate*this->fea_num, 1, this->fea_rank, 1.0, half_hessi[0], this->fea_rank, mid_muti, 1, tuc_lam, gd_grad[0], 1);
	}
}