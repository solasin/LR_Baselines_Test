#include<iostream>
#include<string>
#include<cstdio>
#include<cstring>
#include<mkl.h>
#include"sgd.h"
#include"svrg.h"
#include"gd.h"
#include"saga.h"
#include"newton.h"
#include"quasi_newton.h"
#include"lant.h"
svrg opt_1;
lant opt_2;
int main(){
	opt_1.init_svrg("E:\\Conference\\ICML18\\dataset\\post-processing\\minist\\train-images.in", "E:\\Conference\\ICML18\\dataset\\post-processing\\minist\\train-labele.in",1e-4,0.025,3);
	opt_1.find_opt();
	opt_2.init_lant(&opt_1, 100,200);
	opt_1.relese_svrg();
	opt_2.find_opt(true);

	//---------- SVD Case --------//
	/*double** a;
	a = (double**)malloc(sizeof(double*) * 3);
	a[0] = (double*)malloc(sizeof(double) * 3 * 4);
	for (int i = 1; i < 3; i++)
		a[i] = a[0] + 4 * i;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			a[i][j] = i * 4 + j;
	double** u;
	double** vt;
	double* s;
	double* superb;
	s = (double*)malloc(sizeof(double) * 3);
	superb = (double*)malloc(sizeof(double) * 4);
	u = (double**)malloc(sizeof(double*) * 3);
	u[0] = (double*)malloc(sizeof(double) * 3 * 3);
	for (int i = 1; i < 3; i++)
		u[i] = u[0] + i * 3;
	vt = (double**)malloc(sizeof(double*) * 4);
	vt[0] = (double*)malloc(sizeof(double) * 4 * 4);
	for (int i = 1; i < 4; i++)
		vt[i] = vt[0] + i * 4;
	LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', 3, 4, a[0], 4, s, u[0], 3, vt[0], 4, superb);
	printf("U = \n");
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++)
			printf("%lf ", u[i][j]);
		printf("\n");
	}
	printf("V = \n");
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++)
			printf("%lf ", vt[i][j]);
		printf("\n");
	}
	printf("Sigma = \n");
	for (int i = 0; i < 3; i++)
		printf("%d = %lf\n", i, s[i]);
	*/

	// ----------------- dgemm Trans Case ---------------//
	/*double** a;
	a = (double**)malloc(sizeof(double*) * 4);
	a[0] = (double*)malloc(sizeof(double) * 8);
	for (int i = 1; i < 4; i++)
		a[i] = a[0] + 2 * i;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 2; j++)
			a[i][j] = i * 2 + j;

	double** ans;
	ans = (double**)malloc(sizeof(double*) * 2);
	ans[0] = (double*)malloc(sizeof(double) * 2 * 2);
	ans[1] = ans[0] + 2;

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 2, 2, 4, 1.0, a[0], 2, a[0], 2, 0, ans[0], 2);

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++)
			printf("%lf ", ans[i][j]);
		printf("\n");
	}*/
	return 0;
}