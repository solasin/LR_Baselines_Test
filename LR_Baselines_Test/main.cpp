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
newton opt_1;
int main(){
	opt_1.init_newton("E:\\Conference\\ICML18\\dataset\\post-processing\\minist\\train-images.in", "E:\\Conference\\ICML18\\dataset\\post-processing\\minist\\train-labele.in",1e-4,100);
	opt_1.find_opt();
	/*double* a;
	double* b;
	int* ipv;
	a = (double*)malloc(sizeof(double) * 4);
	for (int i = 0; i < 4; i++)
		a[i] = i*1.0;
	ipv = (int*)malloc(sizeof(int) * 2);
	b = (double*)malloc(sizeof(double) * 4);
	b[0] = b[3] = 1;
	b[1] = b[2] = 0;
	double* c;
	c = (double*)malloc(sizeof(double) * 2);
	//LAPACKE_dgesv(LAPACK_ROW_MAJOR, 2, 2, a, 2, ipv, b, 2);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,2,1,2,1.0,a,2,b,1,0.0,c,1);*/
	return 0;
}