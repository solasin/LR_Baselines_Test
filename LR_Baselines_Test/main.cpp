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
svrg opt_1;
quasi_newton opt_2;
int main(){
	opt_1.init_svrg("E:\\Conference\\ICML18\\dataset\\post-processing\\minist\\train-images.in", "E:\\Conference\\ICML18\\dataset\\post-processing\\minist\\train-labele.in",1e-4,0.0025,3);
	opt_1.find_opt();
	opt_2.init_quasi_newton(&opt_1, 100);
	opt_1.relese_svrg();
	opt_2.find_opt(true);
	return 0;
}