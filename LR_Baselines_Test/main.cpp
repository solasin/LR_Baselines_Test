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
	opt_1.find_opt(false);
	return 0;
}