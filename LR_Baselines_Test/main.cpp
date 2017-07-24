#include<iostream>
#include<string>
#include<cstdio>
#include<cstring>
#include<mkl.h>
#include"sgd.h"
#include"svrg.h"
#include"gd.h"
#include"saga.h"
svrg opt_1;
int main(){
	opt_1.init_svrg("E:\\Conference\\ICML18\\dataset\\post-processing\\minist\\train-images.in", "E:\\Conference\\ICML18\\dataset\\post-processing\\minist\\train-labele.in",1e-4, 0.025,100);
	opt_1.find_opt();
	return 0;
}