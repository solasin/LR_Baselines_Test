#include<iostream>
#include<string>
#include<cstdio>
#include<cstring>
#include<mkl.h>
#include"sgd.h"
#include"svrg.h"


sgd opt_1;
int main(){
	opt_1.init_sgd("E:\\Conference\\ICML18\\dataset\\post-processing\\check_data\\test_feature.in", "E:\\Conference\\ICML18\\dataset\\post-processing\\check_data\\test_label.in",1, 0.005, false, 1000000);
	opt_1.find_opt();
}