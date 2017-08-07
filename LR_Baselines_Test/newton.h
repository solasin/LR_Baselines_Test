#ifndef NEWTON_H
#define NEWTON_H
#include"sgd.h"
#include"saga.h"
#include"svrg.h"
#include"gd.h";
struct newton {
	//input para
	int exp_num, fea_num, cate;
	double** wi;
	double** xi;
	int* yi;

	//model para
	double lambda;
	int iter_num;

	int init_newton(char* fea_file, char* label_file, double lambda, int iter_num);
	int init_newton(svrg* opt_pre, int iter_num);
	int find_opt(bool pre_opted);
	int check_hessian(double** now_hessi,int sam_num);
	int check_inversion(double** ori_mtx, double** inv_mtx, int sam_num);
};
#endif // !NEWTON_H