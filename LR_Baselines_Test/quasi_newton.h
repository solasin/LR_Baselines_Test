#ifndef QUASI_NEWTON_H
#define QUASI_NEWTON_H
#include"sgd.h"
#include"saga.h"
#include"svrg.h"
#include"gd.h";
struct quasi_newton {
	//input para
	int exp_num, fea_num, cate;
	double** wi;
	double** xi;
	int* yi;

	//model para
	double lambda;
	int iter_num;

	int init_quasi_newton(char* fea_file, char* label_file, double lambda, int iter_num);
	int init_quasi_newton(svrg* opt_pre, int iter_num);
	int find_opt(bool pre_opted);
	double backtracking_armijo(double init_step, double** newton_step, double** g_grad);
};


#endif // !QUASI_NEWTON
