#ifndef LANT_H
#define LANT_H
#include"gd.h"
#include"sgd.h"
#include"svrg.h"
#include"saga.h"

struct lant {
	//input para
	int exp_num, fea_num, cate;
	double** wi;
	double** xi;
	int* yi;

	//model para
	double lambda;
	int iter_num;
	int fea_rank;

	int init_lant(char* fea_file, char* label_file, double lambda, int iter_num, int opt_rank, double gau_sigma);
	int init_lant(svrg* opt_pre, int iter_num, int opt_rank);
	int find_opt(bool pre_opted);
	int find_aprx(double** half_hessi, double &tuc_lam);
	int check_aprx();
	
};

#endif LANT_H
