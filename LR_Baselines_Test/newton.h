#ifndef NEWTON_H
#define NEWTON_H
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
	int find_opt();
	bool check_hessian();
};
#endif // !NEWTON_H