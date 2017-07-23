#ifndef SAGA_H 
#define SAGA_H
struct saga{
	// input para
	int exp_num, fea_num,cate;
	double** wi;
	double** xi;
	int* yi;

	// model para
	double lambda;
	double eta;
	int iter_num;
	double*** wi_set;

	int init_saga(char* fea_file, char* label_file, double lambda, double eta, int iter_num);
	int find_opt();
};
#endif