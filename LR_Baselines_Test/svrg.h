#ifndef SVRG_H
#define SVRG_H
struct svrg{
	// input para
	int exp_num, fea_num, cate;
	double** wi;
	double** xi;
	int* yi;

	// model para
	double lambda;
	double eta;
	int epoch;

	int init_svrg(char* fea_file, char* label_file, double lambda, double eta, int epoch);
	int find_opt();
	int relese_svrg();
};
#endif