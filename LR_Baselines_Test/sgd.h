#ifndef SGD_H
#define SGD_H
struct sgd{
	// input para
	int exp_num, fea_num;
	double* wi;
	double** xi;
	double* yi;

	// model para
	double lambda;
	double eta; 
	bool step_dimin;
	int iter_num;

	int init_sgd(char* fea_file, char* label_file, double lambda, double eta, bool step_dimin, int iter_num);
	int find_opt();
	bool check_grad();
};
#endif