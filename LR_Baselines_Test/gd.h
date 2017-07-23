#ifndef GD_H
#define GD_H
struct gd {
	// input para
	int exp_num, fea_num,cate;
	double** wi;
	double** xi;
	int* yi;

	// model para
	double lambda;
	double eta;
	int iter_num;

	int init_gd(char* fea_file, char* label_file, double lambda, double eta, int iter_num);
	int find_opt();
	bool check_grad();
};
#endif
