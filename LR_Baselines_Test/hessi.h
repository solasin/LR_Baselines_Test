#ifndef HESSI_H
#define HESSI_H
int softmax_hessi(int var_idx, int exp_num, int cate, int fea_num, double** wi, double** xi, int* yi, double lambda, double** delta_hessi);
int softmax_sto_hessi(int var_idx, int delta_exp, int cate, int fea_num, double** wi, double** xi, int* yi, double lambda, double** delta_hessi);
#endif // !HESSI_H
