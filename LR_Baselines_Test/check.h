#ifndef CHECK_H
#define CHECK_H
double lr_loss(int exp_num, int fea_num, double* wi, double** xi, double* yi, double lambda);
double softmax_loss(int exp_num, int cate, int fea_num, double** wi, double** xi, int* yi, double lambda);
double softmax_sto_loss(int delta_exp, int cate, int fea_num, double** wi, double** xi, int* yi, double lambda);
#endif