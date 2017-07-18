#ifndef GRAD_H
#define GRAD_H
int lr_grad(int exp_num, int fea_num, double* wi, double** xi, double* yi, double lambda, double* delta_wi);
int lr_sto_grad(int delta_exp, int fea_num, double* wi, double** xi, double* yi, double lambda, double* delta_wi);
int lr_mini_gra(int batch_len, int* batch, int fea_num, double* wi, double** xi, double* yi, double lambda,double* delta_wi);
#endif