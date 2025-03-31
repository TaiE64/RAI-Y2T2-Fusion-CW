clear;clc;
load('C:\Users\33582\Desktop\OptimizationCW\trainingData\task1_1.mat');
[X_Est, P_Est, GT]=myEKF(out);
% save('Task1_4_Estimation.mat', 'X_Est');