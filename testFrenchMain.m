clc;clear all;close all;

addpath C:\Users\Berker\Documents\GitHub\SCE_project
load C:\Users\Berker\Documents\GitHub\SCE_project\EmotionEvents.mat
audioPath='D:\JOKER\FrenchDataset\audio\05\20150216_144243_00.wav';
featurePath='C:\Users\Berker\Documents\GitHub\SCE_project\textSource\';

[pred, real]=test_on_french(audioPath,EmotionEvents,featurePath,2);
