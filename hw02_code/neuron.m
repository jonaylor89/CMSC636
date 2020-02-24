#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Contains: Custom Neuron
# Name: hw_xor_neural.m
# Course Instructor: Milos Manic
# Provided by: John Naylor
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all

#load Truth Table from .txt file#
neuronTT=load("neuron.txt");
dim=size(neuronTT);
num_entries=dim(1);

#Seperate input matrix and output vector from Truth Table#
in=neuronTT(:,1:end-1);
out=neuronTT(:,end);

#Transpose them for Neural Toolbox function usage#
in=in';
out=out';

#Min_max is required for creating a Feed-Forward Network in Octave
minmax_in=min_max(in);

#Create Feed-Forward Network using newff function#
#<For additional help, type "help newff"(without quotes) on the Octave prompt#
MLPnet=newff(minmax_in,[3 1],{"tansig","logsig"},"trainlm","learngdm","mse");

#Save Neural Network Structure in a text file
saveMLPStruct(MLPnet,"neuron_net.txt");

#Show training performance every 1 step#
MLPnet.trainParam.show = 1;

#Train the neural network#
[net]=train(MLPnet,in,out);
