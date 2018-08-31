#include "input.h"

namespace NNET
{

void Input::SetUp(Tensor& Input, Tensor& Output) 
{
	YW = stoi(this->layer_param_[0]);
	YH = stoi(this->layer_param_[1]);
	YC = stoi(this->layer_param_[2]);
	
	Output[0]->Reshape(YC, YH, YW);
}

void Input::L2Net(Net *net)
{
	net->_NH = YH;
	net->_NW = YW;
	net->_NC = YC;
}

void Input::Reshape(Tensor& Input, Tensor& Output)
{
	YC = Output[0]->CC();
	YH = Output[0]->HH();
	YW = Output[0]->WW();
	
	Output[0]->Reshape(YC, YH, YW);
}

void Input::Run(Tensor& Input, Tensor& Output)
{
	
}

}