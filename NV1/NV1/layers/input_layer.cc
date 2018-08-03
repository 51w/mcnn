#include "NV1/layers/input_layer.h"

namespace NV1{

void InputLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	YW = stoi(this->layer_param_[0]);
	YH = stoi(this->layer_param_[1]);
	YC = stoi(this->layer_param_[2]);

	Output[0]->Reshape(YC, YH, YW);
}

void InputLayer::Run(Tensor& Input, Tensor& Output)
{
	//
}

}