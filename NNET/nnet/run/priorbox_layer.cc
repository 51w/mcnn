#include "nnet/run/priorbox_layer.h"

namespace NNET
{

void PriorBoxLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	YH = XH;
	YW = XW;
	YC = XC;
	
	Output[0]->Reshape(YC, YH, YW);
}

void PriorBoxLayer::Run(Tensor& Input, Tensor& Output)
{
	
}

}
