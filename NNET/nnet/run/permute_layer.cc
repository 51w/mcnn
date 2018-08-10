#include "nnet/run/permute_layer.h"
#include <string.h>

namespace NNET
{
// order_type
// 0 = w h c
// 1 = h w c
// 2 = w c h
// 3 = c w h
// 4 = h c w
// 5 = c h w

void PermuteLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	order_type = GetParam_Int32(0, 0);
	
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	YH = XH;
	YW = XW;
	YC = XC;
	
	Output[0]->Reshape(YC, YH, YW);
}

void PermuteLayer::Run(Tensor& Input, Tensor& Output)
{
	
}

}
