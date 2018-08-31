#include "permute.h"
#include <string.h>

namespace NNET
{
// order_type
// 0 = w h c  <--> init
// 1 = h w c
// 2 = w c h
// 3 = c w h
// 4 = h c w
// 5 = c h w
void Permute::SetUp(Tensor& Input, Tensor& Output) 
{
	order_type = GetParam_Int32(0, 0);
	CHECK(order_type == 3);
}

void Permute::Reshape(Tensor& Input, Tensor& Output)
{
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	
	YH = XW;
	YW = XC;
	YC = XH;
	Output[0]->Reshape(YC, YH, YW);
}

void Permute::Run(Tensor& Input, Tensor& Output)
{
	float* dst = Output[0]->mutable_cpu_data();
	const float* src = Input[0]->cpu_data();
	
	for(int q=0; q<YC; q++)
	{
		for(int i=0; i<YH; i++)
		{
			for(int j=0; j<YW; j++)
			{
				const float* ptr = src + j*XW*XH + q*XW;
				dst[i*YW + j] = ptr[i];
			}
		}
		dst += YH*YW;
	}
}

}
