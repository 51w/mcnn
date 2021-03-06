#include "power.h"
#include <math.h>

namespace NNET
{

void Power::SetUp(Tensor& Input, Tensor& Output) 
{
	power = GetParam_f32(0, 1.f);
    scale = GetParam_f32(1, 1.f);
    shift = GetParam_f32(2, 0.f);
}

void Power::Reshape(Tensor& Input, Tensor& Output)
{
	XC = Input[0]->CC();  YC = XC;
	XH = Input[0]->HH();  YH = XH;
	XW = Input[0]->WW();  YW = XW;
	
	Output[0]->Reshape(YC, YH, YW);
	//Output[0]->ShareData(*Input[0]);
}

// Compute y = (shift + scale * x)^power
void Power::Run(Tensor& Input, Tensor& Output)
{
	float* dst = Output[0]->mutable_cpu_data();
	const float* src = Input[0]->cpu_data();
	
	int count = Output[0]->count();
	for(int i=0; i < count; ++i)
	{
		dst[i] = pow((shift + src[i] * scale), power);
	}
}

}
