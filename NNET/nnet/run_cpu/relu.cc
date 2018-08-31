#include "relu.h"

namespace NNET
{

void ReLU::SetUp(Tensor& Input, Tensor& Output) 
{
	slope = GetParam_f32(0, 0.f);
}

void ReLU::Reshape(Tensor& Input, Tensor& Output)
{
	XC = Input[0]->CC();  YC = XC;
	XH = Input[0]->HH();  YH = XH;
	XW = Input[0]->WW();  YW = XW;
	
	Output[0]->Reshape(YC, YH, YW);
	//Output[0]->ShareData(*Input[0]);
}

void ReLU::Run(Tensor& Input, Tensor& Output)
{
	float* dst = Output[0]->mutable_cpu_data();
	const float* src = Input[0]->cpu_data();
	
	int count = Input[0]->count();
	for(int i=0; i < count; ++i)
	{
		dst[i] = std::max(src[i], 0.f) + slope * std::min(src[i], 0.f);
	}
}

}
