#include "NV1/run/relu_layer.h"

namespace NV1 {

void ReLULayer::SetUp(Tensor& Input, Tensor& Output) 
{
	slope = GetParam_f32(0, 0.f);
	//LOG(INFO) << "ReLU: " << slope;
	
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	YH = XH;
	YW = XW;
	YC = XC;
	
	Output[0]->Reshape(YC, YH, YW);
	Output[0]->ShareData(*Input[0]);
}

void ReLULayer::Run(Tensor& Input, Tensor& Output)
{
	LOG(INFO) << "ReLU: " << YC << " " << YH << " " << YW;

	float* dst = Output[0]->mutable_cpu_data();
	const float* src = Input[0]->cpu_data();
	
	int count = Input[0]->count();
	for (int i = 0; i < count; ++i)
	{
		dst[i] = std::max(src[i], 0.f) + slope * std::min(src[i], 0.f);
	}
}

}
