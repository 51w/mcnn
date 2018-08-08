#include "NV1/run/dropout_layer.h"
#include <string.h>

namespace NV1 {

void DropoutLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	//scale = this->layer_param_.size()>0 ? stof(this->layer_param_[0]) : 1.f;
	scale = GetParam_f32(0, 1.f);
	
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	YH = XH;
	YW = XW;
	YC = XC;
	
	Output[0]->Reshape(YC, YH, YW);
	
	//LOG(INFO) << "DropoutLayer scale[1.f]: " << scale;
}

void DropoutLayer::Run(Tensor& Input, Tensor& Output)
{
	LOG(INFO) << "DropOUT: " << YC << " " << YH << " " << YW;
	
	float* dst = Output[0]->mutable_cpu_data();
	const float* src = Input[0]->cpu_data();
	
	int count = Input[0]->count();
	memcpy(dst, src, sizeof(float) * count);
}

}
