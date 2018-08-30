#include "nnet/run/dropout_layer.h"
#include <string.h>

namespace NNET
{

void DropoutLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	scale = GetParam_f32(0, 1.f);
}

void DropoutLayer::Reshape(Tensor& Input, Tensor& Output)
{
	XC = Input[0]->CC();  YC = XC;
	XH = Input[0]->HH();  YH = XH;
	XW = Input[0]->WW();  YW = XW;
	
	Output[0]->Reshape(YC, YH, YW);
	Output[0]->ShareData(*Input[0]);
}

void DropoutLayer::Run(Tensor& Input, Tensor& Output)
{
	if(scale != 1.f)
	{
		float* dst = Output[0]->mutable_cpu_data();
		int count  = Output[0]->count();
		
		for(int i=0; i<count; i++)
		dst[i] = dst[i] * scale;
	}
	//memcpy(dst, src, sizeof(float) * count);
}

}
