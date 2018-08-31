#include "prelu.h"

namespace NNET
{

void PReLU::SetUp(Tensor& Input, Tensor& Output) 
{
	num_slope = GetParam_Int32(0, 0);
	
	this->blobs_.resize(1);
	this->blobs_[0].reset(new Blob(num_slope));
}

void PReLU::Reshape(Tensor& Input, Tensor& Output)
{
	XC = Input[0]->CC();  YC = XC;
	XH = Input[0]->HH();  YH = XH;
	XW = Input[0]->WW();  YW = XW;
	
	Output[0]->Reshape(YC, YH, YW);
	Output[0]->ShareData(*Input[0]);
}

void PReLU::Run(Tensor& Input, Tensor& Output)
{
	float* dst = Output[0]->mutable_cpu_data();
	const float* slope_data = this->blobs_[0]->cpu_data();
	
	int size = YW*YH;
	
	for(int q=0; q<YC; q++)
	{
		float slope = num_slope > 1 ? slope_data[q] : slope_data[0];

		for(int i=0; i<size; i++)
		{
			if(dst[i] < 0)	dst[i] *= slope;
		}
		dst += size;
	}
}

}
