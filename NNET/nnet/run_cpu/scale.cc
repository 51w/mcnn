#include "scale.h"
#include <string.h>

namespace NNET
{

void Scale::SetUp(Tensor& Input, Tensor& Output) 
{
	scale_size = GetParam_Int32(0, 0);
    bias_term  = GetParam_Int32(1, 0);
	
	if(bias_term)
	{
		this->blobs_.resize(2);
		this->blobs_[1].reset(new Blob(scale_size));
		this->blobs_[0].reset(new Blob(scale_size));
	}else
	{
		this->blobs_.resize(1);
		this->blobs_[0].reset(new Blob(scale_size));
	}
}

void Scale::Reshape(Tensor& Input, Tensor& Output)
{
	XC = Input[0]->CC();  YC = XC;
	XH = Input[0]->HH();  YH = XH;
	XW = Input[0]->WW();  YW = XW;
	
	Output[0]->Reshape(YC, YH, YW);
	Output[0]->ShareData(*Input[0]);
}

void Scale::Run(Tensor& Input, Tensor& Output)
{
	float* dst = Output[0]->mutable_cpu_data();
	const float* scale_blob = this->blobs_[0]->cpu_data();
	int size = YW*YH;
	
	if(bias_term)
	{
		const float* bias_data  = this->blobs_[1]->cpu_data();
		for(int q=0; q<YC; ++q)
		{
			float s = scale_blob[q];
            float bias = bias_data[q];
				
			for(int i=0; i<size; ++i)
			{
				dst[i] = dst[i] * s + bias;
			}
			dst += size;
		}
	}
	else
	{
		for(int q=0; q<YC; ++q)
		{
			float s = scale_blob[q];
				
			for(int i=0; i<size; ++i)
			{
				dst[i] = dst[i] * s;
			}
			dst += size;
		}
	}
}

}
