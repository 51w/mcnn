#include "nnet/run/batchnorm_layer.h"
#include <math.h>

namespace NNET
{

void BatchNormLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	channels = GetParam_Int32(0, 0);
    eps = GetParam_f32(1, 0.f);
	CHECK(channels == Input[0]->CC());
	
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	
	YH = XH;
	YW = XW;
	YC = XC;
	
	Output[0]->Reshape(YC, YH, YW);
	Output[0]->ShareData(*Input[0]);
	
	init = 0;
	a_data.resize(channels);
	b_data.resize(channels);
	
	this->blobs_.resize(4);
	this->blobs_[0].reset(new Blob(channels));
	this->blobs_[1].reset(new Blob(channels));
	this->blobs_[2].reset(new Blob(channels));
	this->blobs_[3].reset(new Blob(channels));
}

void BatchNormLayer::Run(Tensor& Input, Tensor& Output)
{
	float* dst = Output[0]->mutable_cpu_data();

#if 1	
	if(!init)
	{
		const float* slope_data = this->blobs_[0]->cpu_data();
		const float* mean_data  = this->blobs_[1]->cpu_data();
		const float* var_data   = this->blobs_[2]->cpu_data();
		const float* bias_data  = this->blobs_[3]->cpu_data();
		
		for(int i=0; i<channels; i++)
		{
			float sqrt_var = sqrt(var_data[i] + eps);
			a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
			b_data[i] = slope_data[i] / sqrt_var;
		}
		init = 1;
		
		for(int q=0; q<channels; q++)
		{
			LOG(INFO) << mean_data[q] << " " << var_data[q] << "      "
				<< slope_data[q] << " " << bias_data[q] << " " << eps;
		}
	}
	
	int size = YH*YW;
	for(int q=0; q<YC; q++)
	{
		float a = a_data[q];
		float b = b_data[q];

		for (int i=0; i<size; i++)
		{
			dst[i] = b * dst[i] + a;
		}
		dst += size;
	}
#else 

	const float* mean_data  = this->blobs_[1]->cpu_data();
	const float* var_data   = this->blobs_[2]->cpu_data();
	int size = YH*YW;
	for(int q=0; q<YC; q++)
	{
		for (int i=0; i<size; i++)
		{
			dst[i] = (dst[i]-mean_data[q]) / sqrt(var_data[q]);
		}
		dst += size;
	}
#endif

/*	
	FILE* pp = fopen("000/1.txt", "wb");
	Dtype* out = Output[0]->mutable_cpu_data();
	int count  = Output[0]->count();
	for(int i=0; i<count; i++)
	{
		fprintf(pp, "%f  -- bn\n", out[i]);	
	}
	fclose(pp);
	LOG(FATAL) << "TXT.";
*/
}

}
