#include "lrn.h"
#include <math.h>

namespace NNET
{

void LRN::SetUp(Tensor& Input, Tensor& Output) 
{
	region_type = GetParam_Int32(0, 0);
    local_size  = GetParam_Int32(1, 5);
    alpha = GetParam_f32(2, 1.f);
    beta  = GetParam_f32(3, 0.75f);
    bias  = GetParam_f32(4, 1.f);
	CHECK(region_type == 0) << "LRN only support ACROSS_CHANNELS";
}

void LRN::Reshape(Tensor& Input, Tensor& Output)
{
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	
	if(region_type == NormRegion_ACROSS_CHANNELS)
    {
		YH = XH;
		YW = XW;
		YC = XC;
		Output[0]->Reshape(YC, YH, YW);
		square_blob.Reshape(YC,YH, YW);
		square_sum.Reshape(YC, YH, YW);
	}
}

void LRN::Run(Tensor& Input, Tensor& Output)
{
	float* dst = Output[0]->mutable_cpu_data();
	float* src = Input[0]->mutable_cpu_data();
	
	float* psqr = square_blob.mutable_cpu_data();
	float* psum = square_sum.mutable_cpu_data();
	
	int count = Input[0]->count();
	for(int i=0; i<count; i++)
    {
		psqr[i] = src[i] * src[i];
	}
	
	if(region_type == NormRegion_ACROSS_CHANNELS)
	{
		for(int i=0; i<count; i++)
		{	psum[i] = 0.f;  }
		
		const float alpha_div_size = alpha / local_size;

		for(int q=0; q<YC; q++)
		{
			// square sum
			float* ssptr = psum + q*YW*YH;
			for(int p=q - local_size / 2; p<=q + local_size / 2; p++)
			{
				if(p < 0 || p >= YC)
					continue;

				const float* sptr = psqr + p*YW*YH;
				for(int i=0; i<YW*YH; i++)
				{
					ssptr[i] += sptr[i];
				}
			}

			float* ptrI = src + q*YW*YH;
			float* ptrO = dst + q*YW*YH;
			for(int i=0; i<YW*YH; i++)
			{
				ptrO[i] = ptrI[i] * pow(bias + alpha_div_size * ssptr[i], -beta);
			}
		}
	}
	else if (region_type == NormRegion_WITHIN_CHANNEL)
    {
		
	}
}

} // NNET