#include "nnet/run/eltwise_layer.h"
#include <string.h>

namespace NNET
{

void EltwiseLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	op_type = GetParam_Int32(0, 0);
	coeff_size = GetParam_Int32(1, 0);
	CHECK(coeff_size == 0) << "Eltwise coeff no use.";
	
	coeffs = vector<float>(Input.size(), 1);
	
	for(int i=0; i<Input.size(); i++)
	CHECK(Input[0]->shape() == Input[i]->shape());
	
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	YH = XH;
	YW = XW;
	YC = XC;
	
	Output[0]->Reshape(YC, YH, YW);
	//Output[0]->ShareData(*Input[0]);
}

void EltwiseLayer::Run(Tensor& Input, Tensor& Output)
{
	const int count = Output[0]->count();
	float* dst = Output[0]->mutable_cpu_data();
	
	if(op_type == Operation_PROD)
    {
		const float *src0 = Input[0]->cpu_data();
		const float *src1 = Input[1]->cpu_data();
		for(int i=0; i<count; ++i)
		{
			dst[i] = src0[i] * src1[i];
		}
		
		for(int i=2; i<Input.size(); ++i)
		{
			const float *src = Input[i]->cpu_data();
			for(int j=0; j<count; ++j)
			{
				dst[j] = dst[j] * src[j];
			}
		} 
	}
	else if (op_type == Operation_SUM)
	{
		for(int i=0; i<count; ++i)
		{  dst[i] = 0.f;  }

		for(int i=0; i<Input.size(); ++i)
		{
			const float *src = Input[i]->cpu_data();
			for(int j=0; j<count; ++j)
			{
			  dst[j] += coeffs[i] * src[j];
			}
		}  
	}
	else if (op_type == Operation_MAX)
    {
		LOG(FATAL) << "Unknown elementwise operation.";
	}
	else
	{
		LOG(FATAL) << "Unknown elementwise operation.";
	}
}

}
