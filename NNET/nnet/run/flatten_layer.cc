#include "nnet/run/flatten_layer.h"

namespace NNET
{

void FlattenLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	
}

void FlattenLayer::Reshape(Tensor& Input, Tensor& Output)
{
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	
	YC = XC*XH*XW;
	Output[0]->Reshape(YC);
}

void FlattenLayer::Run(Tensor& Input, Tensor& Output)
{
	float* dst = Output[0]->mutable_cpu_data();
	const float* src = Input[0]->cpu_data();	
	int size = XW*XH;
	
	for(int q=0; q<XC; q++)
    {
        for(int i=0; i<size; i++)
        {
            dst[q*size + i] = src[q*size + i];
        }
    }
}

}
