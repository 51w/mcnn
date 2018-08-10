#include "nnet/layer/upsample_layer.h"

namespace NNET
{

void UpsampleLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	_stride = stoi(this->layer_param_[0]);
	
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	
	YH = XH * _stride;
	YW = XW * _stride;
	YC = XC;
	
	Output[0]->Reshape(YC, YH, YW);
}

void UpsampleLayer::Run(Tensor& Input, Tensor& Output)
{
	float* dst = Output[0]->mutable_cpu_data();
	const float* src = Input[0]->cpu_data();
	
	for(int k = 0; k < XC; k++){
	for(int j = 0; j < YH; j++){
	for(int i = 0; i < YW; i++){
		int in_index   = k*XW*XH + (j/2)*XW + i/2;
		int out_index  = k*YH*YW + j*YW + i;
		dst[out_index] = src[in_index];			
		}
	  }
	}
}

}