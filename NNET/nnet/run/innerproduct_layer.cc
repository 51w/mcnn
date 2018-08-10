#include "nnet/run/innerproduct_layer.h"

namespace NNET 
{

void InnerProductLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	num_output = GetParam_Int32(0, 0);
    bias_term  = GetParam_Int32(1, 0);
    weight_data_size = GetParam_Int32(2, 0);
	
	//LOG(INFO) << "InnerProduct: " << num_output << " " << bias_term << " " << weight_data_size;
	
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	
	Output[0]->Reshape(num_output);
	
	if (bias_term) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
	this->blobs_[0].reset(new Blob(XC*XH*XW * num_output));
	this->blobs_[1].reset(new Blob(num_output));
	
	LOG(INFO) << "InnerProduct: " << XC << " " << XH << " " << XW;
}

void InnerProductLayer::Run(Tensor& Input, Tensor& Output)
{
	float *src = Input[0]->mutable_cpu_data();
	float *dst = Output[0]->mutable_cpu_data();
	float *weight_data = this->blobs_[0]->mutable_cpu_data();
	float *bias_data   = this->blobs_[1]->mutable_cpu_data();
	
	for(int p=0; p<num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        // channels
        for(int q=0; q<XC; q++)
        {
            const float* w = weight_data + XW*XH*XC * p + XW*XH * q;
            const float* m = src + q*XW*XH;

            for(int i = 0; i < XW*XH; i++)
            {
                sum += m[i] * w[i];
            }
        }
        dst[p] = sum;
    }
}

}
