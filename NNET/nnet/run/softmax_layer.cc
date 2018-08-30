#include "nnet/run/softmax_layer.h"
#include "nnet/util/math_functions.h"
#include <string.h>

namespace NNET
{
// value = exp( value - global max value )
// sum 	 = all value
// value = value / sum

void SoftmaxLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	axis = GetParam_Int32(0, 0);
	outer_num_ = Input[0]->count(0, axis);
	inner_num_ = Input[0]->count(axis + 1);
	
	vector<int> mult_dims(1, Input[0]->shape(axis));
	sum_multiplier_.Reshape(mult_dims);
	float* multiplier_data = sum_multiplier_.mutable_cpu_data();
	for (int z=0; z<sum_multiplier_.count(); ++z)
		multiplier_data[z] = 1.f;

	vector<int> scale_dims = Input[0]->shape();
	scale_dims[axis] = 1;
	scale_.Reshape(scale_dims);
	
	YW = Input[0]->WW();
	YH = Input[0]->HH();
	YC = Input[0]->CC();
	
	Output[0]->Reshape(YC, YH, YW);
	LOG(INFO) << "Softmax: " << outer_num_ << " " << inner_num_;
	LOG(INFO) << YC << " " << YH << " " << YW;
}

void SoftmaxLayer::Reshape(Tensor& Input, Tensor& Output)
{
	
}

void SoftmaxLayer::Run(Tensor& Input, Tensor& Output)
{
	//LOG(INFO) << "SoftMax: " << YC << " " << YH << " " << YW;
	
	float* dst = Output[0]->mutable_cpu_data();
	float* src = Input[0]->mutable_cpu_data();
	float* scale_data = scale_.mutable_cpu_data();
	
	int channels = Input[0]->shape(axis);
	int dim = Input[0]->count() / outer_num_;
	
	int count = Input[0]->count();
	memcpy(dst, src, sizeof(float) * count);
	
	for(int i = 0; i < outer_num_; ++i)
	{
		memcpy(scale_data, src + i * dim, sizeof(float) * inner_num_);
		for (int j = 0; j < channels; j++) {
		  for (int k = 0; k < inner_num_; k++) {
			scale_data[k] = std::max(scale_data[k],
				src[i * dim + j * inner_num_ + k]);
		  }
		}
		
		// subtraction
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels, inner_num_,
		  1, -1., sum_multiplier_.cpu_data(), scale_data, 1., dst);
		
		for (int z=0; z<dim; ++z)
			dst[z] = exp(dst[z]);
		// sum after exp
		caffe_cpu_gemv(CblasTrans, channels, inner_num_, 1.,
			dst, sum_multiplier_.cpu_data(), 0., scale_data);
		
		for (int j = 0; j < channels; j++) {
			for (int z=0; z<inner_num_; ++z)
			dst[z] = dst[z] / scale_data[z];
			dst += inner_num_;
		}
	}
}

}
