#include "nnet/run/concat_layer.h"
#include <string.h>

namespace NNET
{

void ConcatLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	LOG(INFO) << "1111111111";
	for(int i = 0; i < Input.size(); ++i)
	{
		LOG(INFO) << Input[i]->CC() << " " << Input[i]->HH() << " " << Input[i]->WW() << " " << Input[i]->count();
	}
	
	concat_axis_ = GetParam_Int32(0, 0);
	
	const int num_axes = Input[0]->num_axes();
	
	// Initialize with the first blob.
	vector<int> top_shape = Input[0]->shape();
	num_concats_ = Input[0]->count(0, concat_axis_);
	concat_input_size_ = Input[0]->count(concat_axis_ + 1);
	int bottom_count_sum = Input[0]->count();
	
	for (int i = 1; i < Input.size(); ++i)
	{
		CHECK(num_axes == Input[i]->num_axes()) << "All inputs must have the same #axes.";
		for (int j = 0; j < num_axes; ++j)
		{
		  if (j == concat_axis_) { continue; }
		  CHECK(top_shape[j] == Input[i]->shape(j)) << "All inputs must have the same shape, except at concat_axis.";
		}
		bottom_count_sum += Input[i]->count();
		top_shape[concat_axis_] += Input[i]->shape(concat_axis_);
	}
	Output[0]->Reshape(top_shape);
	
	CHECK(bottom_count_sum == Output[0]->count());
	if (Input.size() == 1)
	{ Output[0]->ShareData(*Input[0]); }

	LOG(INFO) << "22222222222";
	LOG(INFO) << "Output " << Output[0]->CC() << " " << Output[0]->HH() << " " << Output[0]->WW() << " " << Output[0]->count();
}

void ConcatLayer::Run(Tensor& Input, Tensor& Output)
{
	if(Input.size() == 1) { return; }	
	
	float* top_data = Output[0]->mutable_cpu_data();

	int offset_concat_axis = 0;
	const int top_concat_axis = Output[0]->shape(concat_axis_);
	
	for(int i=0; i<Input.size(); ++i)
	{
		const float* bottom_data = Input[i]->cpu_data();
		const int bottom_concat_axis = Input[i]->shape(concat_axis_);
		
		for(int n=0; n<num_concats_; ++n)
		{
			memcpy(top_data + (n * top_concat_axis + offset_concat_axis) * concat_input_size_,
			bottom_data + n * bottom_concat_axis * concat_input_size_, 
			sizeof(float) * bottom_concat_axis * concat_input_size_);
		}
		offset_concat_axis += bottom_concat_axis;
	}
}

}
