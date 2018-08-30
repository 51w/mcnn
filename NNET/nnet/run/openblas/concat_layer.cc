#include "nnet/run/concat_layer.h"
#include <string.h>

namespace NNET
{

void ConcatLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	concat_axis = GetParam_Int32(0, 0);
}

void ConcatLayer::Reshape(Tensor& Input, Tensor& Output)
{
	int all_count = Input[0]->count();
	
	int num_axes = Input[0]->num_axes();
	concat_num   = Input[0]->count(0, concat_axis);
	concat_size  = Input[0]->count(concat_axis + 1);
	
	vector<int> top_shape = Input[0]->shape();
	for(int i=1; i < Input.size(); ++i)
	{
		CHECK(num_axes == Input[i]->num_axes());
		
		for(int j=0; j < num_axes; ++j)
		{
			if (j == concat_axis) continue;
			CHECK(top_shape[j] == Input[i]->shape(j)) << "Concat check.";
		}
		all_count += Input[i]->count();
		top_shape[concat_axis] += Input[i]->shape(concat_axis);
	}
	
	Output[0]->Reshape(top_shape);
	CHECK(all_count == Output[0]->count());
	
	if(Input.size() == 1)
	Output[0]->ShareData(*Input[0]);
}

void ConcatLayer::Run(Tensor& Input, Tensor& Output)
{
	if(Input.size() == 1) return;
	
	float* dst = Output[0]->mutable_cpu_data();
	int dst_axis = Output[0]->shape(concat_axis);
	int offset = 0;
	
	LOG(INFO) << concat_num << " " << concat_size;
	
	for(int i=0; i<Input.size(); ++i)
	{
		const float* src = Input[i]->cpu_data();
		const int src_axis = Input[i]->shape(concat_axis);
		
		for(int n=0; n<concat_num; ++n)
		{
			memcpy(	dst + (n * dst_axis + offset) * concat_size,
					src + n * src_axis * concat_size, 
							src_axis * concat_size * sizeof(float));
		}
		offset += src_axis;
	}
}

}
