#include "NV1/run/pooling_layer.h"
#include <cfloat>

namespace NV1 {

void PoolingLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	pooling_type = GetParam_Int32(0, 0);
    kernel	= GetParam_Int32(1, 0);
    stride	= GetParam_Int32(2, 1);
    pad		= GetParam_Int32(3, 0);
    global_pooling	= GetParam_Int32(4, 0);
    pad_mode = GetParam_Int32(5, 0);
	
	//LOG(INFO) << "Pool: " << pooling_type << " " << kernel << " " << stride 
	//	<< " " << pad << " " << global_pooling << " " << pad_mode;
		
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	
	YH = (XH + 2*pad - kernel) / stride + 1;
	YW = (XW + 2*pad - kernel) / stride + 1;
	YC = XC;
	
	mask.resize(YC*YH*YW);
	Output[0]->Reshape(YC, YH, YW);
	//LOG(INFO) << "Pool: " << YH << " " << YW << " " << YC;
}

void PoolingLayer::Run(Tensor& Input, Tensor& Output)
{
	LOG(INFO) << "PooL: " << YC << " " << YH << " " << YW;
	
	const float* bottom_data = Input[0]->cpu_data();
	float* top_data = Output[0]->mutable_cpu_data();
	const int top_count = Output[0]->count();
	
	if(global_pooling)
    {
		LOG(FATAL) << "PooL global Not Implemented Yet";
	}
	
	if (pooling_type == PoolMethod_MAX)
    {
		// Initialize
		for(int i=0; i<top_count; i++)
		{
			mask[i] = -1;
			top_data[i] = -FLT_MAX;
		}
		
		// The main loop
		for (int c = 0; c < YC; ++c) 
		{
			for (int ph = 0; ph < YH; ++ph)
			{
				for (int pw = 0; pw < YW; ++pw)
				{
					int hstart = ph * stride - pad;
					int wstart = pw * stride - pad;
					int hend = std::min(hstart + kernel, XH);
					int wend = std::min(wstart + kernel, XW);
					hstart = std::max(hstart, 0);
					wstart = std::max(wstart, 0);
					const int pool_index = ph * YW + pw;
					
					for (int h = hstart; h < hend; ++h) {
					  for (int w = wstart; w < wend; ++w) {
						const int index = h * XW + w;
						if (bottom_data[index] > top_data[pool_index]) {
						  top_data[pool_index] = bottom_data[index];
						  
						  mask[c*YW*YH + pool_index] = index;
						  
						}
					  }
					}
				}
			}
			
			// compute offset
			bottom_data += XW*XH;
			top_data += YW*YH;
			//mask += YW*YH;
		}
	}
	else
	{
		LOG(FATAL) << "PooL global Not Implemented Yet";
	}
}

}
