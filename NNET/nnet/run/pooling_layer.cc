#include "nnet/run/pooling_layer.h"
#include <cfloat>

namespace NNET
{

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
	
	if (global_pooling) 
	{ 
		kernel = XH;
		CHECK(stride == 1 && pad == 0);
	}
	YH = static_cast<int>(ceil(static_cast<float>(XH + 2*pad - kernel) / stride)) + 1;
	YW = static_cast<int>(ceil(static_cast<float>(XW + 2*pad - kernel) / stride)) + 1;
	YC = XC;
	
	mask.resize(YC*YH*YW);
	Output[0]->Reshape(YC, YH, YW);
}

void PoolingLayer::Run(Tensor& Input, Tensor& Output)
{
	// FILE* pp1 = fopen("Pool0.txt", "wb");
	// float *out1 = Input[0]->mutable_cpu_data();
	// for (int j=0; j < Input[0]->count(); j++)
		// fprintf(pp1, "%f  -- Pool0\n", out1[j]);
	// fclose(pp1);
	
	
	LOG(INFO) << "PooL: XX " << XC << " " << XH << " " << XW;
	LOG(INFO) << "PooL: YY " << YC << " " << YH << " " << YW;
	
	const float* bottom_data = Input[0]->cpu_data();
	float* top_data = Output[0]->mutable_cpu_data();
	const int top_count = Output[0]->count();
	
	if(global_pooling)
    {
		//LOG(FATAL) << "PooL global Not Implemented Yet";
		int size = XH*XW;
		
		if (pooling_type == PoolMethod_MAX)
        {
			for(int c=0; c < YC; ++c) 
			{
				float max = bottom_data[0];
				for(int i=0; i<size; i++)
                {
                    max = std::max(max, bottom_data[i]);
                }
				top_data[c] = max;
				
				bottom_data += size;
			}
		}
		else if (pooling_type == PoolMethod_AVE)
        {
			for(int c=0; c < YC; ++c) 
			{
				float sum = 0.f;
				for(int i=0; i<size; i++)
                {
					sum += bottom_data[i];
                }
				top_data[c] = sum / size;
				
				bottom_data += size;
			}
		}
	}
	else
	{
	
	if(pooling_type == PoolMethod_MAX)
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
	else if(pooling_type == PoolMethod_AVE)
	{
		//LOG(FATAL) << "PooL PoolMethod_AVE Not Implemented Yet";
		for (int i = 0; i < top_count; ++i) {
		  top_data[i] = 0;
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
					int hend = std::min(hstart + kernel, XH + pad);
					int wend = std::min(wstart + kernel, XW + pad);
					int pool_size = (hend - hstart) * (wend - wstart);
					hstart = std::max(hstart, 0);
					wstart = std::max(wstart, 0);
					hend = std::min(hend, XH);
					wend = std::min(wend, XW);
					for (int h = hstart; h < hend; ++h) {
					  for (int w = wstart; w < wend; ++w) {
						top_data[ph * YW + pw] += bottom_data[h * XW + w];
					  }
					}
					top_data[ph * YW + pw] /= pool_size;
				}
			}		
			// compute offset
			bottom_data += XW*XH;
			top_data += YW*YH;
			//mask += YW*YH;
		}
	}
	
	}
	
	
	
	// FILE* pp2 = fopen("Pool1.txt", "wb");
	// float *out2 = Output[0]->mutable_cpu_data();
	// for (int j=0; j < Output[0]->count(); j++)
		// fprintf(pp2, "%f  -- Pool1\n", out2[j]);
	// fclose(pp2);
}

}
