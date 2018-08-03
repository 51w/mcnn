#include "NV1/layers/maxpool_layer.h"
#include <cfloat>

namespace NV1{

void MaxpoolLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	_kernel = stoi(this->layer_param_[0]);
	_stride = stoi(this->layer_param_[1]);
	_pad    = stoi(this->layer_param_[2]);
	
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	
	YH = (XH + 2*_pad) / _stride;
	YW = (XW + 2*_pad) / _stride;
	YC = XC;
	
	Output[0]->Reshape(YC, YH, YW);
}

void MaxpoolLayer::Run(Tensor& Input, Tensor& Output)
{
	const float* src = Input[0]->cpu_data();
	float* dst = Output[0]->mutable_cpu_data();
	
	const int maxk = _kernel * _kernel;	
	int space_ofs[maxk];
	{
		int p1 = 0;
		int p2 = 0;
		int gap = XW - _kernel;
		for (int i = 0; i < _kernel; i++)
		{
			for (int j = 0; j < _kernel; j++)
			{
				space_ofs[p1] = p2;
				p1++;
				p2++;
			}
			p2 += gap;
		}
	}
	
	for (int q=0; q<YC; q++)
	{
		for (int i = 0; i < YH; i++)
		{
			for (int j = 0; j < YW; j++)
			{
				const float* sptr = src + i*XW*_stride + j*_stride;
				
				float max = sptr[0];				 
				for (int k = 0; k < maxk; k++)
				{
					float val;
					if(i*_stride + k/_kernel >= XH || j*_stride + k%_kernel >= XW)
						float val = -FLT_MAX;
					else val = sptr[ space_ofs[k] ];
					max = std::max(max, val);
				}		
				dst[i*YW + j] = max;
			}
		}		
		src += Input[0]->offset(1);
        dst += Output[0]->offset(1);
	}
}
	
}