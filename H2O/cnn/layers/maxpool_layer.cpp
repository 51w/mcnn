#include "cnn/layers/maxpool_layer.hpp"
#include <cfloat>

namespace H2O {

void MaxpoolLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	_kernel = stoi(this->layer_param_[0]);
	_stride = stoi(this->layer_param_[1]);
	_pad    = stoi(this->layer_param_[2]);
	
	XC = bottom[0]->channels();
	XH = bottom[0]->height();
	XW = bottom[0]->width();
	
	YH = (XH + 2*_pad) / _stride;
	YW = (XW + 2*_pad) / _stride;
	YC = XC;
	
	top[0]->Reshape(YC, YH, YW);
	//fprintf(stderr, "Maxpl: %d %d %d\n", top[0]->channels(), top[0]->height(), top[0]->width());
}

void MaxpoolLayer::Forward_cpu(const vector<Blob*>& bot, const vector<Blob*>& top)
{
	//fprintf(stderr, "@@@@@@@@@@@@@@@@%f  %d %d\n", this->_NetThresh, this->NetH, this->NetW);
	//fprintf(stderr, "Maxpl: %d %d %d\n", _kernel, _stride, _pad);
	
	const float* src = bot[0]->cpu_data();
	float* dst = top[0]->mutable_cpu_data();
	
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
		src += bot[0]->offset(1);
        dst += top[0]->offset(1);
	}
}
	
REGISTER_LAYER_CLASS(Maxpool);
}