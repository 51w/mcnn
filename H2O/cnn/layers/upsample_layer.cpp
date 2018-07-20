#include "cnn/layers/upsample_layer.hpp"

namespace H2O {

void UpsampleLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	_stride = stoi(this->layer_param_[0]);
	
	XC = bottom[0]->channels();
	XH = bottom[0]->height();
	XW = bottom[0]->width();
	
	YH = XH * _stride;
	YW = XW * _stride;
	YC = XC;
	
	top[0]->Reshape(YC, YH, YW);
	//fprintf(stderr, "UpsampleLayer: %d %d %d\n", top[0]->channels(), top[0]->height(), top[0]->width());
}

void UpsampleLayer::Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	float* dst = top[0]->mutable_cpu_data();
	const float* src = bottom[0]->cpu_data();
	
	for(int k=0; k< XC; k++){
	for(int j = 0; j < YH; j++){
	for(int i = 0; i < YW; i++){
		int in_index   = k*XW*XH + (j/2)*XW + i/2;
		int out_index  = k*YH*YW + j*YW + i;
		dst[out_index] = src[in_index];				
		}
	  }
	}
}

REGISTER_LAYER_CLASS(Upsample);
}