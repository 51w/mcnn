#pragma once
#include "NV1/layer.h"

namespace NV1{

class ConvolutionLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Convolution);
	
	
	void bias_inplace(Blob& bottom);
	void scale_inplace(Blob& bottom);
	void normalize_inplace(Blob& bottom);
	void activate_inplace(Blob& bottom, int type);

protected:	
	int  _kernel;
	int  _pad;
	int  _stride;
	int  _filters;
	int  _activation;
	bool _normalize;
	
	Blob _im2col;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;	
	Blob blob_bordered;
};

REGISTER_LAYER_CLASS(Convolution);
}