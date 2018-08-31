#pragma once
#include "nnet/layer.h"

namespace NNET
{

class Convolution : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Convolution);

protected:
	int  num_output;
	int  kernel;
	int  dilation;
	int  stride;
	int  pad;
	bool bias_term;
	int  weight_data_size;
	
	Blob _im2col;
	Blob blob_bordered;

	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Convolution);
}