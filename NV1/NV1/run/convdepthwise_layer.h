#pragma once
#include "NV1/layer.h"

namespace NV1{

class ConvolutionDepthWiseLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(ConvolutionDepthWise);

protected:
	// param
    int num_output;
    int kernel;
    int dilation;
    int stride;
    int pad;
    int bias_term;
    int weight_data_size;
	
    int group;
	Blob _im2col;
	//Blob blob_bordered;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(ConvolutionDepthWise);
}