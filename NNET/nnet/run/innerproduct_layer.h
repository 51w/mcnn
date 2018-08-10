#pragma once
#include "nnet/layer.h"

namespace NNET
{

class InnerProductLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(InnerProduct);

protected:
	int num_output;
    int bias_term;
    int weight_data_size;
	
	int XC;
	int XH;
	int XW;
};

REGISTER_LAYER_CLASS(InnerProduct);
}