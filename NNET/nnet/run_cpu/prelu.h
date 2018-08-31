#pragma once
#include "nnet/layer.h"

namespace NNET
{

class PReLU : public Layer
{
	
public:
	REGISTER_LAYER_INIT(PReLU);

protected:
	int num_slope;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(PReLU);
}