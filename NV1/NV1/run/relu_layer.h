#pragma once
#include "NV1/layer.h"

namespace NV1{

class ReLULayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(ReLU);

protected:
	float slope;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(ReLU);
}