#pragma once
#include "nnet/layer.h"

namespace NNET
{

class Power : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Power);

protected:	
	float power;
    float scale;
    float shift;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Power);
}