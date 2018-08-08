#pragma once
#include "NV1/layer.h"

namespace NV1{

class DropoutLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Dropout);

protected:
	float scale;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Dropout);
}