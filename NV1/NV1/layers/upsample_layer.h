#pragma once
#include "NV1/layer.h"

namespace NV1 {

class UpsampleLayer : public Layer
{

public:
	REGISTER_LAYER_INIT(Upsample);
	
protected:
	int _stride;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Upsample);
}