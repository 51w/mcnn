#pragma once
#include "NV1/layer.h"

namespace NV1{

class MaxpoolLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Maxpool);

protected:
	int _kernel;
    int _stride;
    int _pad;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(Maxpool);
}