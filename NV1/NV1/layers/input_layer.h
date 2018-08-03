#pragma once
#include "NV1/layer.h"

namespace NV1{

class InputLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Input);

protected:
	int YC;
	int YH;
	int YW;
};

REGISTER_LAYER_CLASS(Input);
}