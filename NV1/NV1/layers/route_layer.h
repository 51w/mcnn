#pragma once
#include "NV1/layer.h"

namespace NV1{

class RouteLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Route);

protected:
	int YC;
	int YH;
	int YW;

};

REGISTER_LAYER_CLASS(Route);
}