#pragma once
#include "nnet/layer.h"

namespace NNET
{

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