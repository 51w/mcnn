#pragma once
#include "NV1/layer.h"

namespace NV1{

class SplitLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Split);

protected:
	int CC;
	int HH;
	int WW;

};

REGISTER_LAYER_CLASS(Split);
}