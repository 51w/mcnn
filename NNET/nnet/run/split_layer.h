#pragma once
#include "nnet/layer.h"

namespace NNET
{

class SplitLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Split);
	
};

REGISTER_LAYER_CLASS(Split);
}