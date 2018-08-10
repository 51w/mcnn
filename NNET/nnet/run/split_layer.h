#pragma once
#include "nnet/layer.h"

namespace NNET
{

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