#pragma once
#include "nnet/layer.h"

namespace NNET
{

class DetectionOutputLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(DetectionOutput);

protected:
	

	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(DetectionOutput);
}