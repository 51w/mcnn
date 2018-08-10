#pragma once
#include "nnet/layer.h"

namespace NNET
{

class Yolo2Layer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(Yolo2);

protected:
	float _thresh;
	int _num;
	int _classes;
	int _coords;
	float _biases[10];
	
};
}