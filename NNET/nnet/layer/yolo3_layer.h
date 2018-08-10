#pragma once
#include "nnet/layer.h"

namespace NNET
{

class Yolo3Layer : public Layer
{

public:
	REGISTER_LAYER_INIT(Yolo3);
	
	virtual void L2Net(Net *net);

protected:
	float _thresh;
	int _num;
	int _classes;
	
	int _mask[3];
	int _anchors[12];
	
	int _netH, _netW;
};

REGISTER_LAYER_CLASS(Yolo3);
}