#pragma once
#include "NV1/layer.h"

namespace NV1{

class Yolo3Layer : public Layer
{

public:
	REGISTER_LAYER_INIT(Yolo3);

protected:
	float _thresh;
	int _num;
	int _classes;
	
	int _mask[3];
	int _anchors[12];
};

REGISTER_LAYER_CLASS(Yolo3);
}