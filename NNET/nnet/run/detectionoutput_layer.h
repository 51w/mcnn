#pragma once
#include "nnet/layer.h"

namespace NNET
{

class DetectionOutputLayer : public Layer
{
	
public:
	REGISTER_LAYER_INIT(DetectionOutput);

protected:
	int num_class;
    float nms_threshold;
    int nms_top_k;
    int keep_top_k;
    float confidence_threshold;
	
	vector<float> bboxes;

	int XC, YC;
	int XH, YH;
	int XW, YW;
};

REGISTER_LAYER_CLASS(DetectionOutput);
}