#pragma once
#include "nnet/layer.h"

namespace NNET
{

class LRN : public Layer
{
	
public:
	REGISTER_LAYER_INIT(LRN);

protected:
	int region_type;
    int local_size;
    float alpha;
    float beta;
    float bias;
	
	Blob square_blob;
	Blob square_sum;
	
	int XC, YC;
	int XH, YH;
	int XW, YW;
	enum { NormRegion_ACROSS_CHANNELS = 0, NormRegion_WITHIN_CHANNEL = 1 };
};

REGISTER_LAYER_CLASS(LRN);
}