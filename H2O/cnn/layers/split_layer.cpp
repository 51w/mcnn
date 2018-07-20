#include "cnn/layers/split_layer.hpp"

namespace H2O {

void SplitLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	CC = bottom[0]->CC();
	HH = bottom[0]->HH();
	WW = bottom[0]->WW();
}

void SplitLayer::Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	fprintf(stderr, "SplitLayer\n");
}

REGISTER_LAYER_CLASS(Split);
}
