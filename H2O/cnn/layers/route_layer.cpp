#include "cnn/layers/route_layer.hpp"
#include <string.h>

namespace H2O {
	
void RouteLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if((bottom.size()>1) && (bottom[0]->WW() != bottom[1]->WW() || bottom[0]->HH() != bottom[1]->HH()))
	{
		fprintf(stderr, "RouteLayer LayerSetUp width failed!\n");
		exit(0);
	}
	int Num_output = 0;
	for(int i=0; i<bottom.size(); i++)
	{
		Num_output += bottom[i]->CC();
	}
	
	YC = Num_output;
	YH = bottom[0]->HH();
	YW = bottom[0]->WW();
	
	top[0]->Reshape(YC, YH, YW);	
}

void RouteLayer::Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	float* dst = top[0]->mutable_cpu_data();
	
	int index = 0;
	for(int i=0; i<bottom.size(); i++)
	{
		const float* src = bottom[i]->cpu_data();
		
		int count = bottom[i]->count();
		memcpy(dst+index, src, sizeof(float) * count);
		index += count;
	}
}

REGISTER_LAYER_CLASS(Route);
}