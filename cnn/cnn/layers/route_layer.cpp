#include "cnn/layers/route_layer.hpp"
#include <string.h>

namespace cnn 
{
	
void RouteLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if((bottom.size()>1) && (bottom[0]->width() != bottom[1]->width() || bottom[0]->height() != bottom[1]->height()))
	{
		fprintf(stderr, "RouteLayer LayerSetUp width failed!\n");
		exit(0);
	}
	
	int Num_output = 0;
	for(int i=0; i<bottom.size(); i++)
	{
		Num_output += bottom[i]->channels();
	}
	
	top[0]->Reshape(Num_output, bottom[0]->height(), bottom[0]->width());	
	//fprintf(stderr, "LayerSetUp! %d %d %d  [%d]\n", top[0]->height(), top[0]->width(), top[0]->channels(), Num_output);
}

void RouteLayer::Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	//fprintf(stderr, "RouteLayer! %d %d %d\n", top[0]->height(), top[0]->width(), top[0]->channels());
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