#include "NV1/layers/route_layer.h"
#include <string.h>

namespace NV1{
	
void RouteLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	if((Input.size()>1) && (Input[0]->WW() != Input[1]->WW() || Input[0]->HH() != Input[1]->HH()))
	{
		fprintf(stderr, "RouteLayer LayerSetUp width failed!\n");
		exit(0);
	}
	int Num_output = 0;
	for(int i=0; i<Input.size(); i++)
	{
		Num_output += Input[i]->CC();
	}
	
	YC = Num_output;
	YH = Input[0]->HH();
	YW = Input[0]->WW();
	
	Output[0]->Reshape(YC, YH, YW);	
}

void RouteLayer::Run(Tensor& Input, Tensor& Output)
{
	float* dst = Output[0]->mutable_cpu_data();
	
	int index = 0;
	for(int i=0; i<Input.size(); i++)
	{
		const float* src = Input[i]->cpu_data();
		
		int count = Input[i]->count();
		memcpy(dst+index, src, sizeof(float) * count);
		index += count;
	}
}

}