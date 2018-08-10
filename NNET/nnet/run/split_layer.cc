#include "nnet/run/split_layer.h"

namespace NNET
{

void SplitLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	CC = Input[0]->CC();
	HH = Input[0]->HH();
	WW = Input[0]->WW();
	
	for(int i = 0; i < Output.size(); ++i)
	{
		CHECK(Output[i] != Input[0]) 
			<< "SplitLayer does not allow in-place computation.";
	
		Output[i]->ReshapeLike(*Input[0]);	
		CHECK(Input[0]->count() == Output[i]->count());
	}
	
	for (int i = 0; i < Output.size(); ++i)
	{
		Output[i]->ShareData(*Input[0]);
	}
}

void SplitLayer::Run(Tensor& Input, Tensor& Output)
{
	
}

}
