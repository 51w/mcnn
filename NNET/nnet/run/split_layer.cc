#include "nnet/run/split_layer.h"

namespace NNET
{

void SplitLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	
}

void SplitLayer::Reshape(Tensor& Input, Tensor& Output)
{
	for(int i=0; i<Output.size(); ++i)
	{
		Output[i]->ReshapeLike(*Input[0]);
	}
}

void SplitLayer::Run(Tensor& Input, Tensor& Output)
{
	for(int i=0; i<Output.size(); ++i)
	{
		Output[i]->ShareData(*Input[0]);
	}
}

}
