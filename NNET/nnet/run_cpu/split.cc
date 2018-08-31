#include "split.h"

namespace NNET
{

void Split::SetUp(Tensor& Input, Tensor& Output) 
{
	
}

void Split::Reshape(Tensor& Input, Tensor& Output)
{
	int count = Input[0]->count();
	
	for(int i=0; i<Output.size(); ++i)
	{
		CHECK(Output[i] != Input[0]);
		
		Output[i]->ReshapeLike(*Input[0]);
		CHECK(Output[i]->count() == count);
	}
}

void Split::Run(Tensor& Input, Tensor& Output)
{
	for(int i=0; i<Output.size(); ++i)
	{
		Output[i]->ShareData(*Input[0]);
	}
}

}