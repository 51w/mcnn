#include "nnet/run/reshape_layer.h"
#include <string.h>

namespace NNET
{
	
void ReshapeLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	dim[0] = GetParam_Int32(0, -233);
    dim[1] = GetParam_Int32(1, -233);
    dim[2] = GetParam_Int32(2, -233);
    permute = GetParam_Int32(3, 0);
	CHECK(dim[0] != -233 && dim[1] != -233 && dim[2] != -233);

	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	
	 //........//
    
	YC = dim[1];
	YH = dim[0];
	YW = XW;
    
	if(YW == 0)	YW = XW;
	if(YH == 0)	YH = XH;
	if(YC == 0)	YC = XC;

	if(YW == -1) YW = XC*XH*XW / YC / YH;
	if(YH == -1) YH = XC*XH*XW / YC / YW;
	if(YC == -1) YC = XC*XH*XW / YH / YW;
	
	LOG(INFO) << XC << " " << XH << " " << XW;
	LOG(INFO) << YC << " " << YH << " " << YW;
	
	CHECK(YC*YH*YW == XC*XH*XW);
	Output[0]->Reshape(YC, YH, YW);
	Output[0]->ShareData(*Input[0]);
	
	LOG(INFO) << XC << " " << XH << " " << XW;
	LOG(INFO) << YC << " " << YH << " " << YW;
}

void ReshapeLayer::Run(Tensor& Input, Tensor& Output)
{
	
}

}
