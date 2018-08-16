#include "nnet/run/priorbox_layer.h"

namespace NNET
{

void PriorBoxLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	variances[0] = GetParam_f32(3, 0.f);
    variances[1] = GetParam_f32(4, 0.f);
    variances[2] = GetParam_f32(5, 0.f);
    variances[3] = GetParam_f32(6, 0.f);
    flip = GetParam_Int32(7, 1);
    clip = GetParam_Int32(8, 0);
    image_width  = GetParam_Int32(9, 0);
    image_height = GetParam_Int32(10,0);
    step_width   = GetParam_f32(11, -233.f);
    step_height  = GetParam_f32(12, -233.f);
    offset = GetParam_f32(13, 0.f);
	
	LOG(INFO) << variances[0] << " " << variances[1] << " " << variances[2] << " " << variances[3] << " "
	 << flip << " " << clip << " " << image_width << " " << image_height << " "
	 << step_width << " " << step_height << " " << offset;
	
	
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	YH = XH;
	YW = XW;
	YC = XC;
	
	Output[0]->Reshape(1, 1, 1);
}

void PriorBoxLayer::Run(Tensor& Input, Tensor& Output)
{
	// FILE* pp = fopen("Flatten0.txt", "wb");
	// float *data = Input[0]->mutable_cpu_data();
	// for (int j=0; j < Input[0]->count(); j++)
		// fprintf(pp, "%f  -- Flatte\n", data[j]);
	// fclose(pp);
	
	
	
	// FILE* pp1 = fopen("Flatten1.txt", "wb");
	// float *out = Output[0]->mutable_cpu_data();
	// for (int j=0; j < Output[0]->count(); j++)
		// fprintf(pp1, "%f  -- Flatte\n", out[j]);
	// fclose(pp1);
	// LOG(FATAL) << "Flatten";
}

}
