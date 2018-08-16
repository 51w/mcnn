#include "nnet/run/detectionoutput_layer.h"

namespace NNET
{

void DetectionOutputLayer::SetUp(Tensor& Input, Tensor& Output) 
{

}

void DetectionOutputLayer::Run(Tensor& Input, Tensor& Output)
{
	FILE* pp1 = fopen("DetectionOutput1.txt", "wb");
	float *out1 = Input[0]->mutable_cpu_data();
	for (int j=0; j < Input[0]->count(); j++)
		fprintf(pp1, "%f  -- 11\n", out1[j]);
	fclose(pp1);
	
	FILE* pp2 = fopen("DetectionOutput2.txt", "wb");
	float *out2 = Input[1]->mutable_cpu_data();
	for (int j=0; j < Input[1]->count(); j++)
		fprintf(pp2, "%f  -- 22\n", out2[j]);
	fclose(pp2);
	
	
	FILE* pp3 = fopen("DetectionOutput3.txt", "wb");
	float *out3 = Input[2]->mutable_cpu_data();
	for (int j=0; j < Input[2]->count(); j++)
		fprintf(pp3, "%f  -- 33\n", out3[j]);
	fclose(pp3);
	LOG(FATAL) << "do";
}

}
