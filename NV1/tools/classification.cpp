#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>
#include "NV1/net.h"

using namespace std;
using namespace cv;

typedef std::pair<string, float> Prediction;

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

int main(int argc, char** argv)
{
	NV1::Net yolov3;

	yolov3.ncnn_param("alexnet.param");
    yolov3.load_model("alexnet.bin");
	
	std::cout << std::endl;
	LOG(INFO) << yolov3._NH << " " << yolov3._NW << " " << yolov3._NC;
	
	
	//**********************************//
	cv::Mat img = cv::imread(argv[1]);
	if (img.empty())
    {
       fprintf(stderr, "cv::imread %s failed\n", argv[1]);
       return -1;
    }
	int inputH = yolov3._NH;
	int inputW = yolov3._NW;
	int height = img.rows;
	int width = img.cols;
	Size input_geometry_ = cv::Size(inputH, inputW);
	
	std::vector<cv::Mat> input_channels;
	NV1::Blob* data = yolov3.net_input_blobs_[0];
	float* input_data = data->mutable_cpu_data();
	for (int i = 0; i < data->CC(); ++i) {
		cv::Mat channel(inputH, inputW, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += inputH * inputW;
	}
	cv::Mat sample_resized;
	cv::resize(img, sample_resized, input_geometry_);
	
	cv::Mat sample_float;
	sample_resized.convertTo(sample_float, CV_32FC3);
	
	Mat mean_;
	//Scalar channel_mean = {104.007, 116.669, 122.679, 0};
	Scalar channel_mean;
	channel_mean[0] = 104.007;
	channel_mean[1] = 116.669;
	channel_mean[2] = 122.679;
	mean_ = cv::Mat(inputH, inputW, CV_32FC3, channel_mean);
	
	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);
	cv::split(sample_normalized, input_channels);
	
	//**********************************//
	LOG(INFO) << "Blob_SIZE: " << yolov3.blobs_.size();
	
	yolov3.Run();
	for (int i=0; i < yolov3.blobs_.size(); i++)
	{
		char name[20];
		sprintf(name, "000/blob_%d.txt", i);
		FILE* pp = fopen(name, "wb");

		shared_ptr<NV1::Blob> aa = yolov3.blobs_[i];
		float *data = aa->mutable_cpu_data();
		for (int j=0; j < aa->count(); j++)
			fprintf(pp, "%f  -- %d\n", data[j], i-1);

		fclose(pp);
	}
	
	//LOG(FATAL) << "start..";
	
	NV1::Blob* result = yolov3.net_output_blobs_[0];			
	const float* result_data = result->cpu_data();
	vector<float> output;
	for (int i = 0; i < result->count(); ++i)
	{
		//LOG(INFO) << result_data[i];
		output.push_back(result_data[i]);
	}
	LOG(INFO) << "Total: " << output.size();

	vector<int> maxN = Argmax(output, 5);

	for (int i = 0; i < maxN.size(); ++i)
	{
		LOG(INFO) << ">>" << maxN[i];
	}
	
	
	//
	cv::imshow("result",img);
	cv::waitKey(30);
	return 0;
}