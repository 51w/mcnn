#include <iostream>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>
#include "nnet/net.h"

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
	NNET::Net yolov3;

	yolov3.ncnn_param(argv[1]);
	// for(int i=0; i < yolov3.blobs_.size(); i++)
	// {
	// 	 shared_ptr<NNET::Blob> aa = yolov3.blobs_[i];
	// 	 LOG(INFO) << aa->CC() << " " << aa->HH() << " " << aa->WW();
	// }
    yolov3.load_model(argv[2]);
	
	std::cout << std::endl;
	LOG(INFO) << yolov3._NH << " " << yolov3._NW << " " << yolov3._NC;
	
	
	vector<string> labels_;
	std::ifstream labels("synset_words.txt");
	//labels.open("synset_words.txt");
	CHECK(labels) << "Unable to open labels file synset_words.txt";
	string line;
	while(std::getline(labels, line))
	labels_.push_back(string(line));
	//**********************************//
	cv::Mat img = cv::imread(argv[3]);
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
	NNET::Blob* data = yolov3.net_input_blobs_[0];
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
	
	cv::Mat zz(inputH, inputW, CV_32FC3);
	for(int i=0; i<inputH; i++)
	{
		for(int j=0; j<inputW; j++)
		{
			zz.at<Vec3f>(i,j)[0] = sample_normalized.at<Vec3f>(i,j)[0] * 0.017;
			zz.at<Vec3f>(i,j)[1] = sample_normalized.at<Vec3f>(i,j)[1] * 0.017;
			zz.at<Vec3f>(i,j)[2] = sample_normalized.at<Vec3f>(i,j)[2] * 0.017;
		}
	}

	cv::split(zz, input_channels);
	
	//**********************************//	
	yolov3.Run();
	

	// LOG(INFO) << "Blob_SIZE: " << yolov3.blobs_.size();
	// for (int i=0; i < yolov3.blobs_.size(); i++)
	// {
		// shared_ptr<NNET::Blob> aa = yolov3.blobs_[i];
		// LOG(INFO) << yolov3.blob_names_[i] << "  SIZE: " << aa->count();
	// }
	
/*
	for (int i=0; i < yolov3.blobs_.size(); i++)
	//for (int i=0; i < 10; i++)
	{
		char name[20];
		sprintf(name, "000/blob_%d.txt", i);
		FILE* pp = fopen(name, "wb");

		shared_ptr<NNET::Blob> aa = yolov3.blobs_[i];
		float *data = aa->mutable_cpu_data();
		for (int j=0; j < aa->count(); j++)
			fprintf(pp, "%f  -- %d\n", data[j], i-1);

		fclose(pp);
	}
*/
	
	//LOG(FATAL) << "start..";
	
	NNET::Blob* result = yolov3.net_output_blobs_[0];			
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
		LOG(INFO) << ">>" << maxN[i] << "  <-->  " << output[maxN[i]] << "  <-->  " << labels_[maxN[i]]; 
	}
	
	
	//
	cv::imshow("result",img);
	cv::waitKey();
	return 0;
}