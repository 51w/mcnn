#include "mtcnn.h"

MTCNN::MTCNN(const string &model_path)
{
	std::cout << "Det1..." << std::endl;
	Pnet.ncnn_param(model_path + "/det1.proto");
	Pnet.load_model(model_path + "/det1.bin");
	
	std::cout << "Det2..." << std::endl;
	Rnet.ncnn_param(model_path + "/det2.proto");
	Rnet.load_model(model_path + "/det2.bin");
	
	std::cout << "Det3..." << std::endl;
	Onet.ncnn_param(model_path + "/det3.proto");
	Onet.load_model(model_path + "/det3.bin");
	std::cout << "OK." << std::endl;
}

void MTCNN::detect(Mat& image, vector<Bbox>& finalBbox_)
{
	Mat sample_single, resized;
	image.convertTo(sample_single, CV_32FC3);
	cvtColor(sample_single, sample_single, cv::COLOR_BGR2RGB);
	sample_single = sample_single.t();
	
	int height = image.rows;
	int width  = image.cols;
	int minWH = std::min(height, width);
	int factor_count = 0;
	double m = 12. / minsize;
	minWH *= m;
	vector<double> scales;
	while(minWH >= 12)
	{
		scales.push_back(m * std::pow(pre_facetor, factor_count));
		minWH *= pre_facetor;
		++factor_count;
	}
}