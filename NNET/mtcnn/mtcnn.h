#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "nnet/net.h"
using namespace std;
using namespace cv;

struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    float ppoint[10];
    float regreCoord[4];
};

class MTCNN {

public:
	MTCNN(const string &model_path);
	~MTCNN() {}
	
	void detect(Mat& src, vector<Bbox>& finalBbox);
	
	NNET::Net Pnet;
	NNET::Net Rnet;
	NNET::Net Onet;
	
private:
	const float threshold[3] = { 0.8f, 0.8f, 0.6f };
	int minsize = 40;
	const float pre_facetor = 0.709f;	
};