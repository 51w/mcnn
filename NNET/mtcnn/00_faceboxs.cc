#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "mtcnn.h"

int main(int argc, char** argv)
{
	MTCNN mtcnn("cnn");
	
	Mat image = cv::imread(argv[1]);
	
	vector<Bbox> finalBbox;

	mtcnn.detect(image, finalBbox);
	
	std::cout << ">>> " << finalBbox.size() << std::endl;
	
	return 0;
}