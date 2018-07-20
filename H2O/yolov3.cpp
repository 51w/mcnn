#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "cnn/nbox.h"
#include "cnn/net.hpp"
using namespace std;

struct Object{
    cv::Rect rec;
    int class_id;
    float prob;
};

void correct_boxes(float *out, int size, int w, int h, int netw, int neth)
{
	int new_w = 0;
    int new_h = 0;
	
    if(((float)netw/w) < ((float)neth/h)) 
	{
        new_w = netw;
        new_h = (h * netw)/w;
    }else{
        new_h = neth;
        new_w = (w * neth)/h;
    }
	
	for(int i=0; i<size; i+=6)
	{		
		out[i+2] = (out[i+2] - (netw - new_w)/2./netw) / ((float)new_w/netw); 
		out[i+3] = (out[i+3] - (neth - new_h)/2./neth) / ((float)new_h/neth);
		out[i+4] *= (float)netw/new_w;
		out[i+5] *= (float)neth/new_h;
	}
}

int main(int argc, char** argv)
{
	H2O::Net yolov3;
    yolov3.load_param("yolov3-tiny.param");
    yolov3.load_model("yolov3-tiny.weights");
	//yolov3.load_param("V_5000.param");
    //yolov3.load_model("V_5000.bin");

	

	cv::Mat img = cv::imread(argv[1], 3);
	if (img.empty())
    {
       fprintf(stderr, "cv::imread %s failed\n", argv[1]);
       return -1;
    }
	
	cv::namedWindow("result", 0);
	
// cv::VideoCapture cam(argv[1]);
// cv::Mat img;

// while (1) {
// cam >> img;
// if (img.empty()) break;	

	int w = img.cols;
	int h = img.rows;
	
	image input = Exchange_ncnnMat(img.cols, img.rows, img.channels(), img.data);
	shared_ptr<H2O::Blob> data = yolov3.blob_by_name("data");
	float *in = data->mutable_cpu_data();
	for(int i=0; i<data->count(); i++)
		in[i] = input.data[i];
	
	
	yolov3.forward();
	
	std::vector<Object> objects;
	
	shared_ptr<H2O::Blob> prob1 = yolov3.blob_by_name("prob1");
	if(prob1->count()>5)
	{
		float *out = prob1->mutable_cpu_data();
		correct_boxes(out, prob1->count(), img.cols, img.rows, 416, 416);
		
		for(int i=0; i<prob1->count(); i+=6)
		{
			Object object;
			object.class_id = out[i];
			object.prob = out[i+1];
			
			fprintf(stderr, ">>%f %f %f %f\n", out[i+2], out[i+3], out[i+4], out[i+5]);
			
			int left  = (out[i+2]-out[i+4]/2.)*w;
			int right = (out[i+2]+out[i+4]/2.)*w;
			int top   = (out[i+3]-out[i+5]/2.)*h;
			int bot   = (out[i+3]+out[i+5]/2.)*h;
				
			if(left < 0) left = 0;
			if(right > w-1) right = w-1;
			if(top < 0) top = 0;
			if(bot > h-1) bot = h-1;
			
			object.rec.x = left;
			object.rec.y = top;
			object.rec.width = right - left;
			object.rec.height = bot - top;
			objects.push_back(object);
		}
	}
	
	shared_ptr<H2O::Blob> prob2 = yolov3.blob_by_name("prob2");
	if(prob2->count()>5)
	{
		float *out = prob2->mutable_cpu_data();
		correct_boxes(out, prob2->count(), img.cols, img.rows, 416, 416);
		
		for(int i=0; i<prob2->count(); i+=6)
		{
			Object object;
			object.class_id = out[i];
			object.prob = out[i+1];
			
			fprintf(stderr, ">>%f %f %f %f\n", out[i+2], out[i+3], out[i+4], out[i+5]);
			
			int left  = (out[i+2]-out[i+4]/2.)*w;
			int right = (out[i+2]+out[i+4]/2.)*w;
			int top   = (out[i+3]-out[i+5]/2.)*h;
			int bot   = (out[i+3]+out[i+5]/2.)*h;
				
			if(left < 0) left = 0;
			if(right > w-1) right = w-1;
			if(top < 0) top = 0;
			if(bot > h-1) bot = h-1;
			
			object.rec.x = left;
			object.rec.y = top;
			object.rec.width = right - left;
			object.rec.height = bot - top;
			objects.push_back(object);
		}
	}
	
	
	
	
	for(int i = 0;i<objects.size();++i)
    {
		Object object = objects.at(i);
		
		cv::rectangle(img, object.rec, cv::Scalar(0, 0, 255), 3);
	}
	
	
	//
	cv::imshow("result",img);
	cv::waitKey();
//}
	
	return 0;
}