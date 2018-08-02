#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "cnn/nbox.h"
#include "cnn/net.hpp"
using namespace std;

struct Object{
    cv::Rect rec;
    int class_id;
    float prob;
};

struct YoloBox{
	int sort_class;
    float x;
	float y;
	float w;
	float h;
	vector<float> prob;
};

float overlap3(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection3(YoloBox a, YoloBox b)
{
    float w = overlap3(a.x, a.w, b.x, b.w);
    float h = overlap3(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union3(YoloBox a, YoloBox b)
{
    float i = box_intersection3(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float Yolo_IOU(YoloBox a, YoloBox b)
{
    return box_intersection3(a, b)/box_union3(a, b);
}

bool cmp(YoloBox a, YoloBox b)
{
    return a.prob[b.sort_class] > b.prob[b.sort_class];
}

void correct_boxes(float *out, int size, int w, int h, int netw, int neth, int len)
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
	
	for(int i=0; i<size; i+=len)
	{
		out[i+0] = (out[i+0] - (netw - new_w)/2./netw) / ((float)new_w/netw); 
		out[i+1] = (out[i+1] - (neth - new_h)/2./neth) / ((float)new_h/neth);
		out[i+2] *= (float)netw/new_w;
		out[i+3] *= (float)neth/new_h;
		//fprintf(stderr, "@@@  %f %f %f %f\n", out[i+0], out[i+1], out[i+2], out[i+3]);
	}
}

int main(int argc, char** argv)
{
	H2O::Net yolov3;
    //yolov3.load_param("yolov3-tiny.param");
    //yolov3.load_model("yolov3-tiny.weights");
	yolov3.load_param("V10000.param");
    yolov3.load_model("v7200.bin");

	

	// cv::Mat img = cv::imread(argv[1], 3);
	// if (img.empty())
    // {
       // fprintf(stderr, "cv::imread %s failed\n", argv[1]);
       // return -1;
    // }
	int _classN = 1;
	int _thresh = 0.2;
	
	cv::namedWindow("result", 0);
	
cv::VideoCapture cam(argv[1]);
cv::Mat img;

while (1) {
cam >> img;
if (img.empty()) break;	

	int w = img.cols;
	int h = img.rows;
	
	image input = Exchange_ncnnMat(img.cols, img.rows, img.channels(), img.data);
	shared_ptr<H2O::Blob> data = yolov3.blob_by_name("data");
	float *in = data->mutable_cpu_data();
	for(int i=0; i<data->count(); i++)
		in[i] = input.data[i];
	
	
	yolov3.forward();
	
	shared_ptr<H2O::Blob> prob1 = yolov3.blob_by_name("prob1");
	shared_ptr<H2O::Blob> prob2 = yolov3.blob_by_name("prob2");
	fprintf(stderr, "prob1->count() %d    prob2->count() %d\n", prob1->count(), prob2->count());
	
	// float *out1 = prob1->mutable_cpu_data();
	// correct_boxes(out1, prob1->count(), img.cols, img.rows, 416, 416, 4+80);	
	// float *out2 = prob2->mutable_cpu_data();
	// correct_boxes(out2, prob2->count(), img.cols, img.rows, 416, 416, 4+80);
	
	vector<YoloBox> yolobox;
	if(prob1->count()>0)
	{
		float *out1 = prob1->mutable_cpu_data();
		correct_boxes(out1, prob1->count(), img.cols, img.rows, 416, 416, 4+_classN);	
	
		for(int i=0; i<prob1->count(); i+=4+_classN)
		{
			YoloBox tmp;
			tmp.prob.resize(_classN);
			tmp.x = out1[i];
			tmp.y = out1[i+1];
			tmp.w = out1[i+2];
			tmp.h = out1[i+3];
			for(int j=0; j<_classN; j++)
				tmp.prob[j] = out1[i+4+j];
			
			yolobox.push_back(tmp);
		}
	}
	if(prob2->count()>0)
	{
		float *out2 = prob2->mutable_cpu_data();
		correct_boxes(out2, prob2->count(), img.cols, img.rows, 416, 416, 4+_classN);
	
		for(int i=0; i<prob2->count(); i+=4+_classN)
		{
			YoloBox tmp;
			tmp.prob.resize(_classN);
			tmp.x = out2[i];
			tmp.y = out2[i+1];
			tmp.w = out2[i+2];
			tmp.h = out2[i+3];
			for(int j=0; j<_classN; j++)
				tmp.prob[j] = out2[i+4+j];
			
			yolobox.push_back(tmp);
		}
	}
	fprintf(stderr, "yolobox.size %ld\n", yolobox.size());

	if(yolobox.size() > 0)
	{
	for(int k = 0; k < _classN; ++k)
	{
        for(int i = 0; i < yolobox.size(); ++i)
		{
            yolobox[i].sort_class = k;
		}	
		
		sort(yolobox.begin(), yolobox.begin() + yolobox.size(), cmp);
        
		for(int i = 0; i < yolobox.size(); ++i)
		{
            if(yolobox[i].prob[k] == 0) 
				continue;
			
            for(int j = i+1; j < yolobox.size(); ++j)
			{
                if (Yolo_IOU(yolobox[i], yolobox[j]) > 0.45)
				{
                    yolobox[j].prob[k] = 0;
                }
            }
        }
    }
	}
	
	//for(int i=0; i<yolobox.size(); i++)
	//fprintf(stderr, "YOLO %f %f %f %f\n", yolobox[i].x, yolobox[i].y, yolobox[i].w, yolobox[i].h);
	
	
	
	std::vector<Object> objects;
	for(int i=0; i<yolobox.size(); i++)
	{
		int idclass = -1;
		Object object;
		
		for(int n = 0; n < _classN; ++n)
		{
			if (yolobox[i].prob[n] > _thresh)
			{
				if(idclass < 0) 
				  idclass = n;
			  
				printf("%d  %.0f%%\n", n, yolobox[i].prob[n]*100);
				object.prob = yolobox[i].prob[n];
			}
		}
		if(idclass >= 0)
		{		
			object.class_id = idclass;
			
			float xxx = yolobox[i].x;
			float yyy = yolobox[i].y;
			float www = yolobox[i].w;
			float hhh = yolobox[i].h;
			
			int left  = (xxx-www/2.)*w;
			int right = (xxx+www/2.)*w;
			int top   = (yyy-hhh/2.)*h;
			int bot   = (yyy+hhh/2.)*h;
			
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
	
/*
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
*/
/*	
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
*/	
	
	
	
	for(int i = 0;i<objects.size();++i)
    {
		Object object = objects.at(i);
		
		cv::rectangle(img, object.rec, cv::Scalar(0, 0, 255), 3);
	}
	
	
	//
	cv::imshow("result",img);
	cv::waitKey(30);
}
	
	return 0;
}