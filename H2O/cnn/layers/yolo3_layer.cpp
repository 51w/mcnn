#include "cnn/layers/yolo3_layer.hpp"
#include <cmath>
#include <cfloat>
#include "cnn/nbox.h"

namespace H2O {

void Yolo3Layer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	_mask[0] = stoi(this->layer_param_[0]);
	_mask[1] = stoi(this->layer_param_[1]);
	_mask[2] = stoi(this->layer_param_[2]);
	
	_num     = stof(this->layer_param_[3]);
	_classes = stoi(this->layer_param_[4]);
	_thresh  = stof(this->layer_param_[5]);
	
	int anchors[12] = {10,14, 23,27, 37,58, 81,82, 135,169, 344,319};
	for(int i=0; i<12; i++) _anchors[i] = anchors[i];	
}

float logistic_activate3(float x) 
{ 
	return 1.F / (1.F + exp(-x)); 
}


void softmax3(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

box get_region_box3(float *x, int *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

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

float box_intersection3(box a, box b)
{
    float w = overlap3(a.x, a.w, b.x, b.w);
    float h = overlap3(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union3(box a, box b)
{
    float i = box_intersection3(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou3(box a, box b)
{
    return box_intersection3(a, b)/box_union3(a, b);
}

int nms_comparator3(const void *pa, const void *pb)
{
    Detection a = *(Detection *)pa;
    Detection b = *(Detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

void do_nms_sort3(Detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            Detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;
	//fprintf(stderr, ">>>>>>>>>>%d \n", total);

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(Detection), nms_comparator3);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou3(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

box get_yolo_box(float *x, int *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}


int Yolo_Box(float *box, float *x, int *biases, int n, int index, 
				int i, int j, int lw, int lh, int w, int h, int stride)
{
    box[0] = (i + x[index + 0*stride]) / lw;
    box[1] = (j + x[index + 1*stride]) / lh;
    box[2] = exp( x[index + 2*stride]) * biases[2*n]   / w;
    box[3] = exp( x[index + 3*stride]) * biases[2*n+1] / h;
	
    return 0;
}

void Yolo3Layer::Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top)
{	
	float* bottom_data = bottom[0]->mutable_cpu_data();
	
	int ccc = _classes+4+1;
	int YW = bottom[0]->WW();
	int YH = bottom[0]->HH();
	int size = YW*YH;
	
	for(int n = 0; n < _num; ++n)
	{
	 for(int i = 0; i < size; ++i)
	 {
		int index = n*size*ccc + i;
		Dtype x = bottom_data[index];
		bottom_data[index] = logistic_activate3(x);
		
		index = (n*ccc + 1)*size + i;
		x = bottom_data[index];
		bottom_data[index] = logistic_activate3(x);
		
		
		for(int j=0; j<_classes+1; j++)
		{
			index = (n*ccc + 4)*size + i*(_classes+1) + j;
			x = bottom_data[index];
			bottom_data[index] = logistic_activate3(x);
		}
	 }
	}
	
	// output
	vector<float> Yolo3;
	for(int i = 0; i < YH; i++)
	{
	 for(int j = 0; j < YW; j++)
	 {
		int index = i*YW + j;
		for(int n = 0; n < _num; n++)
		{
			int   obj_index  = n*size*(4+_classes+1) + 4*size + index;
			float objectness = bottom_data[obj_index];
			if(objectness <= _thresh) continue;
			
			int box_index  = n*size*(4+_classes+1) + index;
			float rec[4];
			Yolo_Box(rec, bottom_data, _anchors, _mask[n], box_index, j, i, YW, YH, 416, 416, size);
			//fprintf(stderr, "@@@  %f %f %f %f\n", rec[0], rec[1], rec[2], rec[3]);
			
			Yolo3.push_back(rec[0]);
			Yolo3.push_back(rec[1]);
			Yolo3.push_back(rec[2]);
			Yolo3.push_back(rec[3]);		
			//Yolo3.push_back(objectness);
			
			int class_inx = n*size*(4+_classes+1) + 5*size + index;
			for(int j = 0; j < _classes; ++j)
			{
				int class_index = class_inx + j*size;
				float prob = objectness*bottom_data[class_index];
				float pp = (prob > _thresh) ? prob : 0;
				
				Yolo3.push_back(pp);
			}
		}
	 }
	}
	fprintf(stderr, "Yolo3.size %ld\n", Yolo3.size());
	
	top[0]->Reshape(0);
	if(Yolo3.size()>0)
	{
		top[0]->Reshape(Yolo3.size());
		float *output = top[0]->mutable_cpu_data();
		for(int i=0; i<top[0]->count(); i++)
		{
			output[i] = Yolo3[i];
		}
	}
}

REGISTER_LAYER_CLASS(Yolo3);
}