#include "nnet/layer/yolo3_layer.h"
#include <cmath>
#include <cfloat>
#include "nnet/nbox.h"

namespace NNET
{

void Yolo3Layer::SetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
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

void Yolo3Layer::L2Net(Net *net)
{
	_netH = net->_NH;
	_netW = net->_NW;
	
	net->_Nclass = _classes;
	net->_Nthresh = _thresh;
	
	LOG(INFO) << _netH << "X" << _netW << "   " << net->_Nclass << " " << net->_Nthresh;
}

float logistic_activate3(float x) 
{ 
	return 1.F / (1.F + exp(-x)); 
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

void Yolo3Layer::Run(const vector<Blob*>& bottom, const vector<Blob*>& top)
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
			Yolo_Box(rec, bottom_data, _anchors, _mask[n], box_index, j, i, YW, YH, _netH, _netW, size);
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
	//fprintf(stderr, "Yolo3.size %ld\n", Yolo3.size());
	
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

}