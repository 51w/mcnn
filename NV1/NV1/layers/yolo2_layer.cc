#include "NV1/layers/yolo2_layer.h"
#include <cmath>
#include <cfloat>
#include "NV1/nbox.h"

namespace NV1 {

void Yolo2Layer::SetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	_thresh 	= stof(this->layer_param_[0]);
	_classes	= stoi(this->layer_param_[1]);
	_num	   	= stoi(this->layer_param_[2]);
	_coords	   	= stoi(this->layer_param_[3]);
	
	float tmp[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
	for(int i=0; i<10; i++) _biases[i] = tmp[i];

	//fprintf(stderr, "Yolo2: %f %d %d %d\n", _thresh, _classes, _num, _coords);
}

float logistic_activate(float x) 
{ 
	return 1.F / (1.F + exp(-x)); 
}

void softmax(float *input, int n, float temp, int stride, float *output)
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

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

int nms_comparator(const void *pa, const void *pb)
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

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

void do_nms_sort(Detection *dets, int total, int classes, float thresh)
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

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(Detection), nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void Yolo2Layer::Run(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	fprintf(stderr, "Yolo2: %f %d %d %d\n", _thresh, _classes, _num, _coords);
	
	// FILE* pp = fopen("AAAA.txt", "wb");
	// Dtype* aaa = bottom[0]->mutable_cpu_data();
	// for(int i=0; i<bottom[0]->count(); i++){
		// fprintf(pp, "%f  v05\n", aaa[i]);
	// }
	// fclose(pp);
	// exit(0);
	
	Dtype* bottom_data = bottom[0]->mutable_cpu_data();
	//Dtype* top_data = top[0]->mutable_cpu_data();
	
	int ccc = _classes+_coords+1; //80+4+1
	int w = bottom[0]->WW();
	int h = bottom[0]->HH();
	int _outputs = w*h*_num*ccc;
	
	for(int n = 0; n < _num; ++n)
	{
		for(int i = 0; i < w*h; ++i)
		{
			int index = n*w*h*ccc + i;
			Dtype x = bottom_data[index];
			bottom_data[index] = logistic_activate(x);
			
			index = (n*ccc + 1)*w*h + i;
			x = bottom_data[index];
			bottom_data[index] = logistic_activate(x);
			
			index = (n*ccc + _coords)*w*h + i;
			x = bottom_data[index];
			bottom_data[index] = logistic_activate(x);
		}
	}
	
	// softmax activation
	for(int i=0; i<_num; ++i)
	{
		for(int j=0; j<w*h; ++j)
		{
			softmax(bottom_data + (_coords + 1)*w*h + i*_outputs/_num + j,\
			_classes, 1, w*h, bottom_data + (_coords + 1)*w*h + i*_outputs/_num + j);
		}
	}
	
	//detection result
	int _numbox = _num*w*h;
	Detection *_box = (Detection *)calloc(_numbox, sizeof(Detection));
	for(int i = 0; i < _num*w*h; ++i){
        _box[i].prob = (float *)calloc(_classes, sizeof(float));
    }
	
	float *predictions = bottom_data;	
	for(int i = 0; i < w*h; ++i)
	{
        int row = i / w;
        int col = i % w;
        for(int n = 0; n < _num; ++n){
            int index = n*w*h + i;
            for(int j = 0; j < _classes; ++j){
                _box[index].prob[j] = 0;
            }
			
			int obj_index  = n*w*h*(4+_classes+1) + _coords*w*h + i;
			int box_index  = n*w*h*(4+_classes+1) + i;		
			float scale = predictions[obj_index];
			_box[index].bbox = get_region_box(predictions, _biases, n, box_index, col, row, w, h, w*h);
            _box[index].objectness = scale > _thresh ? scale : 0;
			
			int class_inx = n*w*h*(4+_classes+1) + 5*w*h + i;
			if(_box[index].objectness){
				for(int j = 0; j < _classes; ++j)
				{
					int class_index = class_inx + j*w*h;
					float prob = scale*predictions[class_index];
					_box[index].prob[j] = (prob > _thresh) ? prob : 0;
				}
			}
		}
	}
	
	float _NSMthresh = 0.45;
	if(_NSMthresh) 
	do_nms_sort(_box, _numbox, _classes, _NSMthresh);

	
	vector<float> result;
	for(int m = 0; m < _numbox; ++m)
	{
		int idclass = -1;
		for(int n = 0; n < _classes; ++n)
		{
			if (_box[m].prob[n] > _thresh)
			{
				if(idclass < 0) idclass = n;
			   //fprintf(stderr, "classes %d  %d\n", i, n);
			   printf("%d  %.0f%%\n", n, _box[m].prob[n]*100);
			}
		}
		if(idclass >= 0)
		{
			box b = _box[m].bbox;
			fprintf(stderr, "###%f %f %f %f\n", b.x, b.y, b.w, b.h);
			result.push_back(idclass);
			result.push_back(_box[m].prob[idclass]);
			result.push_back(b.x);
			result.push_back(b.y);
			result.push_back(b.w);
			result.push_back(b.h);
		}
	}
	
	fprintf(stderr, "result:%ld\n", result.size());
	
	top[0]->Reshape(result.size());
	float *output = top[0]->mutable_cpu_data();
	for(int i=0; i<top[0]->count(); i++)
	{
		output[i] = result[i];
	}
	
}

REGISTER_LAYER_CLASS(Yolo2);

}