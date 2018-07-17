#include "cnn/layers/conv_layer.hpp"

namespace cnn 
{

void ConvolutionLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	_normalize 	= stoi(this->layer_param_[0]);
	_filters	= stoi(this->layer_param_[1]);
	_kernel   	= stoi(this->layer_param_[2]);
	_stride 	= stoi(this->layer_param_[3]);
	_pad 		= stoi(this->layer_param_[4]);
	_activation = stoi(this->layer_param_[5]);
	if(_pad) _pad = _kernel/2;
	
	// load model
	if(_normalize)
	{
		this->blobs_.resize(5);
		this->blobs_[2].reset(new Blob(_filters));
		this->blobs_[3].reset(new Blob(_filters));
		this->blobs_[4].reset(new Blob(_filters));
	}
	else
	{
		this->blobs_.resize(2);
	}
	this->blobs_[0].reset(new Blob(bottom[0]->channels()*_kernel*_kernel*_filters));
	this->blobs_[1].reset(new Blob(_filters));
	
	
	XC = bottom[0]->channels();
	XH = bottom[0]->height();
	XW = bottom[0]->width();
	
	YH = (XH + 2*_pad - _kernel) / _stride + 1;
	YW = (XW + 2*_pad - _kernel) / _stride + 1;
	YC = _filters;
	
	top[0]->Reshape(YC, YH, YW);
	//fprintf(stderr, "ConvolutionLayer: %d %d %d\n", top[0]->channels(), top[0]->height(), top[0]->width());
}

void ConvolutionLayer::Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	//fprintf(stderr, "Conv:  %d %d %d  %d %d %d\n", _normalize, _filters, _kernel, _stride, _pad, _activation);
	if(_pad > 0)
	{
		int newh = XH + 2*_pad;
		int neww = XW + 2*_pad;
		blob_bordered.Reshape(XC, newh, neww);
		
		Dtype* data = blob_bordered.mutable_cpu_data();
		Dtype* bottom_blob = bottom[0]->mutable_cpu_data();
		
		for (int q = 0; q < XC; q++)
		{
			for (int i = 0; i < newh; i++)
			{
				for (int j = 0; j < neww; j++)
				{
					if(i < _pad || i > newh-_pad-1 || j < _pad || j > neww-_pad-1)
						data[q*neww*newh + i*neww + j] = 0.0f;
					else
						data[q*neww*newh + i*neww + j] = \
					bottom_blob[q*XW*XH + (i-_pad)*XW + j-_pad];
				}
			}
		}
	}
	else
	{
		blob_bordered.ReshapeLike(*bottom[0]);
		blob_bordered.ShareData(*bottom[0]);
	}
	
	const float* inptr = blob_bordered.cpu_data();
	float* outptr = top[0]->mutable_cpu_data();
	
	const int maxk = _kernel * _kernel;	
	int space_ofs[maxk];
	{
		int p1 = 0;
		int p2 = 0;
		int gap = blob_bordered.width() - _kernel;
		for (int i = 0; i < _kernel; i++)
		{
			for (int j = 0; j < _kernel; j++)
			{
				space_ofs[p1] = p2;
				p1++;
				p2++;
			}
			p2 += gap;
		}
	}
	
	float *_weights = this->blobs_[0]->mutable_cpu_data();
	for (int p=0; p<YC; p++)
    {
        for (int i = 0; i < YH; i++)
        {
            for (int j = 0; j < YW; j++)
            {
                float sum = 0.f;

                const float* kptr = _weights + maxk * XC * p;

                for (int q=0; q<XC; q++)
                {
					const float* aaa = inptr + q * blob_bordered.offset(1);
                    const float* sptr = aaa + (i*_stride*blob_bordered.width()) + j*_stride;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[ space_ofs[k] ]; // 20.72
                        float weight = kptr[k];
                        sum += val * weight; // 41.45
                    }

                    kptr += maxk;
                }

				int conv_index = i * YW + j;
				outptr[conv_index] = sum;
            }
        }
        outptr += top[0]->offset(1);
    }
	
	if(_normalize)
	{
		normalize_inplace(*top[0]);
		scale_inplace(*top[0]);
    } 
	
	bias_inplace(*top[0]);
	activate_inplace(*top[0], _activation);
}

void ConvolutionLayer::bias_inplace(Blob& bottom)
{
	float *data = bottom.mutable_cpu_data();
	float *_biases = this->blobs_[1]->mutable_cpu_data();
	
	for (int q=0; q<bottom.channels(); q++)
    {
        float* ptr = data + q*bottom.offset(1);

        float bias = _biases[q];

        for (int i=0; i<bottom.offset(1); i++)
        {
            ptr[i] += bias;
        }
    }
}

void ConvolutionLayer::scale_inplace(Blob& bottom)
{
	float *data = bottom.mutable_cpu_data();
	float *_scales = this->blobs_[2]->mutable_cpu_data();
	
	for (int q=0; q<bottom.channels(); q++)
    {
        float* ptr = data + q*bottom.offset(1);

        float s = _scales[q];

        for (int i=0; i<bottom.offset(1); i++)
        {
            ptr[i] = ptr[i] * s;
        }
    }
}

void ConvolutionLayer::normalize_inplace(Blob& bottom)
{
	float *data = bottom.mutable_cpu_data();
	float *_rolling_mean = this->blobs_[3]->mutable_cpu_data();
	float *_rolling_variance = this->blobs_[4]->mutable_cpu_data();
	
	for (int q=0; q<bottom.channels(); q++)
    {
        float* ptr = data + q*bottom.offset(1);

        float a = _rolling_mean[q];
		float b = _rolling_variance[q];

        for (int i=0; i<bottom.offset(1); i++)
        {
            ptr[i] = (ptr[i] - a) / (sqrt(b) + .000001f);
        }
    }
}

void ConvolutionLayer::activate_inplace(Blob& bottom, int type)
{
	float *data = bottom.mutable_cpu_data();
	
	if(type == 1)
	{
		for(int i=0; i<bottom.count(); i++)
		{
			if(data[i] < 0) data[i] *= .1;
		}
	}
}

REGISTER_LAYER_CLASS(Convolution);

}