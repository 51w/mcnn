#include "cnn/layers/conv_layer.hpp"
#include "cblas.h"

namespace H2O {

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
	this->blobs_[0].reset(new Blob(bottom[0]->CC()*_kernel*_kernel*_filters));
	this->blobs_[1].reset(new Blob(_filters));
	
	
	XC = bottom[0]->CC();
	XH = bottom[0]->HH();
	XW = bottom[0]->WW();
	
	YH = (XH + 2*_pad - _kernel) / _stride + 1;
	YW = (XW + 2*_pad - _kernel) / _stride + 1;
	YC = _filters;
	
	top[0]->Reshape(YC, YH, YW);
	//_im2col.Reshape(6230016);
	_im2col.Reshape(YW*YH* XC * _kernel*_kernel);
	//fprintf(stderr, "ConvolutionLayer: %d %d %d\n", top[0]->channels(), top[0]->height(), top[0]->width());
	//fprintf(stderr, "_im2col size: %d\n", _im2col.count() );
}

float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
				//std::cout << channels_col << " " << col_index << std::endl;
            }
        }
    }
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    //#pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB){
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);	
	}	
    else if(TA && !TB)
        ;//gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        ;//gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        ;//gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
	//std::cout << "    >>>gemm_nn\n";
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb, 0,C,ldc);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void ConvolutionLayer::Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	//fprintf(stderr, "Conv:  %d %d %d  %d %d %d\n", _normalize, _filters, _kernel, _stride, _pad, _activation);
#if 0
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
#else
	float *inptr = bottom[0]->mutable_cpu_data();
	float *_weights = this->blobs_[0]->mutable_cpu_data();
	float *outptr   = top[0]->mutable_cpu_data();
	float *_workspace = _im2col.mutable_cpu_data();

	int m = _filters;
    int k = _kernel*_kernel*XC;
    int n = YW*YH;
	
	float *a = _weights;
	float *b = _workspace;
	float *c = outptr;
	
	for(int i=0; i<top[0]->count(); i++)
		outptr[i] = 0;


//fprintf(stderr, "111111111\n");
	im2col_cpu(inptr, XC, XH, XW, _kernel, _stride, _pad, b);
	gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
//fprintf(stderr, "222222222\n");

#endif


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
	
	for (int q=0; q<bottom.CC(); q++)
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
	
	for (int q=0; q<bottom.CC(); q++)
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
	
	for (int q=0; q<bottom.CC(); q++)
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