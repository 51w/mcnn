#include "NV1/run/conv_layer.h"
#include "cblas.h"

namespace NV1{

void ConvolutionLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	num_output	= GetParam_Int32(0, 0);
	kernel		= GetParam_Int32(1, 0);
	dilation	= GetParam_Int32(2, 1);
	stride		= GetParam_Int32(3, 1);
	pad			= GetParam_Int32(4, 0);
	bias_term	= GetParam_Int32(5, 0);
	weight_data_size = GetParam_Int32(6, 0);
	
	//LOG(INFO) << "Conv: " << num_output << " " << kernel << " " << dilation << " "
	//	<< stride << " " << pad << " " << bias_term << " " << weight_data_size;
		 
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	
	int kernel_extent = dilation * (kernel - 1) + 1;
	YH = (XH + 2*pad - kernel_extent) / stride + 1;
	YW = (XW + 2*pad - kernel_extent) / stride + 1;
	YC = num_output;
	
	Output[0]->Reshape(YC, YH, YW);
	_im2col.Reshape(XW*XH* XC * kernel*kernel);
	//LOG(INFO) << "Conv: " << YC << " " << YW << " " << YH;
	
	if (bias_term) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
	this->blobs_[0].reset(new Blob(Input[0]->CC() *kernel*kernel* num_output));
	this->blobs_[1].reset(new Blob(num_output));
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

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void ConvolutionLayer::Run(Tensor& Input, Tensor& Output)
{
	LOG(INFO) << "Convolution: " << YC << " " << YH << " " << YW;
	
	float *inptr = Input[0]->mutable_cpu_data();
	float *_weights = this->blobs_[0]->mutable_cpu_data();
	float *outptr   = Output[0]->mutable_cpu_data();
	float *_workspace = _im2col.mutable_cpu_data();

	int m = num_output;
    int k = kernel*kernel*XC;
    int n = YW*YH;
	
	float *a = _weights;
	float *b = _workspace;
	float *c = outptr;
	
	for(int i=0; i<Output[0]->count(); i++)
		outptr[i] = 0;

	im2col_cpu(inptr, XC, XH, XW, kernel, stride, pad, b);
	gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
	
	
	float *_biases = this->blobs_[1]->mutable_cpu_data();
	if (this->bias_term) 
	{
		for(int q=0; q<Output[0]->CC(); q++)
		{
			float* ptr = outptr + q*YW*YH;

			float bias = _biases[q];

			for (int i=0; i<YW*YH; i++)
			{
				ptr[i] += bias;
			}
		}
	}
}

}