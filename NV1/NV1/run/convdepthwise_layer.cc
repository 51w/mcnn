#include "NV1/run/convdepthwise_layer.h"
#include "cblas.h"

namespace NV1 {

void ConvolutionDepthWiseLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	num_output	= GetParam_Int32(0, 0);
	kernel		= GetParam_Int32(1, 0);
	dilation	= GetParam_Int32(2, 1);
	stride		= GetParam_Int32(3, 1);
	pad			= GetParam_Int32(4, 0);
	bias_term	= GetParam_Int32(5, 0);
	weight_data_size = GetParam_Int32(6, 0);
	group 		= GetParam_Int32(7, 1);
	CHECK(Input[0]->CC()%group == 0 || num_output%group == 0);
	
	//LOG(INFO) << "ConvDW: " << num_output << " " << kernel << " " << dilation << " " << stride << " "
	//	 << pad << " " << bias_term << " " << weight_data_size << " " <<group;
		 
	XC = Input[0]->CC();
	XH = Input[0]->HH();
	XW = Input[0]->WW();
	
	int kernel_extent = dilation * (kernel - 1) + 1;
	YH = (XH + 2*pad - kernel_extent) / stride + 1;
	YW = (XW + 2*pad - kernel_extent) / stride + 1;
	YC = num_output;
	
	Output[0]->Reshape(YC, YH, YW);
	_im2col.Reshape(XW*XH* XC * kernel*kernel);
	//LOG(INFO) << "ConvDW: " << YC << " " << YW << " " << YH;
	
	if (bias_term) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
	this->blobs_[0].reset(new Blob(Input[0]->CC() *kernel*kernel* num_output/group));
	this->blobs_[1].reset(new Blob(num_output));
}

float im2col_get_pixelDW(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col_cpu0(float* data_im,
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
                data_col[col_index] = im2col_get_pixelDW(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
				//std::cout << channels_col << " " << col_index << std::endl;
            }
        }
    }
}

void gemm0(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void caffe_cpu_gemm0(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
	const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) 
{
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

void ConvolutionDepthWiseLayer::Run(Tensor& Input, Tensor& Output)
{
	LOG(INFO) << "ConvolutionDepthWiseLayer: " << YC << " " << YH << " " << YW;
	
	float *inptr = Input[0]->mutable_cpu_data();
	float *_weights = this->blobs_[0]->mutable_cpu_data();
	float *outptr   = Output[0]->mutable_cpu_data();
	float *_workspace = _im2col.mutable_cpu_data();

	int m = num_output;
    int k = kernel*kernel*XC/group;
    int n = YW*YH;
	
	float *a = _weights;
	float *b = _workspace;
	float *c = outptr;
	
	for(int i=0; i<Output[0]->count(); i++)
		outptr[i] = 0;

	im2col_cpu0(inptr, XC, XH, XW, kernel, stride, pad, b);
	//gemm0(0,0,m,n,k,1,a,k,b,n,1,c,n);
	LOG(INFO) << "im2col OK";
	
	int col_offset_ = n*k;
	int output_offset_ = m*n/group;
	int weight_offset_ = m*k/group;
	for (int g = 0; g < group; ++g)
	{
		caffe_cpu_gemm0(CblasNoTrans, CblasNoTrans, m/group, n, k, 
			1., a + weight_offset_ * g, b + col_offset_ * g,
			0., c + output_offset_ * g);
	}
	
	LOG(INFO) << "convdw OK";
	
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
