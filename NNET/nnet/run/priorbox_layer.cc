#include "nnet/run/priorbox_layer.h"

namespace NNET
{

void PriorBoxLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	GetParam_vec(0, min_sizes);
	GetParam_vec(1, max_sizes);
	GetParam_vec(2, aspect_ratios);

	variances[0] = GetParam_f32(3, 0.f);
    variances[1] = GetParam_f32(4, 0.f);
    variances[2] = GetParam_f32(5, 0.f);
    variances[3] = GetParam_f32(6, 0.f);
    flip = GetParam_Int32(7, 1);
    clip = GetParam_Int32(8, 0);
    image_width  = GetParam_Int32(9, 0);
    image_height = GetParam_Int32(10,0);
    step_width   = GetParam_f32(11, -233.f);
    step_height  = GetParam_f32(12, -233.f);
    offset = GetParam_f32(13, 0.f);
}

void PriorBoxLayer::Reshape(Tensor& Input, Tensor& Output)
{
	X0C = Input[0]->CC();
	X0H = Input[0]->HH();
	X0W = Input[0]->WW();
	X1C = Input[1]->CC();
	X1H = Input[1]->HH();
	X1W = Input[1]->WW();
	
	int num_min_size = min_sizes.size();
    int num_max_size = max_sizes.size();
    int num_aspect_ratio = aspect_ratios.size();

    num_prior = num_min_size * num_aspect_ratio + num_min_size + num_max_size;
    if (flip)
        num_prior += num_min_size * num_aspect_ratio;
	
	Output[0]->Reshape(2, 4 * X0W * X0H * num_prior);
}

void PriorBoxLayer::Run(Tensor& Input, Tensor& Output)
{
	int image_w = image_width;
    int image_h = image_height;
    if (image_w == -233)
        image_w = X1W;
    if (image_h == -233)
        image_h = X1H;
	
	float step_w = step_width;
    float step_h = step_height;
    if(step_w == -233)
        step_w = (float)image_w / X0W;
    if(step_h == -233)
        step_h = (float)image_h / X0H;
	
	float* dst = Output[0]->mutable_cpu_data();
	int dim = X0W * X0H * num_prior * 4;
	
	int idx = 0;
	for(int h=0; h < X0H; ++h)
	{
		for(int w=0; w < X0W; ++w)
		{
			float center_x = (w + offset) * step_w;
			float center_y = (h + offset) * step_h;
			float box_width, box_height;
			for(int s = 0; s < min_sizes.size(); ++s)
			{
				int min_size = min_sizes[s];
				box_width = box_height = min_size;

				dst[idx++] = (center_x - box_width  / 2.) / image_w;
				dst[idx++] = (center_y - box_height / 2.) / image_h;
				dst[idx++] = (center_x + box_width  / 2.) / image_w;
				dst[idx++] = (center_y + box_height / 2.) / image_h;

				if(max_sizes.size() > 0)
				{
					CHECK(min_sizes.size() == max_sizes.size());
					int max_size = max_sizes[s];
					box_width = box_height = sqrt(min_size * max_size);

					dst[idx++] = (center_x - box_width  / 2.) / image_w;
					dst[idx++] = (center_y - box_height / 2.) / image_h;
					dst[idx++] = (center_x + box_width  / 2.) / image_w;
					dst[idx++] = (center_y + box_height / 2.) / image_h;
				}

				// rest of priors
				for(int r = 0; r < aspect_ratios.size(); ++r)
				{
					float ar = aspect_ratios[r];
					if(fabs(ar - 1.) < 1e-6) {
						continue;
					}
					box_width  = min_size * sqrt(ar);
					box_height = min_size / sqrt(ar);

					dst[idx++] = (center_x - box_width  / 2.) / image_w;
					dst[idx++] = (center_y - box_height / 2.) / image_h;
					dst[idx++] = (center_x + box_width  / 2.) / image_w;
					dst[idx++] = (center_y + box_height / 2.) / image_h;
					
					if(flip)
					{
						dst[idx++] = (center_x - box_height / 2.) / image_w;
						dst[idx++] = (center_y - box_width  / 2.) / image_h;
						dst[idx++] = (center_x + box_height / 2.) / image_w;
						dst[idx++] = (center_y + box_width  / 2.) / image_h;
					}
				}
			}
		}
	}
	
	if(clip)
    {
        for(int i=0; i<dim; i++)
        {
            dst[i] = std::min(std::max(dst[i], 0.f), 1.f);
        }
    }

    // set variance
    dst += Output[0]->offset(1);
    for(int i=0; i < dim/4; i++)
    {
        dst[0] = variances[0];
        dst[1] = variances[1];
        dst[2] = variances[2];
        dst[3] = variances[3];

        dst += 4;
    }
}

}
