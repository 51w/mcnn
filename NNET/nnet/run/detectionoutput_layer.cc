#include "nnet/run/detectionoutput_layer.h"

namespace NNET
{

void DetectionOutputLayer::SetUp(Tensor& Input, Tensor& Output) 
{
	num_class     = GetParam_Int32(0, 0);
    nms_threshold = GetParam_f32(1, 0.05f);
    nms_top_k     = GetParam_Int32(2, 300);
    keep_top_k    = GetParam_Int32(3, 100);
    confidence_threshold = GetParam_f32(4, 0.5f);
	
	bboxes.resize(Input[2]->HH());
}

void DetectionOutputLayer::Reshape(Tensor& Input, Tensor& Output)
{
	
}

struct BBoxRect
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    int   label;
};

static inline float intersection_area(const BBoxRect& a, const BBoxRect& b)
{
    if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
    float inter_height = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);

    return inter_width * inter_height;
}

template <typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores, int left, int right)
{
    int i = left;
    int j = right;
    float p = scores[(left + right) / 2];

    while (i <= j)
    {
        while (scores[i] > p)
            i++;

        while (scores[j] < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(datas[i], datas[j]);
            std::swap(scores[i], scores[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(datas, scores, left, j);

    if (i < right)
        qsort_descent_inplace(datas, scores, i, right);
}

template <typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores)
{
    if (datas.empty() || scores.empty())
        return;

    qsort_descent_inplace(datas, scores, 0, scores.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<BBoxRect>& bboxes, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = bboxes.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        const BBoxRect& r = bboxes[i];

        float width = r.xmax - r.xmin;
        float height = r.ymax - r.ymin;

        areas[i] = width * height;
    }

    for (int i = 0; i < n; i++)
    {
        const BBoxRect& a = bboxes[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const BBoxRect& b = bboxes[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
//             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void DetectionOutputLayer::Run(Tensor& Input, Tensor& Output)
{
	//LOG(INFO) << num_class << " " << nms_threshold << " " << nms_top_k << " " << keep_top_k << " " << confidence_threshold;
	// LOG(INFO) << Input[0]->CC() << " " << Input[0]->HH() << " " << Input[0]->WW();
	// LOG(INFO) << Input[1]->CC() << " " << Input[1]->HH() << " " << Input[1]->WW();
	// LOG(INFO) << Input[2]->CC() << " " << Input[2]->HH() << " " << Input[2]->WW();
	// vector<int> top_shape(2, 1);
	// LOG(INFO) << top_shape[0] << " " << top_shape[1];
	
	float *location   = Input[0]->mutable_cpu_data();
	float *confidence = Input[1]->mutable_cpu_data();
	float *priorbox   = Input[2]->mutable_cpu_data();
	
	const int num_prior = Input[2]->HH() / 4;
	
	int idx = 0;
	for(int i=0; i<num_prior; i++)
    {
        const float* loc = location + i * 4;
        const float* pb  = priorbox + i * 4;
        const float* var = priorbox + i * 4 + Input[2]->HH();

        // CENTER_SIZE
        float pb_w = pb[2] - pb[0];
        float pb_h = pb[3] - pb[1];
        float pb_cx = (pb[0] + pb[2]) * 0.5f;
        float pb_cy = (pb[1] + pb[3]) * 0.5f;

        float bbox_cx = var[0] * loc[0] * pb_w + pb_cx;
        float bbox_cy = var[1] * loc[1] * pb_h + pb_cy;
        float bbox_w = exp(var[2] * loc[2]) * pb_w;
        float bbox_h = exp(var[3] * loc[3]) * pb_h;

        bboxes[idx++] = bbox_cx - bbox_w * 0.5f;
        bboxes[idx++] = bbox_cy - bbox_h * 0.5f;
        bboxes[idx++] = bbox_cx + bbox_w * 0.5f;
        bboxes[idx++] = bbox_cy + bbox_h * 0.5f;
    }

	// sort and nms for each class
    std::vector< std::vector<BBoxRect> > all_class_bbox_rects;
    std::vector< std::vector<float> > all_class_bbox_scores;
    all_class_bbox_rects.resize(num_class);
    all_class_bbox_scores.resize(num_class);
	
	for(int i=1; i<num_class; i++)
    {
        std::vector<BBoxRect> class_bbox_rects;
        std::vector<float> class_bbox_scores;

        for(int j = 0; j < num_prior; j++)
        {
            float score = confidence[j * num_class + i];

            if(score > confidence_threshold)
            {
                BBoxRect c = { bboxes[0 + 4*j], bboxes[1 + 4*j], bboxes[2 + 4*j], bboxes[3 + 4*j], i };
                class_bbox_rects.push_back(c);
                class_bbox_scores.push_back(score);
            }
        }

        // sort inplace
        qsort_descent_inplace(class_bbox_rects, class_bbox_scores);

        // keep nms_top_k
        if(nms_top_k < (int)class_bbox_rects.size())
        {
            class_bbox_rects.resize(nms_top_k);
            class_bbox_scores.resize(nms_top_k);
        }

        // apply nms
        std::vector<int> picked;
        nms_sorted_bboxes(class_bbox_rects, picked, nms_threshold);

        // select
        for(int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            all_class_bbox_rects[i].push_back(class_bbox_rects[z]);
            all_class_bbox_scores[i].push_back(class_bbox_scores[z]);
        }
    }
	
	//LOG(INFO) << "===============";
	
	std::vector<BBoxRect> bbox_rects;
    std::vector<float> bbox_scores;

    for(int i = 1; i < num_class; i++)
    {
        const std::vector<BBoxRect>& class_bbox_rects = all_class_bbox_rects[i];
        const std::vector<float>& class_bbox_scores = all_class_bbox_scores[i];

        bbox_rects.insert(bbox_rects.end(), class_bbox_rects.begin(), class_bbox_rects.end());
        bbox_scores.insert(bbox_scores.end(), class_bbox_scores.begin(), class_bbox_scores.end());
    }

	qsort_descent_inplace(bbox_rects, bbox_scores);

    if (keep_top_k < (int)bbox_rects.size())
    {
        bbox_rects.resize(keep_top_k);
        bbox_scores.resize(keep_top_k);
    }
	
	int num_detected = bbox_rects.size();
	float *dst;
	
	if(num_detected == 0)
	{
		LOG(INFO) << "Couldn't find any detections";	
		Output[0]->Reshape(6, 1);	
		dst = Output[0]->mutable_cpu_data();
		
		for(int i=0; i<6; ++i) dst[i] = -1;
	}
	else
	{
		Output[0]->Reshape(6, num_detected);	
		dst = Output[0]->mutable_cpu_data();
	}

	for(int i = 0; i < num_detected; i++)
    {
        const BBoxRect& r = bbox_rects[i];
        float score = bbox_scores[i];

        dst[0] = r.label;
        dst[1] = score;
        dst[2] = r.xmin;
        dst[3] = r.ymin;
        dst[4] = r.xmax;
        dst[5] = r.ymax;
		dst += 6;
    }
	
	// FILE* pp1 = fopen("DetectionOutput1.txt", "wb");
	// float *out1 = Input[0]->mutable_cpu_data();
	// for (int j=0; j < Input[0]->count(); j++)
		// fprintf(pp1, "%f  -- 11\n", out1[j]);
	// fclose(pp1);
	
	// FILE* pp2 = fopen("DetectionOutput2.txt", "wb");
	// float *out2 = Input[1]->mutable_cpu_data();
	// for (int j=0; j < Input[1]->count(); j++)
		// fprintf(pp2, "%f  -- 22\n", out2[j]);
	// fclose(pp2);
	
	
	// FILE* pp3 = fopen("DetectionOutput3.txt", "wb");
	// float *out3 = Input[2]->mutable_cpu_data();
	// for (int j=0; j < Input[2]->count(); j++)
		// fprintf(pp3, "%f  -- 33\n", out3[j]);
	// fclose(pp3);
	// LOG(FATAL) << "do";
}

}
