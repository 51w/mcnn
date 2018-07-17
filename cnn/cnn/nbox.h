#pragma once

typedef struct{
    float x, y, w, h;
}box;

typedef struct{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
}Detection;


typedef struct{
    int L, R, Top, Bot;
}Rect;

typedef struct{
    Rect box;
    int nclass;
    int percent;
}Drawbox;

typedef struct{
    int w;
    int h;
    int c;
    float *data;
}image;

void correct_region_boxes(Detection *dets, int n, int w, int h, int netw, int neth, int relative);

image get_image(int w, int h, int c, unsigned char *ptr);

image letterbox_image(image im, int w, int h);

image Exchange_ncnnMat(int Yw, int Yh, int Yc, unsigned char *inptr);