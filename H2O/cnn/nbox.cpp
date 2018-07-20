#include <stdio.h>
#include <stdlib.h>
#include "nbox.h"
	
static float get_pixel(image m, int x, int y, int c)
{
    //assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    //assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}
static void add_pixel(image m, int x, int y, int c, float val)
{
    //assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}
void fill_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}
image make_image(int w, int h, int c)
{
    image out;// = make_empty_image(w,h,c);
	out.h = h;
	out.c = c;
	out.w = w;
    out.data = (float*)calloc(h*w*c, sizeof(float));
    return out;
}

image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);   
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}


image letterbox_image(image im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
	image resized = resize_image(im, new_w, new_h);
    image boxed = make_image(w, h, im.c);
	
	fill_image(boxed, .5);
	embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2);
	
    free_image(resized);
    return boxed;
}

image get_image(int w, int h, int c, unsigned char *ptr)
{
	image im;
	im.w = w;
	im.h = h;
	im.c = c;
	im.data = (float*)calloc(h*w*c, sizeof(float));
	for(int i = 0; i < h; ++i){
        for(int k= 0; k < c; ++k){
            for(int j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = ptr[i*w*c + j*c + k]/255.;
            }
        }
    }
    for(int i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
	
	return im;
}

// correct boxes over
void correct_region_boxes(Detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int new_w = 0;
    int new_h = 0;
	
    if(((float)netw/w) < ((float)neth/h)) 
	{
        new_w = netw;
        new_h = (h * netw)/w;
    }else{
        new_h = neth;
        new_w = (w * neth)/h;
    }
	
    for(int i = 0; i < n; ++i)
	{
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

image Exchange_ncnnMat(int Yw, int Yh, int Yc, unsigned char *inptr)
{
	image im = get_image(Yw, Yh, Yc, inptr);
	image sized = letterbox_image(im, 416, 416);
	
	return sized;
}

//Debug
//FILE* pp = fopen("image111.txt", "wb");
//for(int i=0; i<input.w * input.h * input.c; i++){
//	fprintf(pp, "%f  v05\n", input.data[i]);
//}
//fclose(pp);