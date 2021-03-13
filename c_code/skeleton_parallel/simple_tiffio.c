/** Simple interface to Libtiff

    Copyright (C) 2019 Cesar Henrique Comin
    chcomin@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <tiffio.h>
#include "simple_tiffio.h"

//#define get_value_gray(img, plane, row, col) (img.data[plane][row*img.width+col])

// Compile with
// gcc -Itiff-4.0.9/libtiff simple_tiffio.c -ltiff -Ltiff-4.0.9/libtiff/.libs

img_st read_img(char const* file_name) {

    uint32_t width, height, depth, plane;
    uint16_t photometricInt, samplesPerPix, bitsPerSample, compression, rowsPerStrip;
    size_t npixels, count;
    uint32_t* raster;
    uint16_t is_RGB;
    uint16_t num_channels;
    uint16_t *plane_p;
    img_st img;

    TIFF* tif = TIFFOpen(file_name, "r");

    if (!tif) {
        printf("Could not read file");
    }

    // Count number of planes
    depth = 0;
    do {
        depth++;
    } while (TIFFReadDirectory(tif));
    TIFFSetDirectory(tif, 0);

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometricInt);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPix);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_COMPRESSION, &compression);
    TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);

    switch(photometricInt) {
        case 0 || 1:
            if (samplesPerPix != 1) {
                printf("Warning, photometric interpretation is %d, but number of samples per pixel is not 1", photometricInt);
            }
            else {
                is_RGB = 0;
            }
            break;
        case 2:
            if (samplesPerPix != 3) {
                printf("Warning, photometric interpretation is 2, but number of samples per pixel is not 3");
            }
            else {
                is_RGB = 1;
            }
            break;
        default:
            printf("Photometric interpretation not recognized");
    }

    if (is_RGB) {
        num_channels = 3;
    }
    else {
        num_channels = 1;
    }

    img.width = width;
    img.height = height;
    img.depth = depth;
    img.num_channels = num_channels;

    alloc_img_data(&img);

    npixels = width * height;
    raster = (uint32_t*) _TIFFmalloc(npixels * sizeof (uint32_t));
    
    plane = 0;
    do {
        if (TIFFReadRGBAImageOriented(tif, width, height, raster, ORIENTATION_TOPLEFT, 0)) {

            plane_p = img.data[plane];
            count = 0;

             if (num_channels==1) {
                uint32_t* raster_reader = raster;
                while (count++<npixels) {
                    *(plane_p++) = (uint16_t) ((*(raster_reader++)) & 0xff);
                }
            }
            else {
                unsigned char * raster_reader = (unsigned char *) raster;
                while (count++<npixels) {
                    *(plane_p++) = (uint16_t) *(raster_reader++);
                    *(plane_p++) = (uint16_t) *(raster_reader++);
                    *(plane_p++) = (uint16_t) *(raster_reader++);
                    raster_reader++;
                }
            }

        }
        plane++;
    } while (TIFFReadDirectory(tif));

    _TIFFfree(raster);
    TIFFClose(tif);

    return img;
}

static int cvt_whole_image(img_st img, TIFF *out, uint32_t page)
{
    unsigned char* raster;			   
    uint32_t  width, height;	        /* image width & height */
    uint32_t  row;
    size_t pixel_count, count;
    uint32_t rowsperstrip = (uint32_t) -1;
    unsigned char* raster_p;
    uint16_t* plane_p;
        
    width = img.width;
    height = img.height;
    pixel_count = width * height;

    rowsperstrip = TIFFDefaultStripSize(out, rowsperstrip);
    TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, rowsperstrip);

    raster = (unsigned char*) _TIFFmalloc(3*pixel_count * sizeof (unsigned char));

    raster_p = (unsigned char*) raster;
    plane_p = img.data[page];
    count = 0;

    if (img.num_channels==1) {
        while (count++<pixel_count) {
            *(raster_p++) = *(plane_p++);
        }
    }
    else if (img.num_channels==3) {
         while (count++<pixel_count) {
            *(raster_p++) = *(plane_p++);
            *(raster_p++) = *(plane_p++);
            *(raster_p++) = *(plane_p++);
        }
    }    
    else {
        printf("Number of channels not supported\n");
    }

    int	bytes_per_pixel;
    if (img.num_channels==1) {
        bytes_per_pixel = 1;
    }
    else if (img.num_channels==3) {
        bytes_per_pixel = 3;
    }
    else {
        printf("Number of channels not supported\n");
    }
    
    for (row = 0; row < height; row += rowsperstrip)
    {
        unsigned char * raster_strip;
        int	rows_to_write;
        
        raster_strip = ((unsigned char *) raster) + bytes_per_pixel * row * width;

        if( row + rowsperstrip > height )
            rows_to_write = height - row;
        else
            rows_to_write = rowsperstrip;

        if( TIFFWriteEncodedStrip( out, row / rowsperstrip, raster_strip,
                             bytes_per_pixel * rows_to_write * width ) == -1 )
        {
            _TIFFfree( raster );
            return 0;
        }
    }

    _TIFFfree( raster );

    return 1;
}


static int tiffcvt(img_st img, TIFF* out, uint32_t page)
{
	TIFFSetField(out, TIFFTAG_IMAGEWIDTH, img.width);
	TIFFSetField(out, TIFFTAG_IMAGELENGTH, img.height);
	TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);
	TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(out, TIFFTAG_SOFTWARE, TIFFGetVersion());

    TIFFSetField(out, TIFFTAG_SUBFILETYPE, 0);
    TIFFSetField(out, TIFFTAG_FILLORDER, 1);

    // In ImageJ, the resolution is the inverse of pixel width
    TIFFSetField(out, TIFFTAG_XRESOLUTION, 1.);
    TIFFSetField(out, TIFFTAG_YRESOLUTION, 1.);
    TIFFSetField(out, TIFFTAG_RESOLUTIONUNIT, RESUNIT_NONE);

    if (img.num_channels==1) {
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
    }
    else if (img.num_channels==3) {
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 3);
    }
    else {
        printf("Number of channels not recognized");
    }
    

    return( cvt_whole_image(img, out, page) );
}

int save_img(img_st img, char const* file_name_out) {

    TIFF *out;
    uint32_t page=0;

	out = TIFFOpen(file_name_out, "w");

    for (page=0; page<img.depth; page++) {
        if (!tiffcvt(img, out, page) || !TIFFWriteDirectory(out)) {
            TIFFClose(out);
            return (1);
        }
    } 

	TIFFClose(out);

    return 0;
}

pix_st get_value_rgb(img_st img, uint32_t plane, uint32_t row, uint32_t col) {

    pix_st pix_val;

    pix_val.r = img.data[plane][3*row*img.width+3*col];
    pix_val.g = img.data[plane][3*row*img.width+3*col+1];
    pix_val.b = img.data[plane][3*row*img.width+3*col+2];

    return pix_val;

}

uint16_t get_value_gray(img_st img, uint32_t plane, uint32_t row, uint32_t col) {

    return img.data[plane][row*img.width+col];
}

void put_value_rgb(img_st img, pix_st pix_val, uint32_t plane, uint32_t row, uint32_t col) {

    img.data[plane][3*row*img.width+3*col] = pix_val.r;
    img.data[plane][3*row*img.width+3*col+1] = pix_val.g;
    img.data[plane][3*row*img.width+3*col+2] = pix_val.b;
}

void put_value_gray(img_st img, uint16_t pix_val, uint32_t plane, uint32_t row, uint32_t col) {

    img.data[plane][row*img.width+col] = pix_val;
}


void print_img(img_st img) {

    uint32_t row, col, plane;
    pix_st pix_val;

    for (plane = 0; plane < img.depth; plane++) {
        for (row = 0; row < img.height; row++) {
            for (col = 0; col < img.width; col++) {
                if (img.num_channels == 1) {
                    printf("%d ", get_value_gray(img, plane, row, col));
                }
                else {
                    pix_val = get_value_rgb(img, plane, row, col);
                    printf("(%d,%d,%d) ", pix_val.r, pix_val.g, pix_val.b);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}

void alloc_img_like(img_st in_img, img_st* out_img) {

    uint32_t plane;
    //img_st out_img = *out_img_p;

    out_img->depth = in_img.depth;
    out_img->height = in_img.height;
    out_img->width = in_img.width;
    out_img->num_channels = in_img.num_channels;   

    out_img->data = (uint16_t**) malloc(in_img.depth * sizeof (uint16_t*));
    for (plane = 0; plane < in_img.depth; plane++) {
        out_img->data[plane] = (uint16_t*) malloc(in_img.width*in_img.height*in_img.num_channels * sizeof (uint16_t));
    }  
}

void alloc_img_data(img_st* img) {

    uint32_t plane;

    img->data = (uint16_t**) malloc(img->depth * sizeof (uint16_t*));
    for (plane = 0; plane < img->depth; plane++) {
        img->data[plane] = (uint16_t*) malloc(img->width*img->height*img->num_channels * sizeof (uint16_t));
    }  
}

void free_img_data(img_st* img) {

    uint32_t plane;

    for (plane = 0; plane < img->depth; plane++) {
        free(img->data[plane]);
    }  
    free(img->data);
}

void copy_img(img_st in_img, img_st* out_img) {

    uint32_t row, col, plane;
    uint16_t val;

    out_img->depth = in_img.depth;
    out_img->height = in_img.height;
    out_img->width = in_img.width;
    out_img->num_channels = in_img.num_channels;

    for (plane = 0; plane < in_img.depth; plane++) {
        for (row = 0; row < in_img.height; row++) {
            for (col = 0; col < in_img.width; col++) {
                val = get_value_gray(in_img, plane, row, col);
                put_value_gray(*out_img, val, plane, row, col);
            }
        }
    }
}

void reescale_to_01(img_st img) {

    uint32_t row, col, plane;
    uint16_t val;

    for (plane = 0; plane < img.depth; plane++) {
        for (row = 0; row < img.height; row++) {
            for (col = 0; col < img.width; col++) {
                val = get_value_gray(img, plane, row, col);
                val = (uint16_t) val/255;
                put_value_gray(img, val, plane, row, col);
            }
        }
    }
}

void reescale_to_0255(img_st img) {

    uint32_t row, col, plane;
    uint16_t val;

    for (plane = 0; plane < img.depth; plane++) {
        for (row = 0; row < img.height; row++) {
            for (col = 0; col < img.width; col++) {
                val = get_value_gray(img, plane, row, col);
                val = 255*val;
                put_value_gray(img, val, plane, row, col);
            }
        }
    }
}

void tests(char* file_name) {

    uint32_t width, height, depth;
    uint16_t photometricInt, samplesPerPix, bitsPerSample;
    size_t npixels;
    uint32_t* raster;

    TIFF* tif = TIFFOpen(file_name, "r");

    if (!tif) {
        printf("Could not read file");
    }

    // Count number of planes
    depth = 0;
    do {
        depth++;
    } while (TIFFReadDirectory(tif));
    TIFFSetDirectory(tif, 0);

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometricInt);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPix);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);

    npixels = width * height;
    raster = (uint32_t*) _TIFFmalloc(npixels * sizeof (uint32_t));


    if (TIFFReadRGBAImageOriented(tif, width, height, raster, ORIENTATION_TOPLEFT, 0)) {
        printf("%d ", (raster[0] >> 16) & 0xff);
    }

    _TIFFfree(raster);
    TIFFClose(tif);

}
