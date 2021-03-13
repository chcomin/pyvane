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

#ifndef SIMPLETIFFIO_H
#define SIMPLETIFFIO_H

#include <stdint.h>
 
typedef struct img_st {
    uint16_t **data;
    uint32_t depth;
    uint32_t width;
    uint32_t height;
    uint16_t num_channels;
} img_st;

typedef struct pix_st {
    uint16_t r;
    uint16_t g;
    uint16_t b;
} pix_st;

/** Read image from a given file */
img_st read_img(char const* file_name);

/** Save image to file */
int save_img(img_st img, char const* file_name_out);

/** Get voxel value from grayscale image */
uint16_t get_value_gray(img_st img, uint32_t plane, uint32_t row, uint32_t col);

/** Get voxel value from RGB image */
pix_st get_value_rgb(img_st img, uint32_t plane, uint32_t row, uint32_t col);

/** Set voxel value of grayscale image */
void put_value_gray(img_st img, uint16_t pix_val, uint32_t plane, uint32_t row, uint32_t col);

/** Set voxel value of RGB image*/
void put_value_rgb(img_st img, pix_st pix_val, uint32_t plane, uint32_t row, uint32_t col);

/** Print image values in the terminal */
void print_img(img_st img);

/** Allocate memory for image. The memory is allocated for member 'data' of
    input 'img'. Parameters depth, height, width and num_channels of img
    must be set before calling this function */
void alloc_img_data(img_st* img);

/** Fill image properties and allocate memory for image 'out_img'. The size
    and number of channels of the generated image is copied from 'in_img'. */
void alloc_img_like(img_st in_img, img_st* out_img);

/** Free memory associated with image 'img' */ 
void free_img_data(img_st* img);

/** Copy image data from 'in_img' to 'out_img' */
void copy_img(img_st in_img, img_st* out_img);

/** Reescale binary image having values 0 and 255 to 0 and 1. Note
    That this function only works for binary images */
void reescale_to_01(img_st img);

/** Reescale binary image having values 0 and 1 to 0 and 255. Note
    That this function only works for binary images */
void reescale_to_0255(img_st img);
   
#endif