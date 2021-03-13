// skeleton.c

/** Skeletonization algorithm

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

/** Compilar excutável utilizando
    gcc skeleton.c simple_tiffio.c -Iinclude -ltiff -Llib -o skeleton

    ou 

    gcc -Wall skeleton.c simple_tiffio.c -ltiff -o skeleton

    Compilar biblioteca utilizando

    gcc -Wall -fpic -shared skeleton.c simple_tiffio.c -Iinclude -ltiff -Llib -o libskeleton.so

    ou

    gcc -Wall -fpic -shared skeleton.c simple_tiffio.c -ltiff -o libskeleton.so
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>
#include "simple_tiffio.h"
#include "skeleton.h"

uint16_t get_mask_value(uint16_t* mask, uint32_t plane, uint32_t row, uint32_t col) {
    return mask[9*plane+3*row+col];
}

uint16_t* get_mask(uint16_t masks[][6][4][27], uint32_t dir, uint32_t i, uint32_t angle) {
    return masks[dir][i][angle];
}

bool match_mask(img_st* img, uint16_t* mask, uint32_t plane, uint32_t row, uint32_t col) {

    int32_t m_plane, m_row, m_col;   // Coordenadas da máscara
    bool flag_mask_match,            // true se todos os valores da máscara encaixam
         flag_01_match,              // true se os valores 0 e 1 da máscara encaixam na imagem
         flag_at_least_1,            // true se valores 2 da máscara encaixam em ao menos um valor 1 da imagem
         flag_has_2;                 // true se máscara possui valor 2
    uint16_t val, mask_val;

    flag_01_match = true;
    flag_at_least_1 = false;
    flag_has_2 = false;
    for (m_plane=-1; m_plane<2; m_plane++) {
        for (m_row=-1; m_row<2; m_row++) {
            for (m_col=-1; m_col<2; m_col++) {

                if (((plane+m_plane)<0) || ((row+m_row)<0) || ((col+m_col)<0) || ((plane+m_plane)>=img->depth) || ((row+m_row)>=img->height) || ((col+m_col)>=img->width))
                    val = 0;
                else
                    val = get_value_gray(*img, plane+m_plane, row+m_row, col+m_col);

                mask_val = get_mask_value(mask, m_plane+1, m_row+1, m_col+1);
                if ((mask_val == 1) || (mask_val == 0)) {
                    flag_01_match = ((val == mask_val) && (flag_01_match));
                }
                else if (mask_val == 2) {
                    flag_has_2 = true;
                    flag_at_least_1 = ((val == 1) || (flag_at_least_1));
                }
            }
        }
    }

    if (flag_has_2) {
        flag_mask_match = (flag_01_match) && (flag_at_least_1);
    }
    else {
        flag_mask_match = flag_01_match;
    }

    return flag_mask_match;
}

bool skel_dir(img_st img_iter, img_st img_final, uint32_t dir) {

    uint32_t idx, angle;            // Índices das máscaras
    uint32_t plane, row, col;       // Coordenadas da imagem
    bool flag_has_changed,          // true se ao menos um voxel da imagem foi deletado
         flag_mask_match;           // true se a máscara encaixou
    uint16_t* mask;

    flag_has_changed = false;
    //#pragma omp parallel for collapse(2) private(idx, angle, mask, plane, row, col, flag_mask_match)
    for (idx=0; idx<6; idx++) {
        for (angle=0; angle<4; angle++) {

            mask = get_mask(masks, dir, idx, angle);

            #pragma omp parallel for collapse(3) private(plane, row, col, flag_mask_match)
            for (plane=0; plane<img_iter.depth; plane++) {
                for (row=0; row<img_iter.height; row++) {
                    for (col=0; col<img_iter.width; col++) {

                        if (get_value_gray(img_iter, plane, row, col) == 1) {
                            flag_mask_match = match_mask(&img_iter, mask, plane, row, col);

                            if (flag_mask_match) {
                                put_value_gray(img_final, 0, plane, row, col);
                                flag_has_changed = true;
                            }
                        }
                    }
                }
            }
        }

    }

    return flag_has_changed;

}

img_st skel(img_st img, uint32_t verbosity) {

    img_st img_iter, img_final, img_save;
    bool flag_has_changed,            // true se um voxel da imagem mudou
         flag_has_changed_in_dir;     // true se um voxel da imagem mudou após aplicar máscaras em uma direção
    uint32_t ind_dir;
    uint32_t num_iterations=1;

    alloc_img_like(img, &img_iter);
    alloc_img_like(img, &img_final);
    copy_img(img, &img_iter);
    copy_img(img, &img_final);
    
    if (verbosity > 1){
        alloc_img_like(img, &img_save);
    }

    flag_has_changed = true;
    while (flag_has_changed==true) {

        flag_has_changed = false;
        for (ind_dir=1;ind_dir<=6;ind_dir++) {

            flag_has_changed_in_dir = skel_dir(img_iter, img_final, ind_dir-1);
            flag_has_changed = flag_has_changed || flag_has_changed_in_dir;

            copy_img(img_final, &img_iter);

        }

        if (verbosity > 0) {
            printf("%d ", num_iterations++);
            fflush(stdout);
            if (verbosity > 1) {
                if (num_iterations%verbosity == 0){
                    copy_img(img_final, &img_save);
                    reescale_to_0255(img_save);
                    save_img(img_save, "temp.tif");
                }
            }
        }

    }

    free_img_data(&img_iter);

    return img_final;
}

/** Funções para a interface com Python */
img_st img_from_array(uint16_t* in_arr, uint32_t depth, uint32_t height, uint32_t width) {

    uint32_t plane;
    img_st out_img;

    out_img.depth = depth;
    out_img.height = height;
    out_img.width = width;
    out_img.num_channels = 1;   

    out_img.data = (uint16_t**) malloc(depth * sizeof (uint16_t*));
    for (plane = 0; plane < depth; plane++) {
        out_img.data[plane] = &in_arr[plane*width*height];
    }  

    return out_img;
}

void array_from_img(img_st in_img, uint16_t* out_arr) {

    uint32_t plane, row, col, k=0, depth, height, width;

    depth = in_img.depth;
    height = in_img.height;
    width = in_img.width;

    for (plane = 0; plane < depth; plane++) {
        for (row = 0; row < height; row++) {
            for (col = 0; col < width; col++) {
                out_arr[k++] = in_img.data[plane][width*row+col];
            }
        }
    }  
}

extern int skel_interface(uint16_t *in_arr, uint16_t *out_arr, uint32_t depth, 
                        uint32_t height, uint32_t width, uint32_t num_threads, uint32_t verbosity) {

    img_st img_final;
    //omp_set_num_threads(num_threads);

    img_st in_img = img_from_array(in_arr, depth, height, width);
    img_final = skel(in_img, verbosity);
    array_from_img(img_final, out_arr);

    return 0;
}
/************************/

int main(int argc, char **argv) {

    char* file_name_out = "out.tif";
    int num_threads;
    img_st img, img_final;

    if (argc>1) {
        img = read_img(argv[1]);
        if (argc>2) {
            file_name_out = argv[2];
            if (argc>3) {
                num_threads = (int) strtol(argv[3], NULL, 10);
                //omp_set_num_threads(num_threads);
            }
        }
    }
    else {
        fprintf(stderr, "No input file");
        return -1;
    }
    reescale_to_01(img);

    img_final = skel(img, 1);
    
    reescale_to_0255(img_final);
    save_img(img_final, file_name_out);

    free_img_data(&img);
    free_img_data(&img_final);

    return 0;
}