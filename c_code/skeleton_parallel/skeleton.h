// skeleton.h

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

/** Retorna valor de uma máscara de tamanho 3x3x3 na posição (plane,row,col) */
uint16_t get_mask_value(uint16_t* mask, uint32_t plane, uint32_t row, uint32_t col);

/** Retorna array 1D contendo a máscara 3x3x3 na direção 'dir', índice 'i' e ângulo 'angle'.
    'dir' deve estar no intervalo [0,5], 'i' deve estar no intervalo [0,5] e 'angle' deve estar
    no intervalo [0,3].
 */
uint16_t* get_mask(uint16_t masks[][6][4][27], uint32_t dir, uint32_t i, uint32_t angle);

/** Verifica se a máscara encaixa no pixel (plane,row,col) da imagem. Retorna true em caso
    positivo e false caso contrário. Casos positivos representam vóxeis que podem ser removidos.
*/
bool match_mask(img_st* img, uint16_t* mask, uint32_t plane, uint32_t row, uint32_t col);

/** Aplica algoritmo de esqueletização em uma direção específica. Máscaras
    são sucessivamente aplicadas am cada voxel da imagem, que são removidos
    se sua vizinhança seguir o padrão da máscara.
 */
bool skel_dir(img_st img_iter, img_st img_final, uint32_t dir);

/** Aplica algoritmo de esqueletização na imagem binária. Retorna uma nova imagem
    binária. Se parâmetro verbosity for 1, imprime na tela a iteração atual. Se ele
    for maior que 1, salva uma imagem temporária a cada verbosity iterações
 */
img_st skel(img_st img, uint32_t verbosity);

/** Funções para a interface com Python */

/** Transforma uma imagem representada por um array uint16_t 1D 
    em uma imagem do tipo img_st. Valores não são copiados.
 */
img_st img_from_array(uint16_t* in_arr, uint32_t depth, uint32_t height, uint32_t width);

/** Transforma uma imagem do tipo img_st em um array 1D do tipo uint16_t.
    Os valores são copiados. 
    Cuidado! 'out_arr' deve ser pré alocado!
 */
void array_from_img(img_st in_img, uint16_t* out_arr);

/** Função a ser chamada por um programa em Python. 
    Parâmetros:
    -----------
    in_arr : uint16_t*
        Imagem de entrada, representada por um array 1D
    out_arr : uint16_t*
        Array de saída. Deve ser pré alocado em Python, utilizando Numpy
    depth : uint32_t
        Profundidade (número de planos) da imagem
    height : uint32_t
        Altura (número de linhas) da imagem
    width : uint32_t
        Largura (número de colunas) da imagem
    verbosity : uint32_t
        Se possuir valor 1, imprime na tela a iteração atual. Se ele for maior que 1, 
        salva uma imagem temporária, com nome temp.tif, a cada verbosity iterações
 */
extern int skel_interface(uint16_t *in_arr, uint16_t *out_arr, uint32_t depth, 
                          uint32_t height, uint32_t width, uint32_t num_threads, 
                          uint32_t verbosity);

/** Definição das máscaras */
static uint16_t masks[6][6][4][27] = {
    // Direção U
    {
        // M1
        {   
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2},  // 0
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2},  // 90
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2},  // 180
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2}   // 270
        },
        // M2
        {

            {3, 3, 3, 0, 0, 0, 0, 0, 0, 3, 1, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3}, 
            {0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {0, 0, 3, 0, 0, 3, 0, 0, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3} 
        },
        // M3
        {
            {3, 3, 3, 0, 0, 3, 0, 0, 3, 3, 1, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3}, 
            {3, 3, 3, 3, 0, 0, 3, 0, 0, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3}, 
            {3, 0, 0, 3, 0, 0, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3}, 
            {0, 0, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3}  
        },
        // M4
        {
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3}, 
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3}  
        },
        // M5
        {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 2, 0, 0, 0, 2, 1, 2, 2, 0, 2, 0, 0, 0}, 
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 1, 0, 2, 2, 0, 2, 2, 0, 1, 0, 0, 2, 2, 0}, 
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 2, 2, 0, 0, 0, 2, 0, 2, 2, 1, 2}, 
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 1, 2, 0, 2, 2, 0, 2, 2, 0, 0, 1, 0, 2, 2}  
        },
        // M6
        {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 1, 3, 0, 0, 3, 2, 1, 2, 0, 0, 1, 0, 0, 2}, 
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 1, 0, 3, 0, 0, 2, 1, 2, 1, 0, 0, 2, 0, 0}, 
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 1, 0, 3, 3, 3, 2, 0, 0, 1, 0, 0, 2, 1, 2}, 
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 3, 3, 3, 3, 0, 0, 2, 0, 0, 1, 2, 1, 2}  
        }
    },

    // Direção D
    {
        {
            {2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}
        },
        {
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 0, 0, 3, 0, 0, 3, 0, 0},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 1, 3, 0, 0, 0, 0, 0, 0, 3, 3, 3},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3}
        },
        {
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 0, 0},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 1, 3, 3, 0, 0, 3, 0, 0, 3, 3, 3},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 3, 0, 0, 3, 0, 0, 3, 3, 3, 3},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 0, 0, 3, 0, 0, 3}
        },
        {
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0}
        },
        {
            {2, 1, 2, 2, 0, 2, 0, 0, 0, 2, 2, 2, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {2, 2, 0, 1, 0, 0, 2, 2, 0, 2, 2, 0, 2, 1, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 2, 0, 2, 2, 1, 2, 0, 0, 0, 2, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 2, 2, 0, 0, 1, 0, 2, 2, 0, 2, 2, 0, 1, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}
        },
        {
            {2, 1, 2, 1, 0, 0, 2, 0, 0, 3, 3, 3, 3, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {2, 0, 0, 1, 0, 0, 2, 1, 2, 3, 0, 0, 3, 1, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 2, 0, 0, 1, 2, 1, 2, 0, 0, 3, 0, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {2, 1, 2, 0, 0, 1, 0, 0, 2, 3, 3, 3, 0, 1, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0}
        }
    },

    // Direção N
    {
        {
            {0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 1, 2, 2, 1, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2},
            {0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 1, 2, 2, 1, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2},
            {0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 1, 2, 2, 1, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2},
            {0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 1, 2, 2, 1, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2}
        },
        {
            {0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 1, 1, 3, 1, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 3, 3, 3, 3, 3},
            {3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 3, 3, 1, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3}
        },
        {
            {0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 1, 1, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 0, 0, 3, 3, 1, 1, 3, 1, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 0, 0, 1, 1, 3, 3, 1, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3},
            {3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3}
        },
        {
            {0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 1, 3, 3, 1, 3, 3, 3},
            {0, 0, 1, 3, 3, 1, 3, 3, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 3, 3, 3, 3, 3},
            {1, 0, 0, 1, 3, 3, 3, 3, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 3, 3, 3, 3, 3},
            {0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 1, 0, 0, 1, 3, 3, 3, 3, 3}
        },
        {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 0, 2, 0, 0, 0, 2, 2, 2, 2, 1, 2},
            {0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 2, 2},
            {0, 0, 0, 2, 2, 2, 2, 1, 2, 0, 0, 0, 2, 1, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0}
        },
        {
            {0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 1, 3, 0, 0, 1, 0, 0, 0, 3, 3, 3, 2, 1, 2},
            {0, 0, 0, 3, 3, 3, 2, 1, 2, 0, 0, 0, 0, 1, 3, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 2},
            {0, 0, 0, 3, 3, 3, 2, 1, 2, 0, 0, 0, 3, 1, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0},
            {0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 3, 1, 0, 1, 0, 0, 0, 0, 0, 3, 3, 3, 2, 1, 2}
        }
    },

    // Direção S
    {
        {
            {2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 1, 2, 2, 1, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0},
            {2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 1, 2, 2, 1, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0},
            {2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 1, 2, 2, 1, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0},
            {2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 1, 2, 2, 1, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0}
        },
        {
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0},
            {3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 1, 3, 1, 1, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0},
            {3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 1, 3, 3, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3}
        },
        {
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 1, 1, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0},
            {3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 1, 3, 1, 1, 3, 3, 0, 0, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 1, 3, 3, 1, 1, 0, 0, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3}
        },
        {
            {3, 3, 3, 3, 3, 1, 0, 0, 1, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0},
            {3, 3, 3, 1, 3, 3, 1, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0},
            {3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 3, 3, 1, 3, 3, 1, 0, 0},
            {3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 3, 3, 3, 3, 1, 0, 0, 1}
        },
        {
            {2, 1, 2, 2, 2, 2, 0, 0, 0, 2, 0, 2, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {2, 2, 0, 2, 2, 0, 0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 1, 2, 0, 0, 0, 2, 1, 2, 2, 2, 2, 0, 0, 0},
            {0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0}
        },
        {
            {2, 1, 2, 3, 3, 3, 0, 0, 0, 0, 0, 1, 0, 1, 3, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0},
            {2, 1, 2, 3, 3, 3, 0, 0, 0, 1, 0, 0, 3, 1, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0},
            {2, 0, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 3, 1, 0, 0, 0, 0, 2, 1, 2, 3, 3, 3, 0, 0, 0},
            {0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 1, 0, 1, 3, 0, 0, 0, 2, 1, 2, 3, 3, 3, 0, 0, 0}
        }
    },
        
    // Direção E
    {
        {
            {2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0},
            {2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0},
            {2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0},
            {2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0}
        },
        {
            {3, 3, 3, 3, 3, 0, 3, 3, 0, 3, 1, 3, 1, 1, 0, 3, 3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 0},
            {3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 1, 1, 0, 3, 3, 0, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {3, 3, 0, 3, 3, 0, 3, 3, 3, 3, 3, 0, 1, 1, 0, 3, 1, 3, 3, 3, 0, 3, 3, 0, 3, 3, 3},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0}
        },
        {
            {3, 3, 3, 3, 3, 0, 3, 3, 0, 3, 1, 3, 1, 1, 0, 3, 3, 0, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {3, 3, 0, 3, 3, 0, 3, 3, 3, 3, 3, 0, 1, 1, 0, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0, 3, 1, 3, 3, 3, 0, 3, 3, 0, 3, 3, 3},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 1, 1, 0, 3, 3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 0}
        },
        {
            {3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 1, 1, 0, 3, 3, 0, 3, 1, 1, 3, 3, 0, 3, 3, 0},
            {3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 1, 1, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 1, 1},
            {3, 3, 0, 3, 3, 0, 3, 1, 1, 3, 3, 0, 1, 1, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0},
            {3, 1, 1, 3, 3, 0, 3, 3, 0, 3, 3, 0, 1, 1, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0}
        },
        {
            {2, 2, 0, 2, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 1, 0, 2, 2, 0, 2, 2, 0, 1, 2, 0, 2, 2, 0},
            {0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0},
            {2, 2, 0, 1, 2, 0, 2, 2, 0, 2, 2, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
        },
        {
            {2, 3, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 1, 0, 0, 0, 0, 2, 3, 0, 1, 3, 0, 2, 3, 0},
            {0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 1, 0, 1, 3, 0, 2, 3, 0, 1, 3, 0, 2, 3, 0},
            {2, 3, 0, 1, 3, 0, 2, 3, 0, 0, 0, 0, 0, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0},
            {2, 3, 0, 1, 3, 0, 2, 3, 0, 1, 3, 0, 0, 1, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0}
        }
    },

    // Direção W
    {
        {
            {0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2},
            {0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2},
            {0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2},
            {0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2}
        },
        {
            {3, 3, 3, 0, 3, 3, 0, 3, 3, 3, 1, 3, 0, 1, 1, 0, 3, 3, 3, 3, 3, 0, 3, 3, 0, 3, 3},
            {0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 1, 1, 0, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {0, 3, 3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 0, 1, 1, 3, 1, 3, 0, 3, 3, 0, 3, 3, 3, 3, 3},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 0, 3, 3, 0, 1, 1, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3}
        },
        {
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 0, 1, 1, 0, 3, 3, 3, 3, 3, 0, 3, 3, 0, 3, 3},
            {3, 3, 3, 0, 3, 3, 0, 3, 3, 3, 1, 3, 0, 1, 1, 0, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {0, 3, 3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 0, 1, 1, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3},
            {3, 3, 3, 3, 1, 3, 3, 3, 3, 0, 3, 3, 0, 1, 1, 3, 1, 3, 0, 3, 3, 0, 3, 3, 3, 3, 3}
        },
        {
            {1, 1, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 1, 1, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3},
            {0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 1, 1, 0, 3, 3, 1, 1, 3, 0, 3, 3, 0, 3, 3},
            {0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 1, 1, 0, 3, 3, 0, 3, 3, 0, 3, 3, 1, 1, 3},
            {0, 3, 3, 0, 3, 3, 1, 1, 3, 0, 3, 3, 0, 1, 1, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3}
        },
        {
            {0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 1, 0, 0, 2, 2, 0, 2, 2, 0, 2, 1, 0, 2, 2},
            {0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 2, 2, 0, 2, 2},
            {0, 2, 2, 0, 2, 1, 0, 2, 2, 0, 2, 2, 0, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}
        },
        {
            {0, 3, 2, 0, 3, 1, 0, 3, 2, 0, 3, 1, 0, 1, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0},
            {0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 1, 0, 0, 0, 0, 0, 3, 2, 0, 3, 1, 0, 3, 2},
            {0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 1, 0, 0, 3, 1, 0, 3, 2, 0, 3, 1, 0, 3, 2},
            {0, 3, 2, 0, 3, 1, 0, 3, 2, 0, 0, 0, 0, 1, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 3, 2}
        }
    }
};