# encoding utf-8

import torch
import torch.nn as nn
import cupy
import re

raw_kernel_code = '''
    extern "C" __global__ 
    
    void UpdateKernelRenderOutput(
        const int nElement,
        const float* image,          // original image
        const float* refocus,        // signed refocus map
        int* refocusDilate,          // signed refocus map after dilating
        float* bokehCum,             // cumulative bokeh image
        float* weightCum             // cumulative weight map
    )
    {
        for (int intIdx = (blockIdx.x * blockDim.x) + threadIdx.x; intIdx < nElement; intIdx += blockDim.x * gridDim.x) 
        {
            const int n = (intIdx / SIZE_3(weightCum) / SIZE_2(weightCum) / SIZE_1(weightCum)) % SIZE_0(weightCum);
         // const int c = (intIdx / SIZE_3(weightCum) / SIZE_2(weightCum))                     % SIZE_1(weightCum);
            const int y = (intIdx / SIZE_3(weightCum))                                         % SIZE_2(weightCum);
            const int x = (intIdx)                                                             % SIZE_3(weightCum);

            float refocusValue = VALUE_4(refocus, n, 0, y, x);
            float radius = fabsf(refocusValue);

            for (int deltaY = -(int)(radius)-1; deltaY <= (int)(radius)+1; ++deltaY) 
            {
                for (int deltaX = -(int)(radius)-1; deltaX <= (int)(radius)+1; ++deltaX) 
                {

                    int neighborY = y + deltaY;
                    int neighborX = x + deltaX;

                    if ((neighborY >= 0) && (neighborY < SIZE_2(bokehCum)) && 
                        (neighborX >= 0) && (neighborX < SIZE_3(bokehCum)))
                    {
                        float dist = sqrtf((float)(deltaY)*(float)(deltaY) + (float)(deltaX)*(float)(deltaX));
                        float weightValue = (0.5 + 0.5 * tanhf(4 * (radius - dist))) / (radius * radius + 0.2);
                        
                        if (radius >= dist) 
                        {
                            atomicMax(&refocusDilate[OFFSET_4(refocusDilate, n, 0, neighborY, neighborX)], 
                                      int(refocusValue));
                        }
                        
                        atomicAdd(&weightCum[OFFSET_4(weightCum, n, 0, neighborY, neighborX)], 
                                  weightValue);
                        atomicAdd(&bokehCum[OFFSET_4(bokehCum, n, 0, neighborY, neighborX)], 
                                  weightValue * VALUE_4(image, n, 0, y, x));
                        atomicAdd(&bokehCum[OFFSET_4(bokehCum, n, 1, neighborY, neighborX)], 
                                  weightValue * VALUE_4(image, n, 1, y, x));
                        atomicAdd(&bokehCum[OFFSET_4(bokehCum, n, 2, neighborY, neighborX)], 
                                  weightValue * VALUE_4(image, n, 2, y, x));
                    }
                }
            }
        }
    }
'''


def modify_cupy_kernel(func_name, obj_values):
    kernel_code = globals()[func_name]

    while True:
        matched_obj = re.search('(SIZE_)([0-4])(\\()([^)]*)(\\))', kernel_code)

        if matched_obj is None:
            break
        # if end

        int_arg = int(matched_obj.group(2))
        str_tensor = matched_obj.group(4)
        int_sizes = obj_values[str_tensor].size()

        kernel_code = kernel_code.replace(matched_obj.group(), str(int_sizes[int_arg]))
    # while end

    while True:
        matched_obj = re.search('(OFFSET_)([0-4])(\\()([^)]+)(\\))', kernel_code)

        if matched_obj is None:
            break
        # if end

        str_tensor, str_idx = set_tensor_and_idx(matched_obj, obj_values)

        kernel_code = kernel_code.replace(matched_obj.group(0), '(' + str.join('+', str_idx) + ')')
    # while end

    while True:
        matched_obj = re.search('(VALUE_)([0-4])(\\()([^)]+)(\\))', kernel_code)

        if matched_obj is None:
            break
        # if end

        str_tensor, str_idx = set_tensor_and_idx(matched_obj, obj_values)

        kernel_code = kernel_code.replace(matched_obj.group(0), str_tensor + '[' + str.join('+', str_idx) + ']')
    # while end

    return kernel_code


def set_tensor_and_idx(matched_obj, obj_values):

    int_arg = int(matched_obj.group(2))
    str_arg = matched_obj.group(4).split(',')
    str_tensor = str_arg[0]
    int_strides = obj_values[str_tensor].stride()
    str_idx = ['((' + str_arg[arg + 1].replace('{', '(').replace('}', ')').strip() + ')*' +
               str(int_strides[arg]) + ')' for arg in range(int_arg)]

    return str_tensor, str_idx


# @cupy.memoize(bool_for_each_device=True)
@cupy.memoize(True)
def cupy_launch(func_name, kernel_code):
    # return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)  <- Deprecated
    mod = cupy.RawModule(code=kernel_code)

    return mod.get_function(func_name)


class RenderData(torch.autograd.Function):
    @staticmethod
    # def forward(self, image, refocus):  <- Not matched the signature param
    def forward(ctx, *args):
        image, refocus = args[0], args[1]

        refocus_dilate = refocus.int()
        bokeh_cum = torch.zeros_like(image)
        weight_cum = torch.zeros_like(refocus)

        data = {'image': image, 'refocus': refocus,
                'refocusDilate': refocus_dilate, 'bokehCum': bokeh_cum, 'weightCum': weight_cum}

        if refocus.is_cuda:
            num_element = weight_cum.nelement()
            cupy_launch('UpdateKernelRenderOutput',
                        modify_cupy_kernel('raw_kernel_code', data))(
                grid=tuple([int((num_element + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=(
                    num_element,
                    image.data_ptr(),
                    refocus.data_ptr(),
                    refocus_dilate.data_ptr(),
                    bokeh_cum.data_ptr(),
                    weight_cum.data_ptr()
                ))
        else:
            raise NotImplementedError('refocus is not a cuda!')
        # if end

        return refocus_dilate.float(), bokeh_cum, weight_cum


class ModuleRenderScatter(nn.Module):
    def __init__(self):
        super(ModuleRenderScatter, self).__init__()

    @staticmethod
    def forward(image, refocus):
        refocus_dilate, bokeh_cum, weight_cum = RenderData.apply(image, refocus)
        bokeh = bokeh_cum / weight_cum

        return bokeh, refocus_dilate
