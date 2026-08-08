#include "rknn_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fp16/Float16.h"

using namespace rknpu2;

namespace dw {

static void dump_tensor_attr(rknn_tensor_attr* attr) {
    char dims[128] = {0};
    for (int i = 0; i < (int)attr->n_dims; ++i) {
        int idx = strlen(dims);
        sprintf(&dims[idx], "%d%s", attr->dims[i], (i == (int)attr->n_dims - 1) ? "" : ", ");
    }
    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, fmt=%s, type=%s, "
           "qnt_type=%s, zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, dims, attr->n_elems, attr->size,
           get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

// NC1HWC2 int8 -> NCHW float (RV1106 量化 NPU 输出布局)
static void NC1HWC2_int8_to_NCHW_float(const int8_t* src, float* dst, int* dims,
                                       int channel, int h, int w, int zp, float scale) {
    int batch  = dims[0];
    int C1     = dims[1];
    int C2     = dims[4];
    int hw_src = dims[2] * dims[3];
    int hw_dst = h * w;
    for (int i = 0; i < batch; i++) {
        const int8_t* src_b = src + i * C1 * hw_src * C2;
        float*        dst_b = dst + i * channel * hw_dst;
        for (int c = 0; c < channel; ++c) {
            int           plane  = c / C2;
            const int8_t* src_c  = plane * hw_src * C2 + src_b;
            int           offset = c % C2;
            for (int cur_h = 0; cur_h < h; ++cur_h)
                for (int cur_w = 0; cur_w < w; ++cur_w) {
                    int cur_hw = cur_h * w + cur_w;
                    dst_b[c * hw_dst + cur_h * w + cur_w] =
                        (src_c[C2 * cur_hw + offset] - zp) * scale;
                }
        }
    }
}

int rknn_model_init(RknnModel* m, const char* path) {
    int ret;
    printf("[RKNN] Loading model: %s\n", path);
    ret = rknn_init(&m->ctx, (void*)path, 0, 0, NULL);
    if (ret < 0) { printf("[RKNN] rknn_init fail! ret=%d\n", ret); return -1; }

    ret = rknn_query(m->ctx, RKNN_QUERY_IN_OUT_NUM, &m->io_num, sizeof(m->io_num));
    if (ret != RKNN_SUCC) return -1;
    printf("  inputs: %d, outputs: %d\n", m->io_num.n_input, m->io_num.n_output);

    m->input_attrs = (rknn_tensor_attr*)calloc(m->io_num.n_input, sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < m->io_num.n_input; i++) {
        m->input_attrs[i].index = i;
        rknn_query(m->ctx, RKNN_QUERY_INPUT_ATTR, &m->input_attrs[i], sizeof(rknn_tensor_attr));
        dump_tensor_attr(&m->input_attrs[i]);
    }

    m->output_attrs = (rknn_tensor_attr*)calloc(m->io_num.n_output, sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < m->io_num.n_output; i++) {
        m->output_attrs[i].index = i;
        rknn_query(m->ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &m->output_attrs[i], sizeof(rknn_tensor_attr));
    }

    m->orig_out_attrs = (rknn_tensor_attr*)calloc(m->io_num.n_output, sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < m->io_num.n_output; i++) {
        m->orig_out_attrs[i].index = i;
        rknn_query(m->ctx, RKNN_QUERY_OUTPUT_ATTR, &m->orig_out_attrs[i], sizeof(rknn_tensor_attr));
    }

    m->input_mems = (rknn_tensor_mem**)calloc(m->io_num.n_input, sizeof(rknn_tensor_mem*));
    m->input_attrs[0].type = RKNN_TENSOR_UINT8;
    m->input_attrs[0].fmt  = RKNN_TENSOR_NHWC;
    m->input_mems[0] = rknn_create_mem(m->ctx, m->input_attrs[0].size_with_stride);

    m->output_mems = (rknn_tensor_mem**)calloc(m->io_num.n_output, sizeof(rknn_tensor_mem*));
    for (uint32_t i = 0; i < m->io_num.n_output; i++)
        m->output_mems[i] = rknn_create_mem(m->ctx, m->output_attrs[i].size_with_stride);

    ret = rknn_set_io_mem(m->ctx, m->input_mems[0], &m->input_attrs[0]);
    if (ret < 0) return -1;
    for (uint32_t i = 0; i < m->io_num.n_output; i++) {
        ret = rknn_set_io_mem(m->ctx, m->output_mems[i], &m->output_attrs[i]);
        if (ret < 0) return -1;
    }
    return 0;
}

void rknn_model_set_input(RknnModel* m, const unsigned char* data,
                          int width, int height, int channels) {
    int stride = m->input_attrs[0].w_stride;
    if (width == stride || stride == 0) {
        memcpy(m->input_mems[0]->virt_addr, data, (size_t)width * height * channels);
    } else {
        const uint8_t* src     = data;
        uint8_t*       dst     = (uint8_t*)m->input_mems[0]->virt_addr;
        int            src_row = width * channels;
        int            dst_row = stride * channels;
        for (int h = 0; h < height; ++h) {
            memcpy(dst, src, src_row);
            src += src_row;
            dst += dst_row;
        }
    }
}

float* rknn_model_get_output_float(RknnModel* m, int idx) {
    int    n_elems = m->orig_out_attrs[idx].n_elems;
    float* out     = (float*)malloc(n_elems * sizeof(float));

    if (m->output_attrs[idx].fmt == RKNN_TENSOR_NC1HWC2 &&
        m->output_attrs[idx].type == RKNN_TENSOR_INT8) {
        int channel = m->orig_out_attrs[idx].dims[1];
        int h = m->orig_out_attrs[idx].n_dims > 2 ? m->orig_out_attrs[idx].dims[2] : 1;
        int w = m->orig_out_attrs[idx].n_dims > 3 ? m->orig_out_attrs[idx].dims[3] : 1;
        NC1HWC2_int8_to_NCHW_float((int8_t*)m->output_mems[idx]->virt_addr, out,
                                   (int*)m->output_attrs[idx].dims, channel, h, w,
                                   m->output_attrs[idx].zp, m->output_attrs[idx].scale);
    } else if (m->output_attrs[idx].type == RKNN_TENSOR_INT8) {
        int8_t* src   = (int8_t*)m->output_mems[idx]->virt_addr;
        int     zp    = m->output_attrs[idx].zp;
        float   scale = m->output_attrs[idx].scale;
        for (int i = 0; i < n_elems; i++)
            out[i] = (src[i] - zp) * scale;
    } else if (m->output_attrs[idx].type == RKNN_TENSOR_UINT8) {
        uint8_t* src  = (uint8_t*)m->output_mems[idx]->virt_addr;
        int      zp   = m->output_attrs[idx].zp;
        float    scale = m->output_attrs[idx].scale;
        for (int i = 0; i < n_elems; i++)
            out[i] = (src[i] - zp) * scale;
    } else if (m->output_attrs[idx].type == RKNN_TENSOR_FLOAT16) {
        const uint16_t* src = (const uint16_t*)m->output_mems[idx]->virt_addr;
        for (int i = 0; i < n_elems; i++)
            out[i] = (float)float16::fromBits(src[i]);
    } else if (m->output_attrs[idx].type == RKNN_TENSOR_FLOAT32) {
        memcpy(out, m->output_mems[idx]->virt_addr, n_elems * sizeof(float));
    } else {
        printf("[RKNN] unsupported output type %d at index %d\n",
               m->output_attrs[idx].type, idx);
        memset(out, 0, n_elems * sizeof(float));
    }
    return out;
}

void rknn_model_input_hw(RknnModel* m, int* w, int* h) {
    if (m->input_attrs[0].fmt == RKNN_TENSOR_NHWC) {
        *h = m->input_attrs[0].dims[1];
        *w = m->input_attrs[0].dims[2];
    } else { // NCHW
        *h = m->input_attrs[0].dims[2];
        *w = m->input_attrs[0].dims[3];
    }
}

void rknn_model_destroy(RknnModel* m) {
    if (m->input_mems) {
        for (uint32_t i = 0; i < m->io_num.n_input; i++)
            if (m->input_mems[i]) rknn_destroy_mem(m->ctx, m->input_mems[i]);
        free(m->input_mems);
    }
    if (m->output_mems) {
        for (uint32_t i = 0; i < m->io_num.n_output; i++)
            if (m->output_mems[i]) rknn_destroy_mem(m->ctx, m->output_mems[i]);
        free(m->output_mems);
    }
    free(m->input_attrs);
    free(m->output_attrs);
    free(m->orig_out_attrs);
    if (m->ctx) rknn_destroy(m->ctx);
    *m = RknnModel{};
}

} // namespace dw
