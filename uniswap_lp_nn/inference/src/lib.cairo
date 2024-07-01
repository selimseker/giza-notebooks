use orion::operators::tensor::{Tensor, TensorTrait};
use orion::operators::tensor::{U32Tensor, I32Tensor, I8Tensor, FP8x23Tensor, FP16x16Tensor, FP32x32Tensor, BoolTensor};
use orion::numbers::{FP8x23, FP16x16, FP32x32};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};
use orion::operators::nn::{NNTrait, FP16x16NN};
use orion::operators::ml;

use node_0_weight::get_node_0_weight;
use node_0_bias::get_node_0_bias;
use node_2_weight::get_node_2_weight;
use node_2_bias::get_node_2_bias;
use node_4_weight::get_node_4_weight;
use node_4_bias::get_node_4_bias;


fn main(node_input: Tensor<FP16x16>) -> Tensor<FP16x16> {
let node__0_gemm_output_0 = NNTrait::gemm(node_input, get_node_0_weight(), Option::Some(get_node_0_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
let node__1_relu_output_0 = NNTrait::relu(@node__0_gemm_output_0);
let node__2_gemm_output_0 = NNTrait::gemm(node__1_relu_output_0, get_node_2_weight(), Option::Some(get_node_2_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);
let node__3_relu_output_0 = NNTrait::relu(@node__2_gemm_output_0);
let node_output = NNTrait::gemm(node__3_relu_output_0, get_node_4_weight(), Option::Some(get_node_4_bias()), Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(FP16x16 { mag: 65536, sign: false }), false, true);

        node_output
    }