
?
1maxwell_scudnn_128x32_stridedB_splitK_large_nn_v0W?P*?2?8???@???H???Xb;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterhu  ?A
?
g_Z17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEviiiPKT_iPS0_S2_18kernel_grad_paramsyifiiiiG?*2	?8???@???H???Xb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhu  /B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*?288???@???H???b(gradient_tape/sequential/conv2d/ReluGradhu  ?B
?
?_ZN5cudnn3ops21pooling_bw_kernel_maxIffNS_15maxpooling_funcIfL21cudnnNanPropagation_t0EEELb0EEEv17cudnnTensorStructPKT_S5_S8_S5_S8_S5_PS6_18cudnnPoolingStructT0_SB_iNS_15reduced_divisorESC_ ?*?2 8ဈ@ဈHဈb:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradhup?B
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2@8ᘇ@ᘇHᘇbsequential/conv2d_1/Reluhu  ?A
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2 8??m@??mH??mXb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhu  ?A
?
5maxwell_scudnn_128x128_stridedB_splitK_interior_nn_v0|??*?2<8??k@??kH??kXb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ?A
?
?_ZN5cudnn3ops21pooling_bw_kernel_maxIffNS_15maxpooling_funcIfL21cudnnNanPropagation_t0EEELb0EEEv17cudnnTensorStructPKT_S5_S8_S5_S8_S5_PS6_18cudnnPoolingStructT0_SB_iNS_15reduced_divisorESC_ ?*?2 8??D@??DH??Db<gradient_tape/sequential/max_pooling2d_1/MaxPool/MaxPoolGradhu ??B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*?288??C@??CH??Cb*gradient_tape/sequential/conv2d_1/ReluGradhu  ?B
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2?8??A@??AH??Absequential/conv2d_2/Reluhu  ?A
?
?_ZN5cudnn3ops20pooling_fw_4d_kernelIffNS_15maxpooling_funcIfL21cudnnNanPropagation_t0EEEL18cudnnPoolingMode_t0ELb0EEEv17cudnnTensorStructPKT_S6_PS7_18cudnnPoolingStructT0_SC_iNS_15reduced_divisorESD_( ?*?2 8??@@??@H??@b sequential/max_pooling2d/MaxPoolhuԔB
d
&maxwell_scudnn_128x32_relu_small_nn_v1??R*@2?~8??;@??;H??;bsequential/conv2d/Reluhu  ?A
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2@8??:@??:H??:Xb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhu  ?A
?
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0???*?2?8??6@??6H??6Xb<gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropInputhu  ?A
?
?_ZN3cub27DeviceSegmentedReduceKernelINS_18DeviceReducePolicyIffiNS_3SumEE9Policy600EPKfPfNS_22TransformInputIteratorIiN10tensorflow7functor9RowOffsetENS_21CountingInputIteratorIixEExEEiS2_fEEvT0_T1_T2_SH_iT4_T5_0*?2?8??/@??/H??/b3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGradhu  ?B
?
4_ZN5cudnn3ops24scalePackedTensor_kernelIffEEvxPT_T0_*?2??8??(@??(H??(b:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28??!@??!H??!b$Adam/Adam/update_8/ResourceApplyAdamhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*?288??@??H??b*gradient_tape/sequential/conv2d_2/ReluGradhu  ?B
?
?_ZN5cudnn3ops20pooling_fw_4d_kernelIffNS_15maxpooling_funcIfL21cudnnNanPropagation_t0EEEL18cudnnPoolingMode_t0ELb0EEEv17cudnnTensorStructPKT_S6_PS7_18cudnnPoolingStructT0_SC_iNS_15reduced_divisorESD_( ?*?2 8??@??H??b"sequential/max_pooling2d_1/MaxPoolhu?O?B
?
?_ZN5cudnn45pooling_bw_kernel_max_nchw_fully_packed_largeIffLi2EL21cudnnNanPropagation_t0EEEv17cudnnTensorStructPKT_S2_S5_S2_S5_S2_PS3_18cudnnPoolingStructT0_S8_NS_15reduced_divisorES9_i" ??*?2 ?8??@??H??b<gradient_tape/sequential/max_pooling2d_2/MaxPool/MaxPoolGradhu  B
V
maxwell_sgemm_128x64_nnx?b*?2$8??@??H??bsequential/conv2d_3/Reluhu  ?A
?
?_ZN3cub27DeviceSegmentedReduceKernelINS_18DeviceReducePolicyIffiNS_3SumEE9Policy600EPKfPfNS_22TransformInputIteratorIiN10tensorflow7functor9RowOffsetENS_21CountingInputIteratorIixEExEEiS2_fEEvT0_T1_T2_SH_iT4_T5_0*?2?8??@??H??b5gradient_tape/sequential/conv2d_1/BiasAdd/BiasAddGradhu  ?B
}
maxwell_sgemm_128x64_ntx?`*?2$8??@??H??Xb=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterhu  ?A
?
4_ZN5cudnn3ops24scalePackedTensor_kernelIffEEvxPT_T0_*?2??8??@??H??b<gradient_tape/sequential/max_pooling2d_1/MaxPool/MaxPoolGradhu  ?B
?
Z_ZN5cudnn17winograd_nonfused21winogradWgradDelta4x4IffEEvNS0_19WinogradDeltaParamsIT_T0_EE@??*?2@ 8??@??H??Xb=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterhu  HB
?
^_ZN5cudnn17winograd_nonfused24winogradForwardOutput4x4IffEEvNS0_20WinogradOutputParamsIT_T0_EE@??*?2?8??@??H??bsequential/conv2d_3/Reluhu  HB
?
?_ZN5cudnn3ops20pooling_fw_4d_kernelIffNS_15maxpooling_funcIfL21cudnnNanPropagation_t0EEEL18cudnnPoolingMode_t0ELb0EEEv17cudnnTensorStructPKT_S6_PS7_18cudnnPoolingStructT0_SC_iNS_15reduced_divisorESD_( ?*?2  8??@??H??b"sequential/max_pooling2d_2/MaxPoolhu?k?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*?288??@??H??bAdam/gradients/AddN_5hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*?288ɶ@ɶHɶb*gradient_tape/sequential/conv2d_3/ReluGradhu  ?B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b,gradient_tape/dense/kernel/Regularizer/Mul_1hu  ?B
?
W_Z18sgemm_largek_lds64ILb0ELb0ELi5ELi5ELi4ELi4ELi4ELi32EEvPfPKfS2_iiiiiiS2_S2_ffiiPiS3_ ?!*218??@??H??Xbsequential/dense/MatMulhu  ?B
?
v_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi1024ELi1024ELi2ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ ?`*?2? 8??@??H??bagradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
f
sgemm_32x32x32_TN_vec???*?2?8??@??H??b'gradient_tape/sequential/dense/MatMul_1hu  ?A
?
?_ZN5cudnn45pooling_bw_kernel_max_nchw_fully_packed_smallIffLi2EL21cudnnNanPropagation_t0EEEv17cudnnTensorStructPKT_S2_S5_S2_S5_S2_PS3_18cudnnPoolingStructT0_S8_NS_15reduced_divisorES9_ ?1*?2 ?8??
@??
H??
b<gradient_tape/sequential/max_pooling2d_3/MaxPool/MaxPoolGradhu  ?B
?
?_ZN3cub27DeviceSegmentedReduceKernelINS_18DeviceReducePolicyIffiNS_3SumEE9Policy600EPKfPfNS_22TransformInputIteratorIiN10tensorflow7functor9RowOffsetENS_21CountingInputIteratorIixEExEEiS2_fEEvT0_T1_T2_SH_iT4_T5_0*?2? 8??
@??
H??
b5gradient_tape/sequential/conv2d_2/BiasAdd/BiasAddGradhu  ?B
?
Z_ZN5cudnn17winograd_nonfused22winogradForwardData4x4IffEEvNS0_18WinogradDataParamsIT_T0_EE@??*?2?8??	@??	H??	bsequential/conv2d_3/Reluhu  HB
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bsequential/rescaling/mulhu  ?B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??b*gradient_tape/dense/kernel/Regularizer/Mulhu  ?B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bdense/kernel/Regularizer/Squarehu  ?B
_
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?2?8??@??H??bsequential/rescaling/addhu  ?B
?
X_ZN5cudnn17winograd_nonfused20winogradWgradData4x4IffEEvNS0_18WinogradDataParamsIT_T0_EE@??*?2  8??@??H??Xb=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterhu  B
f
sgemm_32x32x32_NT_vec???*?2?8??@??H??Xb%gradient_tape/sequential/dense/MatMulhu  ?A
?
?_ZN5cudnn3ops20pooling_fw_4d_kernelIffNS_15maxpooling_funcIfL21cudnnNanPropagation_t0EEEL18cudnnPoolingMode_t0ELb0EEEv17cudnnTensorStructPKT_S6_PS7_18cudnnPoolingStructT0_SC_iNS_15reduced_divisorESD_( ?*?2@ 8??@??H??b"sequential/max_pooling2d_3/MaxPoolhu?eB
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorBroadcastingOpIKNS_5arrayIiLy2EEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*?288??@??H??b+gradient_tape/dense/kernel/Regularizer/Tilehu  ?B
?
k_ZN10tensorflow7functor15RowReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*?2?8??@??H??b5gradient_tape/sequential/conv2d_3/BiasAdd/BiasAddGradhu  ?B
?
?_ZN3cub18DeviceReduceKernelINS_18DeviceReducePolicyIffiN10tensorflow7functor3SumIfEEE9Policy600EPfS8_iS5_EEvT0_T1_T2_NS_13GridEvenShareISB_EET3_0*?2?8??@??H??bdense/kernel/Regularizer/Sumhu  ?B
?
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ ?!*?2?8??@??H??bJsequential/max_pooling2d_3/MaxPool-0-2-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
a_ZN5cudnn17winograd_nonfused22winogradWgradOutput4x4IffEEvNS0_25WinogradWgradOutputParamsIT_T0_EE@?H* 2 8??@??H??Xb=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterhu  HB
?
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ ?!*?2?8??@??H??bbgradient_tape/sequential/max_pooling2d_3/MaxPool/MaxPoolGrad-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28??@??H??b$Adam/Adam/update_6/ResourceApplyAdamhu  ?B
?
^_ZN5cudnn17winograd_nonfused24winogradForwardFilter4x4IffEEvNS0_20WinogradFilterParamsIT_T0_EE ?H* 2 8??@??H??bsequential/conv2d_3/Reluhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28?p@?pH?pXb<gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropInputhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28?g@?gH?gbsequential/conv2d_3/Reluhu  ?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2H8?^@?^H?^b/gradient_tape/conv2d_3/kernel/Regularizer/Mul_1hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi4ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorBroadcastingOpIKNS_5arrayIiLy4EEEKNS4_INS5_IKfLi4ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_0*?288?X@?XH?Xb.gradient_tape/conv2d_3/kernel/Regularizer/Tilehu  HB
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28?X@?XH?XXb=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*?288?X@?XH?XbAdam/gradients/AddN_4hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi4ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorBroadcastingOpIKNS_5arrayIiLy4EEEKNS4_INS5_IKfLi4ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_0*?288?P@?PH?Pb.gradient_tape/conv2d_2/kernel/Regularizer/Tilehu  HB
?
c_ZN5cudnn8winograd27generateWinogradTilesKernelILi1EffEEvNS0_27GenerateWinogradTilesParamsIT0_T1_EE(?D* 2@8?P@?PH?PXb<gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropInputhu ??B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28?O@?OH?Ob$Adam/Adam/update_4/ResourceApplyAdamhu  ?B
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*?2H8?@@?@H?@b"conv2d_3/kernel/Regularizer/Squarehu  ?B
?
c_ZN5cudnn8winograd27generateWinogradTilesKernelILi1EffEEvNS0_27GenerateWinogradTilesParamsIT0_T1_EE(?D* 28?@@?@H?@bsequential/conv2d_1/Reluhu ??B
?
c_ZN5cudnn8winograd27generateWinogradTilesKernelILi1EffEEvNS0_27GenerateWinogradTilesParamsIT0_T1_EE(?D* 28?8@?8H?8Xb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhu ??B
?
c_ZN5cudnn8winograd27generateWinogradTilesKernelILi1EffEEvNS0_27GenerateWinogradTilesParamsIT0_T1_EE(?D* 2 8?8@?8H?8Xb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhu ??B
?
{_Z13gemv2N_kernelIiiffffLi128ELi32ELi4ELi4ELi1ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_8?*?2 8?/@?/H?/b)gradient_tape/sequential/dense_1/MatMul_1hu  aB
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2H8?(@?(H?(b-gradient_tape/conv2d_3/kernel/Regularizer/Mulhu  ?B
?
?_Z17gemv2T_kernel_valIiiffffLi128ELi16ELi2ELi2ELb0ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_T4_S7_$?*?28?(@?(H?(Xbsequential/dense_1/MatMulhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28?(@?(H?(b$Adam/Adam/update_7/ResourceApplyAdamhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28?(@?(H?(bsequential/conv2d_2/Reluhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28?(@?(H?(Xb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28?(@?(H?(Xb<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputhu  ?B
?
?_ZN3cub18DeviceReduceKernelINS_18DeviceReducePolicyIffiN10tensorflow7functor3SumIfEEE9Policy600EPfS8_iS5_EEvT0_T1_T2_NS_13GridEvenShareISB_EET3_0*?2H8?(@?(H?(bconv2d_3/kernel/Regularizer/Sumhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28?'@?'H?'b$Adam/Adam/update_5/ResourceApplyAdamhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28?'@?'H?'b$Adam/Adam/update_2/ResourceApplyAdamhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28?'@?'H?'b%Adam/Adam/update_10/ResourceApplyAdamhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*?288? @? H? bAdam/gradients/AddN_3hu  ?B
?
K_ZN10cask_cudnn20computeOffsetsKernelILb0ELb0EEEvNS_20ComputeOffsetsParamsE*?2?8? @? H? bsequential/conv2d/Reluhu  ?B
?
?_ZN10tensorflow73_GLOBAL__N__43_dynamic_stitch_op_gpu_cu_compute_80_cpp1_ii_46ae2929_1144019DynamicStitchKernelIiEEviiNS_20GpuDeviceArrayStructIiLi8EEENS2_IPKT_Li8EEEPS4_*?28? @? H? b/gradient_tape/binary_crossentropy/DynamicStitchhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28? @? H? b$Adam/Adam/update_3/ResourceApplyAdamhu  ?B
?
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE0*?28? @? H? bconv2d/kernel/Regularizer/Sumhu  ?B
?
c_ZN5cudnn8winograd27generateWinogradTilesKernelILi1EffEEvNS0_27GenerateWinogradTilesParamsIT0_T1_EE(?D* 28? @? H? bsequential/conv2d_2/Reluhu ??B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b/gradient_tape/conv2d_2/kernel/Regularizer/Mul_1hu  ?B
?
Q_ZN10cask_cudnn31computeWgradSplitKOffsetsKernelENS_26ComputeSplitKOffsetsParamsE*?2?8?@?H?Xb;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterhu  ?B
?
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE?!*  28?@?H?b2gradient_tape/sequential/dense/BiasAdd/BiasAddGradhu  ?B
?
Q_ZN10cask_cudnn31computeWgradSplitKOffsetsKernelENS_26ComputeSplitKOffsetsParamsE*?2<8?@?H?Xb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ?B
?
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE0*?2 8?@?H?bconv2d_2/kernel/Regularizer/Sumhu  ?B
j
"Log1p_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b'binary_crossentropy/logistic_loss/Log1phu  ?B
I
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bAdam/Powhu  ?B
d
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b conv2d/kernel/Regularizer/Squarehu  ?B
?
U_Z11scal_kernelIffLi1ELb1ELi6ELi5ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*?28?@?H?Xbsequential/dense/MatMulhu  ?B
?
?_Z13gemmk1_kernelIfLi256ELi5ELb1ELb0ELb0ELb0E30cublasGemvTensorStridedBatchedIKfES2_S0_IfEfEv18cublasGemmk1ParamsIT_T6_T7_T8_T9_N8biasTypeINS8_10value_typeES9_E4typeEE?*?28?@?H?Xb'gradient_tape/sequential/dense_1/MatMulhu  ?B
?
L_ZN10cask_cudnn26computeWgradBOffsetsKernelENS_26ComputeWgradBOffsetsParamsE*?28?@?H?Xb;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterhu  ?B
?
L_ZN10cask_cudnn26computeWgradBOffsetsKernelENS_26ComputeWgradBOffsetsParamsE*?28?@?H?Xb=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28?@?H?b"Adam/Adam/update/ResourceApplyAdamhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28?@?H?b$Adam/Adam/update_1/ResourceApplyAdamhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28?@?H?b$Adam/Adam/update_9/ResourceApplyAdamhu  ?B
?
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE?!*  28?@?H?b3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGradhu  ?B
?
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE?!*  28?@?H?b5gradient_tape/sequential/conv2d_1/BiasAdd/BiasAddGradhu  ?B
?
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE?!*  28?@?H?b5gradient_tape/sequential/conv2d_2/BiasAdd/BiasAddGradhu  ?B
?
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE?!*  28?@?H?b5gradient_tape/sequential/conv2d_3/BiasAdd/BiasAddGradhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28?@?H?bsequential/conv2d/Reluhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28?@?H?Xb;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28?@?H?bsequential/conv2d_1/Reluhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28?@?H?Xb=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterhu  ?B
?
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*?28?@?H?Xb<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputhu  ?B
?
z_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE?!* 28?@?H?b4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradhu ?	B
?
?_ZN3cub28DeviceReduceSingleTileKernelINS_18DeviceReducePolicyIffiN10tensorflow7functor3SumIfEEE9Policy600EPfS8_iS5_fEEvT0_T1_T2_T3_T4_0*?28?@?H?bdense/kernel/Regularizer/Sumhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?b'binary_crossentropy/weighted_loss/valuehu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?b3binary_crossentropy/weighted_loss/num_elements/Casthu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_max_opIKfSB_Li1EEEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bsequential/dense/Reluhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bAdam/gradients/AddN_2hu  ?B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b/gradient_tape/conv2d_1/kernel/Regularizer/Mul_1hu  ?B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bconv2d/kernel/Regularizer/mulhu  ?B
j
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*?28?@?H?bsequential/dense/BiasAddhu  ?B
?
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE0*?2 8?@?H?bconv2d_1/kernel/Regularizer/Sumhu  ?B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bMulhu  ?B
?
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*?28?@?H?b%Adam/Adam/update_11/ResourceApplyAdamhu  ?B
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b"conv2d_1/kernel/Regularizer/Squarehu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKxLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bAdam/Cast_1hu  ?B
d
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b!binary_crossentropy/logistic_losshu  ?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bconv2d_3/kernel/Regularizer/mulhu  ?B
l
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*?28?@?H?bsequential/dense_1/BiasAddhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_14TensorSelectOpIKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EESH_EEEENS_9GpuDeviceEEEiEEvT_T0_*?28?@?H?b8gradient_tape/binary_crossentropy/logistic_loss/Select_2hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opIfEEKS8_EEEENS_9GpuDeviceEEEiEEvT_T0_ *?28?@?H?b:gradient_tape/binary_crossentropy/logistic_loss/zeros_likehu  ?B
v
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b3gradient_tape/binary_crossentropy/logistic_loss/addhu  ?B
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel
*?28?@?H?bAdam/addhu  ?B
j
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b)gradient_tape/binary_crossentropy/truedivhu  ?B
G
!Equal_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?bEqualhu  ?B
f
 Exp_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b%binary_crossentropy/logistic_loss/Exphu  ?B
w
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?b.binary_crossentropy/logistic_loss/GreaterEqualhu  ?B
K
#Greater_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?bGreaterhu  ?B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*?28?@?H?b
LogicalAndhu  ?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bconv2d_1/kernel/Regularizer/mulhu  ?B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bdense/kernel/Regularizer/mulhu  ?B
f
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b%binary_crossentropy/logistic_loss/mulhu  ?B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b3gradient_tape/binary_crossentropy/logistic_loss/mulhu  ?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b-gradient_tape/conv2d/kernel/Regularizer/Mul_1hu  ?B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b-gradient_tape/conv2d_2/kernel/Regularizer/Mulhu  ?B
f
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b%binary_crossentropy/logistic_loss/Neghu  ?B
x
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b7gradient_tape/binary_crossentropy/logistic_loss/sub/Neghu  ?B
f
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b"conv2d_2/kernel/Regularizer/Squarehu  ?B
f
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b%binary_crossentropy/logistic_loss/subhu  ?B
?
n_ZN10tensorflow7functor15CleanupSegmentsIPfS2_NS0_3SumIfEEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE* 28?@?H?bconv2d_1/kernel/Regularizer/Sumhu  HB
?
n_ZN10tensorflow7functor15CleanupSegmentsIPfS2_NS0_3SumIfEEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE* 28?@?H?bconv2d_2/kernel/Regularizer/Sumhu  HB
?
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE0*?28?@?H?bSum_2hu  ?B
?
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE0*?28?@?H?b%binary_crossentropy/weighted_loss/Sumhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_14TensorSelectOpIKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EESH_EEEENS_9GpuDeviceEEEiEEvT_T0_*?28?@?H?b(binary_crossentropy/logistic_loss/Selecthu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_14TensorSelectOpIKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EESH_EEEENS_9GpuDeviceEEEiEEvT_T0_*?28?@?H?b*binary_crossentropy/logistic_loss/Select_1hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_14TensorSelectOpIKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EESH_EEEENS_9GpuDeviceEEEiEEvT_T0_*?28?@?H?b8gradient_tape/binary_crossentropy/logistic_loss/Select_3hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_17scalar_inverse_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*?28?@?H?b:gradient_tape/binary_crossentropy/logistic_loss/Reciprocalhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?b
div_no_nanhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bdiv_no_nan_1hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?b@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nanhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorBroadcastingOpIKNS_5arrayIiLy1EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*?28?@?H?b6gradient_tape/binary_crossentropy/weighted_loss/Tile_1hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKbLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bCast_3hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKbLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bCast_4hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bCasthu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bCast_5hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bbinary_crossentropy/Casthu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?b&gradient_tape/binary_crossentropy/Casthu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bAdam/gradients/AddN_1hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS9_ISC_KNS9_ISC_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EESF_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bAdam/gradients/AddNhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS9_ISC_KNS9_ISC_KNS9_ISC_KNS9_ISC_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EESF_EESF_EESF_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_ *?28?@?H?b(ArithmeticOptimizer/AddOpsRewrite_AddN_1hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bAssignAddVariableOphu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bAssignAddVariableOp_1hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bAssignAddVariableOp_2hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bAssignAddVariableOp_3hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bAdam/Adam/AssignAddVariableOphu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?bAssignAddVariableOp_4hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opIfEEKS8_EEEENS_9GpuDeviceEEEiEEvT_T0_ *?28?@?H?b<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1hu  ?B
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b
Adam/Pow_1hu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_14TensorSelectOpIKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EESH_EEEENS_9GpuDeviceEEEiEEvT_T0_*?28?@?H?b6gradient_tape/binary_crossentropy/logistic_loss/Selecthu  ?B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bconv2d_2/kernel/Regularizer/mulhu  ?B
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b7gradient_tape/binary_crossentropy/logistic_loss/mul/Mulhu  ?B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b5gradient_tape/binary_crossentropy/logistic_loss/mul_1hu  ?B
t
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b3gradient_tape/binary_crossentropy/logistic_loss/Neghu  ?B
?
?_ZN3cub28DeviceReduceSingleTileKernelINS_18DeviceReducePolicyIffiN10tensorflow7functor3SumIfEEE9Policy600EPfS8_iS5_fEEvT0_T1_T2_T3_T4_0*?28?@?H?bconv2d_3/kernel/Regularizer/Sumhu  ?B
?
?_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*?28?@?H?b'gradient_tape/sequential/dense/ReluGradhu  ?B