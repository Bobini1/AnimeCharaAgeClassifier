?	g?UIdmG@g?UIdmG@!g?UIdmG@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCg?UIdmG@??? !? @1w? ݗ?C@A&s,????I'?E'K=@rEagerKernelExecute 0*	P??nB??@2?
jIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::ParallelMapV2 Ӄ?R?"@!??0Ź?L@)Ӄ?R?"@1??0Ź?L@:Preprocessing2?
JIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2??vL/@!? ????X@)??hW!%@1??<ϰC@:Preprocessing2?
XIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip 5????#@!1.j"?(N@)|?y????1?????M@:Preprocessing2?
wIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice ;S???.??!5~?????);S???.??15~?????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle O?\?9#@!e`CyqN@)?R	??1??L??<??:Preprocessing2?
hIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::TensorSlice ???P????!<??T?y??)???P????1<??T?y??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?/?????!c4?s???)?/?????1c4?s???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?3?l??!?Q?ɑ???)??<????1?n1 n??:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImplT ?g?P/@!oB?:?X@)?Ϛi??1?b2!???:Preprocessing2F
Iterator::Model???????!5C?c???)R?=?Ne?1??w)?ސ?:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache*S?A?Q/@!b,????X@)??Z
H?_?1?i>]?Q??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??C?,?-@Q?7b?LU@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??? !? @??? !? @!??? !? @      ??!       "	w? ݗ?C@w? ݗ?C@!w? ݗ?C@*      ??!       2	&s,????&s,????!&s,????:	'?E'K=@'?E'K=@!'?E'K=@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??C?,?-@y?7b?LU@?"-
IteratorGetNext/_1_Send??'n?Կ?!??'n?Կ?"g
;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!?ʓD?:??0"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??ԝ?3??!???=???0"[
:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad2Ͱ?f???!?L?2???"F
(gradient_tape/sequential/conv2d/ReluGradReluGrad??????!a ??Zb??":
sequential/conv2d_1/Relu_FusedConv2D?f????!???-z??"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?
??=??!Jg??	???0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltertw2?Φ?!????Z??0"]
<gradient_tape/sequential/max_pooling2d_1/MaxPool/MaxPoolGradMaxPoolGrad0l?O*??!M??Y???"H
*gradient_tape/sequential/conv2d_1/ReluGradReluGrad?{5>???!)????o??Q      Y@YS???P@aY????A@q[?!61?@y??M?)u?"?

both?Your program is POTENTIALLY input-bound because 4.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B 