	g?UIdmG@g?UIdmG@!g?UIdmG@      ??!       "?
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
	??? !? @??? !? @!??? !? @      ??!       "	w? ݗ?C@w? ݗ?C@!w? ݗ?C@*      ??!       2	&s,????&s,????!&s,????:	'?E'K=@'?E'K=@!'?E'K=@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??C?,?-@y?7b?LU@