# cl_tagger
在本地运行https://huggingface.co/cella110n/cl_tagger

修改自https://huggingface.co/TekeshiX/cl_tagger

批量改为预测后直接在文件夹生成.txt文件，主要用于打标

增加了标签开关

app.py应该没问题，其他的ds写的，有问题我也不会解决

（不会在WD标签器增加模型，被逼急了自己瞎搞的）

运行run.bat出现

* Running on local URL:  http://0.0.0.0:7870
* To create a public link, set `share=True` in `launch()`.

就可以使用了，端口号之类的可以在app.py里自己改

如果出现

* 模型 cl_tagger_1_01 已加载，但未检测到 GPU，使用 CPU 进行推理。

你需要在虚拟环境中安装onnxruntime-gpu，并保证其与你的CUDA和cudnn版本相匹配

可以参考https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
