# XGBoost Inference Benchmark

There are not many resources on deploying XGBoost models, especially optimizing for inference speed. This repo benchmarks inference speed for several libraries that claim to optimize inference speed for tree algorithms. If you know any other methods or libraries, do let me know and I will try to include it.

Only CPU inference is supported.

Included libraries:

- Native [XGBoost](https://xgboost.readthedocs.io/en/latest/prediction.html)
- [ONNX Runtime](https://github.com/Microsoft/onnxruntime) (convert model using [ONNXMLTools](https://github.com/onnx/onnxmltools))
- Intel's [DAAL4PY](https://intelpython.github.io/daal4py/)
- [Treelite Runtime](https://github.com/dmlc/treelite)

## Environment setup

Using conda

```bash
conda create -n xgboost python=3.9
conda activate xgboost
conda install numpy

pip install xgboost
pip install onnxruntime git+https://github.com/onnx/onnxmltools.git
pip install daal4py
pip install treelite treelite-runtime
```

Note: At the time of this writing, the latest release of ONNXMLTools does not support converting models from `xgboost>=1.6.1` without [this patch](https://github.com/onnx/onnxmltools/pull/567). Therefore, you have to install `onnxmltools` from the latest main branch.

## Usage

```bash
python generate_models.py
python benchmark.py
```
