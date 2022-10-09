import time

import daal4py as d4p
import numpy as np
import onnxruntime
import treelite_runtime
import xgboost as xgb


def compare(x1, x2):
    diff = np.abs(x1 - x2)
    rmse = (diff ** 2).mean() ** 0.5
    max_diff = diff.max()
    print("RMSE:", rmse)
    print("Max diff:", max_diff)


def time_fn(fn, *args, **kwargs):
    N = 1000
    time0 = time.time()
    for _ in range(N):
        fn(*args, **kwargs)
    avg_time = (time.time() - time0) / N
    print(f"Time: {avg_time*1e3:.3f}ms")
    print(f"Speed: {1/avg_time:.0f} it/s")


n_samples, n_features = 10_000, 512
np.random.seed(2022)
X = np.random.randn(n_samples, n_features).astype(np.float32)
emb_1 = X[:1]
emb_10 = X[:10]
emb_100 = X[:100]

n_threads = 10
bst = xgb.Booster()
bst.load_model("xgboost_model.json")
params = {
    "predictor": "cpu_predictor",
    "nthread": n_threads,
}
bst.set_param(params)

print("XGBoost")
probs = bst.inplace_predict(X)
time_fn(bst.inplace_predict, emb_1)
time_fn(bst.inplace_predict, emb_10)
time_fn(bst.inplace_predict, emb_100)
print()


print("ONNX Runtime")
sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = n_threads

sess = onnxruntime.InferenceSession("onnx_model.onnx", sess_options=sess_options)
onnx_in_name = sess.get_inputs()[0].name
onnx_out_name = sess.get_outputs()[1].name
def onnx_pred(embs):
    return sess.run([onnx_out_name], {onnx_in_name: embs})[0][:, 1]

onnx_probs = onnx_pred(X)
compare(probs, onnx_probs)
time_fn(onnx_pred, emb_1)
time_fn(onnx_pred, emb_10)
time_fn(onnx_pred, emb_100)
print()


print("DAAL Runtime")
daal_model = d4p.get_gbt_model_from_xgboost(bst)
def get_daal_algo():
    return d4p.gbt_classification_prediction(
        nClasses=2,
        resultsToEvaluate="computeClassProbabilities",
    )
def daal_pred(embs, algo):
    return algo.compute(embs, daal_model).probabilities[:, 1]

daal_probs = daal_pred(X, get_daal_algo())
compare(probs, daal_probs)
time_fn(daal_pred, emb_1, get_daal_algo())
time_fn(daal_pred, emb_10, get_daal_algo())
time_fn(daal_pred, emb_100, get_daal_algo())
print()


print("Treelite Runtime")
predictor = treelite_runtime.Predictor("treelite_model.so", nthread=n_threads)
def treelite_pred(embs):
    return predictor.predict(treelite_runtime.DMatrix(embs))

treelite_probs = treelite_pred(X)
compare(probs, treelite_probs)
time_fn(treelite_pred, emb_1)
time_fn(treelite_pred, emb_10)
time_fn(treelite_pred, emb_100)
print()
