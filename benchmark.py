import time

import daal4py as d4p
import numpy as np
import onnxruntime
import pandas as pd
import treelite_runtime
import xgboost as xgb


def compare(x1, x2):
    diff = np.abs(x1 - x2)
    rmse = (diff ** 2).mean() ** 0.5
    max_error = diff.max()
    print("RMSE:", rmse)
    print("Max diff:", max_error)
    return rmse, max_error


def time_fn(fn, *args, **kwargs):
    N = 1000
    fn(*args, **kwargs) # warm-up
    time0 = time.time()
    for _ in range(N):
        fn(*args, **kwargs)
    speed = N / (time.time() - time0)
    print(f"Speed: {speed:.0f} it/s")
    return speed


d4p_version = ".".join(str(x) for x in eval(d4p._get__version__())[:-1])
print("XGBoost:", xgb.__version__)
print("ONNX Runtime:", onnxruntime.__version__)
print("DAAL4PY:", d4p_version)
print("Treelite Runtime:", treelite_runtime.__version__)


n_samples, n_features = 10_000, 512
np.random.seed(2022)
X = np.random.randn(n_samples, n_features).astype(np.float32)
emb_1 = X[:1]
emb_10 = X[:10]
emb_100 = X[:100]

n_threads = 16
bst = xgb.Booster()
bst.load_model("xgboost_model.json")
params = {
    "predictor": "cpu_predictor",
    "nthread": n_threads,
}
bst.set_param(params)

print("XGBoost")
probs = bst.inplace_predict(X)
output_data = compare(probs, probs)
speed_data = [time_fn(bst.inplace_predict, x) for x in [emb_1, emb_10, emb_100]]
xgboost_data = [*output_data, *speed_data]
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
output_data = compare(probs, onnx_probs)
speed_data = [time_fn(onnx_pred, x) for x in [emb_1, emb_10, emb_100]]
onnx_data = [*output_data, *speed_data]
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
output_data = compare(probs, daal_probs)
speed_data = [time_fn(daal_pred, x, get_daal_algo()) for x in [emb_1, emb_10, emb_100]]
daal_data = [*output_data, *speed_data]
print()


print("Treelite Runtime")
predictor = treelite_runtime.Predictor("treelite_model.so", nthread=n_threads)
def treelite_pred(embs):
    return predictor.predict(treelite_runtime.DMatrix(embs))

treelite_probs = treelite_pred(X)
output_data = compare(probs, treelite_probs)
speed_data = [time_fn(treelite_pred, x) for x in [emb_1, emb_10, emb_100]]
treelite_data = [*output_data, *speed_data]
print()


data = [xgboost_data, onnx_data, daal_data, treelite_data]
names = ["XGBoost", "ONNX", "DAAL", "Treelite"]
columns_1 = ["rmse", "max error"]
columns_2 = [f"speed (batch {i})" for i in (1, 10, 100)]
df = pd.DataFrame(data, index=names, columns=columns_1 + columns_2)
df[columns_1] = df[columns_1].applymap("{:.2e}".format)
df[columns_2] = df[columns_2].applymap("{:,.0f}".format)
print(df)
df.to_markdown("temp.txt", tablefmt="github")
