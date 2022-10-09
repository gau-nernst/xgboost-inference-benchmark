import numpy as np
import onnxmltools
import treelite
import xgboost as xgb
from onnxconverter_common import FloatTensorType


np.random.seed(2022)
n_samples, n_features = 10_000, 512
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randint(2, size=(n_samples,))
dtrain = xgb.DMatrix(X, y)

depth, n_trees = 20, 2000
params = {
    "objective": "binary:logistic",
    "max_depth": depth,
    "tree_method": "gpu_hist",
}
bst = xgb.train(params, dtrain, n_trees)
preds = bst.predict(dtrain)
bst.save_model("xgboost_model.json")


onnx_model = onnxmltools.convert_xgboost(
    bst,
    initial_types=[("input", FloatTensorType(shape=[None, n_features]))],
    target_opset=12,
)
onnxmltools.save_model(onnx_model, "onnx_model.onnx")


treelite_model = treelite.Model.from_xgboost(bst)
treelite_model.export_lib(
    toolchain="gcc",
    libpath="treelite_model.so",
    params={"parallel_comp": 8},
    verbose=True
)
