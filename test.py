import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.pyfunc as fn

remote_server_uri = "https://dagshub.com/yuvaneshkm/mlflow-demo.mlflow"
mlflow.set_tracking_uri(remote_server_uri)

model_name = "ElasticnetWineModel"

model = fn.load_model(
    model_uri=f"models:/{model_name}/Production"
)


rslt = model.predict([[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]])
print(rslt)