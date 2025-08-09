import os

from flask import Flask, request, jsonify

import h2o

# Iniciar H2O y mantenerlo vivo (nunca cerrar)
h2o.init()
h2o.no_progress()  # opcional, para evitar spam en consola

# Cargar modelo (usa tu ruta real)
actual_path = os.getcwd()
folder = "artifact"

glm_model_path = "gbm/gbm_grid_model_2"
gbm_model_path = "modelo_lineal/GLM_model_python_1754730853905_1"

glm_model_path = actual_path + "/" + "artifact" + "/" + glm_model_path
gbm_model_path = actual_path + "/" + "artifact" + "/" + gbm_model_path

glm_model = h2o.load_model(glm_model_path)
gbm_model = h2o.load_model(gbm_model_path)

# Crear la app Flask
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        h2o_df = h2o.H2OFrame(data)
        h2o_df = preprocessing_data(h2o_df)
        glm_pred_value = model_prediction_value(glm_model, h2o_df)
        gbm_pred_value = model_prediction_value(gbm_model, h2o_df)

        # ponderación de la salida del modelo
        pred_value = glm_pred_value * 0.1 + gbm_pred_value * 0.9
        return jsonify({"predictions": pred_value}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def preprocessing_data(data):

    # identificación de las variables factor (variables categóricas)
    data['CHAS'] = data['CHAS'].asfactor()
    data['RAD'] = data['RAD'].asfactor()
    return data


def model_prediction_value(model, data) -> float:
    value = model.predict(data).as_data_frame(use_pandas=False)  # lista directa
    # hay que coger el segundo elemento (el primer elemento es la columna de una hipotética tabla)
    return float(value[1][0])  # la salida es un string por eso se convierte a float


if __name__ == "__main__":
    # Modo debug para desarrollo
    app.run(host="0.0.0.0", port=8000, debug=True)
