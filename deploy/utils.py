from h2o.frame import H2OFrame


def preprocessing_data(data: H2OFrame) -> H2OFrame:

    # identificación de las variables factor (variables categóricas)
    data['CHAS'] = data['CHAS'].asfactor()
    data['RAD'] = data['RAD'].asfactor()
    return data


def model_prediction_value(model, data) -> list[float]:
    value = model.predict(data).as_data_frame(use_pandas=False)  # lista directa
    # hay que coger el segundo elemento (el primer elemento es la columna de una hipotética tabla)
    print(value)   # la salida es un string por eso se convierte a float
    process_value = [float(elem[0]) for elem in value if elem[0] != 'predict']
    return process_value


def input_check(data) -> None:
    # evalúa keys del del diccionario
    keys_value = list(data.keys())
    available_keys_value = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT"
    ]
    cat_keys = ["CHAS", "RAD"]

    not_available_key_value = [key for key in keys_value if key not in available_keys_value]
    for key, value in data.items():
        if key not in available_keys_value:
            raise ValueError(
                f"Los datos de entrada tienen columnas con nombres incorrectos: {not_available_key_value}"
            )
        if not isinstance(value, list):
            raise TypeError("Los valores de cada columna deben ser una lista")
        # chequeamos niveles de las variables categóricas
        if key == cat_keys[0]:
            available_chas_value = [0, 1]
            for v in value:
                if v not in available_chas_value:
                    raise ValueError(
                        f"El valor tomado es {v}, pero "
                        f"la variable {cat_keys[0]} solo puede tener los valores {available_chas_value}"
                    )
        elif key == cat_keys[1]:
            available_rad_value = [1,  2,  3,  5,  4,  8,  6,  7, 24]
            for v in value:
                if v not in available_rad_value:
                    raise ValueError(
                        f"El valor tomado es {v}, pero "
                        f"la variable {cat_keys[1]} solo puede tener los valores {available_rad_value}"
                    )
        else:
            for v in value:
                if not isinstance(v, (int, float)):
                    raise TypeError(f"El valor {v} no es int o float")
