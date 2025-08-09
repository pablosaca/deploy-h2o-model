# Ejemplo uso modelización con H2O en Python y Despliegue con Flask

This repo consist of a deployment a Machine Learning model using Flask package.

The Machine Learning model was built using H2O.

Random Forest

Each of these components are developed within the project in an offline setting inside /artifact. The H2O model will still be needed in a production or testing setting in order to be able to predict user-submitted queries, so they can be serialized via python's h2o functionality and stored within the /artifact folder.

Installation
Create a conda virtual environment in the project directory.

```
conda create -n h2o_env python=3.10
```

Activate the virtual environment.

```
conda activate h2o_env
```

While in the conda environment, install required dependencies from requirements.txt.

```
pip install -r requirements.txt
```

Now we can deploy via

```
python app.py  
```

and navigate to http://127.0.0.1:8000/predict to see it live. I recommend the use of postman to check the correct operation of the api (https://www.postman.com/)

The application may then be terminated with the following commands.

ctrl - c