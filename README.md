# ML2_CEIA
ML2 CEIA 2022


# Deploy de modelo de predicción de falla cardiaca

google Slides: https://docs.google.com/presentation/d/1dnJg6_UfxhJKSvt6cZoJ7Qeswi_UwMrhau43M3V7uIg/edit?usp=sharing

python para GCP compute instance: https://github.com/freddy120/ML2_CEIA/blob/main/TP/ml2.py

Notebook databricks: https://github.com/freddy120/ML2_CEIA/blob/main/TP/notebook/ml2_tp_final.ipynb


## Despliegue en GCP

Arquitectura:

![](https://github.com/freddy120/ML2_CEIA/blob/main/imagenes/ML2-GCP.png)

formato de CSV de entrada:
```csv
Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope
40,M,ATA,140,289,0,Normal,172,N,0,Up
49,F,NAP,160,180,0,Normal,156,N,1,Flat
37,M,ATA,130,283,0,ST,98,N,0,Up
```

formato de CSV de salida:

```csv
Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease
40,M,ATA,140,289,0,Normal,172,N,0.0,Up,0
49,F,NAP,160,180,0,Normal,156,N,1.0,Flat,1
37,M,ATA,130,283,0,ST,98,N,0.0,Up,0
48,F,ASY,138,214,0,Normal,108,Y,1.5,Flat,1
```

folder de instalacion
![](https://github.com/freddy120/ML2_CEIA/blob/main/imagenes/despliegue_gcp.png)


Ejecucion del script:
![](https://github.com/freddy120/ML2_CEIA/blob/main/imagenes/run_gcp.png)



## Despliegue en Databricks

Arquitectura:

![](https://github.com/freddy120/ML2_CEIA/blob/main/imagenes/ML2-Databricks.png)


Log de ejecucciones:
![](https://github.com/freddy120/ML2_CEIA/blob/main/imagenes/experiments.png)


Servir un modelo registrado:

![](https://github.com/freddy120/ML2_CEIA/blob/main/imagenes/serving_model.png)


Ejemplo de JSON de entrada:

```json
[
    {
        "Age": 40,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 172,
        "ExerciseAngina": "N",
        "Oldpeak": 0,
        "ST_Slope": "Up"
    }
]
```

Crear token para uso externo de la API:
![](https://github.com/freddy120/ML2_CEIA/blob/main/imagenes/create_token.png)


Prueba con Postman:

configurar token:
![](https://github.com/freddy120/ML2_CEIA/blob/main/imagenes/postman_token.png)


![](https://github.com/freddy120/ML2_CEIA/blob/main/imagenes/postman.png)



# (EXTRA) Despliegue del script de la clase 5 usando Google Function


codigo fuente: https://github.com/freddy120/ML2_CEIA/blob/main/clase5/ml2_gcp_function.py

configurar activador google Function:
![](https://github.com/freddy120/ML2_CEIA/blob/main/imagenes/activador_lambda.png)


dependencias:
![](https://github.com/freddy120/ML2_CEIA/blob/main/imagenes/lambda_req.png)

codigo:
![](https://github.com/freddy120/ML2_CEIA/blob/main/imagenes/lambda_src.png)


