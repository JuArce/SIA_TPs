# TP4: Métodos de Aprendizaje NO Supervisado
Implementación de métodos de aprendizaje NO supervisado. 

## Dependencias

* **[Python 3.9+](https://www.python.org/downloads/)**
* [Pip](https://pip.pypa.io/en/stable/installation/)
* [Pipenv](https://pipenv.pypa.io/en/latest/)
* [Numpy](https://numpy.org/install/)
* [Matplotlib](https://pypi.org/project/matplotlib/)
* [Seaborn](https://seaborn.pydata.org/installing.html)

## Instalación

Correr, parado en el directorio [`TP4`](/TP4), el siguiente comando para realizar toda instalación necesaria:

```sh
pipenv install
```

## Ejecución

Para ejecutar el programa principal se hace uso del siguiente comando:

```shell
pipenv run main <path_to_config> <path_to_input_training> <path_to_output_expected>
```

Donde se tiene:

| Algoritmo | Config                             | Input                     | Output                        |
|-----------|------------------------------------|---------------------------|-------------------------------|
| Kohonen   | `./resources/config_kohonen.json`  | `./resources/europe.csv`  | -                             |
| Hopfield  | `./resources/config_hopfield.json` | `./resources/letters.txt` | -                             |
| Oja       | `./resources/config_oja.json`      | ` ./resources/europe.csv` | ` ./resources/oja_output.txt` |


## Presentación

El documento de la [presentación](/TP4/docs/TP4-Presentación-Grupo7.pdf) de las conclusiones se encuentra en
formato `pdf` en la sección de documentos.