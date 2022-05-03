# TP3: Perceptron Simple y Multicapa

Implementación de redes neuronales para perceptron simple y multicapa.

## Dependencias

* **[Python 3.9+](https://www.python.org/downloads/)**
* [Pip](https://pip.pypa.io/en/stable/installation/)
* [Pipenv](https://pipenv.pypa.io/en/latest/)
* [Numpy](https://numpy.org/install/)
* [Typing](https://pypi.org/project/typing/)
* [Matplotlib](https://pypi.org/project/matplotlib/)

## Instalación

Correr, parado en el directorio [`TP3`](/TP3), el siguiente comando para realizar toda instalación necesaria:

```sh
pipenv install
```

## Configuración

La configuración del programa se encuentra en el archivo [config.json](/TP3/resources/config_ej1.json). La siguiente tabla
describe las configuraciones posibles con sus opciones:

| Configuración          | Posibles Parámetros                                                                        | Descripción                                                                                                                                                                              | 
|------------------------|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `perceptron_algorithm` | `simple_perceptron`,`lineal_perceptron`,`not_linear_perceptron`,y `multi_layer_perceptron` | Algoritmo de perceptron a utilizar.                                                                                                                                                      |
| `cota`                 | `int`                                                                                      | Cantidad de iteraciones máximas                                                                                                                                                          |
| `eta`                  | `float`                                                                                    | Factor para modificar W (η).                                                                                                                                                             |
| `betha`                | `float`                                                                                    | Valor por el cual se multiplica parámetro de función sigmoidea. Solo sirve para perceptron no lineal.                                                                                    |
| `function`             | `tanh`,`logistic`                                                                          | Función `g()` a utilizar en estado activación para el perceptrón no lineal.                                                                                                              |
| `layers`               | { `int`, `int`, `int`}                                                                     | Cantidad de capas de perceptron multicapa, el primer parametro representa las capas de entrada, el sefundo la cantidad de capas ocultas y el último la cantidad de parametros de salida. |
 | `max_error`            | `float`                                                                                    | Condición de corte de error máximo al que puede llegar.                                                                                                                                  |
| `k`                    | `int`                                                                                      | Cantidad de partes a usar aleatoriamente en validación cruzada por k-partes.                                                                                                             |

## Ejecución

Para ejecutar el programa principal se hace uso del siguiente comando:

```shell
pipenv run main <path_to_config> <path_to_input_training> <path_to_output_expected>
```

donde se tiene que:

* `<path_to_config>` posee la configuración mencionada.
* `<path_to_input_training>` posee el path al directorio donde se encuentran los archivos para entrenar la red.
* `<path_to_output_expected>` posee el path al directorio donde se encuentran los archivos con los resultados esperados
  de dicho entrenamiento.

## Presentación

El documento de la [presentación](/TP3/docs/TP3-Presentación-Grupo7.pdf) de las conclusiones se encuentra en
formato `pdf` en la sección de documentos.