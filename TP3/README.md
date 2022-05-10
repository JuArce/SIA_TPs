# TP3: Perceptron Simple y Multicapa

Implementación de redes neuronales para [perceptron](https://en.wikipedia.org/wiki/Perceptron) simple (lineal y no
lineal) y multicapa.

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

La configuración del programa se encuentra en los archivos de tipo `config.json` en el directorio
de [recursos](/TP3/resources/). La siguiente tabla describe las configuraciones posibles con sus opciones (en caso que
aplique):

| Configuración          | Posibles Parámetros                                                                        | Descripción                                                                                           | 
|------------------------|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `perceptron_algorithm` | `simple_perceptron`,`lineal_perceptron`,`not_linear_perceptron` y `multi_layer_perceptron` | Algoritmo de perceptron a utilizar.                                                                   |
| `cota`                 | `int`                                                                                      | Cantidad de iteraciones máximas                                                                       |
| `eta`                  | `float`                                                                                    | Factor para modificar W (η).                                                                          |
| `betha`                | `float`                                                                                    | Valor por el cual se multiplica parámetro de función sigmoidea. Solo sirve para perceptron no lineal. |
| `function`             | `tanh`,`logistic`                                                                          | Función `g()` a utilizar en estado activación para el perceptrón no lineal.                           |
| `layers`               | [ `int`, `int`, `int`]                                                                     | Cantidad de perceptrones en cada una de las capas ocultas                                             |
| `max_error`            | `float`                                                                                    | Condición de corte de error máximo al que puede llegar.                                               |
| `k`                    | `int`                                                                                      | Cantidad de partes a usar aleatoriamente en validación cruzada por k-partes.                          |

## Ejecución

Para ejecutar el programa principal se hace uso del siguiente comando:

```shell
pipenv run main <path_to_config> <path_to_input_training> <path_to_output_expected>
```

donde se tiene que:

* `<path_to_config>` posee el path a la configuración deseada. Se puede ingresar:
  ### ej1:
    * [`config_ej1.json`](/TP3/resources/config_ej1.json)
  ### ej2:
    * [`config_ej2_linear.json`](/TP3/resources/config_ej2_linear.json)
    * [`config_ej2_not_linear.json`](/TP3/resources/config_ej2_not_linear.json)
  ### ej3:
    * [`config_ej3_1.json`](/TP3/resources/config_ej3_1.json)
    * [`config_ej3_2.json`](/TP3/resources/config_ej3_2.json)


* `<path_to_input_training>` posee el path al directorio donde se encuentran los archivos para entrenar la red.
  Incluyen:
  ### ej1:
    * [`and_input.txt`](/TP3/resources/ej1/and_input.txt)
    * [`xor_input.txt`](/TP3/resources/ej1/xor_input.txt)
  ### ej2:
    * [`training_input.txt`](/TP3/resources/ej2/training_input.txt)
  ### ej3:
    * [`training_input_ej3.txt`](/TP3/resources/ej3/training_input_ej3.txt)


* `<path_to_output_expected>` posee el path al directorio donde se encuentran los archivos con los resultados esperados
  de dicho entrenamiento. Incluyen:
  ### ej1:
    * [`and_output.txt`](/TP3/resources/ej1/and_output.txt)
    * [`xor_output.txt`](/TP3/resources/ej1/xor_output.txt)
  ### ej2:
    * [`training_output.txt`](/TP3/resources/ej2/training_output.txt)
  ### ej3:
    * [`training_output_ej3_3.txt`](/TP3/resources/ej3/training_output_ej3_3.txt)

## Presentación

El documento de la [presentación](/TP3/docs/TP3-Presentación-Grupo7.pdf) de las conclusiones se encuentra en
formato `pdf` en la sección de documentos.