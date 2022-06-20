# TP5: Deep Learning

Implementación de Autoencoder, Denoising Autoencoder y Variational Autoencoder. 

## Dependencias

* **[Python 3.9+](https://www.python.org/downloads/)**
* [Pip](https://pip.pypa.io/en/stable/installation/)
* [Pipenv](https://pipenv.pypa.io/en/latest/)
* [Numpy](https://numpy.org/install/)
* [Matplotlib](https://pypi.org/project/matplotlib/)
* [Seaborn](https://seaborn.pydata.org/installing.html)
* [SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.html)
* [Keras](https://keras.io/getting-started/installation/)

## Instalación

Ejecutar, parado en el directorio [`TP5`](/TP5), el siguiente comando:

```sh
pipenv install
```

## Ejecución

Para ejecutar el programa principal se hace uso del siguiente comando:

```shell
pipenv run <main> <config>
```

Donde `<main>` es el nombre del programa principal y `<config>` incluye la configuración con los argumentos que se le pasan al programa.

Los valores de `<main>` pueden ser:
* `ej1a_autoencoder` - Ejecuta el programa para el autoencoder con `font_2` de `font.h` como conjunto de entrenamiento.
* `ej1b_denoising_autoencoder` - Ejecuta el programa para el DAE con el conjunto de fonts con ruido.
* `ej2_vae` - Ejecuta el programa para el VAE con conjunto `MNIST`.
* `ej2_vae_fashion` - Ejecuta el programa para el VAE con conjunto `Fashion MNIST`.

### Configuración

La configuración del programa se realiza en dos archivos JSON disponible en el directorio `resources`. Estos pueden ser los siguientes:

#### [`config.json`](/TP5/resources/config.json)

| Configuración     | Posibles Parámetros     | Descripción                                                                                           | 
|-------------------|-------------------------|-------------------------------------------------------------------------------------------------------|
| `algorithm`       | `no_linear_perceptron`  | Algoritmo de perceptron a utilizar.                                                                   |
| `learning_rate`   | `float`                 | Tasa de aprendizaje a utilizar.                                                                       |
| `betha`           | `float`                 | Valor por el cuál se multiplica parámetro de función sigmoidea. Sólo sirve para perceptron no lineal. |
| `function`        | `tanh`,`logistic`       | Función `g()` a utilizar en estado activación para el perceptrón no lineal.                           |
| `layers`          | [ `int`, `...` , `int`] | Cantidad de neuronas en cada una de las capas.                                                        |
| `max_iter`        | `int`                   | Condición de corte de iteración a la que puede llegar.                                                |
| `latent_code_len` | `int`                   | Largo de capa latente.                                                                                |
| `min_error`       | `float`                 | Condición de corte de error mínimo al que debe llegar.                                                |
| `k`               | `int`                   | Cantidad de partes a usar aleatoriamente en validación cruzada por k-partes.                          |

#### [`config_vae.json`](/TP5/resources/config_vae.json)
| Configuración       | Posibles Parámetros     | Descripción                                    | 
|---------------------|-------------------------|------------------------------------------------|
| `epochs`            | `int`                   | Cantidad de épocas a utilizar.                 |
| `latent_code_len`   | `int`                   | Largo de capa latente.                         |
| `neurons_per_layer` | [ `int`, `...` , `int`] | Cantidad de neuronas en cada una de las capas. |

## Presentación

El documento de la [presentación](/TP5/docs/TP5-Presentación-Grupo7.pdf) de las conclusiones se encuentra en
formato `pdf` en la sección de documentos.