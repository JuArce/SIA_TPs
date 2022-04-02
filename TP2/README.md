# TP2: El problema de la mochila
Implementación de algoritmos genéticos para el problema de la mochila o bien [knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem#:~:text=The%20knapsack%20problem%20is%20a,is%20as%20large%20as%20possible.). 

## Dependencias
* **[Python 3.9+](https://www.python.org/downloads/)**
* [Pip](https://pip.pypa.io/en/stable/installation/)
* [Pipenv](https://pipenv.pypa.io/en/latest/)
* [Typing](https://pypi.org/project/typing/)

## Instalación
Correr, parado en el directorio `TP2`, el siguiente comando para realizar toda instalación necesaria:
```sh
pipenv install
```

## Configuración
La configuración del programa se encuentra en el archivo [config.json](/TP2/resources/config.json).
La siguiente tabla describe las configuraciones posibles con sus opciones:

| Configuración                       | Posibles Parámetros                                                   | Descripción                                                                                                        | 
|-------------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `selection_algorithm`               | `boltzmann`, `elite`, `rank`, `roulette`, `tournament`, y `truncated` | Algoritmo de selección a usar.                                                                                     |
 | `cross_over_algorithm`              | `simple`, `multiple` y `uniform`                                      | Algoritmo de método de cruza a ser usado.                                                                          | 
 | `multiple_cross_points`             | `int`                                                                 | Puntos a usar con algoritmo de cruza múltiple.                                                                     |
 | `population`                        | `int`                                                                 | Población inicial a usar para el algoritmo.                                                                        |
 | `limit_time`                        | `int`                                                                 | Tiempo a usar (en segundos) como criterio de corte.                                                                |
 | `generations_quantity`              | `int`                                                                 | Cantidad de generaciones a usar como condición de corte.                                                           |
 | `mutation_probability`              | `float`                                                               | Probabilidad a usar al momento de mutar los genes en el algoritmo de mutación.                                     |
| `k_truncated`                       | `int`                                                                 | Cantidad de individuos con menor aptitud a eliminar en algoritmo de selección truncada.                            |
| `tournament_probability`            | `float`                                                               | Probabilidad a usar en método de selección competitiva (tournament).                                               |
 | `max_unchanged_fitness_generations` | `int`                                                                 | Criterio de corte para cantidad de generaciones que no poseen un cambio de fitness máximo.                         |
| `unchanged_percentage`              | `float`                                                               | Criterio de corte en base al porcentaje de generaciones que no poseen cambios.                                     |
| `max_unchanged_generations`         | `int`                                                                 | Criterio de corte en base a la cantidad máxima de generaciones que no poseen cambios en base a el porcentaje dado. |
 | `temperature`                       | `int`                                                                 | Valor T de temperatura a usar en algorito de selección de Boltzmann.                                               |
| `temperature_goal`                  | `int`                                                                 | Temperatura T_c deseada en algoritmo de selección de Boltzmann.                                                    |
 | `decrease_temp_factor`              | `int`                                                                 | Factor de decrecimiento en algoritmo de selección de Boltzmann.                                                    | 

## Ejecución
Para ejecutar el programa se hace uso del siguiente comando:
```shell
pipenv run main <path_to_file> <path_to_config> <path_to_output>
```
donde `<path_to_file>` representa el archivo de datos de la mochila a ser analizada. Por defecto se recomienda usar `./resources/Mochila100Elementos.txt`, el archivo aportado por la cátedra.
`<path_to_output>` representa el directorio donde guardar el archivo de salida.

## Presentación
El documento de la [presentación]() de las conclusiones se encuentra en formato `pdf` en la sección de documentos.