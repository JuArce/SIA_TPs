# TP1: Rompecabezas de números
Implementación de generador de soluciones para el juego [rompecabezas de nú́meros](https://es.wikipedia.org/wiki/Rompecabezas_deslizantes) (8 nú́meros en una grilla de 3 por 3).

## Dependencias
* **[Python 3.9+](https://www.python.org/downloads/)**
* [Pip](https://pip.pypa.io/en/stable/installation/)
* [Pipenv](https://pipenv.pypa.io/en/latest/)
* [Typing](https://pypi.org/project/typing/)

## Instalación
Correr, parado en el directorio `TP1`, el siguiente comando para realizar toda instalación necesaria:
```sh
pipenv install
```

## Configuración
La configuración del programa se encuentra en el archivo [config.json](/TP1/resources/config.json).
Dento del mismo se pueden encontrar los siguientes parámetros:
* `algorithm`: define el algoritmo a usar, este puede ser: `a_star`, `bfs`, `dfs`, `vds`, `local_heuristic` y `global_heuristic`
* `heuristic`: permite probar alguna de las heuristicas admisibles o no admisible. Estas pueden ser: `manhattan`, `hamming` o `not_adm_heu`.
* `initial_state`: Define el estado inicial del tablero, este se encuentra vacío por defecto y se genera automáticamente.
* `initial_depth`: Límite que se aplica a BPPV por defecto.
* `final_state`: Estado para ganar el juego, por defecto es `123456780` como el juego original.
* `qty`: Cantidad de movimientos que se realizan para generar un tablero al azar. Si no se ingresa un número se usará 300 por defecto.

## Ejecución
Para ejecutar el programa existen dos configuraciones posibles:
```shell
pipenv run main ./resources/config.json
```
donde en el mismo directorio se obtiene un archivo con el formato `algorithm-yyyy-mm-dd_hh-mm-ss.txt`.
```shell
pipenv run test ./resources/test1 test1
```
donde se corren casos de prueba de los algoritmos y se obtiene su resulado en formato `csv`. 

Hay 7 tipos de pruebas disponibles y se pueden acceder reemplazando `test1` por el _test_ deseado.
* `test1`: ejecuta todos los algoritmos.
* `test2`: ejecuta los algoritmos de `bfs`, `dfs` y `vds`.
* `test3`: ejecuta el algoritmo `vds` con distintas profundidades.
* `test4`: ejecuta el algoritmo `a_star`, `local_heuristic` y `global_heuristic` con igual función de heurística.
* `test5`: ejecuta el algoritmo `a_star` con las tres heurísticas disponibles.
* `test6`: ejecuta el algoritmo `local_heuristic` con las tres heurísticas disponibles.
* `test7`: ejecuta el algoritmo `global_heuristic` con las tres heurísticas disponibles.

## Presentación
El documento de la presentación oral se puede encontrar en la carpeta [`presentation`](resources/presentation) dentro de `resources`.