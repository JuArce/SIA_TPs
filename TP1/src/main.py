from utils.Config import Config
from algorithms.dfs import dfs
from algorithms.bfs import bfs

f = open('./resources/config.json')
config: Config = Config(f.read())
print(config.algorithm)
print(config)
f.close()
results = bfs(config)

if results.result:
    for p in results.plays_to_win:
        print(p[0:3])
        print(p[3:6])
        print(p[6:9])
        print('-------')


        # Chequear a que algoritmo llamo
        # Llamar al algoritmo con los parametros necesarios

        # Obtengo los resultados
        # Imprimo en pantalla o en archivo resultados
