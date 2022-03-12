from utils.Config import Config
from algorithms.dfs import dfs

f = open('./resources/config.json')
config: Config = Config(f.read())
print(config.algorithm)
print(config)
f.close()
print(dfs(config))
# Chequear a que algoritmo llamo
# Llamar al algoritmo con los parametros necesarios

# Obtengo los resultados
# Imprimo en pantalla o en archivo resultados
