from utils.Config import Config

print("main")


f = open('./resources/config.json')
config = Config(f.read())
print(config.algorithm)
print(config)
# Una clase por algoritmo
# Cargar la cfg
# llamar al algoritmo correspondiente
#
