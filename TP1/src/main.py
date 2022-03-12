from utils.Config import Config
from algorithms.dfs import dfs
from datetime import datetime
from algorithms.bfs import bfs

f = open('./resources/config.json')
config: Config = Config(f.read())
f.close()
results = bfs(config)

file_name = results.config.algorithm + '-' + datetime.now().strftime("%d-%m-%Y %H:%M:%S") + '.txt'

rf = open('./' + file_name, "w+")

rf.write("".join(["Configuration: ", str(results.config), "\n"]))
result = "Solved with success" if results.result else "Failed to solve"
rf.write("".join(["Result: ", result]))
rf.write("".join(["Deep: ", str(results.deep), "\n"]))
rf.write("".join(["Cost: ", str(results.cost), "\n"]))
rf.write("".join(["Expanded Nodes: ", str(results.expandedNodes), "\n"]))
rf.write("".join(["Frontier nodes: ", str(results.frontierNodes), "\n"]))
rf.write("".join(["Time: ", str(results.time), "\n"]))

if results.result:

    rf.write("".join(["Number of plays: ", str(len(results.plays_to_win)), "\n"]))

    for idx, p in enumerate(results.plays_to_win):
        rf.write("".join(["Play: ", str(idx), "\n", p[0:3], "\n", p[3:6], "\n", p[6:9], "\n", '-------', "\n"]))

rf.close()
