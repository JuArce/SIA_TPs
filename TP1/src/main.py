from datetime import datetime
from algorithms.dfs import dfs
from algorithms.bfs import bfs
from algorithms.vds import vds
from algorithms.a_star import a_star
from algorithms.local_heuristic import local_heuristic
from algorithms.global_heuristic import global_heuristic
from utils.Config_sm import Config
from utils.Plays import Plays
import sys

algorithms = {
    "bfs": bfs,
    "dfs": dfs,
    "vds": vds,
    "local_heuristic": local_heuristic,
    "global_heuristic": global_heuristic,
    "a_star": a_star
}


print('Argument List:', str(sys.argv))
assert len(sys.argv) == 2, 'Missing config json'
f = open(sys.argv[1])
config: Config = Config(f.read())
if config.initial_state is None or len(config.initial_state) == 0:
    config.initial_state = Plays.build_initial_play(config.qty)

f.close()

print('Running ' + config.algorithm + '...')
results = algorithms[config.algorithm](config)
print(config.algorithm + ' finished.')

file_name = results.config.algorithm + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'

rf = open('./' + file_name, 'w+')

rf.write("".join(["Configuration: ", str(results.config), "\n"]))
result = "Solved with success" if results.result else "Failed to solve"
rf.write("".join(["Result: ", result, "\n"]))
rf.write("".join(["Deep: ", str(results.deep), "\n"]))
rf.write("".join(["Cost: ", str(results.cost), "\n"]))
rf.write("".join(["Expanded Nodes: ", str(results.expandedNodes), "\n"]))
rf.write("".join(["Frontier nodes: ", str(results.frontierNodes), "\n"]))
rf.write("".join(["Time: ", str(results.time), "\n"]))

if results.result:

    rf.write("".join(["Number of plays: ", str(len(results.plays_to_win)-1), "\n"]))

    for idx, p in enumerate(results.plays_to_win):
        rf.write("".join(["Play: ", str(idx), "\n", p[0:3], "\n", p[3:6], "\n", p[6:9], "\n", '-------', "\n"]))


print('Output printed to ' + file_name)
rf.close()
