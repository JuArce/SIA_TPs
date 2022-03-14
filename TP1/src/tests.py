import os
from datetime import datetime
from algorithms.dfs import dfs
from algorithms.bfs import bfs
from algorithms.vds import vds
from algorithms.a_star import a_star
from algorithms.local_heuristic import local_heuristic
from algorithms.global_heuristic import global_heuristic
from utils.Config import Config
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


assert len(sys.argv) == 3, 'Missing test folder or output filename'
path = sys.argv[1]

file_name = sys.argv[2] + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
output_csv = open('./' + file_name, 'w+')
output_csv.write('initial_state,final_state,algorithm,heuristic,initial_depth,result,deep,cost,expanded_nodes,frontier_nodes,time,number_of_plays\n')

for filename in os.listdir(path):
    with open(os.path.join(path, filename)) as f:
        config: Config = Config(f.read())
        if config.initial_state is None or len(config.initial_state) == 0:
            config.initial_state = Plays.build_initial_play(config.qty)
        f.close()
        results = algorithms[config.algorithm](config)
        output_csv.write(config.initial_state + ',' +
                         config.final_state + ',' +
                         config.algorithm + ',' +
                         (config.heuristic if config.heuristic is not None else '') + ',' +
                         (config.initial_depth if config.initial_depth is not None else '') + ',' +
                         ('success' if results.result else 'failed') + ',' +
                         str(results.deep) + ',' +
                         str(results.cost) + ',' +
                         str(results.expandedNodes) + ',' +
                         str(results.frontierNodes) + ',' +
                         str(results.time) + ',' +
                         str(len(results.plays_to_win) - 1) + '\n')

output_csv.close()