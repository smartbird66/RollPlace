from place_db import PlaceDB_chipbench, PlaceDB_ispd, save_placement
from utils import random_guiding, rank_clusters, greedy_placer_with_init_coordinate, region_k_sample_search_placer
import random
import argparse
import time
import os
import math
import numpy as np
import copy
from concurrent.futures import ProcessPoolExecutor
import treelib

from tensorboardX import SummaryWriter

def init_tree(init_round, node_id_ls, placedb, grid_num, grid_size):
    tree = treelib.Tree()
    tree.create_node(
        tag="root",
        identifier=str([]),
        data={
            "visit_time": 0,
            "Q": -math.inf,
            "V": -math.inf,
            "min_hpwl": math.inf,
        }
    )
    
    init_solutions = []
    for i in range(1, init_round + 1):
        print(f"===================== random init {i}/{init_round} =====================")
        place_record = random_guiding(node_id_ls, placedb, grid_num, grid_size)
        init_place_record, init_solution, init_hpwl = greedy_placer_with_init_coordinate(node_id_ls, placedb, grid_num, grid_size, place_record)
        init_solutions.append((init_solution, init_place_record, init_hpwl))
        

    init_solutions = sorted(init_solutions, key=lambda x: x[2])

    for i in range(len(init_solutions)):
        init_solution, _, init_hpwl = init_solutions[i]
        for j in range(0, len(init_solution) + 1):
            node = tree.get_node(str(init_solution[:j]))
            if node:
                node.data['visit_time'] += 1
                node.data['Q'] += -init_hpwl
                node.data['V'] = node.data['Q'] / node.data['visit_time']
                node.data['min_hpwl'] = min(node.data['min_hpwl'], init_hpwl)
            else:   # node is none
                tree.create_node(
                    tag=str(init_solution[j-1]),
                    identifier=str(init_solution[:j]),
                    parent=str(init_solution[:j-1]),
                    data={
                        "visit_time": 1,
                        "Q": -init_hpwl,   # -sum hpwl
                        "V": -init_hpwl,   # Q_value / visit_time
                        "min_hpwl": init_hpwl,
                    }
                )
    
    return tree

def main():
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--dataset', default='ariane133')
    parser.add_argument('--benchmark_folder', default='ispd2005')
    parser.add_argument('--max_workers', default=3, type=int, help='max workers for parallel')
    parser.add_argument('--grid_num', default=240, type=int, help='grid num for placement')
    parser.add_argument('--seed', default=2000, type=int, help='random seed')
    parser.add_argument('--init_round', default=20, type=int, help='initialization round for MCTS')
    parser.add_argument('--search_round', default=500, type=int,  help='search round for MCTS')
    parser.add_argument('--random_placed_times', default=30, type=int, help='random placed times for each k-sample search placer')
    parser.add_argument('--ch_scale', default=30, type=float, help='Chernoff Hoeffding bound scale factor')
    args = parser.parse_args()
    random_placed_times = int(args.random_placed_times)
    dataset = args.dataset
    random_seed = args.seed
    search_round = args.search_round
    init_round = args.init_round
    max_workers = args.max_workers
    random.seed(random_seed)
    BENCHMARK_CLASSES = {
        "ChiPBench": PlaceDB_chipbench,
        "ispd2005": PlaceDB_ispd,
    }
    
    placedb_cls = BENCHMARK_CLASSES.get(args.benchmark_folder)
    if not placedb_cls:
        print(f"benchmark folder not implemented: {args.benchmark_folder}")
        return
    placedb = placedb_cls(dataset)
    
    ref_file_path = os.path.join(f"benchmarks/{args.benchmark_folder}", dataset, dataset+".pl")
    tb_dir = f"./workspace/{args.benchmark_folder}/{dataset}/{random_seed}/{time.strftime('%Y%m%dT%H%M%S')}"
    # set up tensorboard writer
    tb_writer = SummaryWriter(tb_dir)
    node_id_ls = rank_clusters(placedb)
    
    placement_save_dir = f"result/{args.benchmark_folder}/{dataset}/{random_seed}"
    if not os.path.exists(placement_save_dir):
        os.makedirs(placement_save_dir)

    grid_num = args.grid_num
    grid_size = [int((placedb.max_width - placedb.min_width) / grid_num), int((placedb.max_height - placedb.min_height) / grid_num)]
    print(f"grid num: {grid_num}, grid size: {grid_size}")

    # init the search tree
    tree = init_tree(init_round=init_round, node_id_ls=node_id_ls, placedb=placedb, grid_num=grid_num, grid_size=grid_size)
    
    # execute  monte carlo tree search
    node = tree.get_node(str([]))
    children = tree.children(node.identifier)
    min_hpwls = [child.data['min_hpwl'] for child in children]
    c = min(min_hpwls) / args.ch_scale #/ 30
    best_hpwl = min(min_hpwls)
    for i in range(search_round):
        print(f"===================== loop {i+1}/{search_round} =====================")
        node = tree.get_node(str([]))
        children = tree.children(node.identifier)
        # selection
        while len(children) != 0:
            ucb = [(-child.data['min_hpwl'] + child.data['V'])/2  + c * 1 / math.sqrt(child.data['visit_time']) for child in children]
            node_index = np.argmax(ucb)
            node = children[node_index]
            children = tree.children(node.identifier)
        print(f"selected solution' hpwl: {node.data['min_hpwl']}")

        selected_solution = eval(node.identifier)
        results = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(region_k_sample_search_placer, copy.deepcopy(node_id_ls), copy.deepcopy(placedb), grid_num, grid_size, 
                                       init_solution=selected_solution.copy(), random_placed_times=random_placed_times) for _ in range(max_workers)]
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Rollout search exception: {e}")
        results = sorted(results, key=lambda x: x[2])
        placed_macros, new_solution, new_hpwl = results[0]
        if isinstance(new_solution, np.ndarray):
            new_solution = new_solution.tolist()
        elif new_solution == []:
            print("no solution!")
            continue
        
        # update new solution
        if new_hpwl < node.data['min_hpwl']:
            print(f"update tree: {node.data['min_hpwl']} -> {new_hpwl}")
            for j in range(0, len(new_solution) + 1):
                node = tree.get_node(str(new_solution[:j]))
                if node:
                    node.data['visit_time'] += 1
                    node.data['Q'] += -new_hpwl
                    node.data['V'] = node.data['Q'] / node.data['visit_time']
                    node.data['min_hpwl'] = min(node.data['min_hpwl'], new_hpwl)
                else:   # node is none
                    tree.create_node(
                        tag=str(new_solution[j-1]),
                        identifier=str(new_solution[:j]),
                        parent=str(new_solution[:j-1]),
                        data={
                            "visit_time": 1,
                            "Q": -new_hpwl,   # -sum hpwl
                            "V": -new_hpwl,   # Q_value / visit_time
                            "min_hpwl": new_hpwl, # min hpwl
                        }
                    )
        
        ## record min hpwl
        if new_hpwl < best_hpwl:
            best_hpwl = new_hpwl
            save_file_path = placement_save_dir + f'/{dataset}.pl'
            save_placement(node_id_ls, placedb, new_solution, save_file_path, ref_file_path, grid_num, grid_size)
        print(f"new hpwl: {new_hpwl}, \tbest hpwl :{best_hpwl}")
        
    
        ## logger
        tb_writer.add_scalar('hpwl', new_hpwl, i)
        tb_writer.add_scalar('best_hpwl', best_hpwl, i)
        
    print(f"best hpwl: {best_hpwl}")
    

if __name__ == "__main__":
    main()
