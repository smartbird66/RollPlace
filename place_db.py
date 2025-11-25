import numpy as np
import os
import argparse
from operator import itemgetter
from itertools import combinations
import re
import subprocess
import math, random
import time

def save_placement(macro_name_list, placedb, solution, save_file_path, ref_file_path, grid_num, grid_size):
    f = open(save_file_path, "w")
    with open(ref_file_path, "r") as f2:
        for line in f2:
            line = line.strip()
            l = line.split()
            if line and is_number(l[1]):
                macro_name = l[0]
                if macro_name in placedb.macro_info.keys():
                    action = solution[macro_name_list.index(macro_name)]
                    pos_x = action // grid_num
                    pos_y = action % grid_num
                    l[1] = str(int(pos_x * grid_size[0] + placedb.min_width))
                    l[2] = str(int(pos_y * grid_size[1] + placedb.min_height))
                elif macro_name in placedb.std_info.keys():
                    cluster_name = placedb.stdName2Cluster[macro_name]
                    action = solution[macro_name_list.index(cluster_name)]
                    pos_x = action // grid_num
                    pos_y = action % grid_num
                    l[1] = str(int(pos_x * grid_size[0] + placedb.min_width + random.random() * placedb.cluster_info[cluster_name]['width']))
                    l[2] = str(int(pos_y * grid_size[1] + placedb.min_height + random.random() * placedb.cluster_info[cluster_name]['height']))
                line = '\t'.join(l)
            
            f.write(line)
            f.write('\n')
    print(f"chip placement result has been saved in path: {save_file_path}")

def is_number(s):
    # Match integers, decimals, positive numbers, and negative numbers
    pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)$'
    return bool(re.match(pattern, s))

def read_node_file(fopen, macro_set):
    std_info = {}
    macro_info = {}
    port_info = {}
    macro_cnt, std_cnt = 0, 0
    for line in fopen.readlines():
        line = line.strip().split()
        if (not line) or (not is_number(line[1])):
            continue
        node_name = line[0]
        x = int(line[1])
        y = int(line[2])
        if line[-1] == "terminal_NI":
            port_info[node_name] = {"width": x , "height": y }
        elif node_name in macro_set:
            macro_info[node_name] = {"id": macro_cnt, "width": x , "height": y }
            macro_cnt += 1
        else:
            std_info[node_name] = {"id": std_cnt, "width": x , "height": y }
            std_cnt += 1

    for macro_name in macro_info.keys():
        macro_info[macro_name]["id"] = macro_info[macro_name]["id"] + std_cnt
    print(f"std cell counts:{std_cnt}, macro counts:{macro_cnt}")
        
    return std_info, macro_info, port_info

def read_net_file(fopen, macro_info, std_info, port_info):
    node_info = macro_info | std_info
    node_set = set(node_info.keys())
    port_set = set(port_info.keys())
    net_info = {}
    net_name = None
    net_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith("  ") and \
            not line.startswith("NetDegree"):
            continue
        line = line.strip().split()
        if line[0] == "NetDegree":
            net_name = line[-1]
        else:
            node_name = line[0]
            if node_name in node_info or node_name in port_info:
                if not net_name in net_info:
                    net_info[net_name] = {}
                    net_info[net_name]["nodes"] = {}
                    net_info[net_name]["ports"] = {}
                if node_name in node_set and not node_name in net_info[net_name]["nodes"]:
                    x_offset = float(line[-2])
                    y_offset = float(line[-1])
                    net_info[net_name]["nodes"][node_name] = {}
                    net_info[net_name]["nodes"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
                elif node_name in port_set and not node_name in net_info[net_name]["ports"]:
                    x_offset = float(line[-2])
                    y_offset = float(line[-1])
                    net_info[net_name]["ports"][node_name] = {}
                    net_info[net_name]["ports"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]['id'] = net_cnt
        net_cnt += 1
    print("adjust net size = {}".format(len(net_info)))
    return net_info

def read_macro_net_file(fopen, macro_info, port_info):
    node_info = macro_info
    node_set = set(node_info.keys())
    port_set = set(port_info.keys())
    net_info = {}
    net_name = None
    net_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith("  ") and \
            not line.startswith("NetDegree"):
            continue
        line = line.strip().split()
        if line[0] == "NetDegree":
            net_name = line[-1]
        else:
            node_name = line[0]
            if node_name in node_info:
                if not net_name in net_info:
                    net_info[net_name] = {}
                    net_info[net_name]["nodes"] = {}
                    net_info[net_name]["ports"] = {}
                if node_name in node_set and not node_name in net_info[net_name]["nodes"]:
                    x_offset = float(line[-2])
                    y_offset = float(line[-1])
                    net_info[net_name]["nodes"][node_name] = {}
                    net_info[net_name]["nodes"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
                elif node_name in port_set and not node_name in net_info[net_name]["ports"]:
                    x_offset = float(line[-2])
                    y_offset = float(line[-1])
                    net_info[net_name]["ports"][node_name] = {}
                    net_info[net_name]["ports"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]['id'] = net_cnt
        net_cnt += 1
    print("adjust net size = {}".format(len(net_info)))
    return net_info



def get_node_to_net_dict(node_info, net_info):
    node_to_net_dict = {}
    for node_name in node_info:
        node_to_net_dict[node_name] = set()
    for net_name in net_info:
        for node_name in net_info[net_name]["nodes"]:
            node_to_net_dict[node_name].add(net_name)
    return node_to_net_dict


def get_port_to_net_dict(port_info, net_info):
    port_to_net_dict = {}
    for port_name in port_info:
        port_to_net_dict[port_name] = set()
    for net_name in net_info:
        for port_name in net_info[net_name]["ports"]:
            port_to_net_dict[port_name].add(net_name)
    return port_to_net_dict


def read_pl_file(fopen):
    pl_info = {}
    for line in fopen.readlines():
        line = line.strip().split()
        if not line or (not is_number(line[1])):
            continue
        node_name = line[0]
        place_x = int(line[1])
        place_y = int(line[2])
        if node_name not in pl_info:
            pl_info[node_name] = {}
        pl_info[node_name]["x"] = place_x
        pl_info[node_name]["y"] = place_y

    return pl_info

def read_scl_file(fopen):
    context = fopen.read()
    scl_pattern = r'NumRows\s+:\s+\d+'
    match = re.search(scl_pattern, context)
    NumRows = int(match.group().split()[-1])
    match = re.search(r'Height\s+:\s+\d+', context)
    Height = int(match.group().split()[-1])
    
    match = re.search(r'Sitewidth\s+:\s+\d+', context)
    Sitewidth = int(match.group().split()[-1])
    match = re.search(r'NumSites\s+:\s+\d+', context)
    NumSites = int(match.group().split()[-1])

    
    min_height = math.inf
    min_width  = math.inf
    for line in context.splitlines():
        if "Coordinate" in line:
            line = line.strip().split()
            min_height = min(min_height, int(line[-1]))
        elif "SubrowOrigin" in line:
            line = line.strip().split()
            min_width = min(min_width, int(line[2]))

    max_height = NumRows * Height + min_height
    max_width = NumSites * Sitewidth + min_width
    
    print(f"min_height: {min_height}, max_height: {max_height}, min_width: {min_width}, max_width: {max_width}")
    return max_height, max_width, min_height, min_width


def get_pin_cnt(net_info):
    pin_cnt = 0
    for net_name in net_info:
        pin_cnt += len(net_info[net_name]["nodes"])
    return pin_cnt


def get_total_area(node_info):
    area = 0
    for node_name in node_info:
        area += node_info[node_name]["height"] * node_info[node_name]["height"]
    return area

class clusterCell():
    def __init__(self, index):
        self.index = index
        self.stdCells = []
        self.cluster_area = 0
    
    def append(self, cell, size_x, size_y):
        self.stdCells.append(cell)
        self.cluster_area += size_x * size_y

def cluster(benchmark, std_info, macro_info, port_info, net_info, algorithm="shmetis", num_clusters=64, debug=False):
    verbose = False
    ubfactor = 5
    index2Std = {}
    size_info = std_info | macro_info
    std_count = len(std_info)
    for std_name in std_info.keys():
        index2Std[std_info[std_name]["id"]] = std_name
    
    # Cluster std cells using hmetis
    std_set = set(std_info.keys())
    hypeGraph = []
    for net_name in net_info.keys():
        nodes = net_info[net_name]["nodes"]
        net = ()
        for node_name in nodes.keys():
            if node_name in std_set:
                tmpIndex = std_info[node_name]["id"] + 1
                net += (tmpIndex,)
        if len(net) > 1:
            hypeGraph.append(net)
    hypeGraph = set(hypeGraph)
    
    # write to temp file
    temp_dir = os.path.join("temp", benchmark)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    salt = np.random.randint(1e12)
    hmetis_input_file = os.path.join(temp_dir, f'hypeGraph-{salt}.graph')
    with open(hmetis_input_file, 'w') as f:
        f.write(f'{len(hypeGraph)} {len(std_info.keys())}\n')
        for net in hypeGraph:
            f.write(' '.join(map(str, net)) + '\n')
    
    if not os.path.exists("./shmetis"):
        print("The file ./shmetis does not exist. Please download ./shmetis first and try again.")
        exit()
    if algorithm == "shmetis":
        result = subprocess.run([
            './shmetis', 
            hmetis_input_file, 
            str(num_clusters), 
            str(int(ubfactor)),
            ], capture_output = not verbose, text=True)
        
        if debug:
            print("return code:", result.returncode)
            print("std out:", result.stdout)
            print("std error:", result.stderr)
    
    
    hmetis_output_file = f'{hmetis_input_file}.part.{num_clusters}'
    # Read the results and reorder the indices of all cells such that the indices of std cells come first and the indices of macros come later.
    cells = [clusterCell(i) for i in range(num_clusters)]
    stdName2Cluster = {}
    with open(hmetis_output_file, 'r') as f:
        # i: 0, 1, 2, n-1
        for i, line in enumerate(f.readlines()):
            stdCellName = index2Std[i]
            std_size = [size_info[stdCellName]["width"], size_info[stdCellName]["height"]]
            clusterIndex = int(line.strip())
            cells[clusterIndex].append(i, std_size[0], std_size[1])
            stdName2Cluster[stdCellName] = clusterIndex

    # Remove intermediate files
    print(f"remove temp file {hmetis_input_file}, {hmetis_output_file}")
    os.remove(hmetis_input_file)
    os.remove(hmetis_output_file)
    
    # merge macros and standard cell clusters
    x_scale, y_scale = 1., 1.
    cluster_info = {}
    for i, cluster in enumerate(cells):
        cluster_info[i] = {}
        cluster_info[i]['width'] = math.sqrt(cluster.cluster_area)*x_scale
        cluster_info[i]['height'] = math.sqrt(cluster.cluster_area)*y_scale
    for macro_name in macro_info.keys():
        macro_size = [size_info[macro_name]["width"], size_info[macro_name]["height"]]
        # macro_id = macro_info[macro_name]["id"]
        cluster_info[macro_name] = {}
        cluster_info[macro_name]['width'] = macro_size[0]
        cluster_info[macro_name]['height'] = macro_size[1]
    
    # Merge macros. Adjust the cluster net info based on the clustering results.
    cluster_net_info = {}
    for net_name in net_info.keys():
        nodes = net_info[net_name]["nodes"]
        ports = net_info[net_name]["ports"]
        new_nodes = {}
        new_ports = {}
        for node_name in nodes.keys():
            if node_name in stdName2Cluster:
                cluster_index = stdName2Cluster[node_name]
                if cluster_index not in new_nodes:
                    new_nodes[cluster_index] = {"x_offset": 0, "y_offset": 0}
            elif node_name in macro_info:
                # cluster_index = macro_info[node_name]["id"]
                if node_name not in new_nodes:
                    new_nodes[node_name] = {"x_offset": nodes[node_name]["x_offset"], "y_offset": nodes[node_name]["y_offset"]}
        for port_name in ports.keys():
            if port_name in stdName2Cluster:
                cluster_index = stdName2Cluster[port_name]
                if cluster_index not in new_ports:
                    new_ports[cluster_index] = {"x_offset": 0, "y_offset": 0}
                new_ports[cluster_index]["x_offset"] += ports[port_name]["x_offset"]
                new_ports[cluster_index]["y_offset"] += ports[port_name]["y_offset"]
        if len(new_nodes) > 1:
            cluster_net_info[net_name] = {}
            cluster_net_info[net_name]["nodes"] = new_nodes
            cluster_net_info[net_name]["ports"] = new_ports
    
    
    return cells, stdName2Cluster, cluster_info, cluster_net_info


class PlaceDB_ispd():
    def __init__(self, benchmark):
        self.benchmark = benchmark
        assert os.path.exists(os.path.join("benchmarks/ispd2005", benchmark))
        # read all macros
        self.macro_set = self.FilterMacro()
        
        node_file = open(os.path.join("benchmarks/ispd2005", benchmark, benchmark+".nodes"), "r")
        self.std_info, self.macro_info, self.port_info = read_node_file(node_file, self.macro_set)
        node_file.close()
        self.node_info = self.macro_info
        
        net_file = open(os.path.join("benchmarks/ispd2005", benchmark, benchmark+".nets"), "r")
        self.net_info = read_macro_net_file(net_file, self.macro_info,  self.port_info)
        self.net_cnt = len(self.net_info)
        net_file.close()
        self.max_height, self.max_width, self.min_height, self.min_width = \
            self.read_boundary()
        print("bounadry:", self.max_height, self.max_width, self.min_height, self.min_width)

        self.port_to_net_dict = get_port_to_net_dict(self.port_info, self.net_info)
        
        self.cluster_info, self.cluster_net_info = self.macro_info, self.net_info
        self.stdName2Cluster = {}
        self.std_info = {}
            
        self.node_to_net_dict = get_node_to_net_dict(self.cluster_info, self.cluster_net_info)

    def FilterMacro(self):
        # read all macros
        node_info = {}
        with open(f"benchmarks/ispd2005/{self.benchmark}/{self.benchmark}.nodes", "r") as f:
            for line in f:
                line = line.strip().split() 
                if line and line[-1] == "terminal":
                    node_info[line[0]] = {'x': int(line[1]), 'y': int(line[2])}
                    
        node_id_ls = list(node_info.keys()).copy()
        if self.benchmark == "bigblue2" or self.benchmark == "bigblue4":
            node_area = {}
            for node_id in node_id_ls:
                node_area[node_id] = node_info[node_id]["x"] * node_info[node_id]["y"]
            node_id_ls.sort(key = lambda x: - node_area[x])
            node_id_ls = node_id_ls[:256] if self.benchmark == "bigblue2" else node_id_ls[:1024]
        print(f"benchmark: {self.benchmark}, macro counts: {len(node_id_ls)}")
        
        return set(node_id_ls)
        
    
    def read_boundary(self):
        max_height = -math.inf
        max_width = -math.inf
        min_height = math.inf
        min_width = math.inf
        node_info = {}
        with open(f"benchmarks/ispd2005/{self.benchmark}/{self.benchmark}.nodes") as fopen:
            for line in fopen.readlines():
                line = line.strip().split()
                if (not line) or line[-1] != "terminal":
                    continue
                node_name = line[0]
                x = int(line[1])
                y = int(line[2])
                node_info[node_name] = {"width": x , "height": y }
            
        with open(f"benchmarks/ispd2005/{self.benchmark}/{self.benchmark}.pl") as fopen:
            for line in fopen.readlines():
                if not line.startswith('o'):
                    continue
                line = line.strip().split()
                if (not line) or (not line[0] in node_info):
                    continue
                node_name = line[0]
                place_x = int(line[1])
                place_y = int(line[2])
                max_height = max(max_height, node_info[node_name]["width"] + place_x)
                max_width = max(max_width, node_info[node_name]["height"] + place_y)
                min_height = min(min_height, place_x)
                min_width = min(min_width, place_y)
                node_info[node_name]["raw_x"] = place_x
                node_info[node_name]["raw_y"] = place_y

        return max(max_height, max_width), max(max_height, max_width), min(min_height, min_width), min(min_height, min_width)
    
    
# First, read all the pl and nets files. Then, use hmetis for hypergraph partitioning. 
# Finally, convert the partitioned hypergraph back into a macro placement task, with a one-to-one correspondence for ID mapping.
class PlaceDB_chipbench():
    def __init__(self, benchmark, num_cluster=64):
        self.benchmark = benchmark
        assert os.path.exists(os.path.join("benchmarks/ChiPBench", benchmark))
        self.macro_set = self.read_macros(benchmark)
        
        pl_file = open(os.path.join("benchmarks/ChiPBench", benchmark, benchmark+".pl"), "r")
        self.pl_info = read_pl_file(pl_file)
        pl_file.close()
        node_file = open(os.path.join("benchmarks/ChiPBench", benchmark, benchmark+".nodes"), "r")
        self.std_info, self.macro_info, self.port_info = read_node_file(node_file, self.macro_set)
        self.node_info = self.std_info | self.macro_info
        node_file.close()
        net_file = open(os.path.join("benchmarks/ChiPBench", benchmark, benchmark+".nets"), "r")
        self.net_info = read_net_file(net_file, self.macro_info, self.std_info,  self.port_info)
        self.net_cnt = len(self.net_info)
        net_file.close()
        scl_file = open(os.path.join("benchmarks/ChiPBench", benchmark, benchmark+".scl"), "r")
        self.max_height, self.max_width, self.min_height, self.min_width = read_scl_file(scl_file)
        scl_file.close()
        
        self.cells, self.stdName2Cluster, self.cluster_info, self.cluster_net_info = \
            cluster(self.benchmark, self.std_info, self.macro_info, self.port_info, self.net_info, algorithm="shmetis", num_clusters=num_cluster)
        
        self.port_to_net_dict = get_port_to_net_dict(self.port_info, self.cluster_net_info)
        self.node_to_net_dict = get_node_to_net_dict(self.cluster_info, self.cluster_net_info)
    
    def read_macros(self, dataset):
        macro_set = set()
        macro_file_path = os.path.join("benchmarks/ChiPBench", dataset, dataset+".macros")
        with open(macro_file_path, "r") as f:
            for line in f:
                line = line.strip()
                l = line.split()
                if line and line == "0":
                    continue
                macro_set = set(l) | macro_set
        return macro_set

    def debug_str(self):
        print("node_cnt = {}".format(len(self.node_info)))
        print("net_cnt = {}".format(len(self.net_info)))
        print("max_height = {}".format(self.max_height))
        print("max_width = {}".format(self.max_width))
        print("min_height = {}".format(self.min_height))
        print("min_width = {}".format(self.min_width))
        print("pin_cnt = {}".format(get_pin_cnt(self.net_info)))
        print("port_cnt = {}".format(len(self.port_info)))
        print("area_ratio = {}".format(get_total_area(self.node_info)/(self.max_height*self.max_height)))
        #print("node_info:", self.node_info)
        #print("net_info:", self.net_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--dataset', default='ariane133')
    args = parser.parse_args()
    dataset = args.dataset
    placedb = PlaceDB_chipbench(dataset)
    placedb.debug_str()
    
