# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
from typing import Dict, List, Union

import numpy as np
import torch

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import model_learnable_to_dxo, ModelLearnable
from nvflare.app_common.aggregators.assembler import Assembler
from nvflare.app_common.app_constant import AppConstants
from collections import deque, defaultdict


class TreeNode:
    def __init__(self, site, parent=None, update_freq=5):
        self.site = site
        self.parent = parent
        self.children: List[TreeNode] = []
        self.update_step = 0
        self.update_freq = update_freq

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.site) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def is_leaf(self):
        return len(self.children) <= 0


class DependenceTree:
    def __init__(self, assemble_map):
        self.assemble_map = assemble_map
        self.root = "root"
        self.sites = []
        self.site2level = {}
        self.site2node = {}
        self.root_node = self._add_node(self.root, parent=None, level=0)
        self.build_tree()

    def _add_node(self, site: str, parent: TreeNode = None, level: int = 0) -> TreeNode:
        node = TreeNode(site)
        if parent is not None:
            parent.children.append(node)
        self.site2node[site] = node
        self.site2level[site] = level
        return node

    def _get_node(self, site: str) -> TreeNode:
        return self.site2node[site]

    def _build(self, site_list: Union[List, str], parent: TreeNode, level: int = 0) -> TreeNode:
        if isinstance(site_list, str):
            this_node = self._add_node(site_list, parent=parent, level=level)
            return this_node
        count = 0
        while f"parent_level{level}_{count}" in self.site2node:
            count += 1
        name = f"parent_level{level}_{count}"
        this_node = self._add_node(name, parent=parent, level=level)
        for site_group in site_list:
            # this_node.children.append(self._build(site_group, parent, level=level + 1))
            self._build(site_group, this_node, level=level + 1)
        return this_node

    def build_tree(self):
        for site_group in self.assemble_map:
            self._build(site_group, self.root_node, level=1)

    def traverse_leaf_to_node(self):
        leaf2root = sorted(self.site2level.items(), key=lambda x: -x[1])
        return [self._get_node(item[0]) for item in leaf2root]


class HierarchicalAssembler(Assembler):
    def __init__(self, assemble_map):
        super().__init__(data_kind=DataKind.WEIGHTS)
        self.assemble_map = assemble_map
        self.dependence_tree = DependenceTree(assemble_map)

    def get_model_params(self, dxo: DXO):
        data = dxo.data
        # return {data[site] for site in self.dependence_tree.sites}
        return data

    def _agg_subset(self, data: Dict[str, dict],  fl_ctx: FLContext) -> dict:
        global_weights = data[list(data.keys())[0]].copy()
        for name in global_weights:
            # global_weights[name] = np.zeros_like(torch.as_tensor(global_weights[name]))
            global_weights[name] = np.zeros_like(global_weights[name])

        for site in data:
            for name in global_weights:
                # weight = torch.as_tensor(data[site][name])
                weight = data[site][name]
                global_weights[name] += weight / len(data)

        for name in global_weights:
            global_weights[name] = global_weights[name]
        return global_weights

    def assemble(self, data: Dict[str, dict], fl_ctx: FLContext) -> DXO:
        results_weights = copy.deepcopy(data)
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        leaf2root: List[TreeNode] = self.dependence_tree.traverse_leaf_to_node()
        for i in range(len(leaf2root)):
            node = leaf2root[i]
            node.update_step += 1
            # if the node is a leaf node, we do not need to touch it
            if node.is_leaf():
                continue
            # # only aggregate and update the parent with a frequency
            # if node.update_step % node.update_freq != 0:
            #     continue
            # if the node is a non-leaf node, we need to aggregate its children nodes and propogate these weights back to the leaf nodes
            children = node.children
            children_sites = [child.site for child in children]
            weights_subset = {k: v for k, v in data.items() if k in children_sites}
            avg_weights = self._agg_subset(weights_subset, fl_ctx)
            results_weights[node.site] = avg_weights
        assert len(results_weights) == len(leaf2root), f"results weights keys: {str(list(results_weights.keys()))}; leaf2node keys: {[n.site for n in leaf2root]}"
        dxo = DXO(data_kind=self.expected_data_kind, data=results_weights[self.dependence_tree.root])
        for k in results_weights:
            if k == self.dependence_tree.root:
                continue
            dxo.set_data_prop(k, results_weights[k])
        return dxo
