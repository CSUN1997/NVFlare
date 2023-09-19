from typing import Dict, List, Union, Set
from collections import deque


class TreeNode:
    def __init__(self, site, parent=None, update_freq=5):
        self.site = site
        self.children: List[str] = []
        self.update_step = 0
        self.update_freq = update_freq

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.site) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def is_leaf(self):
        return len(self.children) > 0


class DependeceTree:
    def __init__(self, assemble_map):
        self.assemble_map = assemble_map
        self.root = "root"
        self.root_node = TreeNode(self.root)
        self.site2node = {self.root: self.root_node}
        self.site2level = {self.root: 0}
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


if __name__ == "__main__":
    tree = DependeceTree(["site-1", ["site-2", "site-3"]])
    print(tree.root_node)
    print(tree.site2level)
    print(tree.traverse_leaf_to_node())