"""Merkle 包含证明：生成与「仅依据请求体」的离线重算。

本模块是无状态纯函数集合，**不读取数据库**：生成侧（service 层）在同一读事务
内取齐事件与链尾快照后调用 build_inclusion_proof；校验侧（POST /proofs/verify）
仅凭请求体重算叶子、兄弟路径与 snapshot_digest，两侧共用同一套哈希规则，因此
校验请求甚至可以直接回填证明响应。

哈希规则（全部为 SHA-256 的 64 位十六进制摘要）
------------------------------------------------
叶子（绑定 chain_id、seq、event_hash，防止跨链/跨序号替换证明）::

    leaf = SHA256("merkle-leaf:v1:" + sort_keys_json({
        "chain_id": ..., "event_hash": ..., "seq": ...,
    }))

内部节点（左/右子节点均为定长 64 位十六进制，拼接无歧义；域前缀与叶子、
快照分离，防第二原像）::

    node = SHA256("merkle-node:v1:" + left_hex + right_hex)

奇数层末节点复制配对：某一层节点数为奇数时，最后一个节点与自身配对
（right 方向兄弟就是它自己），不引入其他填充值。单叶树（event_count=1）的
merkle_root 即该叶哈希，兄弟路径为空。

链尾快照摘要（把证明锚定到「某条链在某次读事务下的事件集合 + 当前链尾」）::

    snapshot_digest = SHA256("merkle-snapshot:v1:" + sort_keys_json({
        "chain_id": ..., "event_count": ...,
        "merkle_root": ..., "tail_hash": ...,
    }))

批量可选择披露证明（Merkle multiproof）
--------------------------------------
对去重且升序的 1～MAX_BATCH_PROOF 个序号生成**一份** multiproof：证明节点以
坐标 (level, index) 寻址——level 0 为叶子层、index 为该层内位置，根位于
(level=tree_depth(event_count), index=0)。proof_nodes 是「各被选叶子兄弟
路径的并集减去可由被选叶子自行算出的节点」这一最小集合，按 (level, index)
排序且不重复；其中只含节点哈希，不含任何未选择事件的内容。校验侧仅凭请求体
按坐标逐层重建根：每层用已知（被选叶子/已算出的父节点）与证明节点配对上推，
缺失节点、多余节点、重复/越界/冲突坐标、以及为奇数层末节点提供「兄弟」等
不符合自复制规则的证明一律在 proof_nodes 阶段被拒绝。
"""
from __future__ import annotations

from typing import Any, Optional

from .hashing import canonical_json, digest_hex

# 固定域分隔前缀：叶子 / 内部节点 / 链尾快照分属不同域，互不构成第二原像。
LEAF_DOMAIN = "merkle-leaf:v1:"
NODE_DOMAIN = "merkle-node:v1:"
SNAPSHOT_DOMAIN = "merkle-snapshot:v1:"

# 批量证明单次最多选择的序号个数（去重后计）。
MAX_BATCH_PROOF = 20

# 校验失败阶段（首个失败阶段即停止，顺序见 verify_event_proof）。
STAGE_EVENT_COUNT = "event_count"
STAGE_SEQ = "seq"
STAGE_PATH = "path"
STAGE_ROOT = "merkle_root"
STAGE_SNAPSHOT = "snapshot_digest"
# 批量证明特有的失败阶段：证明节点集合不合法（缺失/多余/坐标冲突/违反
# 奇数层自复制规则）。批量校验的阶段顺序为
# event_count -> seq -> proof_nodes -> merkle_root -> snapshot_digest。
STAGE_PROOF_NODES = "proof_nodes"


def leaf_hash(chain_id: str, seq: int, event_hash: str) -> str:
    """叶子哈希：域分隔前缀 + 绑定 chain_id/seq/event_hash 的规范 JSON。"""
    payload = {
        "chain_id": chain_id,
        "event_hash": event_hash,
        "seq": seq,
    }
    return digest_hex(LEAF_DOMAIN + canonical_json(payload))


def internal_hash(left: str, right: str) -> str:
    """内部节点哈希：域分隔前缀 + 左、右子节点定长十六进制摘要拼接。"""
    return digest_hex(f"{NODE_DOMAIN}{left}{right}")


def snapshot_digest(
    chain_id: str, event_count: int, tail_hash: str, merkle_root: str
) -> str:
    """链尾快照摘要：把 root 锚定到 chain_id / 事件数 / 当前链尾。"""
    payload = {
        "chain_id": chain_id,
        "event_count": event_count,
        "merkle_root": merkle_root,
        "tail_hash": tail_hash,
    }
    return digest_hex(SNAPSHOT_DOMAIN + canonical_json(payload))


def tree_depth(event_count: int) -> int:
    """从叶子到根的层数（即兄弟路径长度）：n=1 时为 0。"""
    if event_count < 1:
        raise ValueError("event_count 必须 >= 1")
    return (event_count - 1).bit_length()


def _parent_level(level: list[str]) -> list[str]:
    """上一层：两两配对；奇数层末节点与自身复制配对。"""
    nxt: list[str] = []
    for i in range(0, len(level), 2):
        left = level[i]
        right = level[i + 1] if i + 1 < len(level) else level[i]
        nxt.append(internal_hash(left, right))
    return nxt


def build_inclusion_proof(
    chain_id: str, event_hashes: list[str], seq: int
) -> dict[str, Any]:
    """按当前事件集合（seq 升序的 event_hash 列表）构造包含证明。

    返回 {"event_count", "leaf_hash", "merkle_root", "path"}，path 自叶子向根，
    每项 {"direction": "left"|"right", "sibling_hash": ...}：direction 表示兄弟
    相对当前节点的方向（left=先拼兄弟再拼当前节点）。调用方须保证 seq 在界内、
    事件与链尾取自同一读事务快照。
    """
    event_count = len(event_hashes)
    if event_count < 1:
        raise ValueError("事件集合为空，无法构造 Merkle 证明")
    if not isinstance(seq, int) or not (0 <= seq < event_count):
        raise ValueError(f"seq 越界: {seq}（事件数 {event_count}）")

    level = [
        leaf_hash(chain_id, index, event_hash)
        for index, event_hash in enumerate(event_hashes)
    ]
    target_leaf = level[seq]

    path: list[dict[str, str]] = []
    pos = seq
    while len(level) > 1:
        size = len(level)
        if pos % 2 == 1:
            # 当前节点是右孩子：兄弟在左。
            path.append(
                {"direction": "left", "sibling_hash": level[pos - 1]}
            )
        elif size % 2 == 1 and pos == size - 1:
            # 奇数层末节点：与自身复制配对，兄弟（右）即当前节点。
            path.append(
                {"direction": "right", "sibling_hash": level[pos]}
            )
        else:
            path.append(
                {"direction": "right", "sibling_hash": level[pos + 1]}
            )
        level = _parent_level(level)
        pos //= 2

    return {
        "event_count": event_count,
        "leaf_hash": target_leaf,
        "merkle_root": level[0],
        "path": path,
    }


def evaluate_inclusion(
    leaf: str, seq: int, path: list[tuple[str, str]], event_count: int
) -> tuple[Optional[str], Optional[str]]:
    """仅依据请求体重算 Merkle 包含关系（不访问数据库）。

    返回 (recomputed_root, failed_stage)：成功时 failed_stage 为 None；
    路径结构不合法（层数不符、方向与序号不一致、奇数末位未正确复制配对或在
    非复制位置伪造复制）时阶段为 "path"；event_count < 1 / seq 越界分别为
    "event_count" / "seq"，此时 root 为 None。
    """
    if not isinstance(event_count, int) or event_count < 1:
        return None, STAGE_EVENT_COUNT
    if not isinstance(seq, int) or not (0 <= seq < event_count):
        return None, STAGE_SEQ

    if len(path) != tree_depth(event_count):
        return None, STAGE_PATH

    node = leaf
    pos = seq
    level_size = event_count
    for direction, sibling in path:
        if direction == "left":
            # 兄弟在左意味着当前节点必须是右孩子（pos 为奇数）。
            if pos % 2 != 1:
                return None, STAGE_PATH
            node = internal_hash(sibling, node)
        elif direction == "right":
            if pos % 2 != 0:
                return None, STAGE_PATH
            odd_last = level_size % 2 == 1 and pos == level_size - 1
            if odd_last:
                # 奇数层末节点必须与自身复制配对。
                if sibling != node:
                    return None, STAGE_PATH
            elif sibling == node:
                # 其余位置不允许复制配对（不同节点哈希相同仅在碰撞时发生）。
                return None, STAGE_PATH
            node = internal_hash(node, sibling)
        else:
            return None, STAGE_PATH

        pos //= 2
        level_size = (level_size + 1) // 2

    return node, None


# ---------------------------------------------------------------------------
# 批量可选择披露证明（Merkle multiproof，按 (level, index) 坐标寻址）
# ---------------------------------------------------------------------------


def _level_sizes(event_count: int) -> list[int]:
    """各层节点数：sizes[0] 为叶子层，sizes[-1] == 1 为根层。"""
    sizes = [event_count]
    while sizes[-1] > 1:
        sizes.append((sizes[-1] + 1) // 2)
    return sizes


def _sibling_index(index: int, level_size: int) -> int:
    """配对兄弟的层内位置；奇数层末节点与自身配对（返回其自身）。"""
    if index % 2 == 1:
        return index - 1
    return index + 1 if index + 1 < level_size else index


def build_multi_proof(
    chain_id: str, event_hashes: list[str], seqs: list[int]
) -> dict[str, Any]:
    """为去重且升序的 1～MAX_BATCH_PROOF 个序号构造一份 Merkle multiproof。

    返回 {"event_count", "merkle_root", "proof_nodes"}；proof_nodes 按
    (level, index) 升序且不重复，是「各被选叶子兄弟路径的并集减去可由被选
    叶子自行算出的节点」的最小集合，每项 {"level", "index", "hash"}。
    调用方须保证 seqs 全部在界内、事件与链尾取自同一读事务快照。
    """
    event_count = len(event_hashes)
    if event_count < 1:
        raise ValueError("事件集合为空，无法构造 Merkle 证明")
    if not seqs:
        raise ValueError("至少需要选择一个序号")
    if len(seqs) > MAX_BATCH_PROOF:
        raise ValueError(f"一次最多选择 {MAX_BATCH_PROOF} 个序号")
    if len(set(seqs)) != len(seqs):
        raise ValueError("序号存在重复")
    for seq in seqs:
        if not isinstance(seq, int) or not (0 <= seq < event_count):
            raise ValueError(f"seq 越界: {seq}（事件数 {event_count}）")

    levels = [
        [leaf_hash(chain_id, index, event_hash)
         for index, event_hash in enumerate(event_hashes)]
    ]
    while len(levels[-1]) > 1:
        levels.append(_parent_level(levels[-1]))

    # 逐层收集「已知节点（可由被选叶子算出）的兄弟中、自身不可算出的」坐标。
    proof: dict[tuple[int, int], str] = {}
    known = sorted(set(seqs))
    for level in range(len(levels) - 1):
        size = len(levels[level])
        known_set = set(known)
        parents: set[int] = set()
        for index in known:
            sibling = _sibling_index(index, size)
            if sibling != index and sibling not in known_set:
                proof[(level, sibling)] = levels[level][sibling]
            parents.add(index // 2)
        known = sorted(parents)

    return {
        "event_count": event_count,
        "merkle_root": levels[-1][0],
        "proof_nodes": [
            {"level": level, "index": index, "hash": node_hash}
            for (level, index), node_hash in sorted(proof.items())
        ],
    }


def evaluate_multi_proof(
    chain_id: str,
    leaves: list[tuple[int, str]],
    proof_nodes: list[tuple[int, int, str]],
    event_count: int,
) -> tuple[Optional[str], Optional[str]]:
    """仅依据请求体按节点坐标重建 Merkle 根（不访问数据库）。

    leaves 为 (seq, event_hash) 列表（被选事件声明）；proof_nodes 为
    (level, index, hash) 列表。返回 (recomputed_root, failed_stage)：
    成功时 failed_stage 为 None；event_count < 1 为 "event_count"；序号
    为空 / 超过 MAX_BATCH_PROOF / 重复 / 越界为 "seq"；证明节点缺失、多余、
    坐标重复 / 越界 / 与被选叶子冲突、或违反奇数层末节点自复制规则（为该
    位置提供「兄弟」节点）均为 "proof_nodes"，此时 root 为 None。
    """
    if not isinstance(event_count, int) or event_count < 1:
        return None, STAGE_EVENT_COUNT

    seqs = [seq for seq, _ in leaves]
    if not (1 <= len(seqs) <= MAX_BATCH_PROOF):
        return None, STAGE_SEQ
    if len(set(seqs)) != len(seqs):
        return None, STAGE_SEQ
    if any(
        not isinstance(seq, int) or not (0 <= seq < event_count) for seq in seqs
    ):
        return None, STAGE_SEQ

    sizes = _level_sizes(event_count)
    depth = len(sizes) - 1

    # 坐标合法性：层/位置必须在树形范围内，且坐标不得重复（重复坐标无论
    # 哈希是否一致都视为冲突）。
    coords: dict[tuple[int, int], str] = {}
    for level, index, node_hash in proof_nodes:
        if not isinstance(level, int) or not isinstance(index, int):
            return None, STAGE_PROOF_NODES
        if not (0 <= level <= depth) or not (0 <= index < sizes[level]):
            return None, STAGE_PROOF_NODES
        if (level, index) in coords:
            return None, STAGE_PROOF_NODES
        coords[(level, index)] = node_hash

    # 证明节点不得与被选叶子坐标冲突（被选叶子的哈希由声明重算，不采信证明）。
    if any((0, seq) in coords for seq in seqs):
        return None, STAGE_PROOF_NODES

    # 逐层重建：known 为当前层已可算出的 {index: hash}，每层用已知节点 +
    # 证明节点配对上推；奇数层末节点与自身复制配对，无需也不得有兄弟节点。
    known: dict[int, str] = {
        seq: leaf_hash(chain_id, seq, event_hash) for seq, event_hash in leaves
    }
    used: set[tuple[int, int]] = set()
    for level in range(depth):
        size = sizes[level]
        parents: dict[int, str] = {}
        for index in sorted(known):
            node = known[index]
            sibling = _sibling_index(index, size)
            if sibling == index:
                # 奇数层末节点：与自身复制配对。
                parents[index // 2] = internal_hash(node, node)
                continue
            if sibling in known:
                pair = (known[sibling], node) if sibling < index else (node, known[sibling])
            elif (level, sibling) in coords:
                used.add((level, sibling))
                sibling_hash = coords[(level, sibling)]
                pair = (sibling_hash, node) if sibling < index else (node, sibling_hash)
            else:
                # 缺失节点：无法上推。
                return None, STAGE_PROOF_NODES
            parents[index // 2] = internal_hash(pair[0], pair[1])
        known = parents

    # 多余节点：未被重建过程消费的证明节点（含与可算出坐标冲突的节点）。
    if len(used) != len(coords):
        return None, STAGE_PROOF_NODES

    return known[0], None
