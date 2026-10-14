"""本教程的扩展代码（不是原书代码）：第 4 章用它做对照实验，看清“为什么结果不变”和“时间花在哪里”。

  train_full         ch04/01 的 merge 循环（每步全量重算），但“挑哪个 pair”可以换
  train_incremental  ch04/02 的 merge 循环（增量更新），同样可以换挑选规则，还能顺便记几个计数器
  BOOK_KEY / GET_KEY / BYTES_KEY
                     三种挑选规则：原书的 (count, id0, id1)；第 1 章早期的 max(key=counts.get)；
                     CS336 规定的 (count, bytes0, bytes1)
  train_bpe_heap     ch04/02 + 惰性删除的最大堆（代替每步一次的 max() 全表扫描）
  MemoTokenizer      给 storybot 的 BPETokenizer 加一个“预分词 -> ids”的缓存
用 BOOK_KEY 时，train_full / train_incremental / train_bpe_heap 与原书的 train_bpe 逐项相同，
MemoTokenizer 与 BPETokenizer 逐 id 相同；verify/test_ch04.py 负责检查。
"""
import heapq
from collections import defaultdict

from bpe_book import BPETokenizer, count_pairs, merge, pretokenize

BOOK_KEY = lambda pc, vocab: max(pc, key=lambda pair: (pc[pair], pair[0], pair[1]))        # noqa: E731
GET_KEY = lambda pc, vocab: max(pc, key=pc.get)                                           # noqa: E731
BYTES_KEY = lambda pc, vocab: max(pc, key=lambda pair: (pc[pair], vocab[pair[0]], vocab[pair[1]]))  # noqa: E731


def ids_counts_of(input_text, end_token="<|endoftext|>"):
    """ch04/01-02 的预分词与去重（逐行相同）：{预分词的字节元组: 出现次数}。"""
    pretoken_counts = defaultdict(int)
    for text in input_text.split(end_token):
        for pretoken in pretokenize(text):
            pretoken_counts[pretoken] += 1
    return {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}


def init_cache(ids_counts):
    """ch04/02 的初始化：全量数一次 pair，同时建好倒排索引 pair -> 含有它的序列。"""
    pair_to_ids = defaultdict(set)
    pair_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        count_pairs(ids, count, pair_counts)
        for pair in zip(ids, ids[1:]):
            pair_to_ids[pair].add(ids)
    return pair_counts, pair_to_ids


def apply_merge(ids_counts, pair_counts, pair_to_ids, best_pair, new_id, changed=None):
    """ch04/02 循环体里“只更新受影响的序列”那一段（逐行相同）。
    返回 (动过的唯一序列数, 改动 pair 计数的次数)。"""
    affected_ids = pair_to_ids[best_pair]
    del pair_to_ids[best_pair]
    updates = 0
    for ids in affected_ids:
        ids_count = ids_counts[tuple(ids)]
        new_ids = merge(ids, best_pair, new_id)
        del ids_counts[tuple(ids)]
        ids_counts[tuple(new_ids)] = ids_count
        old_counts = count_pairs(ids)
        for pair, count in old_counts.items():
            pair_counts[pair] -= count * ids_count
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
            pair_to_ids[pair].discard(tuple(ids))
        new_counts = count_pairs(new_ids)
        for pair, count in new_counts.items():
            pair_counts[pair] += count * ids_count
            pair_to_ids[pair].add(tuple(new_ids))
        updates += len(old_counts) + len(new_counts)
        if changed is not None:
            changed.update(old_counts)
            changed.update(new_counts)
    return len(affected_ids), updates


def _record(vocab, merge_rules, best_pair, new_id):
    merge_rules[best_pair] = new_id
    vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]


def train_full(input_text, vocab_size, select=BOOK_KEY, end_token="<|endoftext|>"):
    """ch04/01 的循环：每一步都把所有唯一序列重新数一遍、合并一遍。"""
    ids_counts = ids_counts_of(input_text, end_token)
    vocab = {i: bytes([i]) for i in range(256)}
    merge_rules = {}
    for step in range(vocab_size - 256 - 1):
        pair_counts = defaultdict(int)
        for ids, count in ids_counts.items():
            count_pairs(ids, count, pair_counts)
        if not pair_counts:
            break
        best_pair = select(pair_counts, vocab)
        _record(vocab, merge_rules, best_pair, 256 + step)
        new_ids_counts = defaultdict(int)
        for ids, count in ids_counts.items():
            new_ids_counts[tuple(merge(ids, best_pair, 256 + step))] += count
        ids_counts = new_ids_counts
    return merge_rules


def train_incremental(input_text, vocab_size, select=BOOK_KEY, end_token="<|endoftext|>", stats=None):
    """ch04/02 的循环。传入 stats={} 时额外记录：每步 max() 要看的 pair 数（scanned）、
    动过的唯一序列数（affected）、改动计数的次数（updates）、并列第一的 pair（ties）。"""
    ids_counts = ids_counts_of(input_text, end_token)
    pair_counts, pair_to_ids = init_cache(ids_counts)
    vocab = {i: bytes([i]) for i in range(256)}
    merge_rules = {}
    if stats is not None:
        stats.update(unique=len(ids_counts), scanned=[], affected=[], updates=[], ties={})
    for step in range(vocab_size - 256 - 1):
        if not pair_counts:
            break
        best_pair = select(pair_counts, vocab)
        if stats is not None:
            stats["scanned"].append(len(pair_counts))
            tied = [p for p, c in pair_counts.items() if c == pair_counts[best_pair]]
            if len(tied) > 1:
                stats["ties"][step] = tied
        _record(vocab, merge_rules, best_pair, 256 + step)
        n_affected, n_updates = apply_merge(ids_counts, pair_counts, pair_to_ids, best_pair, 256 + step)
        if stats is not None:
            stats["affected"].append(n_affected)
            stats["updates"].append(n_updates)
    return merge_rules


def train_bpe_heap(input_text, vocab_size, end_token="<|endoftext|>"):
    """ch04/02 + 惰性最大堆。堆里放 (-count, -id0, -id1)：弹出顺序与原书的 key (count, id0, id1) 完全一致。
    计数变了不去改堆里的旧条目，只在每一步结束时为变过的 pair 压入新条目；弹出时计数对不上就丢掉。"""
    return heap_merge_loop(ids_counts_of(input_text, end_token), vocab_size - 256 - 1)


def heap_merge_loop(ids_counts, num_merges):
    """train_bpe_heap 的合并循环部分（预分词之后）。ids_counts 会被修改。"""
    pair_counts, pair_to_ids = init_cache(ids_counts)
    heap = [(-c, -a, -b) for (a, b), c in pair_counts.items()]
    heapq.heapify(heap)
    merge_rules = {}
    for step in range(num_merges):
        while heap and pair_counts.get((-heap[0][1], -heap[0][2])) != -heap[0][0]:
            heapq.heappop(heap)                              # 过期条目：这个 pair 的计数已经变了
        if not heap:
            break
        _, a, b = heap[0]
        best_pair = (-a, -b)
        merge_rules[best_pair] = 256 + step
        changed = set()
        apply_merge(ids_counts, pair_counts, pair_to_ids, best_pair, 256 + step, changed)
        for pair in changed:
            if pair in pair_counts:
                heapq.heappush(heap, (-pair_counts[pair], -pair[0], -pair[1]))
    return merge_rules


class MemoTokenizer(BPETokenizer):
    """同一个预分词永远编码成同一串 id，所以可以缓存。TinyStories 里不同的预分词只有一万多个。"""

    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        super().__init__(merge_rules, end_token)
        self.memo = {}

    def _encode_text(self, text):
        ids = self.memo.get(text)
        if ids is None:
            ids = self.memo[text] = super()._encode_text(text)
        return ids
