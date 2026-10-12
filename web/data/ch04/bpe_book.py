"""原书的 BPE 代码（浏览器里的验证单元直接 import 这个文件）。

逐行取自 third_party/dlfs6，只有两处机械改动：
  1. 去掉 tqdm 进度条（浏览器里没有终端）；
  2. 不同版本的同名函数加了后缀：
       pretokenize_v1 / count_pairs_v1 / train_bpe_naive / BPETokenizerV1  <- codebot/tokenizer.py（第 1 章）
       train_bpe_dedup <- ch04/01_bpe_optimize.py    train_bpe_cache <- ch04/02_bpe_cache.py
       train_bpe_chunk <- ch04/03_bpe_chunk.py       其余          <- storybot/tokenizer.py
由 verify/_ch04_data.py 生成；verify/test_ch04.py::test_bpe_book_is_verbatim 检查它与原书逐个语法节点相同。
"""
import os
import pickle
import shutil
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import regex as re


# --- codebot/tokenizer.py :: pretokenize  (改名为 pretokenize_v1)
def pretokenize_v1(text):
    pattern = "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    return re.findall(pattern, text)


# --- codebot/tokenizer.py :: count_pairs  (改名为 count_pairs_v1)
def count_pairs_v1(ids, counts=None):
    if counts is None:
        counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


# --- codebot/tokenizer.py :: train_bpe  (改名为 train_bpe_naive)
def train_bpe_naive(input_text, vocab_size, end_token='<|endoftext|>'):
    texts = input_text.split(end_token)
    ids_list = []
    for text in texts:
        for pretoken in pretokenize_v1(text):
            ids_list.append(list(pretoken.encode('utf-8')))
    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    for step in range(num_merges):
        counts = defaultdict(int)
        for ids in ids_list:
            counts = count_pairs_v1(ids, counts)
        if not counts:
            break
        best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id
        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], best_pair, new_id)
    return merge_rules


# --- codebot/tokenizer.py :: BPETokenizer  (改名为 BPETokenizerV1)
class BPETokenizerV1:

    def __init__(self, merge_rules, end_token='<|endoftext|>'):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        self.id_to_bytes[self.end_token_id] = self.end_token.encode('utf-8')
        self.vocab_size = len(self.id_to_bytes)

    @staticmethod
    def load_from(filepath):
        with open(filepath, 'rb') as f:
            merge_rules = pickle.load(f)
        return BPETokenizerV1(merge_rules)

    def _encode_text(self, text):
        ids = list(text.encode('utf-8'))
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)
        return ids

    def encode(self, input_text, show_progress=False):
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []
        texts = texts if show_progress else texts
        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                for pretoken in pretokenize_v1(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)
        return all_ids

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b''.join(byte_list)
        text = text_bytes.decode('utf-8', errors='replace')
        return text


# --- storybot/tokenizer.py :: pretokenize
def pretokenize(text):
    pattern = "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    for m in re.finditer(pattern, text):
        yield m.group(0)


# --- storybot/tokenizer.py :: count_pairs
def count_pairs(ids, weight=1, counts=None):
    if counts is None:
        counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts


# --- storybot/tokenizer.py :: merge
def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids


# --- ch04/01_bpe_optimize.py :: train_bpe  (改名为 train_bpe_dedup)
def train_bpe_dedup(input_text, vocab_size, end_token='<|endoftext|>'):
    texts = input_text.split(end_token)
    pretoken_counts = defaultdict(int)
    for text in texts:
        for pretoken in pretokenize(text):
            pretoken_counts[pretoken] += 1
    ids_counts = {tuple(pretoken.encode('utf-8')): count for pretoken, count in pretoken_counts.items()}
    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    for step in range(num_merges):
        pair_counts = defaultdict(int)
        for ids, count in ids_counts.items():
            count_pairs(ids, count, pair_counts)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id
        new_ids_counts = defaultdict(int)
        for ids, count in ids_counts.items():
            new_ids = merge(ids, best_pair, new_id)
            new_ids_counts[tuple(new_ids)] += count
        ids_counts = new_ids_counts
    return merge_rules


# --- ch04/02_bpe_cache.py :: train_bpe  (改名为 train_bpe_cache)
def train_bpe_cache(input_text, vocab_size, end_token='<|endoftext|>'):
    texts = input_text.split(end_token)
    pretoken_counts = defaultdict(int)
    for text in texts:
        for pretoken in pretokenize(text):
            pretoken_counts[pretoken] += 1
    ids_counts = {tuple(pretoken.encode('utf-8')): count for pretoken, count in pretoken_counts.items()}
    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    pair_to_ids = defaultdict(set)
    pair_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        count_pairs(ids, count, pair_counts)
        for pair in zip(ids, ids[1:]):
            pair_to_ids[pair].add(ids)
    for step in range(num_merges):
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id
        affected_ids = pair_to_ids[best_pair]
        del pair_to_ids[best_pair]
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
    return merge_rules


# --- storybot/tokenizer.py :: find_chunk_boundaries
def find_chunk_boundaries(file_path, num_chunks, end_token='<|endoftext|>'):
    byte_end_token = end_token.encode('utf-8')
    with open(file_path, 'rb') as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        chunk_size = file_size // num_chunks
        chunk_boundaries = [i * chunk_size for i in range(num_chunks)]
        chunk_boundaries.append(file_size)
        buffer_size = 4096
        for bi in range(1, len(chunk_boundaries) - 1):
            chunk_position = chunk_boundaries[bi]
            file.seek(chunk_position)
            while True:
                buffer = file.read(buffer_size)
                if buffer == b'':
                    chunk_boundaries[bi] = file_size
                    break
                end_position = buffer.find(byte_end_token)
                if end_position != -1:
                    chunk_boundaries[bi] = chunk_position + end_position
                    break
                chunk_position += buffer_size
    return sorted(set(chunk_boundaries))


# --- ch04/03_bpe_chunk.py :: train_bpe  (改名为 train_bpe_chunk)
def train_bpe_chunk(file_path, vocab_size, end_token='<|endoftext|>'):
    chunk_boundaries = find_chunk_boundaries(file_path, num_chunks=64)
    pretoken_counts = defaultdict(int)
    with open(file_path, 'rb') as f:
        total_chunks = len(chunk_boundaries) - 1
        for i in range(total_chunks):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i + 1]
            f.seek(start)
            chunk_byte = f.read(end - start)
            chunk_text = chunk_byte.decode('utf-8', errors='ignore')
            texts = chunk_text.split(end_token)
            for text in texts:
                for pretoken in pretokenize(text):
                    pretoken_counts[pretoken] += 1
    ids_counts = {tuple(pretoken.encode('utf-8')): count for pretoken, count in pretoken_counts.items()}
    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    pair_to_ids = defaultdict(set)
    pair_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        count_pairs(ids, count, pair_counts)
        for pair in zip(ids, ids[1:]):
            pair_to_ids[pair].add(ids)
    for step in range(num_merges):
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id
        affected_ids = pair_to_ids[best_pair]
        del pair_to_ids[best_pair]
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
    return merge_rules


# --- storybot/tokenizer.py :: pretoken_chunk
def pretoken_chunk(args):
    file_path, start, end, end_token = args
    pretoken_counts = defaultdict(int)
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode('utf-8', errors='ignore')
        texts = chunk_text.split(end_token)
        for text in texts:
            for pretoken in pretokenize(text):
                pretoken_counts[pretoken] += 1
    return pretoken_counts


# --- storybot/tokenizer.py :: BPETokenizer
class BPETokenizer:

    def __init__(self, merge_rules, end_token='<|endoftext|>'):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        self.id_to_bytes[self.end_token_id] = self.end_token.encode('utf-8')
        self.vocab_size = len(self.id_to_bytes)

    @staticmethod
    def load_from(filepath):
        with open(filepath, 'rb') as f:
            merge_rules = pickle.load(f)
        return BPETokenizer(merge_rules)

    def _encode_text(self, text):
        ids = list(text.encode('utf-8'))

        def get_merge_priority(pair):
            return self.merge_rules.get(pair, float('inf'))
        while len(ids) > 1:
            counts = count_pairs(ids)
            best_pair = min(counts, key=get_merge_priority)
            if best_pair not in self.merge_rules:
                break
            new_id = self.merge_rules[best_pair]
            ids = merge(ids, best_pair, new_id)
        return ids

    def encode(self, input_text, show_progress=False):
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []
        texts = texts if show_progress else texts
        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                for pretoken in pretokenize(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)
        return all_ids

    def _encode_chunk(self, args):
        """チャンクを処理してディスクにキャッシュ"""
        file_path, start, end, cache_dir, chunk_idx = args
        with open(file_path, 'rb') as f:
            f.seek(start)
            chunk_byte = f.read(end - start)
            chunk_text = chunk_byte.decode('utf-8', errors='ignore')
            ids = self.encode(chunk_text)
        cache_file = os.path.join(cache_dir, f'chunk_{chunk_idx:05d}.npy')
        np.array(ids, dtype=np.uint16).tofile(cache_file)
        return (cache_file, len(ids))

    def encode_file(self, file_path, output_file, num_processes=4, num_chunks=64, cache_dir='bpe_cache'):
        os.makedirs(cache_dir, exist_ok=True)
        try:
            chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
            total_chunks = len(chunk_boundaries) - 1
            chunk_info_list = []
            for i in range(total_chunks):
                start = chunk_boundaries[i]
                end = chunk_boundaries[i + 1]
                chunk_info_list.append((file_path, start, end, cache_dir, i))
            with Pool(processes=num_processes) as pool:
                cache_results = list(pool.imap(self._encode_chunk, chunk_info_list))
            cache_files = [r[0] for r in cache_results]
            token_counts = [r[1] for r in cache_results]
            total_tokens = sum(token_counts)
            dtype = np.uint16
            arr = np.memmap(output_file, dtype=dtype, mode='w+', shape=(total_tokens,))
            idx = 0
            for cache_file in cache_files:
                chunk_data = np.fromfile(cache_file, dtype=dtype)
                arr[idx:idx + len(chunk_data)] = chunk_data
                idx += len(chunk_data)
            arr.flush()
            del arr
        finally:
            shutil.rmtree(cache_dir)
        return total_tokens

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b''.join(byte_list)
        text = text_bytes.decode('utf-8', errors='replace')
        return text
