"""
SimpleBPE - 一个用于教学演示的迷你 BPE 分词器。
完整复刻 BPE 的核心思路：统计相邻 token 对频率，反复合并最高频的对。
"""
import re
from collections import Counter


class SimpleBPE:
    def __init__(self):
        self.merges = {}       # (a, b) -> merge_rank
        self.vocab = {}        # token -> id
        self.inverse_vocab = {}
        self.special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    def _words_to_tokens(self, texts):
        """把文本切成初始的"字符级 token 序列"，词边界用 </w> 标记。"""
        words = Counter()
        for text in texts:
            for word in re.findall(r"\S+", text):
                words[" ".join(list(word)) + " </w>"] += 1
        return words

    @staticmethod
    def _get_pairs(word_tokens):
        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i + 1]))
        return pairs

    def train(self, texts, vocab_size=300, verbose=False):
        """在语料上训练 BPE，学到 merge 规则与词表。"""
        word_freqs = self._words_to_tokens(texts)     # {"字 符 </w>": freq}
        base_vocab = set()
        for w in word_freqs:
            base_vocab.update(w.split())
        for st in self.special_tokens:
            base_vocab.add(st)

        merges = {}
        # 反复合并最高频相邻对，直到词表达到目标大小
        while len(base_vocab) < vocab_size:
            pair_freqs = Counter()
            for word_str, freq in word_freqs.items():
                tokens = word_str.split()
                for pair in self._get_pairs(tokens):
                    pair_freqs[pair] += freq
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best_pair] < 2:
                break
            merges[best_pair] = len(merges)
            new_token = best_pair[0] + best_pair[1]
            base_vocab.add(new_token)

            # 应用这次合并到所有词
            new_word_freqs = {}
            for word_str, freq in word_freqs.items():
                tokens = word_str.split()
                out, i = [], 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        out.append(new_token)
                        i += 2
                    else:
                        out.append(tokens[i])
                        i += 1
                new_word_freqs[" ".join(out)] = freq
            word_freqs = new_word_freqs

        self.merges = merges
        # 构建词表：特殊 token 占 0-3，其余按字母序
        self.vocab = {st: i for i, st in enumerate(self.special_tokens)}
        for token in sorted(base_vocab - set(self.special_tokens)):
            self.vocab[token] = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        if verbose:
            print(f"[BPE] vocab_size={len(self.vocab)}, merges={len(self.merges)}")
        return self

    def _bpe(self, word):
        tokens = list(word) + ["</w>"]
        while True:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            merged = best[0] + best[1]
            out, i = [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                    out.append(merged); i += 2
                else:
                    out.append(tokens[i]); i += 1
            tokens = out
        return tokens

    def encode(self, text):
        """返回 (tokens, ids, trace)。trace 记录切分细节供前端展示。"""
        tokens = []
        for word in re.findall(r"\S+", text):
            tokens.extend(self._bpe(word))
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        # 学习到的合并规则（用于展示"哪些相邻 token 本可被合并"）
        applied = []
        for a, b in sorted(self.merges, key=self.merges.get)[:8]:
            if a + b in self.vocab:
                applied.append((a, b, a + b))
        return tokens, ids, applied

    def decode(self, ids):
        s = "".join(self.inverse_vocab.get(i, "<UNK>") for i in ids)
        return s.replace("</w>", " ").strip()

    def info(self):
        return {
            "vocab_size": len(self.vocab),
            "merge_rules": len(self.merges),
            "special_tokens": self.special_tokens,
        }
