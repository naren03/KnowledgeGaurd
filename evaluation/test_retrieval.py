import unittest

import numpy as np

from retrieval import retrieve_chunks


class FakeEmbeddingModel:
    def __init__(self):
        self.normalize_embeddings = None

    def encode(self, questions, normalize_embeddings):
        self.normalize_embeddings = normalize_embeddings
        return [[1.0, 0.0]]


class FakeIndex:
    def __init__(self, ranked_indexes):
        self.ranked_indexes = ranked_indexes
        self.ntotal = len(ranked_indexes)
        self.searched_k = None

    def search(self, vector, k):
        self.searched_k = k
        return np.zeros((1, k), dtype="float32"), np.asarray(
            [self.ranked_indexes[:k]], dtype="int64"
        )


def make_chunk(source, groups, content=None):
    return {
        "content": content or f"content from {source}",
        "metadata": {"source_path": source, "allowed_groups": groups},
    }


class RetrieveChunksTests(unittest.TestCase):
    def setUp(self):
        self.embedding_model = FakeEmbeddingModel()

    def test_skips_unauthorized_ranked_chunks_and_finds_authorized(self):
        chunks = [
            make_chunk("docs/private-budget.xlsx", ["Management"]),
            make_chunk("docs/engineering-guide.pdf", ["Engineering"]),
        ]
        index = FakeIndex([0, 1])

        selected = retrieve_chunks(
            "engineering question", self.embedding_model, index, chunks, ["Engineering"]
        )

        self.assertEqual(
            [chunk["metadata"]["source_path"] for chunk in selected],
            ["docs/engineering-guide.pdf"],
        )
        self.assertNotIn("private-budget", str(selected))
        self.assertTrue(self.embedding_model.normalize_embeddings)
        self.assertEqual(index.searched_k, index.ntotal)

    def test_missing_or_malformed_permissions_fail_closed(self):
        chunks = [
            {"content": "no metadata"},
            {"content": "bad groups", "metadata": {"allowed_groups": "HR"}},
            make_chunk("docs/valid.txt", ["HR"]),
        ]
        selected = retrieve_chunks(
            "question", self.embedding_model, FakeIndex([0, 1, 2]), chunks, ["HR"]
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["metadata"]["source_path"], "docs/valid.txt")

    def test_empty_user_groups_return_no_chunks(self):
        chunks = [make_chunk("docs/shared.txt", ["HR", "Engineering"])]

        selected = retrieve_chunks(
            "question", self.embedding_model, FakeIndex([0]), chunks, []
        )

        self.assertEqual(selected, [])

    def test_skips_invalid_index_results_and_caps_authorized_chunks(self):
        chunks = [
            make_chunk(f"docs/engineering-{i}.txt", ["Engineering"])
            for i in range(6)
        ]
        index = FakeIndex([-1, 9, 0, 1, 2, 3, 4, 5])

        selected = retrieve_chunks(
            "question", self.embedding_model, index, chunks, ["Engineering"], limit=4
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(
            [chunk["metadata"]["source_path"] for chunk in selected],
            [f"docs/engineering-{i}.txt" for i in range(4)],
        )

    def test_empty_index_and_nonpositive_limit_do_not_encode(self):
        empty_index = FakeIndex([])
        self.assertEqual(
            retrieve_chunks("question", self.embedding_model, empty_index, [], ["HR"]),
            [],
        )
        self.assertEqual(
            retrieve_chunks(
                "question",
                self.embedding_model,
                FakeIndex([0]),
                [make_chunk("docs/shared.txt", ["HR"])],
                ["HR"],
                limit=0,
            ),
            [],
        )
        self.assertIsNone(self.embedding_model.normalize_embeddings)


if __name__ == "__main__":
    unittest.main()