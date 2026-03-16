import numpy as np
from collections import defaultdict
from scipy.sparse import coo_array
from scipy.sparse.csgraph import maximum_bipartite_matching

from utils import *
from nonword_filters import *

class NeologismControlMatcher:
    def __init__(self, historical_model_object, modern_model_projected_object, count_df, 
                 freq_historical_normalized, freq_modern_normalized, timestep_totals):
        self.count_df = count_df
        self.model_historical = historical_model_object
        self.model_modern_projected = modern_model_projected_object
        self.timestep_totals = timestep_totals
        self.freq_historical_normalized = freq_historical_normalized
        self.freq_modern_normalized = freq_modern_normalized

        self.edges = []

    def pair_neologisms_with_controls(self, neologism_list, outfile):
        """
        Collecting a set of control words by pairing each neologism with a control counterpart,
        controlling for overall frequency and word length
        :param frequency_growth_dict: word - frequency growth rate dictionary
        :param neologism_list: list of neologisms to pair
        :param outfile: file path to output neologism - control word pairs
        :return: neologism - control word pair dictionary
        """

        pairs_dict = {}

        candidate_controls_by_length = defaultdict(list)
        candidate_word2idx = {}
        candidate_idx2word = []

        for word in self.model_historical.wv.index_to_key:
            if word in neologism_list:
                continue
            # Additional filtering for COHA tokens
            if not is_all_alpha_num_hyphen(word) or is_all_numbers_and_hyphens(word):
                continue
            candidate_controls_by_length[len(word)].append(word)
            candidate_word2idx[word] = len(candidate_idx2word)
            candidate_idx2word.append(word)

        print(f"Found {len(candidate_idx2word)} candidate control words")
        print(candidate_idx2word)

        # Find all possible matches
        def get_edges(neologism_idx):
            edges = []
            neologism = neologism_list[neologism_idx]
            neologism_len = len(neologism)
            candidate_controls = candidate_controls_by_length[neologism_len]
            # Pairing condition (length)
            for diff in range(1, 3):
                candidate_controls += candidate_controls_by_length[neologism_len + diff]
                candidate_controls += candidate_controls_by_length[neologism_len - diff]
            if not candidate_controls:
                print(f"Failed to pair with a control (length): {neologism}")
                return []

            try:
                neologism_vector = self.model_modern_projected.wv[neologism]
                neologism_rank = (1 + self.model_modern_projected.wv.key_to_index[neologism]) / len(self.model_modern_projected.wv)
            except KeyError:
                print(f"Neologism not in index: {neologism}")
                return []
            
            matched_flag = False
            for control in candidate_controls:
                control_vector = self.model_historical.wv[control]
                control_rank = (1 + self.model_historical.wv.key_to_index[control]) / len(self.model_historical.wv)

                cosine_similarity = np.dot(neologism_vector, control_vector) / \
                    np.linalg.norm(neologism_vector) / np.linalg.norm(control_vector)

                # Pairing conditions (rank, distance)
                if not abs(control_rank - neologism_rank) <= 0.01: 
                    continue
                if not cosine_similarity >= 0.4:
                    continue
                edges.append([neologism_idx, candidate_word2idx[control]])
                matched_flag = True

            if not matched_flag:
                print(f"Failed to pair with a control (rank/distance/frequency): {neologism}")
            return edges

        results = Parallel(n_jobs=50)(delayed(get_edges)(i) for i in range(len(neologism_list)))
        for edges in results:
            self.edges += edges # type: ignore

        data = [1] * len(self.edges)
        coords = list(zip(*self.edges))
        graph = coo_array((data, coords), shape=(len(neologism_list), len(candidate_word2idx)))
        matching = maximum_bipartite_matching(graph.tocsr(), perm_type='column')
        for i, j in enumerate(matching):
            if j != -1:
                pairs_dict[neologism_list[i]] = candidate_idx2word[j]
        print(f"Created {len(pairs_dict)} neologism-control pairs")
        with open(outfile, 'w') as fout:
            fout.write(f"neologism,control\n")
            for neologism in pairs_dict.keys():
                fout.write(f"{neologism},{pairs_dict[neologism]}\n")
        fout.close()
        return pairs_dict
