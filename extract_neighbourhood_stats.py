import numpy as np
import pandas as pd
import math
from scipy.stats import linregress, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from utils import *

class BaseNeighbourhoodStatsExtractor:
    def __init__(self):
        self.count_df = pd.DataFrame()
        self.timestep_totals = pd.Series()
        pass

    def fetch_neighbours_cosine(self, word, use_modern_projected):
        pass

    def compute_neighbourhood_stats_cosine(self, word_pair_dict, output_file):
        """
        Computing density and average frequency growth rate for a range of neighbourhoods
        of each neologism and control word
        :param word_pair_dict: neologism - control pair dictionary

        :return:
        """

        if len(word_pair_dict) > 20:
            rows = parallelize_func(list(word_pair_dict.items()), self._neighbourhood_stats_helper, 
                                                                chunksz=50, n_jobs=20)
        else:
            rows = self._neighbourhood_stats_helper(list(word_pair_dict.items()))
        
        df = pd.DataFrame(rows) # type: ignore
        df.to_csv(f"{output_file}", na_rep="NaN", index=False)

        df_density = df[["IsNeologism"] + [f"DensityAtRadius{r:.3f}" for r in COSINE_RADIUS_RANGE]].copy()
        mean_density = df_density.groupby("IsNeologism").mean()
        print(mean_density)

        df_growth_mono = df[["IsNeologism"] + [f"GrowthMonotonicityAtRadius{r:.3f}" for r in COSINE_RADIUS_RANGE]].copy()
        mean_growth_mono = df_growth_mono.groupby("IsNeologism").mean()
        print(mean_growth_mono)

        df_growth_slope = df[["IsNeologism"] + [f"GrowthSlopeAtRadius{r:.3f}" for r in COSINE_RADIUS_RANGE]].copy()
        mean_growth_slope = df_growth_slope.groupby("IsNeologism").mean()
        print(mean_growth_slope)
        
        results = df_growth_slope.groupby("IsNeologism").agg(['mean', 'sem'])
        for row in results.iterrows():
            print(row[0])
            for c in COSINE_RADIUS_RANGE:
                print(f'({c}, {row[1][f"GrowthSlopeAtRadius{c:.3f}"]["mean"]}) +- (0.0, {row[1][f"GrowthSlopeAtRadius{c:.3f}"]["sem"]})')
        
    def _neighbourhood_stats_helper(self, word_pair_list):
        results = []

        keys = list(self.count_df.columns)
        time_steps = list(range(1, len(keys) + 1))

        for neologism, control in word_pair_list:
            try:
                neologism_neighbours_dict = self.fetch_neighbours_cosine(neologism, use_modern_projected=True)
            except KeyError:
                print(f"{neologism} not found in the modern embedding space vocabulary")
                continue

            try:
                control_neighbours_dict = self.fetch_neighbours_cosine(control, use_modern_projected=False)
            except KeyError:
                print(f"{control} not found in the historical embedding space vocabulary")
                continue

            neologism_row = {"Word": neologism, "IsNeologism": 1}
            control_row = {"Word": control, "IsNeologism": 0}

            for r in COSINE_RADIUS_RANGE:
                neologism_row[f"DensityAtRadius{r:.3f}"] = math.log(len(neologism_neighbours_dict[r]) + 1) # type: ignore
                control_row[f"DensityAtRadius{r:.3f}"] = math.log(len(control_neighbours_dict[r]) + 1) # type: ignore

                neologism_neighbours = neologism_neighbours_dict[r] # type: ignore
                control_neighbours = control_neighbours_dict[r] # type: ignore

                # if either neighbourhood is empty, we are not adding it or its counterpart
                if len(neologism_neighbours) == 0 or len(control_neighbours) == 0:
                    neologism_row[f"GrowthSlopeAtRadius{r:.3f}"] = np.nan
                    neologism_row[f"GrowthMonotonicityAtRadius{r:.3f}"] = np.nan
                    control_row[f"GrowthSlopeAtRadius{r:.3f}"] = np.nan
                    control_row[f"GrowthMonotonicityAtRadius{r:.3f}"] = np.nan
                
                else:
                    neologism_neighbourhood_percentages = \
                        [self.count_df.loc[neologism_neighbours, t].sum() / len (neologism_neighbours) /   # type: ignore
                        self.timestep_totals[t] for t in keys]
                    neologism_row[f"GrowthMonotonicityAtRadius{r:.3f}"] = \
                        spearmanr(time_steps, neologism_neighbourhood_percentages).statistic  # type: ignore
                    neologism_row[f"GrowthSlopeAtRadius{r:.3f}"] = \
                        linregress(time_steps, neologism_neighbourhood_percentages)[0]

                    control_neighbourhood_percentages = \
                        [self.count_df.loc[control_neighbours, t].sum() / len (control_neighbours) / # type: ignore
                        self.timestep_totals[t] for t in keys]
                    control_row[f"GrowthMonotonicityAtRadius{r:.3f}"] = \
                        spearmanr(time_steps, control_neighbourhood_percentages).statistic # type: ignore
                    control_row[f"GrowthSlopeAtRadius{r:.3f}"] = \
                        linregress(time_steps, control_neighbourhood_percentages)[0]

            results += [neologism_row, control_row]

        return results


class Word2VecNeighbourhoodStatsExtractor(BaseNeighbourhoodStatsExtractor):
    def __init__(self, model_historical_object, model_modern_projected_object, count_df, timestep_totals):
        """
        Loading and aligning the embedding models
        :param 
        """
        self._word_pairs = {}
        self.count_df = count_df
        self.timestep_totals = timestep_totals

        self.model_historical = model_historical_object
        self.model_modern_projected = model_modern_projected_object

    def fetch_neighbours_cosine(self, word, use_modern_projected=False):
        """
        Retrieving a set of nearest neighbours for of the word using cosine similarity metric,
        removing itself (in case of projection) and non-vocabulary words
        :param word: word to center the neighbourhood around
        :param use_modern_projected: toggles between projected modern embeddings (used is 'word' is a neologism)
        and historical embeddings (used if 'word' is a control word)
        :return: dict {radius: word_list}
        """

        if use_modern_projected:
            similarity_scores = self.model_historical.wv.similar_by_vector(self.model_modern_projected.wv[word], topn=None)
        else:
            similarity_scores = self.model_historical.wv.most_similar(word, topn=None)

        neighbours_by_radius = {}
        for cosine_threshold in COSINE_RADIUS_RANGE:
            neighbour_indices = np.where(similarity_scores >= cosine_threshold)[0]
            selected_sim_scores = similarity_scores[neighbour_indices]
            index_order = np.argsort(selected_sim_scores)[::-1]
            neighbours_by_radius[cosine_threshold] = [self.model_historical.wv.index_to_key[i] 
                for i in neighbour_indices[index_order]
                if self.model_historical.wv.index_to_key[i] != word]
        return neighbours_by_radius     


class RobertaNeighbourhoodStatsExtractor(BaseNeighbourhoodStatsExtractor):
    def __init__(self, embeddings_historical, embeddings_modern, count_df, timestep_totals):
        """
        :param 
        """
        self._word_pairs = {}
        self.count_df = count_df
        self.timestep_totals = timestep_totals

        self.embeddings_historical = embeddings_historical
        self.embeddings_modern = embeddings_modern

    def fetch_neighbours_cosine(self, word, use_modern_projected=False):

        if use_modern_projected:
            target_vector = self.embeddings_modern[word]
        else:
            target_vector = self.embeddings_historical[word]
        
        idx2vocab = list(self.embeddings_historical.keys())
        vocab2idx = {w: i for i, w in enumerate(idx2vocab)}
        similarity_scores = np.zeros(len(idx2vocab))
        for candidate in self.embeddings_historical:
            sim = cosine_similarity(target_vector.reshape(1, -1), self.embeddings_historical[candidate].reshape(1, -1))[0][0]
            similarity_scores[vocab2idx[candidate]] = sim

        neighbours_by_radius = {}
        for cosine_threshold in COSINE_RADIUS_RANGE:
            neighbour_indices = np.where(similarity_scores >= cosine_threshold)[0]
            selected_sim_scores = similarity_scores[neighbour_indices]
            index_order = np.argsort(selected_sim_scores)[::-1]

            neighbours_by_radius[cosine_threshold] = [idx2vocab[i] 
                for i in neighbour_indices[index_order]
                if idx2vocab[i] != word]
            
        return neighbours_by_radius
