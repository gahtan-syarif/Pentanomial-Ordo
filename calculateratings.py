import random
import chess.pgn
from collections import defaultdict, deque
from scipy.optimize import minimize
import subprocess
import os
import re
import io
import numpy as np
import argparse
import time

did_not_converge = False

def calculate_percentile_intervals(engine_ratings, percentile=95):
    """
    Calculate the percentile-based confidence intervals for the ratings of each engine based on multiple simulations.

    :param engine_ratings: Dictionary where keys are simulation indices and values are dictionaries of engine ratings.
    :param percentile: Percentile level for the interval (default is 95).
    :return: Dictionary with engines as keys and tuples of (mean, lower_bound, upper_bound) as values.
    """
    # Collect all ratings for each engine
    all_ratings = {}
    
    for sim_id, ratings in engine_ratings.items():
        for engine, rating in ratings.items():
            if engine not in all_ratings:
                all_ratings[engine] = []
            all_ratings[engine].append(rating)
    
    percentile_intervals = {}
    
    # Calculate percentiles for each engine
    for engine, ratings in all_ratings.items():
        lower_bound = np.percentile(ratings, (100 - percentile) / 2)
        upper_bound = np.percentile(ratings, 100 - (100 - percentile) / 2)
        #mean_rating = np.mean(ratings)
        percentile_intervals[engine] = (round(lower_bound, 1), round(upper_bound, 1))
    
    return percentile_intervals

def parse_pgn(pgn_file_path):
    rounds = defaultdict(list)
    engines = set()
    
    with open(pgn_file_path, 'r') as pgn_file:
        while True:
            pgn = chess.pgn.read_game(pgn_file)
            if pgn is None:
                break
            round_tag = pgn.headers.get('Round', 'Unknown')
            white_engine = pgn.headers.get('White', 'Unknown')
            black_engine = pgn.headers.get('Black', 'Unknown')
            result = pgn.headers.get('Result', 'Unknown')

            rounds[round_tag].append({
                'white': white_engine,
                'black': black_engine,
                'result': result
            })
            engines.add(white_engine)
            engines.add(black_engine)
    
    return rounds, list(engines)

def update_game_pairs_pgn(results, rounds):
    """
    Update the count of games played between each pair of engines based on games within the same round.
    
    :param results: Dictionary to store game counts.
    :param rounds: Dictionary with round tags as keys and lists of game results as values.
    """
    for round_tag, games in rounds.items():
        # Group games by the engine pair (white, black)
        round_results = defaultdict(list)
        
        for game in games:
            white = game['white']
            black = game['black']
            result = game['result']
            
            # Store results for each game in the current round
            if white != black:  # Ensure it's a valid pair
                round_results[(white, black)].append(result)

        # Process results for each engine pair
        for (engine1, engine2), results_list in round_results.items():
            results_list_opponent = round_results.get((engine2, engine1), [])
            result1 = results_list[0]
            result2 = results_list_opponent[0]
            if result1 == '1-0' and result2 == '0-1':
                results[engine1][engine2] = (results[engine1][engine2][0], results[engine1][engine2][1], results[engine1][engine2][2], results[engine1][engine2][3], results[engine1][engine2][4] + 1)
            elif (result1 == '1-0' and result2 == '1/2-1/2') or (result1 == '1/2-1/2' and result2 == '0-1'):
                results[engine1][engine2] = (results[engine1][engine2][0], results[engine1][engine2][1], results[engine1][engine2][2], results[engine1][engine2][3] + 1, results[engine1][engine2][4])
            elif (result1 == '1/2-1/2' and result2 == '1/2-1/2') or (result1 == '1-0' and result2 == '1-0') or (result1 == '0-1' and result2 == '0-1'):
                results[engine1][engine2] = (results[engine1][engine2][0], results[engine1][engine2][1], results[engine1][engine2][2] + 1, results[engine1][engine2][3], results[engine1][engine2][4])
            elif (result1 == '1/2-1/2' and result2 == '1-0') or (result1 == '0-1' and result2 == '1/2-1/2'):
                results[engine1][engine2] = (results[engine1][engine2][0], results[engine1][engine2][1] + 1, results[engine1][engine2][2], results[engine1][engine2][3], results[engine1][engine2][4])
            elif result1 == '0-1' and result2 == '1-0':
                results[engine1][engine2] = (results[engine1][engine2][0] + 1, results[engine1][engine2][1], results[engine1][engine2][2], results[engine1][engine2][3], results[engine1][engine2][4])

def simulate_match(probabilities, engine1, engine2):
    LL_prob, LD_prob, WLDD_prob, WD_prob, WW_prob = probabilities[engine1][engine2]
    outcome = random.choices(['LL', 'LD', 'WLDD', 'WD', 'WW'], [LL_prob, LD_prob, WLDD_prob, WD_prob, WW_prob])[0]
    return outcome
    
def calculate_probabilities(results):
    probabilities = {}
    
    for engine1 in results:
        probabilities[engine1] = {}
        for engine2 in results[engine1]:
            LL, LD, WLDD, WD, WW = results[engine1][engine2]
            total_pairs = LL + LD + WLDD + WD + WW
            
            if total_pairs == 0:
                # If no games have been played between these engines, assume equal probability
                probabilities[engine1][engine2] = (1/5, 1/5, 1/5, 1/5, 1/5)
            else:
                LL_prob = LL / total_pairs
                LD_prob = LD / total_pairs
                WLDD_prob = WLDD / total_pairs
                WD_prob = WD / total_pairs
                WW_prob = WW / total_pairs
                probabilities[engine1][engine2] = (LL_prob, LD_prob, WLDD_prob, WD_prob, WW_prob)
    
    return probabilities

def simulate_tournament(probabilities, engines, num_pairs_per_pairing):
    sim_results = {engine: {opponent: (0, 0, 0, 0, 0) for opponent in engines if opponent != engine} for engine in engines}
    
    for _ in range(num_pairs_per_pairing):
        for i in range(len(engines)):
            for j in range(i + 1, len(engines)):
                engine1 = engines[i]
                engine2 = engines[j]
                
                outcome = simulate_match(probabilities, engine1, engine2)
                update_results(sim_results, engine1, engine2, outcome)
    
    return sim_results

def update_results(results, engine1, engine2, outcome):
    if outcome == 'LL':
        results[engine1][engine2] = (results[engine1][engine2][0] + 1, results[engine1][engine2][1], results[engine1][engine2][2], results[engine1][engine2][3], results[engine1][engine2][4])
        results[engine2][engine1] = (results[engine2][engine1][0], results[engine2][engine1][1], results[engine2][engine1][2], results[engine2][engine1][3], results[engine2][engine1][4] + 1)
    elif outcome == 'LD':
        results[engine1][engine2] = (results[engine1][engine2][0], results[engine1][engine2][1] + 1, results[engine1][engine2][2], results[engine1][engine2][3], results[engine1][engine2][4])
        results[engine2][engine1] = (results[engine2][engine1][0], results[engine2][engine1][1], results[engine2][engine1][2], results[engine2][engine1][3] + 1, results[engine2][engine1][4])
    elif outcome == 'WLDD':
        results[engine1][engine2] = (results[engine1][engine2][0], results[engine1][engine2][1], results[engine1][engine2][2] + 1, results[engine1][engine2][3], results[engine1][engine2][4])
        results[engine2][engine1] = (results[engine2][engine1][0], results[engine2][engine1][1], results[engine2][engine1][2] + 1, results[engine2][engine1][3], results[engine2][engine1][4])
    elif outcome == 'WD':
        results[engine1][engine2] = (results[engine1][engine2][0], results[engine1][engine2][1], results[engine1][engine2][2], results[engine1][engine2][3] + 1, results[engine1][engine2][4])
        results[engine2][engine1] = (results[engine2][engine1][0], results[engine2][engine1][1] + 1, results[engine2][engine1][2], results[engine2][engine1][3], results[engine2][engine1][4])
    elif outcome == 'WW':
        results[engine1][engine2] = (results[engine1][engine2][0], results[engine1][engine2][1], results[engine1][engine2][2], results[engine1][engine2][3], results[engine1][engine2][4] + 1)
        results[engine2][engine1] = (results[engine2][engine1][0] + 1, results[engine2][engine1][1], results[engine2][engine1][2], results[engine2][engine1][3], results[engine2][engine1][4])
    else:
        raise ValueError("Result must be 'LL', 'LD', 'WLDD', 'WD', or 'WW'")
    
def update_pentanomial(results, engine1, engine2, pentanomial):
    results[engine1][engine2] = tuple(pentanomial)
    results[engine2][engine1] = tuple(pentanomial[::-1])

def format_ratings_result(ratings_with_error_bars):
    # Determine the maximum width for each column
    max_engine_length = max(len(engine) for engine in ratings_with_error_bars.keys())
    max_mean_length = max(len(f"{mean_rating:.1f}") for mean_rating, _, _ in ratings_with_error_bars.values())
    max_error_length = max(len(f"{round(mean_rating-lower_bound, 1):.1f}") for mean_rating, lower_bound, _ in ratings_with_error_bars.values())
    max_plus_error_length = max(len(f"{round(upper_bound-mean_rating, 1):.1f}") for mean_rating, _, upper_bound in ratings_with_error_bars.values())
    max_interval_length = max(len(f"[{lower_bound:.1f}, {upper_bound:.1f}]") for _, lower_bound, upper_bound in ratings_with_error_bars.values())

    print("-" * (max_engine_length + max_mean_length + max_error_length + max_plus_error_length + max_interval_length + 10))

    # Print each engine's ratings with formatted errors and confidence intervals
    for engine, (mean_rating, lower_bound, upper_bound) in ratings_with_error_bars.items():
        error_down = round(lower_bound - mean_rating, 1)
        error_up = round(upper_bound - mean_rating, 1)
        mean_rating_str = f"{mean_rating:.1f}"
        error_down_str = f"{error_down:.1f}"
        error_up_str = f"{error_up:.1f}"
        interval_str = f"[{lower_bound:.1f}, {upper_bound:.1f}]"

        print(f"{engine:<{max_engine_length}}: {mean_rating_str:>{max_mean_length}} ({error_down_str:>{max_error_length}}/+{error_up_str:<{max_plus_error_length}}) {interval_str:>{max_interval_length}}")
    print("-" * (max_engine_length + max_mean_length + max_error_length + max_plus_error_length + max_interval_length + 10))

def sort_engines_by_mean(ratings_with_error_bars):
    """
    Sorts the engines by their mean ratings from highest to lowest.

    :param ratings_with_error_bars: Dictionary where keys are engine names and values are tuples
                                    containing (mean_rating, lower_bound, upper_bound).
    :return: A new dictionary sorted by mean rating in descending order.
    """
    # Convert dictionary to a list of tuples (engine_name, (mean_rating, lower_bound, upper_bound))
    items = list(ratings_with_error_bars.items())
    
    # Sort the list of tuples by the mean_rating (second element in the tuple) in descending order
    sorted_items = sorted(items, key=lambda x: x[1][0], reverse=True)
    
    # Convert sorted list of tuples back to a dictionary
    sorted_dict = dict(sorted_items)
    
    return sorted_dict

def calculate_expected_scores(results):
    scores = {}
    
    for engine1 in results:
        scores[engine1] = {}
        for engine2 in results[engine1]:
            LL, LD, WLDD, WD, WW = results[engine1][engine2]
            total_pairs = LL + LD + WLDD + WD + WW
            
            if total_pairs == 0:
                # If no games have been played between these engines, assume 0.5
                scores[engine1][engine2] = 0.5
            else:
                scores[engine1][engine2] = (LD * 0.25 + WLDD * 0.5 + WD * 0.75 + WW) / total_pairs
    
    return scores
    
def set_initial_ratings(engines, initial_rating):
    initial_ratings = {}
    
    for engine in engines:
        initial_ratings[engine] = initial_rating
    
    return initial_ratings

def elo_probability(RA, RB):
    """ Calculate the probability of engine A winning against engine B based on Elo ratings. """
    return 1 / (1 + 10**((RB - RA) / 400))

def scores_to_matrix(engines, score_dict):
    """ Convert the score dictionary to a matrix form for optimization. """
    num_engines = len(engines)
    score_matrix = np.zeros((num_engines, num_engines))
    for i, engine in enumerate(engines):
        for j, opponent in enumerate(engines):
            if engine != opponent:
                score_matrix[i, j] = score_dict[engine].get(opponent, 0)
    return score_matrix

def ratings_dict_to_array(ratings_dict, engines):
    """ Convert the Elo ratings dictionary to a NumPy array. """
    return np.array([ratings_dict[engine] for engine in engines])

def ratings_array_to_dict(ratings_array, engines):
    """ Convert a NumPy array of Elo ratings back to a dictionary. """
    return {engine: rating for engine, rating in zip(engines, ratings_array)}

def objective_function(ratings_array, engines, score_matrix):
    """ Calculate the objective function (sum of squared errors) for the given Elo ratings. """
    ratings = ratings_array.reshape(len(engines), 1)
    total_error = 0
    for i in range(len(engines)):
        for j in range(len(engines)):
            if i != j:
                predicted_score = elo_probability(ratings[i], ratings[j])
                actual_score = score_matrix[i, j]
                
                # ignore perfect wins and losses as they're infinite elo, making the simulation hard to converge (same as ordo)
                if actual_score != 0 and actual_score != 1:
                    total_error += (predicted_score - actual_score) ** 2
    return total_error

def optimize_elo_ratings(engines, score_dict, initial_ratings_dict, target_mean, anchor_engine):
    """ Optimize Elo ratings to minimize the discrepancy with expected scores. """
    num_engines = len(engines)
    score_matrix = scores_to_matrix(engines, score_dict)
    initial_ratings_array = ratings_dict_to_array(initial_ratings_dict, engines)
    result = minimize(
        objective_function,
        initial_ratings_array,
        args=(engines, score_matrix),
        method='L-BFGS-B',
        bounds=[(0, None)] * num_engines,
        options={'gtol': 1e-12}  # Increased precision
    )
    # Check if the result converged
    global did_not_converge
    if not result.success:
        did_not_converge = True
    optimized_ratings_array = result.x
    
    # Convert optimized ratings array back to dictionary
    optimized_ratings_dict = ratings_array_to_dict(optimized_ratings_array, engines)
    
    # Normalize ratings to target mean
    normalized_ratings_dict = normalize_ratings_dict_to_target(optimized_ratings_dict, target_mean)
    
    # Normalize rating to anchor rating
    if anchor_engine != "":
        normalized_ratings_dict = normalize_ratings_with_anchor(optimized_ratings_dict, anchor_engine, target_mean)
    
    return normalized_ratings_dict
    
def normalize_ratings_dict_to_target(ratings_dict, target_mean):
    """Normalize ratings so that the average rating is equal to the target mean."""
    ratings_array = np.array(list(ratings_dict.values()))
    current_mean = np.mean(ratings_array)
    adjustment_factor = target_mean - current_mean
    return {engine: rating + adjustment_factor for engine, rating in ratings_dict.items()}

def normalize_ratings_with_anchor(ratings_dict, anchor_engine, anchor_rating):
    """
    Normalize the Elo ratings of all engines so that the specified anchor_engine's rating
    is set to anchor_rating and all other ratings are adjusted relative to it.

    Parameters:
    - ratings_dict: Dictionary with engine names as keys and their Elo ratings as values.
    - anchor_engine: The engine whose rating will be used as the anchor.
    - anchor_rating: The constant rating to which the anchor_engine's rating will be set.

    Returns:
    - A dictionary with normalized Elo ratings.
    """
    # Get the rating of the anchor engine
    if anchor_engine not in ratings_dict:
        raise ValueError(f"{anchor_engine} is not in the ratings dictionary.")
    
    anchor_current_rating = ratings_dict[anchor_engine]

    # Calculate the adjustment factor
    adjustment_factor = anchor_rating - anchor_current_rating

    # Create a new dictionary with normalized ratings
    normalized_ratings_dict = {
        engine: rating + adjustment_factor
        for engine, rating in ratings_dict.items()
    }

    return normalized_ratings_dict
    
parser = argparse.ArgumentParser()
parser.add_argument('--pgnfile', type=str, required=True)
parser.add_argument('--simulations', type=int, default=1000)
parser.add_argument('--average', type=float, default=2300)
parser.add_argument('--anchor', type=str, default="")
parser.add_argument('--rngseed', type=int, default=42)
args = parser.parse_args()
random.seed(args.rngseed)

#engines = ['AlphaZero', 'Stockfish', 'Leela']
#update_pentanomial(results, 'AlphaZero', 'Stockfish', [24, 1, 28, 12, 35])
#update_pentanomial(results, 'AlphaZero', 'Leela', [6, 4, 2, 85, 3])
#update_pentanomial(results, 'Leela', 'Stockfish', [5, 64, 26, 3, 2])
print("Parsing PGN... please wait...")
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, args.pgnfile)
rounds, engines = parse_pgn(filename)
engines.sort()
print("Finished parsing PGN, proceeding to calculate results...")
results = {engine: {opponent: (0, 0, 0, 0, 0) for opponent in engines if opponent != engine} for engine in engines}
update_game_pairs_pgn(results, rounds)


# Calculate probabilities
initial_ratings = set_initial_ratings(engines, 0)
scores = calculate_expected_scores(results)
mean_rating = optimize_elo_ratings(engines, scores, initial_ratings, args.average, args.anchor)
probabilities = calculate_probabilities(results)
num_pairs_per_pairing = results[engines[0]][engines[1]][0] + results[engines[0]][engines[1]][1] + results[engines[0]][engines[1]][2] + results[engines[0]][engines[1]][3] + results[engines[0]][engines[1]][4]

# Simulate the tournament
num_simulations = args.simulations
simulated_ratings = {}
print("Starting simulation...")
start_time = time.time()
for i in range(num_simulations): # 1000 is the typical minimum number of bootstrap samples, but the more the better
    simulated_results = simulate_tournament(probabilities, engines, num_pairs_per_pairing)
    simulated_scores = calculate_expected_scores(simulated_results)
    
    simulated_ratings[i] = optimize_elo_ratings(engines, simulated_scores, initial_ratings, args.average, args.anchor)
    print(f"Finished simulation {i+1} out of {num_simulations}")
end_time = time.time()
elapsed_time = end_time - start_time

print("Calculating confidence intervals...")
confidence_intervals = calculate_percentile_intervals(simulated_ratings)

print("Finalizing results...")
#combine the two dicts
ratings_with_error_bars = {}
for engine in mean_rating:
    mean = mean_rating[engine]
    if engine in confidence_intervals:
        lower_bound, upper_bound = confidence_intervals[engine]
        ratings_with_error_bars[engine] = (mean, lower_bound, upper_bound)

print("Sorting results...")
#print final ratings with confidence intervals
ratings_with_error_bars = sort_engines_by_mean(ratings_with_error_bars)
format_ratings_result(ratings_with_error_bars)
print(f"Total Simulation time: {elapsed_time:.4f} seconds")
if did_not_converge == True:
    print("Warning: optimization did not converge properly")