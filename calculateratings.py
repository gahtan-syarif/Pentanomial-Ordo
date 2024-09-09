from collections import defaultdict, deque, Counter
from scipy.optimize import minimize
import subprocess
import os
import re
import io
import numpy as np
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
        percentile_intervals[engine] = (lower_bound, upper_bound)
    
    return percentile_intervals

def parse_pgn(pgn_file_path):
    rounds = defaultdict(list)
    engines = set()
    
    with open(pgn_file_path, 'r') as pgn_file:
        game_data = pgn_file.read()
    
    # Regex to extract game headers and results
    game_pattern = re.compile(
        r'\[Round\s+"([^"]*)"\]\s*'
        r'\[White\s+"([^"]*)"\]\s*'
        r'\[Black\s+"([^"]*)"\]\s*'
        r'\[Result\s+"([^"]*)"\]',
        re.MULTILINE | re.DOTALL
    )
    
    # Find all matches in the PGN data
    matches = game_pattern.findall(game_data)
    
    for match in matches:
        round_tag, white_engine, black_engine, result = match
        rounds[round_tag].append({
            'white': white_engine,
            'black': black_engine,
            'result': result
        })
        engines.add(white_engine)
        engines.add(black_engine)
    
    return rounds, engines
    
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

def simulate_matches(probabilities, engine1, engine2, num_pairs_per_pairing, rng):
    """
    Simulate matches for a specific pair of engines using a local random state.

    :param probabilities: A dictionary with probabilities for each pair.
    :param engine1: The first engine in the pair.
    :param engine2: The second engine in the pair.
    :param num_pairs_per_pairing: Number of simulations to run for the pair.
    :param seed: Optional seed for reproducibility.
    :return: List of outcomes for all simulations.
    """
    outcomes = ['LL', 'LD', 'WLDD', 'WD', 'WW']
    prob = probabilities[engine1][engine2]
    
    # Simulate matches
    results = []
    if num_pairs_per_pairing > 0:
        outcomes_indices = rng.choice(5, size=num_pairs_per_pairing, p=prob)
        results = [outcomes[index] for index in outcomes_indices]
    
    return results
    
def calculate_probabilities(results):
    probabilities = {}
    
    for engine1 in results:
        probabilities[engine1] = {}
        for engine2 in results[engine1]:
            LL, LD, WLDD, WD, WW = results[engine1][engine2]
            total_pairs = LL + LD + WLDD + WD + WW
            if total_pairs == 0:
                # print(f"Warning: No pairs played between {engine1} and {engine2}")
                probabilities[engine1][engine2] = 0
            else:
                probabilities[engine1][engine2] = (LL / total_pairs, LD / total_pairs, WLDD / total_pairs, WD / total_pairs, WW / total_pairs)
            
    return probabilities

def simulate_tournament(probabilities, engines, rng, results):
    sim_results = {engine: {opponent: (0, 0, 0, 0, 0) for opponent in engines if opponent != engine} for engine in engines}
    
    for i in range(len(engines)):
        for j in range(i + 1, len(engines)):
            engine1 = engines[i]
            engine2 = engines[j]
            LL, LD, WLDD, WD, WW = results[engine1][engine2]
            total_pairs = LL + LD + WLDD + WD + WW
            outcomes = simulate_matches(probabilities, engine1, engine2, total_pairs, rng)
            update_results_batch(sim_results, engine1, engine2, outcomes)
                
    return sim_results

def update_results_batch(results, engine1, engine2, outcomes): 
    # Count occurrences of each outcome
    outcome_counts = Counter(outcomes)
    
    # Extract counts with a default of 0 if the outcome is not present
    LL_count = outcome_counts.get('LL', 0)
    LD_count = outcome_counts.get('LD', 0)
    WLDD_count = outcome_counts.get('WLDD', 0)
    WD_count = outcome_counts.get('WD', 0)
    WW_count = outcome_counts.get('WW', 0)
    
    # Update the results for engine1 vs engine2
    results[engine1][engine2] = (
        results[engine1][engine2][0] + LL_count,
        results[engine1][engine2][1] + LD_count,
        results[engine1][engine2][2] + WLDD_count,
        results[engine1][engine2][3] + WD_count,
        results[engine1][engine2][4] + WW_count
    )
    
    # Mirror the result for engine2 vs engine1
    results[engine2][engine1] = (
        results[engine1][engine2][4],
        results[engine1][engine2][3],
        results[engine1][engine2][2],
        results[engine1][engine2][1],
        results[engine1][engine2][0]
    )
        
def update_pentanomial(results, engine1, engine2, pentanomial):
    results[engine1][engine2] = tuple(pentanomial)
    results[engine2][engine1] = tuple(pentanomial[::-1])

def format_ratings_result(ratings_with_error_bars, penta_stats, performance_stats, summed_results, filename):
    pairs_played = {}
    points = {}
    for engine, pentanomial in summed_results.items():
        LL, LD, WLDD, WD, WW = pentanomial
        pairs_played[engine] = LL + LD + WLDD + WD + WW
        points[engine] = 2 * (LD * 0.25 + WLDD * 0.5 + WD * 0.75 + WW)
        
    # Determine the maximum width for each column
    max_engine_length = max(len(engine) for engine in ratings_with_error_bars.keys())
    max_mean_length = max(len(f"{mean_rating:.1f}") for mean_rating, _, _ in ratings_with_error_bars.values())
    max_error_length = max(len(f"{round(mean_rating-lower_bound, 1):.1f}") for mean_rating, lower_bound, _ in ratings_with_error_bars.values())
    max_plus_error_length = max(len(f"{round(upper_bound-mean_rating, 1):.1f}") for mean_rating, _, upper_bound in ratings_with_error_bars.values())
    penta_stats_length = max(len(penta_string) for penta_string in penta_stats.values())
    performance_stats_length = max(len(performance_string) for performance_string in performance_stats.values())
    points_length = max(max(len(f"{individual_points:.1f}") for individual_points in points.values()), len("POINTS"))
    pairs_length = max(max(len(f"{individual_pairs_played}") for individual_pairs_played in pairs_played.values()), len("PAIRS"))
    
    def output_line(line):
        print(line)
        if filename != "":
            try:
                # Resolve the path and open the file in append mode
                with open(Path(filename).resolve(), "a") as file:
                    file.write(line + "\n")
            except IOError as e:
                # Handle file I/O errors
                print(f"Error writing to file {filename}: {e}")
            
    # Define header strings
    headers = [
        "NAME", 
        "ELO", 
        "ERROR",
        "PENTA", 
        "POINTS",
        "PAIRS",
        "(%)"
    ]
    
    output_line("-" * (max_engine_length + max_mean_length + max_error_length + max_plus_error_length + penta_stats_length + performance_stats_length + points_length + pairs_length + 12))
    output_line(f"{headers[0]:^{max_engine_length}}  {headers[1]:^{max_mean_length}} {headers[2]:^{max_error_length + max_plus_error_length + 5}} {headers[3]:^{penta_stats_length}} {headers[4]:^{points_length}} {headers[5]:^{pairs_length}} {headers[6]:^{performance_stats_length}}")
    output_line("-" * (max_engine_length + max_mean_length + max_error_length + max_plus_error_length + penta_stats_length + performance_stats_length + points_length + pairs_length + 12))
    # Print each engine's ratings with formatted errors and confidence intervals
    for engine, (mean_rating, lower_bound, upper_bound) in ratings_with_error_bars.items():
        error_down = round(lower_bound - mean_rating, 1)
        error_up = round(upper_bound - mean_rating, 1)
        mean_rating_str = f"{mean_rating:.1f}"
        error_down_str = f"{error_down:.1f}"
        error_up_str = f"{error_up:.1f}"
        interval_str = f"[{lower_bound:.1f}, {upper_bound:.1f}]"
        pairs_str = f"{pairs_played[engine]}"
        points_str = f"{points[engine]:.1f}"

        output_line(f"{engine:<{max_engine_length}}: {mean_rating_str:<{max_mean_length}} ({error_down_str:<{max_error_length}}/+{error_up_str:<{max_plus_error_length}}) {penta_stats[engine]:<{penta_stats_length}} {points_str:<{points_length}} {pairs_str:<{pairs_length}} {performance_stats[engine]:>{performance_stats_length}}")
    output_line("-" * (max_engine_length + max_mean_length + max_error_length + max_plus_error_length + penta_stats_length + performance_stats_length + points_length + pairs_length + 12))

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
                scores[engine1][engine2] = 0
            else:
                scores[engine1][engine2] = (LD * 0.25 + WLDD * 0.5 + WD * 0.75 + WW) / total_pairs
    
    return scores

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
    num_engines = len(engines)
    ratings = ratings_array.reshape(num_engines, 1)
    
    # Compute the difference matrix
    rating_diff = ratings.T - ratings
    
    # Compute the predicted scores matrix
    predicted_scores = 1 / (1 + 10 ** (rating_diff / 400))
    
    # Compute the squared errors matrix
    squared_errors = (predicted_scores - score_matrix) ** 2
    
    # Create a mask to exclude perfect scores
    mask = (score_matrix != 0) & (score_matrix != 1)
    
    # Sum the squared errors where mask is True
    total_error = np.sum(squared_errors[mask])
    
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
        options={'gtol': 1e-8}  # Increased precision
    )
    # Check if the result converged
    if not result.success:
        print("Warning: one of the simulations did not converge properly")
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
    
def sum_all_results(results):
    # Create a new dictionary to store the summed results
    summed_results = {}
    
    # Iterate over each engine in the results dictionary
    for engine, opponents in results.items():
        # Initialize a tuple to hold the sum of tuples for this engine
        summed_tuple = (0, 0, 0, 0, 0)
        
        # Iterate over each opponent and sum the tuples
        for opponent, stats in opponents.items():
            # Use zip to add the corresponding elements of the tuples
            summed_tuple = tuple(x + y for x, y in zip(summed_tuple, stats))
        
        # Store the summed tuple in the new dictionary
        summed_results[engine] = summed_tuple
    
    return summed_results
    
def calculate_initial_ratings(summed_results):
    initial_rating = {}
    for engine, pentanomial in summed_results.items():
        LL, LD, WLDD, WD, WW = pentanomial
        performance = (LD * 0.25 + WLDD * 0.5 + WD * 0.75 + WW) / (LL + LD + WLDD + WD + WW)
        if performance == 0:
            initial_rating[engine] = -800
        elif performance == 1:
            initial_rating[engine] = 800
        else:
            initial_rating[engine] = -400 * np.log10(1 / performance - 1)
    return initial_rating
    
def run_simulation(i, probabilities, engines, seed, results, average, anchor, initial_ratings):
    # Each process gets its own RNG with a different seed
    rng = np.random.default_rng(seed + i)
    simulated_results = simulate_tournament(probabilities, engines, rng, results)
    simulated_scores = calculate_expected_scores(simulated_results)
    return i, optimize_elo_ratings(engines, simulated_scores, initial_ratings, average, anchor)
    
def format_penta_stats(summed_results):
    penta_stats = {}
    performance_stats = {}
    for engine, pentanomial in summed_results.items():
        LL, LD, WLDD, WD, WW = pentanomial
        performance = (LD * 0.25 + WLDD * 0.5 + WD * 0.75 + WW) / (LL + LD + WLDD + WD + WW)
        penta_stats[engine] = f"[{LL}, {LD}, {WLDD}, {WD}, {WW}]"
        performance_stats[engine] = f"{round(performance * 100, 1)}%"
    return penta_stats, performance_stats
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgnfile', type=str, nargs='+', required=True)
    parser.add_argument('--simulations', type=int, default=1000)
    parser.add_argument('--average', type=float, default=2300)
    parser.add_argument('--anchor', type=str, default="")
    parser.add_argument('--output', type=str, default="")
    parser.add_argument('--rngseed', type=int, default=42)
    parser.add_argument('--concurrency', type=int, default=os.cpu_count())
    args = parser.parse_args()
    script_start_time = time.time()
    
    # engines = ['AlphaZero', 'Stockfish', 'Leela']
    # results = {engine: {opponent: (0, 0, 0, 0, 0) for opponent in engines if opponent != engine} for engine in engines}
    # update_pentanomial(results, 'AlphaZero', 'Stockfish', [24, 1, 28, 12, 35])
    # update_pentanomial(results, 'AlphaZero', 'Leela', [6, 4, 2, 85, 3])
    # update_pentanomial(results, 'Leela', 'Stockfish', [5, 64, 26, 3, 2])
    
    print("Loading PGN...")
    individual_rounds = []
    engines_set = set()
    for pgnfile in args.pgnfile:
        filepath = Path(pgnfile).resolve()
        file_rounds, file_engines = parse_pgn(filepath)
        engines_set.update(file_engines)
        individual_rounds.append(file_rounds)
    engines = list(engines_set)
    engines.sort()
    results = {engine: {opponent: (0, 0, 0, 0, 0) for opponent in engines if opponent != engine} for engine in engines}
    for rounds in individual_rounds:
        update_game_pairs_pgn(results, rounds)


    # Calculate probabilities
    scores = calculate_expected_scores(results)
    summed_results = sum_all_results(results)
    initial_ratings = calculate_initial_ratings(summed_results)
    mean_rating = optimize_elo_ratings(engines, scores, initial_ratings, args.average, args.anchor)
    probabilities = calculate_probabilities(results)

    # Simulate the tournament
    num_simulations = args.simulations
    simulated_ratings = {}
    print("Commencing simulation...")
    simulation_start_time = time.time()
    with ProcessPoolExecutor(max_workers = args.concurrency) as executor:
        # Pass a different seed to each process
        futures = [
            executor.submit(run_simulation, i, probabilities, engines, args.rngseed, results, args.average, args.anchor, initial_ratings)
            for i in range(num_simulations)
        ]
        for future in as_completed(futures):
            i, rating = future.result()
            simulated_ratings[i] = rating
            # print(f"Finished simulation {i+1} out of {num_simulations}")
    simulation_end_time = time.time()
    simulation_elapsed_time = simulation_end_time - simulation_start_time

    print("Calculating margin of error...")
    confidence_intervals = calculate_percentile_intervals(simulated_ratings)

    print("Finalizing results...")
    #combine the two dicts
    ratings_with_error_bars = {}
    for engine in mean_rating:
        mean = mean_rating[engine]
        if engine in confidence_intervals:
            lower_bound, upper_bound = confidence_intervals[engine]
            ratings_with_error_bars[engine] = (mean, lower_bound, upper_bound)

    #print final ratings with confidence intervals
    ratings_with_error_bars = sort_engines_by_mean(ratings_with_error_bars)
    penta_stats, performance_stats = format_penta_stats(summed_results)
    format_ratings_result(ratings_with_error_bars, penta_stats, performance_stats, summed_results, args.output)
    script_end_time = time.time()
    script_elapsed_time = script_end_time - script_start_time
    print(f"Total simulation time: {simulation_elapsed_time:.4f} seconds")
    print(f"Total elapsed time: {script_elapsed_time:.4f} seconds")
        
if __name__ == "__main__":
    main()

