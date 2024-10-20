from collections import defaultdict, deque, Counter
from scipy.optimize import minimize
import subprocess
import os
import re
import sys
import io
import numpy as np
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv

def calculate_percentile_intervals(engine_ratings, percentile=95.0):
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
    
def calculate_los(engine_ratings, ratings_with_error_bars):
    """
    Calculate the proportion of ratings that exceed the ratings of the next engine in the list.

    :param engine_ratings: Dictionary where keys are simulation indices and values are dictionaries of engine ratings.
    :return: Dictionary with engines as keys and proportion of ratings that exceed the next engine's ratings.
    """
    all_ratings = {}
    
    # Collect all ratings for each engine
    for sim_id, ratings in engine_ratings.items():
        for engine, rating in ratings.items():
            if engine not in all_ratings:
                all_ratings[engine] = []
            all_ratings[engine].append(rating)

    sorted_engines = list(ratings_with_error_bars.keys())
    los = {}  
    # Calculate the proportion of ratings exceeding the next engine's ratings
    for i in range(len(sorted_engines) - 1):
        current_engine = sorted_engines[i]
        next_engine = sorted_engines[i + 1]
        
        current_ratings = all_ratings[current_engine]
        next_ratings = all_ratings[next_engine]

        # Count how many ratings of the current engine exceed the ratings of the next engine
        exceed_count = 0
        for j in range(len(current_ratings)):
            if current_ratings[j] > next_ratings[j]:
                exceed_count += 1
        los[current_engine] = exceed_count * 100.0 / len(current_ratings) if current_ratings else 0.0

    # Handle the last engine which has no next engine to compare
    los[sorted_engines[-1]] = 0.0  # No next engine to compare

    return los
    
def parse_pgn(pgn_file_path):
    rounds = defaultdict(list)
    engines = set()
    
    # Regex pattern to extract game headers and results
    game_pattern = re.compile(
        r'\[Round\s+"([^"]*)"\]\s*'
        r'\[White\s+"([^"]*)"\]\s*'
        r'\[Black\s+"([^"]*)"\]\s*'
        r'\[Result\s+"([^"]*)"\]',
        re.MULTILINE | re.DOTALL
    )
    
    # Use a buffer to accumulate relevant lines
    buffer = []
    tags = ("[Round","[White","[Black","[Result")
    with open(pgn_file_path, 'r') as pgn_file:
        for line in pgn_file:
            if line.startswith(tags):
                buffer.append(line)
            elif buffer and not line.startswith("["):  # When encountering a non-header line, process the accumulated buffer
                # Combine buffer lines into a single string
                buffer_data = ''.join(buffer)
                # Apply regex to the accumulated lines
                matches = game_pattern.findall(buffer_data)
                # Process matches
                for match in matches:
                    round_tag, white_engine, black_engine, result = match
                    rounds[round_tag].append({
                        'white': white_engine,
                        'black': black_engine,
                        'result': result
                    })
                    engines.add(white_engine)
                    engines.add(black_engine)
                # Clear the buffer
                buffer = []
    
    # Process any remaining lines in the buffer after finishing the file read
    if buffer:
        buffer_data = ''.join(buffer)
        matches = game_pattern.findall(buffer_data)
        for match in matches:
            round_tag, white_engine, black_engine, result = match
            rounds[round_tag].append({
                'white': white_engine,
                'black': black_engine,
                'result': result
            })
            engines.add(white_engine)
            engines.add(black_engine)
        buffer = []

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
            if white == black:  # Ensure it's a valid pair
                print("Error: PGN contains match between two players/engines with the same name", file=sys.stderr)
                exit(1)
            round_results[(white, black)].append(result)

        # Process results for each engine pair
        for (engine1, engine2), results_list in round_results.items():
            results_list_opponent = round_results.get((engine2, engine1), [])
            if len(results_list) != 1 or len(results_list_opponent) != 1:
                print("Error: PGN 'Round' header tag is incorrectly formatted. Make sure each gamepair has a unique 'Round' header tag.", file=sys.stderr)
                exit(1)
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
            elif result1 == '*' or result2 == '*':
                print("Error: PGN contains an undecided game.", file=sys.stderr)
                exit(1)
            else:
                print("Error: incorrectly formatted 'Result' header tag in PGN", file=sys.stderr)
                exit(1)

def simulate_matches(prob, num_pairs_per_pairing, rng):
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
            LL, LD, WLDD, WD, WW = results[engines[i]][engines[j]]
            total_pairs = LL + LD + WLDD + WD + WW
            outcomes = simulate_matches(probabilities[engines[i]][engines[j]], total_pairs, rng)
            update_results_batch(sim_results, engines[i], engines[j], outcomes)
                
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

def format_ratings_result(ratings_with_error_bars, penta_stats, performance_stats, summed_results, filename, decimal, los):
    pairs_played = {}
    points = {}
    number_of_engines = 0
    for engine, pentanomial in summed_results.items():
        LL, LD, WLDD, WD, WW = pentanomial
        pairs_played[engine] = LL + LD + WLDD + WD + WW
        points[engine] = 2 * (LD * 0.25 + WLDD * 0.5 + WD * 0.75 + WW)
        number_of_engines += 1
        
    def format_error_str(mean_rating, lower_bound, upper_bound):
        error_down_str = ""
        error_up_str = ""
        if mean_rating >= lower_bound:
            error_down_str = f"-{abs(mean_rating - lower_bound):.{decimal}f}"
        else:
            error_down_str = f"+{abs(lower_bound - mean_rating):.{decimal}f}"
            
        if mean_rating <= upper_bound:
            error_up_str = f"+{abs(upper_bound - mean_rating):.{decimal}f}"
        else:
            error_up_str = f"-{abs(mean_rating - upper_bound):.{decimal}f}"
            
        return f"({error_down_str}/{error_up_str})"
        
    # Determine the maximum width for each column
    max_engine_length = max(len(engine) for engine in ratings_with_error_bars.keys())
    max_mean_length = max(len(f"{mean_rating:.{decimal}f}") for mean_rating, _, _ in ratings_with_error_bars.values())
    max_los_length = max(max(len(f"{percent_los:.{decimal}f}") for percent_los in los.values()), len("LOS(%)"))
    penta_stats_length = max(len(penta_string) for penta_string in penta_stats.values())
    performance_stats_length = max(max(len(performance_string) for performance_string in performance_stats.values()), len("(%)"))
    points_length = max(max(len(f"{individual_points:.1f}") for individual_points in points.values()), len("POINTS"))
    pairs_length = max(max(len(f"{individual_pairs_played}") for individual_pairs_played in pairs_played.values()), len("PAIRS"))
    total_error_length = max(len(format_error_str(mean_rating, lower_bound, upper_bound)) for mean_rating, lower_bound, upper_bound in ratings_with_error_bars.values())
    rank_length = max(len(f"{number_of_engines}"), len("RANK"))
    
    def output_line(line):
        print(line)
        if filename != "":
            try:
                # Resolve the path and open the file in append mode
                with open(Path(filename).resolve(), "a") as file:
                    file.write(line + "\n")
            except IOError as e:
                # Handle file I/O errors
                print(f"Error writing to file {filename}: {e}", file=sys.stderr)
                exit(1)
            
    # Define header strings
    headers = [
        "RANK",
        "NAME", 
        "ELO", 
        "ERROR",
        "LOS(%)",
        "PENTANOMIAL", 
        "POINTS",
        "PAIRS",
        "(%)"
    ]
    
    output_line("=" * (rank_length + max_engine_length + max_mean_length + total_error_length + max_los_length + penta_stats_length + performance_stats_length + points_length + pairs_length + 19))
    output_line(f"{headers[0]:<{rank_length}}  {headers[1]:<{max_engine_length}}  :  {headers[2]:>{max_mean_length}}  {headers[3]:<{total_error_length}}  {headers[4]:>{max_los_length}}  {headers[5]:<{penta_stats_length}}  {headers[6]:>{points_length}}  {headers[7]:>{pairs_length}}  {headers[8]:>{performance_stats_length}}")
    output_line("=" * (rank_length + max_engine_length + max_mean_length + total_error_length + max_los_length + penta_stats_length + performance_stats_length + points_length + pairs_length + 19))
    # Print each engine's ratings with formatted errors and confidence intervals
    i = 0
    for engine, (mean_rating, lower_bound, upper_bound) in ratings_with_error_bars.items():
        i += 1
        mean_rating_str = f"{mean_rating:.{decimal}f}"
        error_str = format_error_str(mean_rating, lower_bound, upper_bound)
        interval_str = f"[{lower_bound:.{decimal}f}, {upper_bound:.{decimal}f}]"
        pairs_str = f"{pairs_played[engine]}"
        points_str = f"{points[engine]:.1f}"
        los_str = f"{los[engine]:.{decimal}f}"

        output_line(f"{i:<{rank_length}}  {engine:<{max_engine_length}}  :  {mean_rating_str:>{max_mean_length}}  {error_str:<{total_error_length}}  {los_str:>{max_los_length}}  {penta_stats[engine]:<{penta_stats_length}}  {points_str:>{points_length}}  {pairs_str:>{pairs_length}}  {performance_stats[engine]:>{performance_stats_length}}")
    output_line("=" * (rank_length + max_engine_length + max_mean_length + total_error_length + max_los_length + penta_stats_length + performance_stats_length + points_length + pairs_length + 19))

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

def calculate_expected_scores(results, purge):
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
                
                # regularize perfect scores
                if purge == False:
                    if scores[engine1][engine2] == 0:
                        # scores[engine1][engine2] = ((LD + 1) * 0.25 + WLDD * 0.5 + WD * 0.75 + WW) / total_pairs
                        scores[engine1][engine2] = 1e-15
                    elif scores[engine1][engine2] == 1:
                        # scores[engine1][engine2] = (LD * 0.25 + WLDD * 0.5 + (WD + 1) * 0.75 + (WW - 1)) / total_pairs
                        scores[engine1][engine2] = 1 - 1e-15
    
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
    
def objective_function(ratings_array, num_engines, score_matrix, mask):
    ratings = ratings_array.reshape(num_engines, 1)
    
    # Compute the difference matrix
    rating_diff = ratings.T - ratings
    
    # Compute the predicted scores matrix
    predicted_scores = np.zeros((num_engines, num_engines))
    predicted_scores[mask] = 1 / (1 + 10 ** (rating_diff[mask] / 400))
    
    # We need to clip predicted_scores to avoid log(0)
    predicted_scores = np.clip(predicted_scores, 1e-15, 1 - 1e-15)
    
    # Calculate the binary cross-entropy loss
    cross_entropy_loss = np.mean(- (score_matrix[mask] * np.log(predicted_scores[mask]) + (1 - score_matrix[mask]) * np.log(1 - predicted_scores[mask])))
    
    return cross_entropy_loss
    
def optimize_elo_ratings(engines, score_dict, initial_ratings_dict, target_mean, anchor_engine, poolrelative):
    """ Optimize Elo ratings to minimize the discrepancy with expected scores. """
    num_engines = len(engines)
    score_matrix = scores_to_matrix(engines, score_dict)
    # Create a mask to exclude perfect scores
    mask = (score_matrix != 0) & (score_matrix != 1) & np.triu(np.ones_like(score_matrix, dtype=bool), k=1)
    initial_ratings_array = ratings_dict_to_array(initial_ratings_dict, engines)
    result = minimize(
        objective_function,
        initial_ratings_array,
        args=(num_engines, score_matrix, mask),
        method='L-BFGS-B',
        bounds=[(-np.inf, np.inf)] * num_engines,
        options={'gtol': 1e-8}  # Increased precision
    )
    # Check if the result converged
    # if not result.success:
        # print("Warning: one of the simulations did not converge properly")
    optimized_ratings_array = normalize_ratings_to_target(result.x, target_mean)
    
    # Convert optimized ratings array back to dictionary
    optimized_ratings_dict = ratings_array_to_dict(optimized_ratings_array, engines)
    
    # Normalize rating to anchor rating
    if anchor_engine != "" and not poolrelative:
        optimized_ratings_dict = normalize_ratings_with_anchor(optimized_ratings_dict, anchor_engine, target_mean)
    
    return optimized_ratings_dict
    
def normalize_ratings_to_target(ratings_array, target_mean):
    """Normalize ratings so that the average rating is equal to the target mean."""
    current_mean = np.mean(ratings_array)
    adjustment_factor = target_mean - current_mean
    return ratings_array + adjustment_factor

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
        if (summed_results[engine][4] == (summed_results[engine][0] + summed_results[engine][1] + summed_results[engine][2] + summed_results[engine][3] + summed_results[engine][4])):
            raise ValueError(f"Rating for {engine} cannot be calculated as it won all games. Please remove from the list using --exclude.")
            
        if (summed_results[engine][0] == (summed_results[engine][0] + summed_results[engine][1] + summed_results[engine][2] + summed_results[engine][3] + summed_results[engine][4])):
            raise ValueError(f"Rating for {engine} cannot be calculated as it lost all games. Please remove from the list using --exclude.")
            
    return summed_results
    
def set_initial_ratings(engines):
    initial_rating = {}
    for engine in engines:
        initial_rating[engine] = 0
    return initial_rating
    
def run_simulation(i, probabilities, engines, seed, results, average, anchor, initial_ratings, purge, poolrelative):
    rng = np.random.Generator(np.random.PCG64(seed))
    simulated_results = simulate_tournament(probabilities, engines, rng, results)
    simulated_scores = calculate_expected_scores(simulated_results, purge)
    return i, optimize_elo_ratings(engines, simulated_scores, initial_ratings, average, anchor, poolrelative)
    
def format_penta_stats(summed_results, decimal):
    penta_stats = {}
    performance_stats = {}
    for engine, pentanomial in summed_results.items():
        LL, LD, WLDD, WD, WW = pentanomial
        total_pairs = (LL + LD + WLDD + WD + WW)
        if total_pairs == 0:
            performance = 0
        else:
            performance = (LD * 0.25 + WLDD * 0.5 + WD * 0.75 + WW) / total_pairs
        penta_stats[engine] = f"[{LL}, {LD}, {WLDD}, {WD}, {WW}]"
        performance_stats[engine] = f"{(performance * 100):.{decimal}f}%"
    return penta_stats, performance_stats
    
def output_to_csv(summed_results, ratings_with_error_bars, filename, decimal, los):
    try:
        os.remove(Path(filename).resolve())
    except OSError:
        pass
        
    def write_line(line):
        if filename != "":
            try:
                # Resolve the path and open the file in append mode
                with open(Path(filename).resolve(), "a") as file:
                    file.write(line + "\n")
            except IOError as e:
                # Handle file I/O errors
                print(f"Error writing to file {filename}: {e}", file=sys.stderr)
                exit(1)
    LL = {}
    LD = {}
    WLDD = {}
    WD = {}
    WW = {}
    for engine, pentanomial in summed_results.items():
        LL[engine], LD[engine], WLDD[engine], WD[engine], WW[engine] = pentanomial
    write_line("Rank,Name,Elo,Error_Lower,Error_Upper,LOS_Percent,LL,LD,WL_and_DD,WD,WW")
    i=0
    for engine, (mean_rating, lower_bound, upper_bound) in ratings_with_error_bars.items():
        i += 1
        write_line(f'{i},"{engine}",{mean_rating:.{decimal}f},{(lower_bound-mean_rating):.{decimal}f},{(upper_bound-mean_rating):.{decimal}f},{(los[engine]):.{decimal}f},{LL[engine]},{LD[engine]},{WLDD[engine]},{WD[engine]},{WW[engine]}')
    
def head_to_head(results, filename):
    try:
        os.remove(Path(filename).resolve())
    except OSError:
        pass
        
    def write_line(line):
        if filename != "":
            try:
                # Resolve the path and open the file in append mode
                with open(Path(filename).resolve(), "a") as file:
                    file.write(line + "\n")
            except IOError as e:
                # Handle file I/O errors
                print(f"Error writing to file {filename}: {e}", file=sys.stderr)
                exit(1)
                
    engines_str_length = max(len(f"{engine} vs {opponent}") for engine, opponents in results.items() for opponent, _ in opponents.items())
    penta_str_length = max(len(f"{scores}") for engine, opponents in results.items() for opponent, scores in opponents.items())
    write_line("head-to-head pentanomial results:")
    for engine, opponents in results.items():
        for opponent, scores in opponents.items():
            engines_str = f"{engine} vs {opponent}"
            penta_str = f"[{scores[0]}, {scores[1]}, {scores[2]}, {scores[3]}, {scores[4]}]"
            pairs = scores[0] + scores[1] + scores[2] + scores[3] + scores[4]
            write_line(f"{engines_str:<{engines_str_length}}  : {penta_str:<{penta_str_length}} : {pairs} Pairs")
    
def los_matrix(simulated_ratings, ratings_with_error_bars, filename, decimals):
    if filename == '':
        return
        
    engines = list(ratings_with_error_bars.keys())
        
    # Collect all ratings for each engine
    all_ratings = {}
    for sim_id, ratings in simulated_ratings.items():
        for engine, rating in ratings.items():
            if engine not in all_ratings:
                all_ratings[engine] = []
            all_ratings[engine].append(rating)
            
    losmatrix = {}
    for player1 in engines:
        if player1 not in losmatrix:
            losmatrix[player1] = {}
        for player2 in engines:
            if (player1 != player2):
                exceed_count = 0
                for i in range(len(all_ratings[player2])):
                    if (all_ratings[player1][i] > all_ratings[player2][i]):
                        exceed_count += 1
                losmatrix[player1][player2] = exceed_count * 100.0 / len(all_ratings[player2])
            
    # Create a CSV file
    try:
        with open(Path(filename).resolve(), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_STRINGS)
            
            # Write the header (player names)
            writer.writerow(['N'] + ['NAME'] + list(range(len(engines))))
            
            # Write the rows
            increment = 0
            
            for engine1 in engines:
                row = [increment, engine1]  # Start the row with engine1 name
                for engine2 in engines:
                    los = losmatrix[engine1].get(engine2, '')
                    # Format the los if it's a float
                    if isinstance(los, float):
                        los = f'{los:.{decimals}f}'  # Format to the desired decimal places
                    row.append(los)
                writer.writerow(row)
                increment += 1
    except IOError as e:
        print(f"Error writing to file {filename}: {e}", file=sys.stderr)
        exit(1)
        
def pool_relative_error(ratings_with_error_bars, poolrelative, anchor, average):
    if (poolrelative == False or anchor == ''):
        return ratings_with_error_bars
        
    delta = average - ratings_with_error_bars[anchor][0]
    
    ratings_with_error_bars_updated = {}
    for engine in ratings_with_error_bars.keys():
        ratings_with_error_bars_updated[engine] = (ratings_with_error_bars[engine][0] + delta, ratings_with_error_bars[engine][1] + delta, ratings_with_error_bars[engine][2] + delta)
    return ratings_with_error_bars_updated
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgnfile', type=str, nargs='+', default=[])
    parser.add_argument('--pgndirectory', type=str, default="")
    parser.add_argument('--simulations', type=int, default=1000)
    parser.add_argument('--average', type=float, default=2300)
    parser.add_argument('--anchor', type=str, default="")
    parser.add_argument('--output', type=str, default="")
    parser.add_argument('--csv', type=str, default="")
    parser.add_argument('--rngseed', type=int, default=321140339834632891350088547258043785703)
    parser.add_argument('--concurrency', type=int, default=os.cpu_count())
    parser.add_argument('--purge', action='store_true')
    parser.add_argument('--confidence', type=float, default=95.0)
    parser.add_argument('--exclude', type=str, nargs='+', default=[])
    parser.add_argument('--include', type=str, nargs='+', default=[])
    parser.add_argument('--decimal', type=int, default=1)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--poolrelative', action='store_true')
    parser.add_argument('--head2head', type=str, default="")
    parser.add_argument('--losmatrix', type=str, default="")
    args = parser.parse_args()
    script_start_time = time.time()
    if args.confidence <= 0.0 or args.confidence >=100.0:
        parser.error("Invalid confidence interval.")
    if not args.pgnfile and not args.pgndirectory:
        parser.error("At least one of --pgnfile or --pgndirectory must be specified.")
    
    # engines = ['AlphaZero', 'Stockfish', 'Leela']
    # results = {engine: {opponent: (0, 0, 0, 0, 0) for opponent in engines if opponent != engine} for engine in engines}
    # update_pentanomial(results, 'AlphaZero', 'Stockfish', [24, 1, 28, 12, 35])
    # update_pentanomial(results, 'AlphaZero', 'Leela', [6, 4, 2, 85, 3])
    # update_pentanomial(results, 'Leela', 'Stockfish', [5, 64, 26, 3, 2])
    
    if not args.quiet:
        print("Loading PGN...")
    pgnfiles = args.pgnfile.copy()
    if args.pgndirectory:
        pgndirectory = Path(args.pgndirectory).resolve()
        if pgndirectory.is_dir():
            pgnfiles.extend(pgndirectory.glob('*.pgn'))
        else:
            print(f"Error: {pgndirectory} is not a valid directory.", file=sys.stderr)
            exit(1)
    individual_rounds = []
    engines_set = set()
    for pgnfile in pgnfiles:
        filepath = Path(pgnfile).resolve()
        file_rounds, file_engines = parse_pgn(filepath)
        engines_set.update(file_engines)
        individual_rounds.append(file_rounds)
    engines = list(engines_set)
    engines.sort()
    results = {engine: {opponent: (0, 0, 0, 0, 0) for opponent in engines if opponent != engine} for engine in engines}
    for rounds in individual_rounds:
        update_game_pairs_pgn(results, rounds)

    # exclude specified engines
    engines_to_remove = []
    if not args.include:
        engines_to_remove = args.exclude
    else:
        for engine_to_keep in args.include:
            if engine_to_keep not in engines:
                raise ValueError(f"{engine_to_keep} not found in engines list.")
        engines_to_remove = [engine for engine in engines if engine not in args.include]
    for engine_to_remove in engines_to_remove:
        if engine_to_remove not in engines:
            raise ValueError(f"{engine_to_remove} not found in engines list.")
        engines = [engine for engine in engines if engine != engine_to_remove]
        results.pop(engine_to_remove, None)
        for engine in results.keys():
            results[engine].pop(engine_to_remove, None)
        
    # Calculate probabilities
    scores = calculate_expected_scores(results, args.purge)
    summed_results = sum_all_results(results)
    initial_ratings = set_initial_ratings(engines)
    if args.anchor != '' and args.anchor not in engines:
        raise ValueError(f"{args.anchor} is not in the ratings dictionary.")
    mean_rating = optimize_elo_ratings(engines, scores, initial_ratings, args.average, args.anchor, args.poolrelative)
    probabilities = calculate_probabilities(results)

    # Simulate the tournament
    num_simulations = args.simulations
    simulated_ratings = {}
    if not args.quiet:
        print("Commencing simulation...")
    simulation_start_time = time.time()
    ss = np.random.SeedSequence(args.rngseed)
    seeds = ss.spawn(args.simulations)
    with ProcessPoolExecutor(max_workers = args.concurrency) as executor:
        # Pass a different seed to each process
        futures = [
            executor.submit(run_simulation, i, probabilities, engines, seeds[i], results, args.average, args.anchor, initial_ratings, args.purge, args.poolrelative)
            for i in range(num_simulations)
        ]
        for future in as_completed(futures):
            i, rating = future.result()
            simulated_ratings[i] = rating
            # print(f"Finished simulation {i+1} out of {num_simulations}")
    simulation_end_time = time.time()
    simulation_elapsed_time = simulation_end_time - simulation_start_time

    if not args.quiet:
        print("Finalizing results...")
    confidence_intervals = calculate_percentile_intervals(simulated_ratings, args.confidence)
    #combine the two dicts
    ratings_with_error_bars = {}
    for engine in mean_rating:
        mean = mean_rating[engine]
        if engine in confidence_intervals:
            lower_bound, upper_bound = confidence_intervals[engine]
            ratings_with_error_bars[engine] = (mean, lower_bound, upper_bound)

    #print final ratings with confidence intervals
    ratings_with_error_bars = sort_engines_by_mean(ratings_with_error_bars)
    ratings_with_error_bars = pool_relative_error(ratings_with_error_bars, args.poolrelative, args.anchor, args.average)
    los = calculate_los(simulated_ratings, ratings_with_error_bars)
    los_matrix(simulated_ratings, ratings_with_error_bars, args.losmatrix, args.decimal)
    penta_stats, performance_stats = format_penta_stats(summed_results, args.decimal)
    output_to_csv(summed_results, ratings_with_error_bars, args.csv, args.decimal, los)
    head_to_head(results, args.head2head)
    format_ratings_result(ratings_with_error_bars, penta_stats, performance_stats, summed_results, args.output, args.decimal, los)
    script_end_time = time.time()
    script_elapsed_time = script_end_time - script_start_time
    if not args.quiet:
        print(f"Total simulation time: {simulation_elapsed_time:.4f} seconds")
        print(f"Total elapsed time: {script_elapsed_time:.4f} seconds")
        
if __name__ == "__main__":
    main()

