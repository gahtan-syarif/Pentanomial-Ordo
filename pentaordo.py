import random
import chess.pgn
from collections import defaultdict, deque
import subprocess
import os
import re
import io
import numpy as np
import argparse

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

def process_pgn_with_ordo(pgn_file_path, ordoargs):
    """
    Process the PGN file using the ordo.exe program.

    :param pgn_file_path: Path to the PGN file to be processed.
    """
    # Path to ordo.exe executable
    dirname = os.path.dirname(__file__)
    ordo_exe_path = os.path.join(dirname, 'ordo.exe')

    # Construct the command to run
    ordoargs_list = ordoargs.split()
    command = [ordo_exe_path, "-p", pgn_file_path] + ordoargs_list 

    try:
        # Run the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Write standard output to file
        #with open(os.path.join(dirname, 'ordoresult.txt'), 'w') as stdout_file:
        #    stdout_file.write(result.stdout)

        # Extract ratings from the output
        ratings = extract_ratings(result.stdout)
        return ratings

        # Print standard output and standard error
        #print("Standard Output:", result.stdout)
        #print("Standard Error:", result.stderr)

    except subprocess.CalledProcessError as e:
        # Handle errors in the subprocess call
        print("Return Code:", e.returncode)
        print("Standard Output:", e.stdout)
        print("Standard Error:", e.stderr)
        raise RuntimeError("Error occurred while processing the PGN file with ordo.exe.") from e

def extract_ratings(output):
    """
    Extract ratings from the output of ordo.exe.

    :param output: Output string from ordo.exe.
    :return: Dictionary of engine ratings.
    """
    # Regular expression to match player ratings
    #rating_pattern = re.compile(r'\d+\s+(\S+)\s+:\s+([\d.]+)')
    rating_pattern = re.compile(r'\d+\s+(\S+)\s+:\s+([-]?\d+\.\d+|\d+)')
    ratings = {}
    
    # Find all matches in the output
    for match in rating_pattern.finditer(output):
        engine_name = match.group(1)
        rating = float(match.group(2))
        ratings[engine_name] = rating

    return ratings

def write_results_to_pgn(results, engines, output_file_path):
    """
    Write the results of the simulated games to a PGN file.

    :param results: Dictionary containing the results of the simulated games.
    :param output_file_path: Path to the output PGN file.
    """
    with open(output_file_path, 'w') as pgn_file:
        for i in range(len(engines)):
            for j in range(i + 1, len(engines)):
                engine1, engine2 = engines[i], engines[j]
                LL, LD, WLDD, WD, WW = results[engine1][engine2]
                total_games = LL + LD + WLDD + WD + WW

                if total_games == 0:
                    continue

                # Construct games for each result type
                for _ in range(LL):
                    game = chess.pgn.Game()
                    game.headers["White"] = engine1
                    game.headers["Black"] = engine2
                    game.headers["Result"] = "0-1"
                    #game.add_line(chess.Move.from_uci("0000"))
                    pgn_file.write(str(game) + "\n\n")

                    game2 = chess.pgn.Game()
                    game2.headers["White"] = engine2
                    game2.headers["Black"] = engine1
                    game2.headers["Result"] = "1-0"
                    #game2.add_line(chess.Move.from_uci("0000"))
                    pgn_file.write(str(game2) + "\n\n")

                for _ in range(LD):
                    game = chess.pgn.Game()
                    game.headers["White"] = engine1
                    game.headers["Black"] = engine2
                    game.headers["Result"] = "0-1"
                    #game.add_line(chess.Move.from_uci("0000"))
                    pgn_file.write(str(game) + "\n\n")

                    game2 = chess.pgn.Game()
                    game2.headers["White"] = engine2
                    game2.headers["Black"] = engine1
                    game2.headers["Result"] = "1/2-1/2"
                    #game2.add_line(chess.Move.from_uci("0000"))
                    pgn_file.write(str(game2) + "\n\n")

                for _ in range(WLDD):
                    game = chess.pgn.Game()
                    game.headers["White"] = engine1
                    game.headers["Black"] = engine2
                    game.headers["Result"] = "1/2-1/2"
                    #game.add_line(chess.Move.from_uci("0000"))
                    pgn_file.write(str(game) + "\n\n")

                    game2 = chess.pgn.Game()
                    game2.headers["White"] = engine2
                    game2.headers["Black"] = engine1
                    game2.headers["Result"] = "1/2-1/2"
                    #game2.add_line(chess.Move.from_uci("0000"))
                    pgn_file.write(str(game2) + "\n\n")

                for _ in range(WD):
                    game = chess.pgn.Game()
                    game.headers["White"] = engine1
                    game.headers["Black"] = engine2
                    game.headers["Result"] = "1-0"
                    #game.add_line(chess.Move.from_uci("0000"))
                    pgn_file.write(str(game) + "\n\n")

                    game2 = chess.pgn.Game()
                    game2.headers["White"] = engine2
                    game2.headers["Black"] = engine1
                    game2.headers["Result"] = "1/2-1/2"
                    #game2.add_line(chess.Move.from_uci("0000"))
                    pgn_file.write(str(game2) + "\n\n")

                for _ in range(WW):
                    game = chess.pgn.Game()
                    game.headers["White"] = engine1
                    game.headers["Black"] = engine2
                    game.headers["Result"] = "1-0"
                    #game.add_line(chess.Move.from_uci("0000"))
                    pgn_file.write(str(game) + "\n\n")

                    game2 = chess.pgn.Game()
                    game2.headers["White"] = engine2
                    game2.headers["Black"] = engine1
                    game2.headers["Result"] = "0-1"
                    #game2.add_line(chess.Move.from_uci("0000"))
                    pgn_file.write(str(game2) + "\n\n")

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
        error_down = round(mean_rating - lower_bound, 1)
        error_up = round(upper_bound - mean_rating, 1)
        mean_rating_str = f"{mean_rating:.1f}"
        error_down_str = f"{error_down:.1f}"
        error_up_str = f"{error_up:.1f}"
        interval_str = f"[{lower_bound:.1f}, {upper_bound:.1f}]"

        print(f"{engine:<{max_engine_length}}: {mean_rating_str:>{max_mean_length}} (-{error_down_str:>{max_error_length}}/+{error_up_str:<{max_plus_error_length}}) {interval_str:>{max_interval_length}}")

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

def reorder_pgn_by_pairs_with_rounds(pgn_file_path, output_file_path):
    # Read and parse the PGN file
    with open(pgn_file_path, 'r') as pgn_file:
        pgn_data = pgn_file.read()

    # Create a PGN reader
    pgn_io = io.StringIO(pgn_data)
    
    # Read all games
    games = []
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        games.append(game)
    
    # Create a dictionary to store games by (white_player, black_player) pair
    pairs_dict = defaultdict(list)
    for game in games:
        white_player = game.headers.get('White', 'Unknown')
        black_player = game.headers.get('Black', 'Unknown')
        pairs_dict[(white_player, black_player)].append(game)
    
    # Reorder games based on pairs
    ordered_games = []
    visited_pairs = set()
    
    for (white_player, black_player), games_list in pairs_dict.items():
        if (black_player, white_player) in pairs_dict:
            # Alternate between (white, black) and (black, white) games
            reversed_games = pairs_dict[(black_player, white_player)]
            while games_list and reversed_games:
                ordered_games.append(games_list.pop(0))
                ordered_games.append(reversed_games.pop(0))
            ordered_games.extend(games_list)
            ordered_games.extend(reversed_games)
    
    # Assign round numbers
    rounds = 1
    for i in range(0, len(ordered_games), 2):
        for j in range(i, min(i + 2, len(ordered_games))):
            ordered_games[j].headers['Round'] = str(rounds)
        rounds += 1
    
    # Write the reordered and updated games to the output PGN file
    with open(output_file_path, 'w') as output_file:
        for game in ordered_games:
            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
            output_file.write(game.accept(exporter))
            output_file.write("\n\n")

def reorder_pgn_by_pairs_with_rounds_nomoves(pgn_file_path, output_file_path):
    # Read and parse the PGN file
    with open(pgn_file_path, 'r') as pgn_file:
        pgn_data = pgn_file.read()

    # Create a PGN reader
    pgn_io = io.StringIO(pgn_data)
    
    # Read all games
    games = []
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        games.append(game)
    
    # Create a dictionary to store games by (white_player, black_player) pair
    pairs_dict = defaultdict(list)
    for game in games:
        white_player = game.headers.get('White', 'Unknown')
        black_player = game.headers.get('Black', 'Unknown')
        pairs_dict[(white_player, black_player)].append(game)
    
    # Reorder games based on pairs
    ordered_games = []
    
    for (white_player, black_player), games_list in pairs_dict.items():
        if (black_player, white_player) in pairs_dict:
            # Alternate between (white, black) and (black, white) games
            reversed_games = pairs_dict[(black_player, white_player)]
            while games_list and reversed_games:
                ordered_games.append(games_list.pop(0))
                ordered_games.append(reversed_games.pop(0))
            ordered_games.extend(games_list)
            ordered_games.extend(reversed_games)
    
    # Assign round numbers
    rounds = 1
    for i in range(0, len(ordered_games), 2):
        for j in range(i, min(i + 2, len(ordered_games))):
            ordered_games[j].headers['Round'] = str(rounds)
        rounds += 1
    
    # Write the reordered and updated games to the output PGN file
    with open(output_file_path, 'w') as output_file:
        for game in ordered_games:
            # Construct PGN output including headers and result, but excluding moves
            def export_headers_with_result(game):
                headers = []
                for key, value in game.headers.items():
                    if key not in ['Moves']:  # Exclude moves
                        headers.append(f"[{key} \"{value}\"]")
                # Ensure the Result tag is included
                if 'Result' not in game.headers:
                    headers.append(f"[Result \"*\"]")
                return "\n".join(headers) + "\n\n"

            # Append the result tag manually
            result = game.headers.get('Result', '*')
            output_file.write(export_headers_with_result(game))
            output_file.write(f"{result}\n")  # Append the result at the end of the game
            output_file.write("\n")

def reorder_pgn_by_pairs_with_rounds_nomoves_faster(pgn_file_path, output_file_path):
    # Read and parse the PGN file
    with open(pgn_file_path, 'r') as pgn_file:
        pgn_data = pgn_file.read()

    # Create a PGN reader
    pgn_io = io.StringIO(pgn_data)
    
    # Read all games
    games = []
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        games.append(game)
    
    # Create a dictionary to store games by (white_player, black_player) pair
    pairs_dict = defaultdict(deque)
    for game in games:
        white_player = game.headers.get('White', 'Unknown')
        black_player = game.headers.get('Black', 'Unknown')
        pairs_dict[(white_player, black_player)].append(game)
    
    # Reorder games based on pairs
    ordered_games = []
    used_pairs = set()

    for (white_player, black_player), games_list in pairs_dict.items():
        if (black_player, white_player) in pairs_dict and (black_player, white_player) not in used_pairs:
            reversed_games = pairs_dict[(black_player, white_player)]
            while games_list and reversed_games:
                ordered_games.append(games_list.popleft())
                ordered_games.append(reversed_games.popleft())
            ordered_games.extend(games_list)
            ordered_games.extend(reversed_games)
            used_pairs.add((white_player, black_player))
            used_pairs.add((black_player, white_player))
    
    # Assign round numbers
    rounds = 1
    for i in range(0, len(ordered_games), 2):
        for j in range(i, min(i + 2, len(ordered_games))):
            ordered_games[j].headers['Round'] = str(rounds)
        rounds += 1
    
    # Write the reordered and updated games to the output PGN file
    with open(output_file_path, 'w') as output_file:
        for game in ordered_games:
            def export_headers_with_result(game):
                headers = []
                for key, value in game.headers.items():
                    if key not in ['Moves']:
                        headers.append(f"[{key} \"{value}\"]")
                if 'Result' not in game.headers:
                    headers.append(f"[Result \"*\"]")
                return "\n".join(headers) + "\n\n"

            result = game.headers.get('Result', '*')
            output_file.write(export_headers_with_result(game))
            output_file.write(f"{result}\n")
            output_file.write("\n")

parser = argparse.ArgumentParser()
parser.add_argument('--pgnfile', type=str, required=True)
parser.add_argument('--simulations', type=int, default=1000)
parser.add_argument('--ordoargs', type=str, default="")
args = parser.parse_args()

#engines = ['AlphaZero', 'Stockfish', 'Leela']
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, args.pgnfile)
# try:
#     os.remove(os.path.join(dirname, 'processed_pgn.pgn'))
# except OSError:
#     pass
# reorder_pgn_by_pairs_with_rounds_nomoves_faster(filename, os.path.join(dirname, 'processed_pgn.pgn'))
mean_rating = process_pgn_with_ordo(filename, args.ordoargs)
print("Parsing PGN... please wait...")
rounds, engines = parse_pgn(filename)
print("Finished parsing PGN, proceeding to calculate results...")
# os.remove(os.path.join(dirname, 'processed_pgn.pgn'))
results = {engine: {opponent: (0, 0, 0, 0, 0) for opponent in engines if opponent != engine} for engine in engines}
#update_pentanomial(results, 'AlphaZero', 'Stockfish', [24, 1, 28, 12, 35])
#update_pentanomial(results, 'AlphaZero', 'Leela', [6, 4, 2, 85, 3])
#update_pentanomial(results, 'Leela', 'Stockfish', [5, 64, 26, 3, 2])
update_game_pairs_pgn(results, rounds)
print("Calculating probabilities...")

# Calculate probabilities
probabilities = calculate_probabilities(results)
simulated_ratings = {}
num_pairs_per_pairing = results[engines[0]][engines[1]][0] + results[engines[0]][engines[1]][1] + results[engines[0]][engines[1]][2] + results[engines[0]][engines[1]][3] + results[engines[0]][engines[1]][4]
print("Starting simulation...")
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'simulated_results.pgn')
# Simulate the tournament
num_simulations = args.simulations
for i in range(num_simulations): # 1000 is the typical minimum number of bootstrap samples, but the more the better
    simulated_results = simulate_tournament(probabilities, engines, num_pairs_per_pairing)
    
    try:
        os.remove(filename)
    except OSError:
        pass
    write_results_to_pgn(simulated_results, engines, filename)
    simulated_ratings[i] = process_pgn_with_ordo(filename, args.ordoargs)
    os.remove(filename)
    print(f"Finished simulation {i+1} out of {num_simulations}")
    
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
#for engine, (mean_rating, lower_bound, upper_bound) in ratings_with_error_bars.items():
 #   print(f"{engine} : {mean_rating} (-{round(mean_rating-lower_bound, 1)}/+{round(upper_bound-mean_rating, 1)})[{lower_bound}, {upper_bound}]")
format_ratings_result(ratings_with_error_bars)
print()