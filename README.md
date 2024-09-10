This is a Python script that calculates the pentanomial elo ratings of chess engines/players in a tournament using a similar methodology to [Ordo](https://github.com/michiguel/Ordo). Ratings are calculated using an [optimization algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS) with its error margins being calculated through non-parametric [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
How to use:
- This script only works for tournaments where games are played in pairs where each engine/player swaps colors.
- Required Python dependencies: numpy, scipy
- Run the command: `python calculateratings.py --pgnfile FILE` or `python calculateratings.py --pgnfile FILE1 FILE2 FILE3 ...` for multiple PGN's.
- To automatically input all PGN files within a directory, use the argument `--pgndirectory DIRECTORY`.
- To set the number of simulations, add `--simulations NUM` with the default being 1000. For the number of simulations the recommended minimum amount is 1000, although the higher the more accurate the error bar becomes.
- To set the seed for the random number generator, you can add `--rngseed SEED` with the default seed being 42.
- To set the average rating for the pool, use the argument `--average AVG` with the default being 2300.
- If you want to set an anchor engine/player, use the argument `--anchor ANCHOR`.
- To set the number of parallel processes, set `--concurrency N` with the default being the number of CPU hardware threads.
- Use `--output FILE` to output the ratings as a text file.
- Use `--csv FILE` to output the ratings as a csv file.
- Each PGN must be correctly formatted where every unique gamepair between two players/engines within that PGN must have a unique "Round" PGN header tag.
