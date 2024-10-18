*Disclaimer: This python script is not affiliated with the original Ordo project or its author. This script also does not contain any implementations of Ordo code.*

This is a Python script that calculates the pentanomial elo ratings of chess engines/players in a tournament using a similar methodology to [Ordo](https://github.com/michiguel/Ordo). Ratings are calculated using an [optimization algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS) with its error margins and CFS/LOS being calculated through non-parametric stratified [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).

This script was made to tackle the issue where ordo displays incorrect error margins and CFS/LOS for tournaments where games are played in pairs in which each player/engine swaps sides, especially if unbalanced opening books are used. This is because ordo calculates those values using trinomial (win-draw-loss) probabilities that are inaccurate for paired games. To fix this, this script uses pentanomial (game pair) probabilities that gives a better estimate for the error margins and CFS/LOS.

How to use:
- This script only works for tournaments where games are played in pairs where each engine/player swaps colors.
- Each game in the PGN file must have at minimum a "Round", "White", "Black", and "Result" header tags listed in that order. Games that are played in pairs must share the same "Round" tag. An example of a correctly formatted PGN file is provided above.
- Required Python dependencies: numpy (version >= 1.17.0), scipy
- Run the command: `python calculateratings.py --pgnfile FILE` or `python calculateratings.py --pgnfile FILE1 FILE2 FILE3 ...` for multiple PGN's.
- Use `--pgndirectory DIRECTORY` if you want to automatically input all PGN files within a directory.
- Use `--simulations N` to set the number of simulations, with the default being 1000. For the number of simulations the recommended minimum amount is 1000, although the higher the more accurate the error margin and LOS becomes. Number of simulations greater than 10000 is likely not necessary.
- Use `--average N` to set the average rating for the pool, with the default being 2300.
- Use `--anchor PLAYER` if you want to set an anchor engine/player. The elo of the anchor would be fixed to the value set with `--average`.
- Use `--poolrelative` if you want to make the error margins relative to the pool instead of the anchor when an anchor is selected.
- Use `--confidence N` to set the % confidence interval for the error margin, with the default value being 95.
- Use `--concurrency N` to set the number of parallel processes, with the default being the number of CPU hardware threads.
- Use `--purge` if you want to purge/exclude perfect wins/losses from ratings calculation. This feature is experimental.
- Use `--decimal N` to round ratings results to N decimal places, with the default value being 1.
- Use `--exclude PLAYER1 PLAYER2 PLAYER3 ...` if you want to exclude a list of engines/players from the rating list.
- Use `--include PLAYER1 PLAYER2 PLAYER3 ...` if you want to only include the engines/players listed into the rating list.
- Use `--output FILE` if you want to output the ratings as a text file.
- Use `--csv FILE` if you want to output the ratings as a csv file.
- Use `--head2head FILE` if you want to output head-to-head results to a text file.
- Use `--losmatrix FILE` if you want to output matrix of LOS values as a csv file.
- Use `--quiet` if you want to silence progress updates.
- Use `--rngseed N` to set the seed for the random number generator, with the default seed being 321140339834632891350088547258043785703.
- For very large PGN files (>1GB), it is recommended to shrink the PGN file beforehand to massively reduce the PGN loading/parsing time. this can be done by using the `shrinkpgn.py` script with the command: `python shrinkpgn.py --inputfile FILE --outputfile FILE`

Example output from SPCC [UHO-Top15 Ratinglist](https://www.sp-cc.de/) (average Elo set to 0):
```
===============================================================================================================
RANK  NAME                   :     ELO  ERROR        LOS(%)  PENTANOMIAL                   POINTS  PAIRS    (%)
===============================================================================================================
1     Stockfish 240820 avx2  :   148.6  (-2.0/+2.0)   100.0  [3, 150, 1244, 5840, 263]    10605.0   7500  70.7%
2     Stockfish 16.1 240224  :   139.4  (-1.9/+2.1)   100.0  [1, 229, 1405, 5638, 227]    10430.5   7500  69.5%
3     Torch 3 popavx2        :   111.9  (-2.2/+2.2)   100.0  [3, 586, 1821, 4800, 290]     9894.0   7500  66.0%
4     Berserk 13 avx2        :    40.5  (-2.4/+2.6)   100.0  [25, 1569, 2644, 3106, 156]   8399.5   7500  56.0%
5     Obsidian 13.08 avx2    :    34.7  (-2.3/+2.5)    63.3  [21, 1655, 2729, 2946, 149]   8273.5   7500  55.2%
6     KomodoDragon 3.3 avx2  :    34.2  (-2.3/+2.5)   100.0  [44, 1571, 2785, 3017, 83]    8262.0   7500  55.1%
7     Caissa 1.20 avx2       :    -6.3  (-2.4/+2.4)    78.4  [64, 2378, 2885, 2085, 88]    7377.5   7500  49.2%
8     PlentyChess 2.1 avx2   :    -7.6  (-2.5/+2.4)    59.7  [47, 2380, 2987, 2003, 83]    7347.5   7500  49.0%
9     Ethereal 14.38 avx2    :    -8.1  (-2.6/+2.5)   100.0  [57, 2395, 2942, 2030, 76]    7336.5   7500  48.9%
10    RubiChess 240817 avx2  :   -21.8  (-2.4/+2.4)   100.0  [153, 2550, 2910, 1846, 41]   7036.0   7500  46.9%
11    Alexandria 7.0 avx2    :   -38.5  (-2.5/+2.4)   100.0  [93, 3012, 2897, 1455, 43]    6671.5   7500  44.5%
12    Clover 7.1 avx2        :   -58.6  (-2.7/+2.5)   100.0  [104, 3510, 2752, 1073, 61]   6238.5   7500  41.6%
13    Viridithas 14.0 avx2   :   -76.8  (-2.9/+2.5)   100.0  [347, 3604, 2576, 949, 24]    5849.5   7500  39.0%
14    Lizard 10.5 avx2       :   -94.2  (-2.5/+2.5)    57.9  [196, 4298, 2356, 638, 12]    5486.0   7500  36.6%
15    Motor 0.70 avx2        :   -94.7  (-2.4/+2.4)   100.0  [214, 4266, 2390, 613, 17]    5476.5   7500  36.5%
16    Titan 1.1 avx2         :  -102.5  (-2.5/+2.5)     0.0  [258, 4396, 2319, 510, 17]    5316.0   7500  35.4%
===============================================================================================================
```
Note: Ratings would be slightly different from Ordo. This is because Ordo by default incorrectly calibrates the rating to be 202 elo at 76% expected score. The script fixes this by setting it to the correct value that is 200.24 elo. CFS is also renamed to LOS (likelihood of superiority).
