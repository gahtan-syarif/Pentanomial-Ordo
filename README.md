Python script that recalculates [Ordo](https://github.com/michiguel/Ordo) ratings with pentanomial error bars using a statistical method called bootstrapping.
How to use:
- Place the script inside a folder with ordo.exe and the pgn file.
- Run the command: `python pentaordo.py --pgnfile NAME --simulations NUM --ordoargs ARGS`
- Example command: `python pentaordo.py --pgnfile tournament.pgn --simulations 1000 --ordoargs "-a 2800 -A Stockfish"`
- For the number of simulations the minimum recommended amount is 1000 although the higher the more accurate the error bar.
- This program only works when the tournament is a roundrobin where games are played in pairs where each engine plays both colors and where each engine plays every other engine an equal amount of times.
- PGNs must be correctly formatted where every unique gamepair has a unique "Round" PGN header tag. So if for example a pgn has 100 games then the "Round" tag must be incremented from 1 to 50 for every gamepair.
- Make sure the engine names in the PGN does not contain any whitespace.
