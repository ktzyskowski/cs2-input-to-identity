- [ ] create local cs2 demo to discover mapping to `buttons` field in demoparser2
    - record `filepath`
    - stop

- [ ] abstract/intro/lit review
    - google notebooklm?

- [ ] pass entire games into LSTM per player rather than trajectories
    - reset hidden state (not cell state) at start of each round
    - resetting short term memory but keeping long term
    - equivalent to start/stop tokens in NLP

- [ ] pass entire game per player through same LSTM in parallel, then concat latent signatures to create hidden feature
  vector
    - pass hidden features to feed forward binary classifier

- [ ] start with simple LSTM model (low # layers, # hidden params)

- [ ] experiment with arc distance between mouse positions/angles

- [ ] limit to one map? (de_dust2 if any)

- [ ] train/val/test split
    1. split players into 80/20 split or similar
    2. majority players will be used for train/val, other players for test
    3. create training/val set using filtering on pairs (all same-player pairs, and identical amount of randomly sampled
       different-player pairs) and then 80/20 split or similar on sampled pairs

- [ ] measure using acc, prec, recall, and f1


- [ ] RISE abstract due early feb, poster due by end of feb
    - can present in-progress work or MS project work if not far enough

- [ ] identify conferences of lit review papers to discover conferences i should apply to publish to
    - figure out when deadlines are for paper submissions

DEMO TESTING NOTES

| Action         | Key        | Encoding |
|----------------|------------|----------|
| Move forward   | W          | 8        |
| Move left      | A          | 512      |
| Move backward  | S          | 16       |
| Move right     | D          | 1024     |
| Crouch         | Left Ctrl  | 4        |
| Walk           | Left Shift | 65536    |
| Jump           | Space      | 2        |
| Use            | E          | 32       |
| Fire           | Mouse1     | 1        |
| Secondary Fire | Mouse2     | 2048     |
| Reload         | R          | 8192     |
| Inspect Weapon | C          | _        |
| Scoreboard     | Tab        | _        |

1 fire
2 jump
4 crouch
8 move forward
16 move backward
32 use
64    _
128   _
256   _
512 move left
1024 move right
2048 secondary fire
4096  _
8192 reload
...
65536 walk
...
8589934592 tab?
34359738368 inspect weapon?

Last Weapon Used Q
Drop Weapon G
Inspect Weapon C

Primary Weapon 1
Secondary Weapon 2
Melee Weapon 3
Cycle Grenades 4
Explosives & Traps 5

HE Grenade V
Flashbang C
Smoke Grenade X
Decoy Grenade 9
Molotov Cocktail/Inc. Z

Grafitti Menu T

Scoreboard Tab

test_demo.dem
44.36s
frames = 2960
ticks = 2839
64tick per second!

# IDEAS!

- binary classification between two players, given their weapon (AK47) sprays.
    - can this generalize beyond just AK sprays? m4a1/a4? famas? galil?
    - can this be improved with 1vN over 1v1 comparisons? (i.e. does one spray belong to a group of sprays we know
      belong to the same individual?)  A vs B / A vs B1, B2, ..., BN
    - can this be improved with a bayes classifier over samples from multiple weapon sprays?
      - i.e. do we know if player A is the same as player B given the AK sprays and M4A4 sprays for both A and B?





