# Synthetic Data: Extreme Case for Single Cache <br> Library size: 50
Consider the data where the number of files is 50 and cache size is 5. The request sequence is just file 1 through 50 repeatedly, i.e. `request: 1,2,3,...,50,1,2,3,...,50,1,2,...` for total $50,000$ requests.

Since this is a single user/single cache setup, the LeadCache algorithm reduces to selecting the 5 most frequently occurring files so far. This strategy, as expected, performs poorly. Whereas the algorithms involving a finite state machines are able to exploit the recurring pattern of sequences for better predictions. This is reflected in the hitrates which are as follows:

| Algo                        | Hitrate |
| --------------------------- | ------- |
| LeadCache                   | 0.099   |
| IPLC                        | 0.92    |
| Order-1 Markov FSM (online) | 0.9991  |

The IPLC has a lower hitrate because there are multiple 'copies' of the same chain being formed from the initial state. Populating so many redundant states takes a lot of time.

<br><br><br>


# Movielens Ratings (1M) Data <br> Library size: 300
## Preprocessing
- We take a subset of 1M datapoints of the Movielens Ratings dataset. Data is first sorted by `userId` and then by `timestamp`.

- There are total 3706 unique files, which are renamed so they are in range [0...3705].
  
- For a fixed value of $\mathbf{offset}$, we filter all files with id between $(\mathbf{offset},\mathbf{offset}+300)$ in every simulation. 

- In case of multiple users, we split the whole request sequence into $u$ (no. of users) continuous sequences, each corresponding to a user.

<br>


| offset | (u,c)  | LC     | IPLC  | MM-on | MM-off | t<br> ($\times 1000$) | threshold |
| ------ | ------ | ------ | ----- | ----- | ------ | --------------------- | --------- |
| 0      | (1,1)  | 0.4429 | 0.479 | 0.605 | 0.695  | 82                    | 200       |
| 300    | (1,1)  | 0.4745 | 0.477 | 0.619 | 0.724  | 50                    | 200       |
| 600    | (1,1)  | 0.4924 | 0.502 | 0.628 | 0.729  | 50                    | 200       |
| 900    | (10,4) | 0.4799 | 0.508 | 0.593 | 0.875  | 5                     | 50        |
| 1200   | (15,7) | 0.6023 | 0.616 | 0.72  | 0.889  | 5                     | 50        |

- `(u,c)`: (number of users, number of caches).
- `LC`: hitrate of LeadCache algorithm.
- `IPLC`: hitrate of IPLC algorithm.
- `MM-on`: hitrate of LeadCache used with online order-1 markov FSM.
- `MM-off`: hitrate f LeadCache used with offline order-1 markov FSM.
- `t`: number of timesteps.
- Number of files is 300 and cache size is 30 for all experiments.
- For `IPLC`, `MM-off` and `MM-on` only the states with at least `threshold` visits are considered to compute the hitrate.




<br><br><br>

# Binary Request Sequence for Single Cache

The aim is to check if increasing the order of markov fsm increases the accuracy. 

For the offline markov fsm, the accuracy increases as we increase the order-k. 

In the case of online markov fsm, the accuracy increases initially but stagnates and starts to decrease slightly as we further increase the order. This is expected, since the size of dataset is constant, the number of visits per state decrease as we increase the order.

<br>

## Offline Markov
- We take the movielens ratings dataset (1M) and preprocess it just like above, except that each sequence contains 256 unique files.
- Each request is converted into a 8 binary requests. The hitrates below are for seq 6, but the hitrates for other sequences are of similar magnitude.
- Total number of binary requests is 515,760

The hitrate offline markov fsm are as follows:

| order (k) | 4      | 8      | 12     | 16     | 20     | 24     | 28     | 32     | 36     | 40     | 44     | 48     |
| --------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| hitrate   | 0.5584 | 0.5796 | 0.6209 | 0.7014 | 0.8676 | 0.9637 | 0.9902 | 0.9966 | 0.9988 | 0.9994 | 0.9998 | 0.9999 |


<br>

## Online Markov

- For the online markov fsm, ~500k binary requests were not sufficient, and the maximum hitrate achieved was 0.607 for k=16.
- Hence, we run the online markov fsm on the full movielens dataset (25M). We filter all files with id less than 256 and create 8 binary requests for each file request. There are ~13M binary requests made. 
- As stated above, the hitrate first increases and decreases gradually. Hitrates are as follows

| order (k) | 4      | 8      | 12     | 16     | 20     | 24     | 28     | 32     |
| --------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| hitrate   | 0.5540 | 0.5907 | 0.6312 | 0.6633 | 0.6687 | 0.6457 | 0.6167 | 0.5914 |

<br>

## IPLC
For comparison, the hitrate achieved by IPLC on ~13M binary requests is 0.5916.

# CMU Data

This is CDN1 data from Berger et al. The data is processed in the exact same manner as the MovieLens data.

There are total ~1.4M requests with 9999 unique file IDs 

| offset | (u,c)  | LC      | IPLC  | t<br> ($\times 1000$) | threshold |
| ------ | ------ | ------- | ----- | --------------------- | --------- |
| 3000   | (1,1)  | 0.917   | 0.895 | 135                   | 200       |
| 7500   | (1,1)  | 0.907   | 0.883 | 123                   | 200       |
| 0      | (10,4) | 0.565   | 0.59  | 2.3                   | 50        |
| 7500   | (10,4) | 0.732   | 0.77  | 10                    | 200       |
| 0      | (15,7) | 0.643   | 0.619 | 1.5                   | 50        |
| 3000   | (15,7) | 0.77488 | 0.775 | 5                     | 200       |

