import pstats
p = pstats.Stats("profile.out")
p.sort_stats("cumtime").print_stats(40)