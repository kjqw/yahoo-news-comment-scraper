# %%
import random

import classes
import numpy as np

# %%
user_num = 5
state_dim = 4
k_max = 3
is_random = True
parents = {
    i: {
        j: random.choices([None] + [x for x in range(user_num) if x != i], k=2)
        for j in range(k_max)
    }
    for i in range(user_num)
}
parents
# %%
user_nodes = {
    i: classes.UserNode(i, parents[i], state_dim, is_random=is_random)
    for i in range(user_num)
}
# %%
user_nodes[0].__dict__
# %%
for user_node in user_nodes:
    for k in range(k_max):
        user_nodes[user_node].update_state(k)
# %%
user_nodes[0].__dict__
# %%
