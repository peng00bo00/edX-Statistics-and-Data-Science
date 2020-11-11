import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
## KMeans
for K in range(1, 5):
    cost_min = float("inf")
    for seed in range(5):
        mixture, post = common.init(X, K, seed)
        mixture, post, cost = kmeans.run(X, mixture, post)

        if cost < cost_min:
            cost_min = cost
    
    print(f"K = {K}, cost = {cost_min}")

## EM
print()
for K in range(1, 5):
    logL_max = float("-inf")
    for seed in range(5):
        mixture, post = common.init(X, K, seed)
        mixture, post, logL = naive_em.run(X, mixture, post)

        if logL > logL_max:
            logL_max = logL
    
    print(f"K = {K}, logLikelihood = {logL_max}")

## BIC
print()
for K in range(1, 5):
    BIC_max = float("-inf")
    for seed in range(5):
        mixture, post = common.init(X, K, seed)
        mixture, post, logL = naive_em.run(X, mixture, post)

        BIC = common.bic(X, mixture, logL)

        if BIC > BIC_max:
            BIC_max = BIC
    
    print(f"K = {K}, BIC = {BIC_max}")

## EM for Collaborative Filtering
X = np.loadtxt("netflix_incomplete.txt")
print()
for K in [1, 12]:
    logL_max = float("-inf")
    for seed in range(5):
        mixture, post = common.init(X, K, seed)
        mixture, post, logL = em.run(X, mixture, post)

        if logL > logL_max:
            logL_max = logL
    
    print(f"K = {K}, logLikelihood = {logL_max}")

## Comparing with Gold Targets
X = np.loadtxt("netflix_incomplete.txt")
logL_max = float("-inf")
K = 12

print()
for seed in range(5):
    mixture, post = common.init(X, K, seed)
    mixture, post, logL = em.run(X, mixture, post)

    if logL > logL_max:
        logL_max = logL
        best_mixture = mixture

X_gold = np.loadtxt('netflix_complete.txt')
X_pred = em.fill_matrix(X, best_mixture)
print(f"RMSE = {common.rmse(X_gold, X_pred)}")