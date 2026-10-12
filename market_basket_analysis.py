import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["rice", "dal", "oil"], ["rice", "dal", "sugar"], ["tea", "sugar", "milk"],
    ["rice", "oil", "salt"], ["tea", "milk", "biscuit"], ["rice", "dal", "salt"],
    ["milk", "bread", "butter"], ["tea", "sugar", "biscuit"], ["rice", "dal", "oil", "salt"],
    ["milk", "bread"], ["tea", "milk", "sugar"], ["rice", "dal"],
    ["bread", "butter", "milk"], ["tea", "biscuit"], ["rice", "oil", "dal"],
    ["tea", "sugar"], ["milk", "bread", "butter", "tea"], ["rice", "salt"],
    ["dal", "oil", "salt"], ["tea", "milk", "biscuit", "sugar"],
]

te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

frequent = apriori(df, min_support=0.25, use_colnames=True)
frequent["itemsets"] = frequent["itemsets"].apply(lambda s: ", ".join(sorted(s)))
print("Frequent itemsets (min_support = 0.25):")
print(frequent.sort_values("support", ascending=False).to_string(index=False))

freq_sets = apriori(df, min_support=0.25, use_colnames=True)
rules = association_rules(freq_sets, metric="confidence", min_threshold=0.6)
rules["rule"] = rules.apply(lambda r: ", ".join(sorted(r["antecedents"])) + " -> "
                            + ", ".join(sorted(r["consequents"])), axis=1)
rules = rules.sort_values("lift", ascending=False)
print("\nAssociation rules (min_confidence = 0.6):")
print(rules[["rule", "support", "confidence", "lift"]].round(3).to_string(index=False))

# Graphs: item popularity and rule scatter
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
item_support = df.mean().sort_values()
ax[0].barh(item_support.index, item_support.values, color="darkcyan")
ax[0].set_xlabel("Support (share of baskets)")
ax[0].set_title("How often each item is bought")

sc = ax[1].scatter(rules["support"], rules["confidence"], s=rules["lift"] * 60,
                   c=rules["lift"], cmap="plasma", alpha=0.8, edgecolor="k")
plt.colorbar(sc, ax=ax[1], label="Lift")
ax[1].set_xlabel("Support")
ax[1].set_ylabel("Confidence")
ax[1].set_title("Rules (bubble size and colour = lift)")
plt.tight_layout()
plt.savefig("q6_market_basket.png", dpi=130)
plt.show()
