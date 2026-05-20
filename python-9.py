losses = [0, 1, 2, 5]
probs = [0.90, 0.05, 0.03, 0.02]

cumulative_prob = 0
var_95 = 0

for i in range(len(losses)):
    cumulative_prob = cumulative_prob + probs[i]
    if cumulative_prob >= 0.95:
        var_95 = losses[i]
        break

print("VaR at 95% confidence level: $" + str(var_95) + " million")

total_loss = 0
total_p = 0

for i in range(len(losses)):
    if losses[i] > var_95:
        total_loss = total_loss + (losses[i] * probs[i])
        total_p = total_p + probs[i]

es_95 = total_loss / total_p

print("ES at 95% confidence level: $" + str(es_95) + " million")