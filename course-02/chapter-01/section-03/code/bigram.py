"""
Calculate the probability of the sentence "datawhale agent learns".
"""
import collections

# example corpus
corpus = "datawhale agent learns datawhale agent works"
tokens = corpus.split()
total_tokens = len(tokens)

# calculate P(datawhale)
count_datawhale = tokens.count('datawhale')
p_datawhale = count_datawhale / total_tokens
print(f"P(datawhale) = {count_datawhale}/{total_tokens} = {p_datawhale:.3f}")

# calculate P(agent|datawhale)
bigrams = zip(tokens, tokens[1:])
bigram_counts = collections.Counter(bigrams)
print(f"Bigram counts: {bigram_counts}")
count_datawhale_agent = bigram_counts[('datawhale', 'agent')]
p_agent_given_datawhale = count_datawhale_agent / count_datawhale
print(f"P(agent|datawhale) = {count_datawhale_agent}/{count_datawhale} = {p_agent_given_datawhale:.3f}")

# calculate P(learns|agent)
count_agent_learns = bigram_counts[('agent', 'learns')]
count_agent = tokens.count('agent')
p_learns_given_agent = count_agent_learns / count_agent
print(f"P(learns|agent) = {count_agent_learns}/{count_agent} = {p_learns_given_agent:.3f}")

# calculate P('datawhale agent learns')
p_sentence = p_datawhale * p_agent_given_datawhale * p_learns_given_agent
print(f"P('datawhale agent learns') ≈ {p_datawhale:.3f} * {p_agent_given_datawhale:.3f} * {p_learns_given_agent:.3f} = {p_sentence:.3f}")
