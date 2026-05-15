
### Example: Load the data and preprocess graph (replace with your dataset)
df = pd.read_csv("interactions_ppi")
G = nx.from_pandas_edgelist(df, 'source', 'target')

print(df)
print(G)

###Remove small connected components (Kcc < 10)
components = [c for c in nx.connected_components(G) if len(c) >= 10]
G = G.subgraph(max(components, key=len)).copy()
nodes = list(G.nodes())
node_idx = {node: i for i, node in enumerate(nodes)}

print(nodes)



#### Step 2: Feature Matrix (Centrality + Clustering)
#### ---------------------------------------------
# [DC, BC, CC, EC, Clustering Coefficient]
deg_centrality = nx.degree_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
clustering_coefficient = nx.clustering(G)

X = np.array([
    [
        deg_centrality[n],
        betweenness_centrality[n],
        closeness_centrality[n],
        eigenvector_centrality[n],
        clustering_coefficient[n]
    ]
    for n in nodes
])

#X = torch.tensor(X, dtype=torch.float32)


# In[19]:


print(X)
print(X.dtype)


# In[20]:


##### Step 5: Normalize the Feature Matrix ----
###scaler = StandardScaler()
###X_scaled = scaler.fit_transform(X)
from sklearn.preprocessing import normalize

X_scaled = normalize(X, norm='l2', axis=1)
X_scaled = torch.FloatTensor(X_scaled)


# In[21]:


print(X_scaled.dtype)


# In[22]:


#### Step 3: Ground Truth Label via SIR Model
# ---------------------------------------------
def run_sir(G, seed, beta, gamma, steps=10):
    infected = {seed}
    recovered = set()
    susceptible = set(G.nodes()) - infected
    for _ in range(steps):
        new_infected = set()
        for node in infected:
            for neighbor in G.neighbors(node):
                if neighbor in susceptible and random.random() < beta:
                    new_infected.add(neighbor)
        new_recovered = {n for n in infected if random.random() < gamma}
        infected = (infected | new_infected) - new_recovered
        recovered |= new_recovered
        susceptible -= new_infected
    return len(infected | recovered)

##### Parameters from the paper
avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes()
beta = 1 / avg_deg
gamma = 0.05

### Compute influence score as ground-truth label
y = []
for n in tqdm(G.nodes()):
    influence = np.mean([run_sir(G, n, beta, gamma) for _ in range(10)])
    y.append(influence)
y = torch.tensor(y, dtype=torch.float32)


# In[23]:


print(y.dtype)


# In[24]:


#### Step 4: Adjacency Matrix Normalization
# ---------------------------------------------
A = nx.adjacency_matrix(G, nodelist=list(G.nodes())).astype(float)
A_hat = A + sp.eye(A.shape[0])
D_hat = np.array(A_hat.sum(axis=1)).flatten()
D_inv_sqrt = sp.diags(np.power(D_hat, -0.5))
A_sym = D_inv_sqrt @ A_hat @ D_inv_sqrt
A_sym = torch.tensor(A_sym.toarray(), dtype=torch.float32)


# In[26]:


# Feature propagation: GCN input = A_sym @ X
AX = A_sym @ X_scaled

# Dataset split: 80% train, 20% test
n_nodes = AX.shape[0]
indices = np.arange(n_nodes)
np.random.shuffle(indices)
train_size = int(0.8 * n_nodes)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

# Define a single GCN Layer
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, A_sym, X_scaled):
        return F.relu(self.linear(A_sym @ X_scaled))

# Full Ip-GCN model combining GCN + LSTM
class IpGCN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=8, lstm_hidden=64):
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, 2)  # output dimension = 2
        self.lstm = nn.LSTM(2, lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, A_sym, X_scaled):
        h = self.gcn1(A_sym, X_scaled)
        h = self.gcn2(A_sym, h)
        h_seq = h.unsqueeze(1)  # reshape for LSTM input
        lstm_out, _ = self.lstm(h_seq)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out.squeeze()

# Initialize model, optimizer, and loss
model = IpGCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(A_sym, X_scaled)
    loss = loss_fn(output[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Train Loss: {loss.item():.4f}")

### Testing
model.eval()
with torch.no_grad():
    preds = model(A_sym, X_scaled)
    test_loss = loss_fn(preds[test_idx], y[test_idx])
    print(f"Test Loss: {test_loss.item():.4f}")

### Get top-10 influential nodes
influence_scores = preds.numpy()
top_k = 15
top_indices = np.argsort(-influence_scores)[:top_k]
top_nodes = [nodes[i] for i in top_indices]
print("Top Influential Nodes:", top_nodes)




