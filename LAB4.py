import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.linalg import svd, cond

#Question 1
df = px.data.stocks()
print(df.head().round(2))

# Question 2
fig = px.line(df, x='date', y=df.columns[1:], title="Stock Values - Major Tech Company")
fig.update_layout(
    font_family="Courier New",
    font=dict(size=30, family="Courier New"),
    title_x=0.5,
    title_font_family="Times New Roman",
    title_font_color="red",
    xaxis_title="Time",
    yaxis_title="Normalized ($)",
    xaxis_title_font_color="yellow",
    yaxis_title_font_color ="yellow",
    xaxis=dict(color="yellow"),
    yaxis=dict(color="yellow"),
    legend_title_font_color="green",
    legend_font_color="yellow",
    height = 800,
    width = 2000,
    template="plotly_dark"

)
for trace in fig.data:
    trace.line.width = 4
fig.show(renderer='browser')

#Question 3
fig = make_subplots(rows=3, cols=2)
fig.add_trace(
    go.Histogram(x=df['GOOG'], nbinsx=50, name='GOOG'),
    row=1, col=1,

)
fig.add_trace(
    go.Histogram(x=df['AAPL'], nbinsx=50, name='AAPL'),
    row=1, col=2
)
fig.add_trace(
    go.Histogram(x=df['AMZN'], nbinsx=50, name='AMZN'),
    row=2, col=1
)

fig.add_trace(
    go.Histogram(x=df['FB'], nbinsx=50, name='FB'),
    row=2, col=2
)
fig.add_trace(
    go.Histogram(x=df['NFLX'], nbinsx=50, name='NFLX'),
    row=3, col=1
)

fig.add_trace(
    go.Histogram(x=df['MSFT'], nbinsx=50, name='MSFT'),
    row=3, col=2
)
fig.update_layout(title_text="Side By Side Subplots")

fig.update_layout(
    title="Histogram Plot",
    title_x=0.5,
    title_font_color="red",
    title_font_family="Times New Roman",
    font=dict(size=15, family="Courier New"),
    title_font_size=30,
    xaxis=dict(title_text="Normalized Price ($)", title_font=dict(color="black")),
    yaxis=dict(title_text="Frequency", title_font=dict(color="black")),
    legend_title_font_color="green",
    legend_font_size=30,
    showlegend=True,  # Show the legend
)

fig.update_xaxes(title_text="Normalized Price ($)", title_font=dict(color="black"), row=1, col=1)
fig.update_xaxes(title_text="Normalized Price ($)", title_font=dict(color="black"), row=1, col=2)
fig.update_xaxes(title_text="Normalized Price ($)", title_font=dict(color="black"), row=2, col=1)
fig.update_xaxes(title_text="Normalized Price ($)", title_font=dict(color="black"), row=2, col=2)
fig.update_xaxes(title_text="Normalized Price ($)", title_font=dict(color="black"), row=3, col=1)
fig.update_xaxes(title_text="Normalized Price ($)", title_font=dict(color="black"), row=3, col=2)
fig.update_yaxes(title_text="Frequency", title_font=dict(color="black"), row=1, col=1)
fig.update_yaxes(title_text="Frequency", title_font=dict(color="black"), row=1, col=2)
fig.update_yaxes(title_text="Frequency", title_font=dict(color="black"), row=2, col=1)
fig.update_yaxes(title_text="Frequency", title_font=dict(color="black"), row=2, col=2)
fig.update_yaxes(title_text="Frequency", title_font=dict(color="black"), row=3, col=1)
fig.update_yaxes(title_text="Frequency", title_font=dict(color="black"), row=3, col=2)

fig.show(renderer='browser')
# ---------------------------
# Question 4
# ---------------------------

#a

scaler = StandardScaler()
X = df.iloc[:, 1:]
X_std = scaler.fit_transform(X)

# b
_, S, _ = np.linalg.svd(X_std)
cond_no = np.linalg.cond(X_std)

S_rounded = np.round(S, 2)
cond_no_rounded = round(cond_no, 2)

print("Singular values of original data:", S_rounded)
print("Condition number of original data:", cond_no_rounded)

# c
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(pd.DataFrame(X_std).corr(), annot=True,  ax=ax, fmt=".2f",
            xticklabels=df.columns[1:], yticklabels=df.columns[1:])

plt.title("Correlation Coefficient between features- Original feature space")
plt.show()

# d
pca = PCA(svd_solver="full", random_state=5764)
X_pca = pca.fit_transform(X_std)
print("Explained Variance Ratio of original : ", (pca.explained_variance_ratio_).round(2))

pca = PCA(svd_solver="full", n_components=0.95, random_state=5764)
X_pca = pca.fit_transform(X_std)
print(f'Orginal data shape: {X_std.shape}')
print(f'Reduced data shape: {X_pca.shape}')
print("Explained Variance Ratio of reduced : ", (pca.explained_variance_ratio_).round(2))
print(
    "Number of features needed to explain more than 95% of the dependent variance:",
    np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0] + 1,
)
print("Number of features to be removed:", X_std.shape[1] - X_pca.shape[1])
#e
import numpy as np
import matplotlib.pyplot as plt
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
plt.figure(figsize=(15, 15))
plt.plot(
    np.arange(1, len(cumulative_variance) + 1),
    cumulative_variance,
    label="Cumulative Explained Variance",
)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.axvline(
    x=num_components_95,
    color="black",
    linestyle="--",
    label=f"Components for 95% Variance: {num_components_95}",
)

plt.axhline(y=0.95, color="red", linestyle="--", label="95% Explained Variance")

plt.legend()
plt.tight_layout()
plt.show()


# f
_, S_pca, _ = np.linalg.svd(X_pca)
cond_no_pca = np.linalg.cond(X_pca)
S_roundedp = np.round(S_pca, 2)
cond_no_roundedp = round(cond_no_pca, 2)

print("Singular values of original data:", S_roundedp)
print("Condition number of original data:", cond_no_roundedp)


# g
correlation_matrix = pd.DataFrame(X_pca).corr().round(2)
print(correlation_matrix)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(pd.DataFrame(X_pca).corr(), annot=True, ax=ax, fmt=".2f")
plt.title("Correlation Coefficient Matrix Heatmap of Transformed Data")
plt.show()

# h
df_pca = pd.DataFrame(
    X_pca, columns=["Principal col " + str(i) for i in range(1, X_pca.shape[1] + 1)]
)
print(df_pca.head(5).round(2))

# i
fig = px.line(
    df_pca, x=df.date, y=df_pca.columns, title="Stock Values - Major Tech Companies"
)

fig.update_layout(
    title=dict(
        text="Stock Values - Major Tech Companies",
        font=dict(color="red", size=30, family="Times New Roman"),
        x=0.5,
    ),
    xaxis=dict(
        title="Time",
        title_font=dict(color="yellow", size=30, family="Courier New"),
        tickfont=dict(color="yellow", size=25),
    ),
    yaxis=dict(
        title="Normalized ($)",
        title_font=dict(color="yellow", size=30, family="Courier New"),
        tickfont=dict(color="yellow", size=25),
    ),
    legend=dict(title="variable", title_font=dict(color="green")),
    legend_font_color="yellow",
    width=2000,
    height=800,
    template="plotly_dark",
)

fig.show(renderer="browser")

# j
fig = make_subplots(rows=df_pca.shape[1], cols=1)

for i, company in enumerate(df_pca.columns):
    fig.add_trace(
        go.Histogram(x=df_pca[company], nbinsx=50, name=company), row=i + 1, col=1
    )
    fig.update_xaxes(title_text="Normalized Price ($)", row=i + 1, col=1)
    fig.update_yaxes(title_text="Frequency", row=i + 1, col=1)

fig.update_layout(
    title_text="Histogram Plot",
    title_font=dict(color="red", size=30, family="Times New Roman"),
    title_x=0.5,
    legend_title_font_color="green",
)

fig.show(renderer="browser")

# k

# Original feature space
fig = px.scatter_matrix(df, dimensions=df.columns[1:])

fig.update_layout(
    title_text="Scatter Matrix of Original Feature Space",
    title_font=dict(color="red", size=30, family="Times New Roman"),
    title_x=0.5,
    legend_title_font_color="green"
)
fig.update_traces(diagonal_visible=False)

fig.show(renderer="browser")

# Reduced feature space
fig = px.scatter_matrix(df_pca, dimensions=df_pca.columns)

fig.update_layout(
    title_text="Scatter Matrix of Reduced Feature Space",
    title_font=dict(color="red", size=30, family="Times New Roman"),
    title_x=0.5,
    legend_title_font_color="green"
)

fig.update_traces(diagonal_visible=False)

fig.show(renderer="browser")

