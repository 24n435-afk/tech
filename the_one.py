import numpy as np
import matplotlib.pyplot as plt

# Q1: Single Perceptron (Linear Neuron) for Regression with Gradient Descent

X = np.array([500, 800, 1000, 1200, 1500], dtype=np.float64).reshape(-1, 1)
y = np.array([150, 220, 300, 360, 450], dtype=np.float64).reshape(-1, 1)

# Normalize for stable training
a_min, a_max = X.min(), X.max()
Xn = (X - a_min) / (a_max - a_min)
yn = y / 1000.0

# Manual initialization
w = 0.10
b = 0.05
lr = 0.1
epochs = 1000

loss_history = []

for epoch in range(epochs):
    y_pred = Xn * w + b
    error = y_pred - yn
    loss = np.mean(error ** 2)
    loss_history.append(loss)

    # Gradients for MSE
    dw = np.mean(2.0 * error * Xn)
    db = np.mean(2.0 * error)

    # Update
    w -= lr * dw
    b -= lr * db

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1:4d} | Loss: {loss:.6f} | w: {w:.4f} | b: {b:.4f}")

# Prediction for 1100 sq.ft
def predict_price(area_sqft):
    area_n = (area_sqft - a_min) / (a_max - a_min)
    pred_n = area_n * w + b
    return float(pred_n * 1000.0)

pred_1100 = predict_price(1100)
print(f"\nPredicted price for 1100 sq.ft: {pred_1100:.2f} (thousand)")

# Plot regression line
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_line_n = (x_line - a_min) / (a_max - a_min)
y_line = (x_line_n * w + b) * 1000.0

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="royalblue", label="Data points")
plt.plot(x_line, y_line, color="crimson", label="Regression line")
plt.scatter([1100], [pred_1100], color="green", marker="x", s=80, label="Prediction @ 1100")
plt.title("Q1: Perceptron Regression (From Scratch)")
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price (thousand)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()



import numpy as np

# Q2: MLP (2 -> 3 -> 1), sigmoid activation, manual backpropagation

X = np.array([
    [2, 50],
    [4, 60],
    [6, 70],
    [8, 80]
], dtype=np.float64)

y = np.array([[0], [0], [1], [1]], dtype=np.float64)

# Normalize features for stable training
Xn = X.copy()
Xn[:, 0] = Xn[:, 0] / 10.0       # Study hours
Xn[:, 1] = Xn[:, 1] / 100.0      # Attendance

np.random.seed(42)
W1 = np.random.randn(2, 3) * 0.5
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.5
b2 = np.zeros((1, 1))

lr = 0.5
epochs = 50

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def dsigmoid(a):
    return a * (1.0 - a)

for epoch in range(epochs):
    # Forward
    z1 = Xn @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)

    # Binary cross-entropy loss
    eps = 1e-8
    loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    # Backprop
    dz2 = (y_hat - y)  # BCE + sigmoid simplification
    dW2 = (a1.T @ dz2) / len(Xn)
    db2 = np.mean(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * dsigmoid(a1)
    dW1 = (Xn.T @ dz1) / len(Xn)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch:2d} | Loss: {loss:.6f}")

# Evaluation
probs = sigmoid(sigmoid(Xn @ W1 + b1) @ W2 + b2)
preds = (probs >= 0.5).astype(int)
acc = np.mean(preds == y)

print("\nPredicted labels:", preds.ravel())
print("True labels     :", y.ravel().astype(int))
print(f"Accuracy: {acc * 100:.2f}%")

# Predict for (Study Hours=5, Attendance=65)
new_x = np.array([[5 / 10.0, 65 / 100.0]])
new_prob = sigmoid(sigmoid(new_x @ W1 + b1) @ W2 + b2)[0, 0]
new_cls = int(new_prob >= 0.5)
print(f"Prediction for (5, 65): prob={new_prob:.4f}, class={new_cls}")



import numpy as np
import tensorflow as tf

# Q3: Core Transformer components with TensorFlow ops + manual attention demonstration

X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.float32)  # (seq_len=3, d_model=3)
X = tf.constant(X[None, :, :])  # (batch=1, seq_len=3, d_model=3)


def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(pe[None, :, :], dtype=tf.float32)


def scaled_dot_product_attention(q, k, v):
    logits = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    weights = tf.nn.softmax(logits / tf.math.sqrt(dk), axis=-1)
    out = tf.matmul(weights, v)
    return out, weights


class SimpleMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wo = tf.keras.layers.Dense(d_model)

    def split_heads(self, x):
        b = tf.shape(x)[0]
        s = tf.shape(x)[1]
        x = tf.reshape(x, (b, s, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        q = self.split_heads(self.wq(x))
        k = self.split_heads(self.wk(x))
        v = self.split_heads(self.wv(x))
        attn_out, weights = scaled_dot_product_attention(q, k, v)
        attn_out = tf.transpose(attn_out, perm=[0, 2, 1, 3])
        b = tf.shape(attn_out)[0]
        s = tf.shape(attn_out)[1]
        concat = tf.reshape(attn_out, (b, s, self.d_model))
        return self.wo(concat), weights


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super().__init__()
        self.mha = SimpleMultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        attn_out, weights = self.mha(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, weights


# 1) Positional Encoding + manual attention score computation
pe = positional_encoding(seq_len=3, d_model=3)
X_pe = X + pe

manual_q = X_pe
manual_k = X_pe
manual_v = X_pe
manual_out, manual_weights = scaled_dot_product_attention(manual_q, manual_k, manual_v)
print("Manual attention weights:\n", manual_weights.numpy()[0])
print("Manual attention output:\n", manual_out.numpy()[0])

# 2) Minimal encoder pass
d_model = 3
block = EncoderBlock(d_model=d_model, num_heads=1, dff=8)
enc_out, w = block(X_pe)
print("\nEncoder output:\n", enc_out.numpy()[0])

# 3) Next-vector prediction (sequence-to-next-vector)
# Build toy training pairs from cyclic shifts to keep demo small
train_inputs = tf.concat([X_pe, tf.roll(X_pe, shift=1, axis=1), tf.roll(X_pe, shift=2, axis=1)], axis=0)
train_targets = tf.concat([
    X_pe[:, -1, :],
    tf.roll(X_pe, shift=1, axis=1)[:, -1, :],
    tf.roll(X_pe, shift=2, axis=1)[:, -1, :]
], axis=0)

inp = tf.keras.Input(shape=(3, 3))
z, _ = EncoderBlock(d_model=3, num_heads=1, dff=8)(inp)
last = z[:, -1, :]
out = tf.keras.layers.Dense(3)(last)
model = tf.keras.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mse")
model.fit(train_inputs, train_targets, epochs=20, verbose=0)

pred_next = model.predict(X_pe, verbose=0)[0]
print("\nPredicted next vector for original sequence:", np.round(pred_next, 4))



import numpy as np

# Q4: XOR with MLP (from scratch), compare learning rates 0.01 and 0.1 for 10 epochs

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(a):
    return a * (1 - a)


def tanh(z):
    return np.tanh(z)


def dtanh(a):
    return 1 - a ** 2


def train_xor(lr, epochs=10, hidden_activation="tanh"):
    np.random.seed(7)
    W1 = np.random.randn(2, 4) * 0.5
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 0.5
    b2 = np.zeros((1, 1))

    for epoch in range(epochs):
        z1 = X @ W1 + b1
        if hidden_activation == "tanh":
            a1 = tanh(z1)
            d_hidden = dtanh(a1)
        else:
            a1 = sigmoid(z1)
            d_hidden = dsigmoid(a1)

        z2 = a1 @ W2 + b2
        y_hat = sigmoid(z2)

        loss = np.mean((y - y_hat) ** 2)

        # Backprop
        dz2 = (y_hat - y) * dsigmoid(y_hat)
        dW2 = a1.T @ dz2 / len(X)
        db2 = np.mean(dz2, axis=0, keepdims=True)

        dz1 = (dz2 @ W2.T) * d_hidden
        dW1 = X.T @ dz1 / len(X)
        db1 = np.mean(dz1, axis=0, keepdims=True)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        print(f"lr={lr:.2f} | epoch={epoch + 1:2d} | loss={loss:.6f}")

    preds = (sigmoid((tanh(X @ W1 + b1) if hidden_activation == 'tanh' else sigmoid(X @ W1 + b1)) @ W2 + b2) >= 0.5).astype(int)
    return preds


for learning_rate in [0.01, 0.1]:
    print("\n" + "=" * 60)
    preds = train_xor(lr=learning_rate, epochs=10, hidden_activation="tanh")
    print("Predictions:", preds.ravel())
    print("Targets    :", y.ravel().astype(int))




import numpy as np
import tensorflow as tf

# Q5: Transformer on pixel sequence with positional encoding + scaled dot-product attention

sequence = np.array([1, 0, 1, 1, 0], dtype=np.float32)

# Build next-token dataset from single sequence
X_train = np.array([[1, 0, 1, 1]], dtype=np.float32)
y_train = np.array([[0]], dtype=np.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(i / 2.0)) / tf.cast(d_model, tf.float32))
        angles = pos * angle_rates
        sin = tf.sin(angles[:, 0::2])
        cos = tf.cos(angles[:, 1::2])
        pe = tf.concat([sin, cos], axis=-1)
        pe = pe[tf.newaxis, :, :]
        pe = tf.pad(pe, [[0, 0], [0, 0], [0, tf.maximum(0, d_model - tf.shape(pe)[-1])]])
        pe = pe[:, :, :d_model]
        return x + pe


class ScaledDotAttention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

    def call(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        logits = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        weights = tf.nn.softmax(logits / tf.math.sqrt(dk), axis=-1)
        out = tf.matmul(weights, v)
        return out, weights


def build_model(lr):
    inp = tf.keras.Input(shape=(4, 1))
    x = PositionalEncoding()(inp)
    attn_out, _ = ScaledDotAttention(d_model=8)(x)
    x = tf.keras.layers.Dense(8, activation="relu")(attn_out)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy")
    return model


X_train_tf = X_train[:, :, None]

for lr in [0.001, 0.01]:
    print("\n" + "=" * 60)
    print(f"Training with learning rate = {lr}")
    model = build_model(lr)
    history = model.fit(X_train_tf, y_train, epochs=10, verbose=0)
    print("Loss history:", [round(v, 6) for v in history.history["loss"]])

    pred = model.predict(X_train_tf, verbose=0)[0, 0]
    print(f"Predicted next pixel probability: {pred:.4f}")
    print(f"Predicted next pixel value      : {int(pred >= 0.5)}")



# Q6 is the same problem statement as Q5.
# This file intentionally mirrors Q5 with an independent run script.

import numpy as np
import tensorflow as tf

sequence = np.array([1, 0, 1, 1, 0], dtype=np.float32)
X_train = np.array([[1, 0, 1, 1]], dtype=np.float32)
y_train = np.array([[0]], dtype=np.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(i / 2.0)) / tf.cast(d_model, tf.float32))
        angles = pos * angle_rates
        sin = tf.sin(angles[:, 0::2])
        cos = tf.cos(angles[:, 1::2])
        pe = tf.concat([sin, cos], axis=-1)
        pe = pe[tf.newaxis, :, :]
        pe = tf.pad(pe, [[0, 0], [0, 0], [0, tf.maximum(0, d_model - tf.shape(pe)[-1])]])
        pe = pe[:, :, :d_model]
        return x + pe


class ScaledDotAttention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

    def call(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        logits = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        weights = tf.nn.softmax(logits / tf.math.sqrt(dk), axis=-1)
        out = tf.matmul(weights, v)
        return out, weights


def build_model(lr):
    inp = tf.keras.Input(shape=(4, 1))
    x = PositionalEncoding()(inp)
    attn_out, _ = ScaledDotAttention(d_model=8)(x)
    x = tf.keras.layers.Dense(8, activation="relu")(attn_out)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy")
    return model


X_train_tf = X_train[:, :, None]

for lr in [0.001, 0.01]:
    print("\n" + "=" * 60)
    print(f"Training with learning rate = {lr}")
    model = build_model(lr)
    history = model.fit(X_train_tf, y_train, epochs=10, verbose=0)
    print("Loss history:", [round(v, 6) for v in history.history["loss"]])

    pred = model.predict(X_train_tf, verbose=0)[0, 0]
    print(f"Predicted next pixel probability: {pred:.4f}")
    print(f"Predicted next pixel value      : {int(pred >= 0.5)}")




import numpy as np
import tensorflow as tf

# Q7: GAN using pixel vectors

real_data = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1]
], dtype=np.float32)

noise_dim = 3


def build_generator():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(noise_dim,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(4, activation="sigmoid")
    ])


def build_discriminator():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])


bce = tf.keras.losses.BinaryCrossentropy()


def train_gan(lr, epochs=10):
    gen = build_generator()
    disc = build_discriminator()

    g_opt = tf.keras.optimizers.Adam(lr)
    d_opt = tf.keras.optimizers.Adam(lr)

    batch_size = real_data.shape[0]

    for epoch in range(epochs):
        noise = tf.random.normal((batch_size, noise_dim))

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            fake = gen(noise, training=True)

            real_pred = disc(real_data, training=True)
            fake_pred = disc(fake, training=True)

            d_loss = bce(tf.ones_like(real_pred), real_pred) + bce(tf.zeros_like(fake_pred), fake_pred)
            g_loss = bce(tf.ones_like(fake_pred), fake_pred)

        d_grads = d_tape.gradient(d_loss, disc.trainable_variables)
        g_grads = g_tape.gradient(g_loss, gen.trainable_variables)

        d_opt.apply_gradients(zip(d_grads, disc.trainable_variables))
        g_opt.apply_gradients(zip(g_grads, gen.trainable_variables))

        print(f"lr={lr:.3f} | epoch={epoch + 1:2d} | d_loss={d_loss.numpy():.4f} | g_loss={g_loss.numpy():.4f}")

    # Generate new patterns
    z = tf.random.normal((5, noise_dim))
    generated = gen(z, training=False).numpy()
    binary_generated = (generated >= 0.5).astype(int)
    return generated, binary_generated


for lr in [0.001, 0.01]:
    print("\n" + "=" * 70)
    print(f"Training GAN with learning rate {lr}")
    soft, hard = train_gan(lr=lr, epochs=10)
    print("Generated (probabilities):\n", np.round(soft, 3))
    print("Generated (binary patterns):\n", hard)



import numpy as np
import tensorflow as tf

# Q8: Self-attention on pixel vector

x = tf.constant([[1., 0., 1., 0., 1.]], dtype=tf.float32)  # shape (1, 5)

# Treat each pixel as a token with dim=1 => reshape to (batch, seq, dim)
x_tokens = tf.reshape(x, (1, 5, 1))


def run_attention(lr, steps=10):
    # Trainable projections for Q, K, V
    Wq = tf.Variable(tf.random.normal((1, 4), stddev=0.2), name="Wq")
    Wk = tf.Variable(tf.random.normal((1, 4), stddev=0.2), name="Wk")
    Wv = tf.Variable(tf.random.normal((1, 4), stddev=0.2), name="Wv")

    optimizer = tf.keras.optimizers.Adam(lr)

    for step in range(steps):
        with tf.GradientTape() as tape:
            Q = tf.matmul(x_tokens, Wq)  # (1,5,4)
            K = tf.matmul(x_tokens, Wk)  # (1,5,4)
            V = tf.matmul(x_tokens, Wv)  # (1,5,4)

            scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(4, tf.float32))
            weights = tf.nn.softmax(scores, axis=-1)
            attn_out = tf.matmul(weights, V)  # (1,5,4)

            # Projection to scalar per token and reconstruction loss
            recon = tf.keras.layers.Dense(1, activation="sigmoid")(attn_out)
            loss = tf.reduce_mean((recon - x_tokens) ** 2)

        vars_ = [Wq, Wk, Wv] + [v for v in tf.compat.v1.global_variables() if "dense" in v.name.lower()]
        grads = tape.gradient(loss, vars_)

        # Filter None gradients (possible for graph-created vars)
        grad_var = [(g, v) for g, v in zip(grads, vars_) if g is not None]
        optimizer.apply_gradients(grad_var)

        if (step + 1) % 2 == 0:
            print(f"lr={lr:.3f} | step={step + 1:2d} | loss={loss.numpy():.6f}")

    # Final forward for importance
    Q = tf.matmul(x_tokens, Wq)
    K = tf.matmul(x_tokens, Wk)
    V = tf.matmul(x_tokens, Wv)
    scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(4, tf.float32))
    weights = tf.nn.softmax(scores, axis=-1).numpy()[0]  # (5,5)
    importance = weights.mean(axis=0)
    most_important_idx = int(np.argmax(importance))

    print("Attention weights:\n", np.round(weights, 3))
    print("Pixel importance:", np.round(importance, 3))
    print("Most important pixel index:", most_important_idx)


for lr in [0.001, 0.01]:
    print("\n" + "=" * 65)
    print(f"Self-attention training with lr={lr}")
    run_attention(lr=lr, steps=10)





import numpy as np
import tensorflow as tf

# Q9: RNN on pixel row sequences

# Sequence of rows (3 timesteps, 3 features)
X = np.array([
    [[1, 1, 1],
     [0, 1, 0],
     [1, 1, 1]],
    [[0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]]
], dtype=np.float32)

y = np.array([[1], [0]], dtype=np.float32)  # 1 = plus-like, 0 = non-pattern


def build_model(lr):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3, 3)),
        tf.keras.layers.SimpleRNN(8, activation="tanh"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


for lr in [0.01, 0.1]:
    print("\n" + "=" * 60)
    print(f"Training RNN with learning rate = {lr}")
    model = build_model(lr)
    hist = model.fit(X, y, epochs=10, verbose=0)

    final_loss = hist.history["loss"][-1]
    final_acc = hist.history["accuracy"][-1]
    print(f"Final loss: {final_loss:.6f}, Final accuracy: {final_acc:.4f}")

    probs = model.predict(X, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    print("Predictions:", preds)
    print("Targets    :", y.ravel().astype(int))

# Predict class of the original pattern
sample = np.array([[[1, 1, 1], [0, 1, 0], [1, 1, 1]]], dtype=np.float32)
model = build_model(0.01)
model.fit(X, y, epochs=10, verbose=0)
prob = model.predict(sample, verbose=0)[0, 0]
print(f"\nPredicted class for sample pattern: {int(prob >= 0.5)} (prob={prob:.4f})")





import numpy as np

# Q10: CNN for cross pattern recognition (from scratch)

X_train = np.array([
    [[0, 1, 0, 0],
     [1, 1, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 0]],
    [[0, 0, 0, 0],
     [0, 1, 1, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
], dtype=np.float64)

y_train = np.array([1.0, 0.0], dtype=np.float64)


class SimpleCNNClassifier:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.kernel = rng.normal(0, 0.2, (2, 2))
        self.w = 0.2
        self.b = 0.0

    def conv2d(self, x):
        h, w = x.shape
        kh, kw = self.kernel.shape
        out = np.zeros((h - kh + 1, w - kw + 1))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                region = x[i:i + kh, j:j + kw]
                out[i, j] = np.sum(region * self.kernel)
        return out

    @staticmethod
    def relu(z):
        return np.maximum(z, 0)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, x):
        conv = self.conv2d(x)
        act = self.relu(conv)
        feat = np.sum(act)
        logit = self.w * feat + self.b
        prob = self.sigmoid(logit)
        return conv, act, feat, prob

    def train(self, X, y, lr=0.01, epochs=10):
        for epoch in range(epochs):
            total_loss = 0.0
            for xi, yi in zip(X, y):
                conv, act, feat, prob = self.forward(xi)
                loss = -(yi * np.log(prob + 1e-8) + (1 - yi) * np.log(1 - prob + 1e-8))
                total_loss += loss

                # Binary cross entropy gradient
                dlogit = prob - yi
                dw = dlogit * feat
                db = dlogit

                # Backprop to kernel through sum(ReLU(conv))
                dfeat = dlogit * self.w
                dact = np.ones_like(act) * dfeat
                dconv = dact * (conv > 0)

                dkernel = np.zeros_like(self.kernel)
                kh, kw = self.kernel.shape
                for i in range(dconv.shape[0]):
                    for j in range(dconv.shape[1]):
                        region = xi[i:i + kh, j:j + kw]
                        dkernel += dconv[i, j] * region

                self.w -= lr * dw
                self.b -= lr * db
                self.kernel -= lr * dkernel

            print(f"lr={lr:.2f} | epoch={epoch + 1:2d} | loss={total_loss / len(X):.6f}")


for lr in [0.01, 0.1]:
    print("\n" + "=" * 65)
    model = SimpleCNNClassifier(seed=42)
    model.train(X_train, y_train, lr=lr, epochs=10)

    test_pattern = np.array([
        [0, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.float64)

    conv_map, act_map, _, prob = model.forward(test_pattern)
    pred = int(prob >= 0.5)
    print("\nPredicted class for new pattern:", pred, f"(prob={prob:.4f})")
    print("Activation map:\n", np.round(act_map, 3))




import numpy as np
import tensorflow as tf

# Q11: Noise removal using autoencoder

clean = np.array([[1, 1, 1,
                   1, 0, 1,
                   1, 1, 1]], dtype=np.float32)

noisy = np.array([[1, 0, 1,
                   1, 1, 1,
                   1, 0, 1]], dtype=np.float32)


def build_autoencoder(lr):
    inp = tf.keras.Input(shape=(9,))
    encoded = tf.keras.layers.Dense(6, activation="relu")(inp)
    bottleneck = tf.keras.layers.Dense(3, activation="relu")(encoded)
    decoded = tf.keras.layers.Dense(6, activation="relu")(bottleneck)
    out = tf.keras.layers.Dense(9, activation="sigmoid")(decoded)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


results = {}
for lr in [0.001, 0.01]:
    print("\n" + "=" * 60)
    print(f"Training autoencoder with learning rate = {lr}")
    model = build_autoencoder(lr)
    hist = model.fit(noisy, clean, epochs=10, verbose=0)
    final_loss = hist.history["loss"][-1]
    results[lr] = final_loss

    denoised = model.predict(noisy, verbose=0)[0]
    denoised_bin = (denoised >= 0.5).astype(int)

    print("Loss history:", [round(v, 6) for v in hist.history["loss"]])
    print("Denoised output (prob):", np.round(denoised, 3))
    print("Denoised output (bin) :", denoised_bin)

print("\nLoss difference (0.001 - 0.01):", round(results[0.001] - results[0.01], 6))



# Q12 repeats the same task as Q11. Kept as separate runnable file.

import numpy as np
import tensorflow as tf

clean = np.array([[1, 1, 1,
                   1, 0, 1,
                   1, 1, 1]], dtype=np.float32)

noisy = np.array([[1, 0, 1,
                   1, 1, 1,
                   1, 0, 1]], dtype=np.float32)


def build_autoencoder(lr):
    inp = tf.keras.Input(shape=(9,))
    encoded = tf.keras.layers.Dense(6, activation="relu")(inp)
    bottleneck = tf.keras.layers.Dense(3, activation="relu")(encoded)
    decoded = tf.keras.layers.Dense(6, activation="relu")(bottleneck)
    out = tf.keras.layers.Dense(9, activation="sigmoid")(decoded)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


results = {}
for lr in [0.001, 0.01]:
    print("\n" + "=" * 60)
    print(f"Training autoencoder with learning rate = {lr}")
    model = build_autoencoder(lr)
    hist = model.fit(noisy, clean, epochs=10, verbose=0)
    final_loss = hist.history["loss"][-1]
    results[lr] = final_loss

    denoised = model.predict(noisy, verbose=0)[0]
    denoised_bin = (denoised >= 0.5).astype(int)

    print("Loss history:", [round(v, 6) for v in hist.history["loss"]])
    print("Denoised output (prob):", np.round(denoised, 3))
    print("Denoised output (bin) :", denoised_bin)

print("\nLoss difference (0.001 - 0.01):", round(results[0.001] - results[0.01], 6))



import numpy as np

# Q13: MLP for 2x2 pixel brightness classification (from scratch)

# Data: bright=1, dark=0
X = np.array([
    [255, 255, 255, 255],
    [10, 10, 10, 10]
], dtype=np.float64)

y = np.array([[1.0], [0.0]], dtype=np.float64)

# Normalize to [0, 1]
Xn = X / 255.0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(a):
    return a * (1.0 - a)


def train_mlp(lr, epochs=10):
    np.random.seed(21)
    W1 = np.random.randn(4, 3) * 0.3
    b1 = np.zeros((1, 3))
    W2 = np.random.randn(3, 1) * 0.3
    b2 = np.zeros((1, 1))

    losses = []

    for epoch in range(epochs):
        h = sigmoid(Xn @ W1 + b1)
        out = sigmoid(h @ W2 + b2)

        loss = np.mean((y - out) ** 2)
        losses.append(loss)

        # Backprop (MSE)
        d_out = (out - y) * dsigmoid(out)
        dW2 = h.T @ d_out / len(Xn)
        db2 = np.mean(d_out, axis=0, keepdims=True)

        d_h = (d_out @ W2.T) * dsigmoid(h)
        dW1 = Xn.T @ d_h / len(Xn)
        db1 = np.mean(d_h, axis=0, keepdims=True)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        print(f"lr={lr:.3f} | epoch={epoch + 1:2d} | loss={loss:.6f}")

    probs = sigmoid(sigmoid(Xn @ W1 + b1) @ W2 + b2)
    preds = (probs >= 0.5).astype(int)
    acc = np.mean(preds == y)

    # Decision boundary in terms of average pixel intensity
    vals = np.linspace(0, 255, 100)
    grid = np.stack([vals, vals, vals, vals], axis=1) / 255.0
    grid_prob = sigmoid(sigmoid(grid @ W1 + b1) @ W2 + b2).ravel()
    boundary_idx = np.argmin(np.abs(grid_prob - 0.5))
    boundary_val = vals[boundary_idx]

    print("Predictions:", preds.ravel())
    print("Accuracy:", round(float(acc), 4))
    print("Approx decision boundary avg-intensity ~", round(float(boundary_val), 2))

    # Predict class for new input
    new_pixel = np.array([[120, 120, 120, 120]], dtype=np.float64) / 255.0
    new_prob = sigmoid(sigmoid(new_pixel @ W1 + b1) @ W2 + b2)[0, 0]
    print(f"Prediction for [120,120,120,120]: class={int(new_prob >= 0.5)}, prob={new_prob:.4f}")


for lr in [0.001, 0.01]:
    print("\n" + "=" * 65)
    train_mlp(lr=lr, epochs=10)



import numpy as np

# Q14: Edge detection with manual kernel + learnable CNN layer (from scratch)

img = np.array([
    [10, 10, 10],
    [0, 0, 0],
    [10, 10, 10]
], dtype=np.float64)

# 1) Manual kernel (horizontal edge detector)
manual_kernel = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
], dtype=np.float64)


def conv2d_valid(x, k):
    h, w = x.shape
    kh, kw = k.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i:i + kh, j:j + kw] * k)
    return out


manual_out = conv2d_valid(img, manual_kernel)
print("Manual kernel:\n", manual_kernel)
print("Manual convolution output:\n", manual_out)

# 2) Learnable CNN kernel to match target edge map
# For 3x3 input and 2x2 kernel => feature map is 2x2
target_map = np.array([
    [5, 5],
    [5, 5]
], dtype=np.float64)


def train_kernel(lr, epochs=10):
    rng = np.random.default_rng(0)
    k = rng.normal(0, 0.1, (2, 2))

    for epoch in range(epochs):
        feat = conv2d_valid(img, k)
        loss = np.mean((feat - target_map) ** 2)

        dfeat = 2 * (feat - target_map) / feat.size
        dk = np.zeros_like(k)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                dk += dfeat[i, j] * img[i:i + 2, j:j + 2]

        k -= lr * dk
        print(f"lr={lr:.2f} | epoch={epoch + 1:2d} | loss={loss:.6f}")

    return k


for lr in [0.01, 0.5]:
    print("\n" + "=" * 65)
    learned_k = train_kernel(lr=lr, epochs=10)
    learned_map = conv2d_valid(img, learned_k)
    print("Learned kernel:\n", np.round(learned_k, 3))
    print("Learned feature map:\n", np.round(learned_map, 3))




import numpy as np

# Q15: CNN (Conv + MaxPool) from scratch on 5x5 digit-like patterns

# Digit 0
img0 = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
], dtype=np.float64)

# Digit 1
img1 = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1]
], dtype=np.float64)

X = np.stack([img0, img1])
y = np.array([0.0, 1.0], dtype=np.float64)


def conv2d_valid(x, k):
    h, w = x.shape
    kh, kw = k.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i:i + kh, j:j + kw] * k)
    return out


def maxpool2x2(x):
    # x: 4x4 -> pooled 2x2
    pooled = np.zeros((2, 2))
    mask = np.zeros_like(x)
    for i in range(2):
        for j in range(2):
            patch = x[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2]
            idx = np.unravel_index(np.argmax(patch), patch.shape)
            pooled[i, j] = patch[idx]
            mask[i * 2 + idx[0], j * 2 + idx[1]] = 1
    return pooled, mask


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train(lr, epochs=10):
    rng = np.random.default_rng(1)
    kernel = rng.normal(0, 0.2, (2, 2))
    w = rng.normal(0, 0.2, (4,))
    b = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        for xi, yi in zip(X, y):
            conv = conv2d_valid(xi, kernel)      # 4x4
            relu = np.maximum(conv, 0)
            pool, pool_mask = maxpool2x2(relu)   # 2x2
            feat = pool.reshape(-1)              # 4
            logit = np.dot(feat, w) + b
            prob = sigmoid(logit)

            loss = -(yi * np.log(prob + 1e-8) + (1 - yi) * np.log(1 - prob + 1e-8))
            total_loss += loss

            # Backprop classifier
            dlogit = prob - yi
            dw = dlogit * feat
            db = dlogit

            dfeat = dlogit * w
            dpool = dfeat.reshape(2, 2)

            # Backprop maxpool to relu map
            drelu = np.zeros_like(relu)
            for i in range(2):
                for j in range(2):
                    patch_mask = pool_mask[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2]
                    drelu[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] += dpool[i, j] * patch_mask

            dconv = drelu * (conv > 0)

            dkernel = np.zeros_like(kernel)
            for i in range(dconv.shape[0]):
                for j in range(dconv.shape[1]):
                    dkernel += dconv[i, j] * xi[i:i + 2, j:j + 2]

            w -= lr * dw
            b -= lr * db
            kernel -= lr * dkernel

        print(f"lr={lr:.2f} | epoch={epoch + 1:2d} | loss={total_loss / len(X):.6f}")

    return kernel, w, b


def predict(x, kernel, w, b):
    conv = conv2d_valid(x, kernel)
    relu = np.maximum(conv, 0)
    pool, _ = maxpool2x2(relu)
    feat = pool.reshape(-1)
    prob = sigmoid(np.dot(feat, w) + b)
    return int(prob >= 0.5), prob, conv, pool


for lr in [0.01, 0.1]:
    print("\n" + "=" * 70)
    kernel, w, b = train(lr=lr, epochs=10)

    # New sample (slightly noisy one-like pattern)
    new_pattern = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ], dtype=np.float64)

    cls, prob, conv_map, pool_map = predict(new_pattern, kernel, w, b)
    print(f"Predicted class for new pattern: {cls} (prob={prob:.4f})")
    print("Feature extraction:")
    print("Conv map:\n", np.round(conv_map, 3))
    print("MaxPool map:\n", np.round(pool_map, 3))




import numpy as np
import tensorflow as tf

# Q16: VAE on manually defined 4x4 digit-like patterns

X = np.array([
    [1, 1, 1, 1,
     1, 0, 0, 1,
     1, 0, 0, 1,
     1, 1, 1, 1],
    [0, 1, 0, 0,
     1, 1, 0, 0,
     0, 1, 0, 0,
     1, 1, 1, 0],
    [1, 1, 1, 0,
     0, 0, 1, 0,
     1, 1, 1, 0,
     0, 0, 1, 0],
    [1, 1, 1, 1,
     0, 0, 1, 0,
     0, 1, 0, 0,
     1, 1, 1, 1]
], dtype=np.float32)

latent_dim = 2


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


def build_vae(lr):
    # Encoder
    enc_inp = tf.keras.Input(shape=(16,))
    h = tf.keras.layers.Dense(12, activation="relu")(enc_inp)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(h)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(h)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(enc_inp, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    dec_inp = tf.keras.Input(shape=(latent_dim,))
    d = tf.keras.layers.Dense(12, activation="relu")(dec_inp)
    dec_out = tf.keras.layers.Dense(16, activation="sigmoid")(d)
    decoder = tf.keras.Model(dec_inp, dec_out, name="decoder")

    class VAE(tf.keras.Model):
        def __init__(self, encoder_model, decoder_model):
            super().__init__()
            self.encoder_model = encoder_model
            self.decoder_model = decoder_model
            self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

        @property
        def metrics(self):
            return [self.total_loss_tracker]

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean_v, z_log_var_v, z_v = self.encoder_model(data)
                reconstruction = self.decoder_model(z_v)
                rec_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=-1))
                kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var_v - tf.square(z_mean_v) - tf.exp(z_log_var_v), axis=-1))
                total_loss = rec_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            return {"loss": self.total_loss_tracker.result()}

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(lr))
    return vae, encoder, decoder


for lr in [0.001, 0.01]:
    print("\n" + "=" * 70)
    print(f"Training VAE with learning rate = {lr}")
    vae, encoder, decoder = build_vae(lr)
    history = vae.fit(X, epochs=10, batch_size=4, verbose=0)
    print("Loss history:", [round(float(v), 6) for v in history.history["loss"]])

    # Generate samples from latent space
    z_samples = tf.random.normal((5, latent_dim))
    generated = decoder.predict(z_samples, verbose=0)
    generated_bin = (generated >= 0.5).astype(int)
    print("Generated samples (probabilities):\n", np.round(generated, 3))
    print("Generated samples (binary):\n", generated_bin)



# Q17 repeats the same CNN task as Q15. Separate file as requested.

import numpy as np

img0 = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
], dtype=np.float64)

img1 = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1]
], dtype=np.float64)

X = np.stack([img0, img1])
y = np.array([0.0, 1.0], dtype=np.float64)


def conv2d_valid(x, k):
    h, w = x.shape
    kh, kw = k.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i:i + kh, j:j + kw] * k)
    return out


def maxpool2x2(x):
    pooled = np.zeros((2, 2))
    mask = np.zeros_like(x)
    for i in range(2):
        for j in range(2):
            patch = x[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2]
            idx = np.unravel_index(np.argmax(patch), patch.shape)
            pooled[i, j] = patch[idx]
            mask[i * 2 + idx[0], j * 2 + idx[1]] = 1
    return pooled, mask


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train(lr, epochs=10):
    rng = np.random.default_rng(2)
    kernel = rng.normal(0, 0.2, (2, 2))
    w = rng.normal(0, 0.2, (4,))
    b = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        for xi, yi in zip(X, y):
            conv = conv2d_valid(xi, kernel)
            relu = np.maximum(conv, 0)
            pool, pool_mask = maxpool2x2(relu)
            feat = pool.reshape(-1)
            prob = sigmoid(np.dot(feat, w) + b)

            loss = -(yi * np.log(prob + 1e-8) + (1 - yi) * np.log(1 - prob + 1e-8))
            total_loss += loss

            dlogit = prob - yi
            dw = dlogit * feat
            db = dlogit

            dpool = (dlogit * w).reshape(2, 2)
            drelu = np.zeros_like(relu)
            for i in range(2):
                for j in range(2):
                    drelu[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] += dpool[i, j] * pool_mask[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2]

            dconv = drelu * (conv > 0)
            dkernel = np.zeros_like(kernel)
            for i in range(dconv.shape[0]):
                for j in range(dconv.shape[1]):
                    dkernel += dconv[i, j] * xi[i:i + 2, j:j + 2]

            w -= lr * dw
            b -= lr * db
            kernel -= lr * dkernel

        print(f"lr={lr:.2f} | epoch={epoch + 1:2d} | loss={total_loss / len(X):.6f}")

    return kernel, w, b


def predict(x, kernel, w, b):
    conv = conv2d_valid(x, kernel)
    relu = np.maximum(conv, 0)
    pool, _ = maxpool2x2(relu)
    feat = pool.reshape(-1)
    prob = sigmoid(np.dot(feat, w) + b)
    return int(prob >= 0.5), prob


for lr in [0.01, 0.1]:
    print("\n" + "=" * 70)
    kernel, w, b = train(lr=lr, epochs=10)

    new_pattern = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ], dtype=np.float64)

    cls, prob = predict(new_pattern, kernel, w, b)
    print(f"Predicted class for new pattern: {cls} (prob={prob:.4f})")


import numpy as np
import tensorflow as tf

# Q18: Transformer Encoder for Positive/Negative classification

X = np.array([
    [[1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 1]],  # positive
    [[0, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],  # positive
    [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]],  # negative
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]   # negative
], dtype=np.float32)

y = np.array([[1], [1], [0], [0]], dtype=np.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(i / 2.0)) / tf.cast(d_model, tf.float32))
        angles = pos * angle_rates
        sin = tf.sin(angles[:, 0::2])
        cos = tf.cos(angles[:, 1::2])
        pe = tf.concat([sin, cos], axis=-1)
        pe = pe[tf.newaxis, :, :]
        pe = tf.pad(pe, [[0, 0], [0, 0], [0, tf.maximum(0, d_model - tf.shape(pe)[-1])]])
        pe = pe[:, :, :d_model]
        return x + pe


def build_transformer_classifier(lr):
    inp = tf.keras.Input(shape=(3, 4))
    x = PositionalEncoding()(inp)

    # Multi-head self-attention
    attn = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # Feed-forward
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(4)
    ])(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


for lr in [0.001, 0.01]:
    print("\n" + "=" * 70)
    print(f"Training Transformer classifier with lr={lr}")
    model = build_transformer_classifier(lr)
    hist = model.fit(X, y, epochs=10, verbose=0)

    final_loss = hist.history["loss"][-1]
    final_acc = hist.history["accuracy"][-1]
    print(f"Final loss={final_loss:.6f}, Final accuracy={final_acc:.4f}")

    probs = model.predict(X, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    print("Predictions:", preds)
    print("Targets    :", y.ravel().astype(int))

    # New review prediction
    new_review = np.array([[[1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 0]]], dtype=np.float32)
    new_prob = model.predict(new_review, verbose=0)[0, 0]
    print(f"New review class: {int(new_prob >= 0.5)} (prob={new_prob:.4f})")
