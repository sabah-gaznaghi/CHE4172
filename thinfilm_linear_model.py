import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('thinfilm_spin.csv')
X = df[['spin_speed_rpm']]
y = df['film_thickness_nm']

# Train model
model = LinearRegression()
model.fit(X, y)

print(f'Coefficient (nm per rpm): {model.coef_[0]:.3f}')
print(f'Intercept (nm): {model.intercept_:.1f}')
print(f'R^2: {model.score(X, y):.3f}')

# Predict new values
new_speeds = pd.DataFrame({'spin_speed_rpm': [1000, 2000, 2500]})
print('Predicted thickness (nm):', model.predict(new_speeds).round(1))

# Plot fit
plt.scatter(X, y, label='Data')
plt.plot(X, model.predict(X), label='Linear fit')
plt.xlabel('Spin speed (rpm)')
plt.ylabel('Film thickness (nm)')
plt.legend()
plt.tight_layout()
plt.savefig('thinfilm_fit.png')
