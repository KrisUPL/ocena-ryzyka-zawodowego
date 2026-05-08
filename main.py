import numpy as np
import pandas as pd

class SiecNeuronowa:
    def __init__(self, learning_rate=0.5, epochs=10000):
        self.lr = learning_rate
        self.epochs = epochs
        np.random.seed(42)

        self.W1 = np.random.uniform(-1, 1, (3, 3))
        self.b1 = np.zeros((1, 3))
        self.W2 = np.random.uniform(-1, 1, (3, 1))
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_pochodna(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        for epoch in range(self.epochs):
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.sigmoid(Z2)

            error = y - A2
            d_A2 = error * self.sigmoid_pochodna(A2)
            error_hidden = d_A2.dot(self.W2.T)
            d_A1 = error_hidden * self.sigmoid_pochodna(A1)

            self.W2 += A1.T.dot(d_A2) * self.lr
            self.b2 += np.sum(d_A2, axis=0, keepdims=True) * self.lr
            self.W1 += X.T.dot(d_A1) * self.lr
            self.b1 += np.sum(d_A1, axis=0, keepdims=True) * self.lr

    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        return (A2 > 0.5).astype(int)

def uruchom_program():
        print("Wczytuję dane z plku Ocena_ryzyka.csv...")
        try:
            df = pd.read_csv("Ocena_ryzyka.csv")
        except FileNotFoundError:
            print("BŁĄD: Nie znaleziono pliku Ocena_ryzyka.csv w folderze!")
            return

        X_train = df[['S', 'E', 'P']].values
        y_train = np.where(df['Kategoria'] == 'Akceptowalne', 0, 1)

        max_wartosci = np.array([100.0, 10.0, 10.0])
        X_train_norm = X_train / max_wartosci

        print("Trwa nauka sieci neuronowej...")
        ai_bhp = SiecNeuronowa(learning_rate=0.5)
        ai_bhp.fit(X_train_norm, y_train)
        print("Sieć nauczona! Gotowa do działania.\n")

        print("OCENA RYZYKA BHP (RISK SCORE)")
        try:
            s = float(input("Podaj przewidywane Skutki (S): "))
            e = float(input("Podaj wartość Ekspozycji (E): "))
            p = float(input("Podaj Prawodpodobieństwo (P): "))
        except ValueError:
            print("Błąd: Należy podawaćwyłącznie wartości liczbowe.")
            return

        nowy_przypadek = np.array([[s, e, p]])
        nowy_przypadek_norm = nowy_przypadek / max_wartosci

        werdykt = ai_bhp.predict(nowy_przypadek_norm)

        if werdykt[0][0] == 1:
            print(" WYNIK: RYZYKO NIEAKCEPTOWALNE! Wymagana natychmiastowa interwencja. ")        
        else:
            print(" WYNIK: Ryzyko akceptowalne. Stanowisko bezpieczne. ")

if __name__ == "__main__":
    uruchom_program()        # type: ignore