# Raport z projektu AI

Opis problemu: Automatyzacja oceny ryzyka BHP (RISK SCORE). Zgodnie z założeniami, ograniczono wyniki do klasyfikacji binarnej (Ryzyko akceptowalne / Ryzyko nieakceptowalne).

Metoda: Zamiast pojedynczego perceptronu zaimplementowano od zera prostą sieć neuronową z warstwą ukrytą, opartą na operacjach macierzowych (`numpy`). Dane uczące są pobierane z pliku CSV za pomocą biblioteki `pandas` i poddawane normalizacji.

Wnioski: Algorytm z powodzeniem uczy się nieliniowych zależności i poprawnie klasyfikuje wprowadzane przez użytkownika nowe dane.