
---
title: "Metody głębokiego uczenia, projekt nr 1"
subtitle: "Własna implementacja algorytmu wstecznej propagacji błędu w perceptronie wielowarstwowym (MLP)"
author:
- Tymoteusz Makowski
- Olaf Skrabacz
date: "19 marca 2019"
documentclass: scrartcl
---
\thispagestyle{empty}
\newpage

# Opis zadania

Celem projektu była implementacja perceptronu wielowarstwowego (ang. *multilayer perceptron*) z szeregiem wymaganych funkcjonalności takich jak:

* wybór liczby warstw oraz liczby neuronów ukrytych w każdej warstwie,
* wybór funkcji aktywacji,
* możliwość ustawienia:
    * liczby iteracji,
    * wartości współczynnika nauki (ang. *learning rate*),
    * wartości współczynnika bezwładności,
* możliwość zastosowania sieci zarówno do klasyfikacji, jak i do regresji.

# Implementacja

Do wykonania zadania projektowego wybraliśmy język programowania Python3 i skorzystaliśmy z jego możliwości obiektowych.

## Funkcje aktywacji

Zaimplementowaliśmy wiele funkcji aktywacji, które można wybierać dla poszczególnych warstw. Oprócz funkcji liniowej zaimplementowaliśmy:

### *ReLU* (Rectified Linear Unit)
\begin{equation}
\mathrm{relu}(x) = \begin{cases}
x, & x>0\\
0, & x\leq0
\end{cases}
\end{equation}

### Funkcja sigmoidalna
\begin{equation}
\mathrm{sigmoid}(x) = \frac{\mathrm{e}^x}{1 + \mathrm{e}^x}
\end{equation}

### Funkcja *tanh*
\begin{equation}
\tanh(x) = \frac{2}{1 + \mathrm{e}^{-2x}} - 1
\end{equation}

### Funkcja wektorowa *softmax*
\begin{equation}
\mathrm{softmax}\big( (x_i)_{i=1}^n \big) = \bigg( \frac{\mathrm{e}^{x_i}}{\sum_{j=1}^n \mathrm{e}^{x_j}} \bigg)_{i=1}^n
\end{equation}

## Klasa warstwy *(Layer)*

Podczas tworzenia każdej z warstw podajemy następujące parametry:

* liczba neuronów, którą ma zawierać ta warstwa,
* liczba neuronów poprzedniej warstwy albo, w przypadku pierwszej warstwy, wymiar danych wejściowych,
* jedna z funkcji aktywacji wymienionych powyżej.
 
Przykład tworzenia warstwy o 3 neuronach, gdzie dane wejściowe mają dwa wymiary (albo poprzednia warstwa ma dwa neurony), a funkcją aktywacji jest funkcja sigmoidalna:

>     Layer(3, 2, "sigmoid")

Klasa *Layer* nie zawiera metod, które są wykorzystywane z perspektywy użytkownika.

## Klasa sieci *(NeuralNetwork)*

Konstruktor klasy *NeuralNetwork* przyjmuje następujące parametry:
