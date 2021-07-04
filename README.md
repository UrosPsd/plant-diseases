# Klasifikacija bolesti biljaka na osnovu slike lista

## **Članovi tima:**

- Uroš Petrić, SW61-2018, grupa 4

- Ivan Luburić,  SW13-2018, grupa 1

## **Asistent:**
Veljko Maksimović

## **Problem koji se rešava:**
Sa slike na kojoj se nalazi list potrebno ga je detektovati i na osnovu njega prepoznati njegova oboljenja.

## **Algoritam:**
Konvolucione neuronske mreže

## **Podaci koji se koriste:**
Podaci su preuzeti sa [linka](https://data.mendeley.com/datasets/tywbtsjrjv/1) koji sadrži 62000 različitih slika podeljenih u 39 vrsta koje predstavljaju oboljenja na biljkama.

## **Metrika za merenje performansi:**
Glavni parametar za merenje performansi će biti predstavljen kroz procenat tačnosti koji se dobija poređenjem rezultata algoritma sa datim rezultatima u dataset-u.

## **Validacija rešenja:**

- Trening skup: 60%

- Validacioni skup: 25%

- Test skup: 15%

## **Pokretanje aplikacije:**
Pre pokretanja aplikacije neophodno je preuzeti istrenirani model sa [linka](https://wetransfer.com/downloads/75bbd90e8159869715c543a69d038e2b20210704153021/f5d06e) i sačuvati ga u plant-diseases-project folder. Zatim je potrebno importovati sve navedene biblioteke na početku koda.
Prilikom pokretanja aplikacije je potrebno uneti naziv slike za testiranje u main funkciji main.py file-a a zatim pokrenuti isti taj file.
