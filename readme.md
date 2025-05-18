Toto je repozitář s projektem do předmětu Vědecké výpočty v Pythonu. Autor: Tobijas Stonawski, sto0257
Python skript slouží k generování, načítání, řešení a vizualizaci bludišť reprezentovaných jako binární matice.

Repozitář obsahuje:
    Soubor "readme.md"
    Zdrojový kód "MazePath.py"
    Jupyter Notebook "examples.ipynb"
    Složku "solved" s ukázkovými možnými výstupy

Funkcionality:
    maze_from_csv(path)
    Načte bludiště z CSV souboru a převede ho na binární matici (False = průchozí, True = zeď).

    adj_matrix(maze)
    Vytvoří sousednostní matici grafu na základě průchodnosti v bludišti (4-směrné sousedství).

    shortest_path(adjacency, start, end)
    Najde nejkratší cestu v bludišti mezi zadanými indexy pomocí algoritmu BFS.

    maze_with_path(maze, path)
    Vykreslí bludiště a zvýrazní nalezenou cestu pomocí knihovny PIL. Cesta je označena červeně.

    generate_maze(n, base, density)
    Vygeneruje nové bludiště dané velikosti n × n, založené na šabloně (base) s náhodně přidanými zdmi (density).

    create_template(n, base)
    Vytvoří základní šablonu bludiště. K dispozici jsou typy:
    "hslalom", "vslalom", "sslalom", "snake".

    solve_and_show(maze)
    Pokusí se bludiště vyřešit a vizuálně zobrazí výsledek s cestou.