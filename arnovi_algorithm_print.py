"""
=============================================================================
RED NEURONAL DESDE CERO CON EL "ALGORITMO DE ARNOVI"
=============================================================================
Autor: [Tu nombre]
Descripción: 
    Entrenamiento de una red neuronal clásica (fully connected) para
    clasificar dígitos escritos a mano (MNIST). Implementado desde cero
    usando solo NumPy, sin librerías de deep learning.
    
    Incluye el "Algoritmo de Arnovi": entrenamiento por particiones
    independientes con promediado final de parámetros.

Arquitectura:
    Entrada (784) → Capa Oculta (30, sigmoide) → Salida (10, softmax)

Método de optimización:
    Gradiente descendente completo (batch gradient descent, NO estocástico)

Requisitos:
    pip install numpy matplotlib torchvision rich
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from torchvision import datasets


# Objeto Console de rich: es un "print mejorado" que entiende colores.
# Ejemplo: con.print("[bold red]Error[/bold red]") imprime "Error" en rojo negrita.
# Los colores se ponen entre corchetes como etiquetas HTML: [color]texto[/color]
con = Console()

# =====================================================================
# CONFIGURACIÓN GENERAL
# =====================================================================
NEURONAS_ENTRADA = 784
NEURONAS_OCULTA = 30
NEURONAS_SALIDA = 10
LEARNING_RATE = 0.5
EPOCAS = 50
NUM_PARTICIONES = 2
NUM_REPETICIONES = 10


# =====================================================================
# CARGAR MNIST
# =====================================================================
def cargar_mnist():    
    dataset_train = datasets.MNIST(root='./datos_mnist', train=True, 
                                   download=True, transform=None)
    dataset_test = datasets.MNIST(root='./datos_mnist', train=False, 
                                  download=True, transform=None)
    
    imagenes_train = []
    etiquetas_train = []
    for i in range(len(dataset_train)): 
        imagen, etiqueta = dataset_train[i]
        imagenes_train.append(np.array(imagen).flatten() / 255.0)
        etiquetas_train.append(etiqueta)
    
    imagenes_test = []
    etiquetas_test = []
    for i in range(len(dataset_test)):
        imagen, etiqueta = dataset_test[i]
        imagenes_test.append(np.array(imagen).flatten() / 255.0)
        etiquetas_test.append(etiqueta)
    
    X_train = np.array(imagenes_train).T
    Y_train = np.array(etiquetas_train)
    X_test = np.array(imagenes_test).T
    Y_test = np.array(etiquetas_test)
    
    return X_train, Y_train, X_test, Y_test


# =====================================================================
# CREAR PARTICIONES
# =====================================================================
def crear_particiones(X_train, Y_train, n_particiones):
    m = X_train.shape[1]
    indices = np.random.permutation(m)
    X_mezclado = X_train[:, indices]
    Y_mezclado = Y_train[indices]
    
    tamano_particion = m // n_particiones
    particiones = []
    
    for i in range(n_particiones):
        inicio = i * tamano_particion
        fin = (i + 1) * tamano_particion
        Xi = X_mezclado[:, inicio:fin]
        Yi = Y_mezclado[inicio:fin]
        particiones.append((Xi, Yi))
        
        # Convertimos a int nativo de Python porque np.int64 genera
        # texto como [np.int64(0), ...] y rich interpreta los corchetes []
        # como etiquetas de formato, "comiéndose" el contenido.
        digitos = [int(d) for d in sorted(np.unique(Yi))]
        con.print(f"    Partición [cyan]{i+1}[/cyan]: "
                  f"[white]{Xi.shape[1]}[/white] imágenes, "
                  f"dígitos: [green]{digitos}[/green]")
    
    return particiones


# =====================================================================
# INICIALIZACIÓN DE PARÁMETROS (Xavier)
# =====================================================================
def inicializar_parametros():
    np.random.seed(None)
    W1 = np.random.randn(NEURONAS_OCULTA, NEURONAS_ENTRADA) * np.sqrt(1.0 / NEURONAS_ENTRADA)
    b1 = np.zeros((NEURONAS_OCULTA, 1))
    W2 = np.random.randn(NEURONAS_SALIDA, NEURONAS_OCULTA) * np.sqrt(1.0 / NEURONAS_OCULTA)
    b2 = np.zeros((NEURONAS_SALIDA, 1))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


# =====================================================================
# FUNCIONES DE ACTIVACIÓN
# =====================================================================
def sigmoide(Z):
    return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))

def sigmoide_derivada(A):
    return A * (1 - A)

def softmax(Z):
    Z_estable = Z - np.max(Z, axis=0, keepdims=True)
    exponenciales = np.exp(Z_estable)
    return exponenciales / np.sum(exponenciales, axis=0, keepdims=True)


# =====================================================================
# ONE-HOT
# =====================================================================
def one_hot(Y, num_clases=10):
    m = Y.shape[0]
    Y_one_hot = np.zeros((num_clases, m))
    Y_one_hot[Y, np.arange(m)] = 1
    return Y_one_hot


# =====================================================================
# FORWARD PROPAGATION
# =====================================================================
def forward_propagation(X, parametros):
    Z1 = parametros['W1'].dot(X) + parametros['b1']
    A1 = sigmoide(Z1)
    Z2 = parametros['W2'].dot(A1) + parametros['b2']
    A2 = softmax(Z2)
    return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}


# =====================================================================
# FUNCIÓN DE COSTO
# =====================================================================
def calcular_costo(A2, Y_one_hot):
    m = Y_one_hot.shape[1]
    return -(1.0 / m) * np.sum(Y_one_hot * np.log(np.clip(A2, 1e-10, 1.0)))


# =====================================================================
# BACKWARD PROPAGATION
# =====================================================================
def backward_propagation(X, Y_one_hot, cache, parametros):
    m = X.shape[1]
    dZ2 = cache['A2'] - Y_one_hot
    dW2 = (1.0 / m) * dZ2.dot(cache['A1'].T)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = parametros['W2'].T.dot(dZ2) * sigmoide_derivada(cache['A1'])
    dW1 = (1.0 / m) * dZ1.dot(X.T)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


# =====================================================================
# ACTUALIZACIÓN DE PARÁMETROS
# =====================================================================
def actualizar_parametros(parametros, gradientes, learning_rate):
    parametros['W1'] -= learning_rate * gradientes['dW1']
    parametros['b1'] -= learning_rate * gradientes['db1']
    parametros['W2'] -= learning_rate * gradientes['dW2']
    parametros['b2'] -= learning_rate * gradientes['db2']
    return parametros


# =====================================================================
# CALCULAR PRECISIÓN
# =====================================================================
def calcular_precision(X, Y, parametros):
    cache = forward_propagation(X, parametros)
    predicciones = np.argmax(cache['A2'], axis=0)
    return np.sum(predicciones == Y) / Y.shape[0]


# =====================================================================
# ENTRENAR UNA PARTICIÓN
# =====================================================================
def entrenar_particion(X, Y, parametros, learning_rate, epocas, id_particion):
    Y_one_hot = one_hot(Y)
    historial_precision = []
    historial_costo = []
    
    con.print(f"\n  [bold cyan]▶ Entrenando Partición {id_particion}[/bold cyan] "
              f"([white]{X.shape[1]}[/white] imágenes)")
    
    tabla = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    tabla.add_column("Época", style="cyan", justify="center", width=8)
    tabla.add_column("Costo", style="yellow", justify="center", width=12)
    tabla.add_column("Precisión", justify="center", width=12)
    tabla.add_column("Barra", justify="center", width=18)
    
    for epoca in range(1, epocas + 1):
        cache = forward_propagation(X, parametros)
        costo = calcular_costo(cache['A2'], Y_one_hot)
        gradientes = backward_propagation(X, Y_one_hot, cache, parametros)
        parametros = actualizar_parametros(parametros, gradientes, learning_rate)
        precision = calcular_precision(X, Y, parametros)
        historial_precision.append(precision)
        historial_costo.append(costo)
        
        # Barra visual
        llenos = int(precision * 15)
        barra = "█" * llenos + "░" * (15 - llenos)
        
        # Color según calidad
        if precision >= 0.9:
            cp = "bold green"
        elif precision >= 0.7:
            cp = "yellow"
        else:
            cp = "red"
        
        tabla.add_row(
            f"{epoca}/{epocas}",
            f"{costo:.4f}",
            f"[{cp}]{precision * 100:.2f}%[/{cp}]",
            f"[green]{barra}[/green]"
        )
    
    con.print(tabla)
    return parametros, historial_precision, historial_costo


# =====================================================================
# ALGORITMO DE ARNOVI
# =====================================================================
def algoritmo_de_arnovi(lista_parametros):
    n = len(lista_parametros)
    
    con.print(Panel(
        f"[bold]Promediando [cyan]{n}[/cyan] conjuntos de parámetros[/bold]\n"
        f"Fórmula: W_prom = (1/{n}) × (W₁ + W₂ + ... + W_{n})",
        title="[bold yellow]⚡ ALGORITMO DE ARNOVI[/bold yellow]",
        border_style="yellow"
    ))
    
    parametros_promedio = {}
    for clave in ['W1', 'b1', 'W2', 'b2']:
        suma = np.zeros_like(lista_parametros[0][clave])
        for i in range(n):
            suma += lista_parametros[i][clave]
        parametros_promedio[clave] = suma * (1.0 / n)
        con.print(f"    [green]✓[/green] {clave}: promedio → {parametros_promedio[clave].shape}")
    
    return parametros_promedio


# =====================================================================
# UNA EJECUCIÓN COMPLETA
# =====================================================================
def ejecutar_una_vez(X_train, Y_train, X_test, Y_test, numero_ejecucion):
    con.rule(f"[bold blue] EJECUCIÓN {numero_ejecucion} [/bold blue]")
    
    particiones = crear_particiones(X_train, Y_train, NUM_PARTICIONES)
    parametros_iniciales = inicializar_parametros()
    
    lista_parametros_entrenados = []
    todos_historiales_precision = []
    todos_historiales_costo = []
    
    for i, (Xi, Yi) in enumerate(particiones):
        params_copia = {k: v.copy() for k, v in parametros_iniciales.items()}
        params_ent, hist_prec, hist_costo = entrenar_particion(
            Xi, Yi, params_copia, LEARNING_RATE, EPOCAS, i + 1
        )
        lista_parametros_entrenados.append(params_ent)
        todos_historiales_precision.append(hist_prec)
        todos_historiales_costo.append(hist_costo)
    
    if NUM_PARTICIONES > 1:
        parametros_finales = algoritmo_de_arnovi(lista_parametros_entrenados)
    else:
        parametros_finales = lista_parametros_entrenados[0]
    
    precision_test = calcular_precision(X_test, Y_test, parametros_finales)
    
    if precision_test >= 0.9:
        c = "bold green"
    elif precision_test >= 0.8:
        c = "bold yellow"
    else:
        c = "bold red"
    
    con.print(f"\n  Ejecución {numero_ejecucion} → Precisión test: "
              f"[{c}]{precision_test * 100:.2f}%[/{c}]")
    
    return precision_test, todos_historiales_precision, todos_historiales_costo


# =====================================================================
# PROGRAMA PRINCIPAL
# =====================================================================
def main():
    con.print(Panel(
        f"[bold white]Arquitectura:[/bold white] {NEURONAS_ENTRADA} → {NEURONAS_OCULTA} → {NEURONAS_SALIDA}\n"
        f"[bold white]Learning rate:[/bold white] {LEARNING_RATE}\n"
        f"[bold white]Épocas:[/bold white] {EPOCAS}\n"
        f"[bold white]Particiones:[/bold white] {NUM_PARTICIONES}\n"
        f"[bold white]Repeticiones:[/bold white] {NUM_REPETICIONES}",
        title="[bold cyan]🧠 RED NEURONAL — ALGORITMO DE ARNOVI[/bold cyan]",
        border_style="cyan", expand=False
    ))
    
    # Cargar datos una sola vez
    con.print("\n[bold green]📦 Cargando dataset MNIST...[/bold green]")
    X_train, Y_train, X_test, Y_test = cargar_mnist()
    con.print(f"    Entrenamiento: [cyan]{X_train.shape[1]}[/cyan] imágenes")
    con.print(f"    Prueba: [cyan]{X_test.shape[1]}[/cyan] imágenes")
    
    # Ejecutar N veces
    con.print(f"\n[bold green]🔄 Ejecutando {NUM_REPETICIONES} repeticiones...[/bold green]")
    
    todas_las_precisiones = []
    ultimo_hist_prec = None
    ultimo_hist_costo = None
    
    for rep in range(1, NUM_REPETICIONES + 1):
        prec, hp, hc = ejecutar_una_vez(X_train, Y_train, X_test, Y_test, rep)
        todas_las_precisiones.append(prec)
        ultimo_hist_prec = hp
        ultimo_hist_costo = hc
    
    # Estadísticas
    precisiones = np.array(todas_las_precisiones)
    promedio = np.mean(precisiones)
    desv_est = np.std(precisiones, ddof=0)
    rsd = (desv_est / promedio) * 100 if promedio > 0 else 0.0
    
    # Tabla de resultados individuales
    tabla_res = Table(
        title="📊 Resultados Individuales",
        box=box.DOUBLE_EDGE, show_header=True,
        header_style="bold white on blue"
    )
    tabla_res.add_column("Ejecución", style="cyan", justify="center")
    tabla_res.add_column("Precisión Test", justify="center")
    
    for i, p in enumerate(todas_las_precisiones):
        if p >= 0.9:
            s = "[bold green]"
        elif p >= 0.8:
            s = "[yellow]"
        else:
            s = "[red]"
        tabla_res.add_row(f"{i+1}", f"{s}{p * 100:.2f}%{s.replace('[', '[/')}")
    
    con.print()
    con.print(tabla_res)
    
    # Interpretación RSD
    if rsd < 2:
        interp = "[bold green]MUY ESTABLE[/bold green] — variabilidad mínima"
        interp_plain = "MUY ESTABLE"
    elif rsd < 5:
        interp = "[green]ESTABLE[/green] — variabilidad aceptable"
        interp_plain = "ESTABLE"
    elif rsd < 10:
        interp = "[yellow]MODERADO[/yellow] — variabilidad notable"
        interp_plain = "MODERADO"
    else:
        interp = "[bold red]INESTABLE[/bold red] — alta variabilidad"
        interp_plain = "INESTABLE"
    
    # Panel estadístico
    con.print(Panel(
        f"[bold]Promedio:[/bold]              [bold cyan]{promedio * 100:.2f}%[/bold cyan]\n"
        f"[bold]Desviación estándar:[/bold]   [yellow]{desv_est * 100:.2f}%[/yellow]\n"
        f"[bold]RSD:[/bold]                   [magenta]{rsd:.2f}%[/magenta]\n"
        f"[bold]Mínimo:[/bold]                {np.min(precisiones) * 100:.2f}%\n"
        f"[bold]Máximo:[/bold]                {np.max(precisiones) * 100:.2f}%\n"
        f"[bold]Rango:[/bold]                 {(np.max(precisiones) - np.min(precisiones)) * 100:.2f}%\n"
        f"\n[bold]Interpretación RSD:[/bold]   {interp}",
        title="[bold white on magenta] 📈 RESUMEN ESTADÍSTICO [/bold white on magenta]",
        border_style="magenta"
    ))
    
    # Gráficas
    con.print("\n[bold green]📊 Generando gráficas...[/bold green]")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colores_graf = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    
    for i, hist in enumerate(ultimo_hist_prec):
        c = colores_graf[i % len(colores_graf)]
        axes[0].plot(range(1, EPOCAS + 1), [p * 100 for p in hist],
                     marker='o', markersize=4, color=c,
                     label=f'Partición {i+1}', linewidth=2)
    axes[0].set_xlabel('Época'); axes[0].set_ylabel('Precisión (%)')
    axes[0].set_title('Precisión por Época (Última ejecución)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 100)
    
    for i, hist in enumerate(ultimo_hist_costo):
        c = colores_graf[i % len(colores_graf)]
        axes[1].plot(range(1, EPOCAS + 1), hist,
                     marker='o', markersize=4, color=c,
                     label=f'Partición {i+1}', linewidth=2)
    axes[1].set_xlabel('Época'); axes[1].set_ylabel('Costo')
    axes[1].set_title('Costo por Época (Última ejecución)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    ejecs = range(1, NUM_REPETICIONES + 1)
    precs_pct = [p * 100 for p in todas_las_precisiones]
    bar_cols = ['#4CAF50' if p >= 90 else '#FF9800' if p >= 80 else '#F44336' for p in precs_pct]
    axes[2].bar(ejecs, precs_pct, color=bar_cols, alpha=0.8, edgecolor='black')
    axes[2].axhline(y=promedio*100, color='red', linestyle='--', linewidth=2,
                    label=f'Promedio: {promedio*100:.2f}%')
    axes[2].axhline(y=(promedio+desv_est)*100, color='orange', linestyle=':', linewidth=1.5, label='±1σ')
    axes[2].axhline(y=(promedio-desv_est)*100, color='orange', linestyle=':', linewidth=1.5)
    axes[2].set_xlabel('Ejecución'); axes[2].set_ylabel('Precisión Test (%)')
    axes[2].set_title(f'10 Repeticiones (RSD: {rsd:.2f}% — {interp_plain})')
    axes[2].set_xticks(list(ejecs)); axes[2].legend(); axes[2].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(
        f'Algoritmo de Arnovi — {NUM_PARTICIONES} particiones, '
        f'LR={LEARNING_RATE}, {EPOCAS} épocas\n'
        f'Promedio: {promedio*100:.2f}% ± {desv_est*100:.2f}% (RSD: {rsd:.2f}%)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('grafica_entrenamiento.png', dpi=150, bbox_inches='tight')
    plt.close()
    con.print("    [green]✓[/green] Gráfica guardada como [cyan]'grafica_entrenamiento.png'[/cyan]")
    
    # Resumen final
    con.print(Panel(
        f"[bold]Particiones:[/bold]     {NUM_PARTICIONES}\n"
        f"[bold]Imágenes/part.:[/bold]  {X_train.shape[1] // NUM_PARTICIONES}\n"
        f"[bold]Épocas:[/bold]          {EPOCAS}\n"
        f"[bold]Learning rate:[/bold]   {LEARNING_RATE}\n"
        f"[bold]Repeticiones:[/bold]    {NUM_REPETICIONES}\n"
        f"\n[bold]PROMEDIO TEST:[/bold]   [bold cyan]{promedio*100:.2f}%[/bold cyan]\n"
        f"[bold]DESV. ESTÁNDAR:[/bold]  [yellow]{desv_est*100:.2f}%[/yellow]\n"
        f"[bold]RSD:[/bold]             [magenta]{rsd:.2f}%[/magenta]",
        title="[bold white on green] ✅ RESUMEN FINAL [/bold white on green]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()