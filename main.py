"""
Punto de entrada principal

Ejecuta experimentos de Federated Learning con redes neuronales
y genera visualizaciones estadísticas.
"""

import argparse
import sys
import os

# Asegura que los módulos locales se puedan importar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Analytics.experiment_runner import run_multiple_experiments, compare_configurations
from Analytics.statistics_engine import compute_epoch_statistics, aggregate_experiments
from Analytics.chart_generator import prepare_accuracy_chart_data, prepare_partition_comparison_data


def run_terminal_mode(args):
    """Ejecuta en modo terminal sin interfaz gráfica."""
    print("=" * 70)
    print("NN_PRACTICA - MODO TERMINAL")
    print("=" * 70)
    print(f"Particiones: {args.partitions}")
    print(f"Épocas: {args.epochs}")
    print(f"Experimentos: {args.experiments}")
    print(f"Neuronas ocultas: {args.hidden_neurons}")
    print(f"Tasa de aprendizaje: {args.learning_rate}")
    print(f"Ejemplos de entrenamiento: {args.n_train}")
    print("=" * 70)
    
    # Ejecuta experimentos
    results = run_multiple_experiments(
        num_partitions=args.partitions,
        num_epochs=args.epochs,
        num_experiments=args.experiments,
        hidden_neurons=args.hidden_neurons,
        learning_rate=args.learning_rate,
        n_train=args.n_train,
        verbose=True
    )
    
    # Calcular estadísticas
    stats = compute_epoch_statistics(results['all_histories'])
    
    # Mostrar resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS")
    print("=" * 70)
    print(f"Precisión final promedio: {stats['mean'][-1]:.2f}%")
    print(f"Desviación estándar final: {stats['std'][-1]:.2f}%")
    print(f"Mejor precisión alcanzada: {max(stats['max']):.2f}%")
    
    # Guarda resultados
    output_file = f"results/experiment_{results['timestamp']}.json"
    print(f"\nResultados guardados en: {output_file}")
    
    # Genera gráfico simple en terminal (ASCII)
    print("\nEvolución de precisión (promedio):")
    for epoch, (mean, std) in enumerate(zip(stats['mean'], stats['std'])):
        bar_length = int(mean / 2)
        bar = "█" * bar_length
        print(f"Época {epoch+1:2d}: {mean:5.2f}% ± {std:4.2f}% {bar}")


def run_interactive_mode():
    """Ejecuta la interfaz gráfica interactiva."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.patches as mpatches
    except ImportError as e:
        print(f"Error: No se pueden cargar las librerías gráficas: {e}")
        print("Instala las dependencias con: pip install matplotlib")
        sys.exit(1)
    
    class FederatedLearningApp:
        def __init__(self, root):
            self.root = root
            self.root.title("NN_practica - Análisis de Federated Learning")
            self.root.geometry("1400x900")
            
            # Datos de ejecuciones previas para comparación
            self.previous_results = []
            self.colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', 
                          '#00BCD4', '#FFEB3B', '#795548', '#607D8B', '#E91E63']
            self.color_index = 0
            
            self._create_ui()
        
        def _create_ui(self):
            # Panel izquierdo: Controles
            control_frame = ttk.Frame(self.root, padding="10")
            control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
            
            ttk.Label(control_frame, text="Parámetros", 
                     font=('Helvetica', 14, 'bold')).pack(pady=10)
            
            # Número de particiones
            ttk.Label(control_frame, text="Particiones:").pack(anchor=tk.W)
            self.partitions_var = tk.IntVar(value=2)
            partitions_scale = ttk.Scale(control_frame, from_=1, to=10, 
                                        orient=tk.HORIZONTAL, 
                                        variable=self.partitions_var,
                                        length=200)
            partitions_scale.pack(fill=tk.X, pady=5)
            ttk.Label(control_frame, textvariable=self.partitions_var).pack()
            
            # Número de épocas
            ttk.Label(control_frame, text="Épocas:").pack(anchor=tk.W, pady=(10,0))
            self.epochs_var = tk.IntVar(value=5)
            epochs_scale = ttk.Scale(control_frame, from_=1, to=50, 
                                    orient=tk.HORIZONTAL, 
                                    variable=self.epochs_var,
                                    length=200)
            epochs_scale.pack(fill=tk.X, pady=5)
            ttk.Label(control_frame, textvariable=self.epochs_var).pack()
            
            # Número de experimentos
            ttk.Label(control_frame, text="Experimentos:").pack(anchor=tk.W, pady=(10,0))
            self.experiments_var = tk.IntVar(value=5)
            experiments_scale = ttk.Scale(control_frame, from_=1, to=20, 
                                         orient=tk.HORIZONTAL, 
                                         variable=self.experiments_var,
                                         length=200)
            experiments_scale.pack(fill=tk.X, pady=5)
            ttk.Label(control_frame, textvariable=self.experiments_var).pack()
            
            # Neuronas ocultas
            ttk.Label(control_frame, text="Neuronas ocultas:").pack(anchor=tk.W, pady=(10,0))
            self.hidden_var = tk.IntVar(value=30)
            hidden_scale = ttk.Scale(control_frame, from_=10, to=100, 
                                    orient=tk.HORIZONTAL, 
                                    variable=self.hidden_var,
                                    length=200)
            hidden_scale.pack(fill=tk.X, pady=5)
            ttk.Label(control_frame, textvariable=self.hidden_var).pack()
            
            # Tasa de aprendizaje
            ttk.Label(control_frame, text="Tasa de aprendizaje:").pack(anchor=tk.W, pady=(10,0))
            self.lr_var = tk.StringVar(value="1.0")
            ttk.Entry(control_frame, textvariable=self.lr_var, width=20).pack(pady=5)
            
            # Ejemplos de entrenamiento
            ttk.Label(control_frame, text="Ejemplos de entrenamiento:").pack(anchor=tk.W, pady=(10,0))
            self.n_train_var = tk.StringVar(value="5000")
            ttk.Entry(control_frame, textvariable=self.n_train_var, width=20).pack(pady=5)
            
            # Botones
            ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
            
            ttk.Button(control_frame, text="▶ Ejecutar Experimento", 
                      command=self._run_experiment).pack(fill=tk.X, pady=5)
            
            ttk.Button(control_frame, text="📊 Comparar Configuraciones", 
                      command=self._compare_configurations).pack(fill=tk.X, pady=5)
            
            ttk.Button(control_frame, text="🗑 Limpiar Gráficos", 
                      command=self._clear_plots).pack(fill=tk.X, pady=5)
            
            ttk.Button(control_frame, text="💾 Guardar Resultados", 
                      command=self._save_results).pack(fill=tk.X, pady=5)
            
            # Panel derecho: Gráficos
            self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
            self.fig.suptitle('Análisis de Federated Learning', fontsize=14, fontweight='bold')
            
            plot_frame = ttk.Frame(self.root)
            plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Barra de estado
            self.status_var = tk.StringVar(value="Listo")
            status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                  relief=tk.SUNKEN, anchor=tk.W)
            status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        def _run_experiment(self):
            """Ejecuta un experimento con los parámetros actuales."""
            try:
                self.status_var.set("Ejecutando experimentos...")
                self.root.update()
                
                params = {
                    'num_partitions': self.partitions_var.get(),
                    'num_epochs': self.epochs_var.get(),
                    'num_experiments': self.experiments_var.get(),
                    'hidden_neurons': self.hidden_var.get(),
                    'learning_rate': float(self.lr_var.get()),
                    'n_train': int(self.n_train_var.get()),
                    'verbose': False
                }
                
                # Ejecuta
                results = run_multiple_experiments(**params)
                
                # Guarda para posible comparación
                self.current_results = results
                self.current_params = params
                
                # Visualiza
                self._plot_results(results, params, comparison=False)
                
                self.status_var.set(f"Completado: Precisión final = {results['final_mean_accuracy']:.2f}%")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error ejecutando experimento:\n{str(e)}")
                self.status_var.set("Error en ejecución")
        
        def _compare_configurations(self):
            """Superpone el resultado actual con ejecuciones anteriores."""
            if not hasattr(self, 'current_results'):
                messagebox.showwarning("Advertencia", "Primero ejecuta un experimento")
                return
            
            # Guarda resultado anterior
            self.previous_results.append({
                'results': self.current_results,
                'params': self.current_params,
                'color': self.colors[self.color_index % len(self.colors)]
            })
            self.color_index += 1
            
            # Ejecuta nuevo experimento
            self._run_experiment()
            
            # Visualiza comparación
            self._plot_comparison()
        
        def _plot_results(self, results, params, comparison=False):
            """Visualiza los resultados de un experimento."""
            ax1, ax2, ax3, ax4 = self.axes.flatten()
            
            # Limpia si no es comparación
            if not comparison:
                for ax in self.axes.flatten():
                    ax.clear()
            
            # Color actual
            color = self.colors[self.color_index % len(self.colors)]
            label = f"P={params['num_partitions']}, E={params['num_epochs']}"
            
            # 1. Evolución del promedio con desviación estándar
            stats = compute_epoch_statistics(results['all_histories'])
            epochs = range(1, len(stats['mean']) + 1)
            
            ax1.plot(epochs, stats['mean'], 'o-', color=color, linewidth=2, label=label)
            ax1.fill_between(epochs, 
                            [m - s for m, s in zip(stats['mean'], stats['std'])],
                            [m + s for m, s in zip(stats['mean'], stats['std'])],
                            alpha=0.2, color=color)
            ax1.set_xlabel('Época')
            ax1.set_ylabel('Precisión (%)')
            ax1.set_title('Evolución del Promedio (con desviación estándar)')
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)
            
            # 2. Comparación por partición (último experimento)
            if results['all_histories']:
                last_exp = results['all_histories'][-1]
                if 'partition_accuracies' in last_exp:
                    for p_idx, p_acc in enumerate(last_exp['partition_accuracies']):
                        ax2.plot(range(1, len(p_acc) + 1), p_acc, 
                                'o-', label=f'Partición {p_idx + 1}', alpha=0.7)
            ax2.set_xlabel('Época')
            ax2.set_ylabel('Precisión (%)')
            ax2.set_title('Comparación por Partición (último experimento)')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)
            
            # 3. Distribución de precisión final
            final_accuracies = [h['accuracies'][-1] for h in results['all_histories']]
            ax3.hist(final_accuracies, bins=10, color=color, alpha=0.7, edgecolor='black')
            ax3.axvline(results['final_mean_accuracy'], color='red', 
                       linestyle='--', linewidth=2, label='Promedio')
            ax3.set_xlabel('Precisión Final (%)')
            ax3.set_ylabel('Frecuencia')
            ax3.set_title('Distribución de Precisión Final')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Convergencia (diferencia entre épocas)
            mean_diffs = [0] + [stats['mean'][i] - stats['mean'][i-1] 
                               for i in range(1, len(stats['mean']))]
            ax4.bar(epochs, mean_diffs, color=color, alpha=0.7)
            ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax4.set_xlabel('Época')
            ax4.set_ylabel('Mejora (%)')
            ax4.set_title('Mejora por Época')
            ax4.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            self.canvas.draw()
        
        def _plot_comparison(self):
            """Visualiza comparación de múltiples configuraciones."""
            if len(self.previous_results) < 1:
                return
            
            ax1 = self.axes.flatten()[0]
            ax1.clear()
            
            # Plotea todas las configuraciones anteriores
            for prev in self.previous_results:
                stats = compute_epoch_statistics(prev['results']['all_histories'])
                epochs = range(1, len(stats['mean']) + 1)
                label = f"P={prev['params']['num_partitions']}, E={prev['params']['num_epochs']}"
                ax1.plot(epochs, stats['mean'], 'o-', color=prev['color'], 
                        linewidth=2, label=label)
            
            # Plotea actual
            if hasattr(self, 'current_results'):
                stats = compute_epoch_statistics(self.current_results['all_histories'])
                epochs = range(1, len(stats['mean']) + 1)
                label = f"P={self.current_params['num_partitions']}, E={self.current_params['num_epochs']} (actual)"
                color = self.colors[self.color_index % len(self.colors)]
                ax1.plot(epochs, stats['mean'], 'o-', color=color, 
                        linewidth=3, label=label, linestyle='--')
            
            ax1.set_xlabel('Época')
            ax1.set_ylabel('Precisión (%)')
            ax1.set_title('Comparación de Configuraciones')
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)
            
            self.canvas.draw()
        
        def _clear_plots(self):
            """Limpia todos los gráficos."""
            for ax in self.axes.flatten():
                ax.clear()
            self.previous_results = []
            self.color_index = 0
            self.canvas.draw()
            self.status_var.set("Gráficos limpiados")
        
        def _save_results(self):
            """Guarda los resultados del experimento actual."""
            if not hasattr(self, 'current_results'):
                messagebox.showwarning("Advertencia", "No hay resultados para guardar")
                return
            
            import json
            from datetime import datetime
            
            filename = f"results/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Asegura que existe el directorio
            os.makedirs('results', exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump({
                    'parameters': self.current_params,
                    'results': {
                        'timestamp': self.current_results['timestamp'],
                        'final_mean_accuracy': self.current_results['final_mean_accuracy'],
                        'final_std_accuracy': self.current_results['final_std_accuracy'],
                        'all_histories': self.current_results['all_histories']
                    }
                }, f, indent=2)
            
            messagebox.showinfo("Éxito", f"Resultados guardados en:\n{filename}")
            self.status_var.set(f"Guardado: {filename}")
    
    # Crea y ejecuta aplicación
    root = tk.Tk()
    app = FederatedLearningApp(root)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description='NN_practica - Análisis de Federated Learning para MNIST'
    )
    
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Ejecutar en modo interactivo con interfaz gráfica')
    parser.add_argument('--partitions', '-p', type=int, default=2,
                       help='Número de particiones (default: 2)')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                       help='Número de épocas (default: 5)')
    parser.add_argument('--experiments', '-x', type=int, default=5,
                       help='Número de experimentos a ejecutar (default: 5)')
    parser.add_argument('--hidden-neurons', '-n', type=int, default=30,
                       help='Número de neuronas en capa oculta (default: 30)')
    parser.add_argument('--learning-rate', '-l', type=float, default=1.0,
                       help='Tasa de aprendizaje (default: 1.0)')
    parser.add_argument('--n-train', type=int, default=5000,
                       help='Número de ejemplos de entrenamiento (default: 5000)')
    
    args = parser.parse_args()
    
    # Crea directorios necesarios
    os.makedirs('Data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    if args.interactive:
        run_interactive_mode()
    else:
        run_terminal_mode(args)